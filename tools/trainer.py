import argparse
import time
import logging
import os
import sys
import math
import yaml
import datetime

import numpy as np
import mxnet as mx
from mxnet import gluon, nd
from mxnet import autograd
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms
from nvidia.dali.plugin.mxnet import DALIClassificationIterator

sys.path.insert(0, os.path.expanduser('~/gluon_classifier'))
from lib.data.loader import HybridTrainPipe, HybridValPipe
from lib.modelzoo.modelzoo import get_model
from lib.utils.lr_scheduler import LRSequential, LRScheduler
from lib.utils.logger import build_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model for image classification.')
    parser.add_argument('--rec-train', type=str, default='~/.mxnet/datasets/imagenet/rec/train.rec',
                        help='the training data')
    parser.add_argument('--rec-train-idx', type=str, default='~/.mxnet/datasets/imagenet/rec/train.idx',
                        help='the index of training data')
    parser.add_argument('--rec-val', type=str, default='~/.mxnet/datasets/imagenet/rec/val.rec',
                        help='the validation data')
    parser.add_argument('--rec-val-idx', type=str, default='~/.mxnet/datasets/imagenet/rec/val.idx',
                        help='the index of validation data')
    parser.add_argument('--dali', action='store_true',
                        help='if use dali dataloder.')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='training batch size per device (CPU/GPU).')
    parser.add_argument('--dtype', type=str, default='float32',
                        help='data type for training. default is float32')
    parser.add_argument('--gpus', type=str, default='0,1,2,3',
                        help='number of gpus to use.')
    parser.add_argument('--num-workers', type=int, default=16,
                        help='number of preprocessing workers')
    parser.add_argument('--num-epochs', type=int, default=3,
                        help='number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate. default is 0.1.')
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help='optimizer. default is sgd.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum value for optimizer, default is 0.9.')
    parser.add_argument('--wd', type=float, default=0.0001,
                        help='weight decay rate. default is 0.0001.')
    parser.add_argument('--lr-mode', type=str, default='step',
                        help='learning rate scheduler mode. options are step, poly and cosine.')
    parser.add_argument('--lr-decay', type=float, default=0.1,
                        help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--lr-decay-period', type=int, default=0,
                        help='interval for periodic learning rate decays. default is 0 to disable.')
    parser.add_argument('--lr-decay-epoch', type=str, default='40,60',
                        help='epochs at which learning rate decays. default is 40,60.')
    parser.add_argument('--warmup-lr', type=float, default=0.0,
                        help='starting warmup learning rate. default is 0.0.')
    parser.add_argument('--warmup-epochs', type=int, default=0,
                        help='number of warmup epochs.')
    parser.add_argument('--last-gamma', action='store_true',
                        help='whether to init gamma of the last BN layer in each bottleneck to 0.')
    parser.add_argument('--model', type=str, required=True,
                        help='type of model to use. see vision_model for options.')
    parser.add_argument('--input-size', type=int, default=224,
                        help='size of the input image size. default is 224')
    parser.add_argument('--crop-ratio', type=float, default=0.875,
                        help='Crop ratio during validation. default is 0.875')
    parser.add_argument('--hybridize', action='store_true',
                        help='if hybridize the model.')
    parser.add_argument('--use-pretrained', action='store_true',
                        help='enable using pretrained model from gluon.')
    parser.add_argument('--use_se', action='store_true',
                        help='use SE layers or not in resnext. default is false.')
    parser.add_argument('--mixup', action='store_true',
                        help='whether train the model with mix-up. default is false.')
    parser.add_argument('--mixup-alpha', type=float, default=0.2,
                        help='beta distribution parameter for mixup sampling, default is 0.2.')
    parser.add_argument('--mixup-off-epoch', type=int, default=0,
                        help='how many last epochs to train without mixup, default is 0.')
    parser.add_argument('--label-smoothing', action='store_true',
                        help='use label smoothing or not in training. default is false.')
    parser.add_argument('--no-wd', action='store_true',
                        help='whether to remove weight decay on bias, and beta/gamma for batchnorm layers.')
    parser.add_argument('--teacher', type=str, default=None,
                        help='teacher model for distillation training')
    parser.add_argument('--temperature', type=float, default=20,
                        help='temperature parameter for distillation teacher model')
    parser.add_argument('--hard-weight', type=float, default=0.5,
                        help='weight for the loss of one-hot label for distillation training')
    parser.add_argument('--batch-norm', action='store_true',
                        help='enable batch normalization or not in vgg. default is false.')
    parser.add_argument('--log-interval', type=int, default=50,
                        help='frequency of log showing.')
    parser.add_argument('--save-frequency', type=int, default=10,
                        help='frequency of model saving.')
    parser.add_argument('--save-dir', type=str, default='params',
                        help='directory of saved models')
    parser.add_argument('--resume-epoch', type=int, default=0,
                        help='epoch to resume training from.')
    parser.add_argument('--resume-params', type=str, default='',
                        help='path of parameters to load from.')
    parser.add_argument('--resume-states', type=str, default='',
                        help='path of trainer state to load from.')
    parser.add_argument('--use-gn', action='store_true',
                        help='whether to use group norm.')
    opt = parser.parse_args()
    return opt


def get_lr_scheduler(opt):
    lr_decay = opt.lr_decay
    lr_decay_period = opt.lr_decay_period
    lr_decay_epoch = [int(i) for i in opt.lr_decay_epoch.split(',')]
    lr_decay_epoch = [e - opt.warmup_epochs for e in lr_decay_epoch]
    num_training_samples = 1281167
    num_batches = num_training_samples // batch_size

    lr_scheduler = LRSequential([
        LRScheduler('linear', base_lr=0, target_lr=opt.lr,
                    nepochs=opt.warmup_epochs, iters_per_epoch=num_batches),
        LRScheduler(opt.lr_mode, base_lr=opt.lr, target_lr=0,
                    nepochs=opt.num_epochs - opt.warmup_epochs,
                    iters_per_epoch=num_batches,
                    step_epoch=lr_decay_epoch,
                    step_factor=lr_decay, power=2)
    ])

    return lr_scheduler

def build_net(ctx, opt):
    model_name = opt.model

    # if model_name == 'shufflenet':
    # net = get_shufflenet0_5(num_classes=1000)
    net = get_model(opt.model, num_classes=1000)
    logging.info('network {} built.'.format(opt.model))
    # else:
    # kwargs = {'ctx': ctx, 'classes': 1000}
    # net = get_model(model_name, **kwargs)
    # net.cast(opt.dtype)

    net.initialize(mx.init.MSRAPrelu(), ctx=ctx)
    logging.info('net built.')

    return net

def get_dali_dataloder(batch_size, ctx, opt):
    rec_train = os.path.expanduser(opt.rec_train)
    rec_train_idx = os.path.expanduser(opt.rec_train_idx)
    rec_val = os.path.expanduser(opt.rec_val)
    rec_val_idx = os.path.expanduser(opt.rec_val_idx)
    input_size = opt.input_size
    num_devices = len(ctx)
    
    trainpipes = [HybridTrainPipe(rec_path=rec_train,
                                  index_path=rec_train_idx,
                                  batch_size=batch_size,
                                  input_size=input_size,
                                  num_gpus=num_devices,
                                  num_threads=16,
                                  device_id=i) for i in range(num_devices)]
    valpipes = [HybridValPipe(rec_path=rec_val,
                              index_path=rec_val_idx,
                              batch_size=batch_size,
                              input_size=input_size,
                              num_gpus=num_devices,
                              num_threads=16,
                              device_id=i) for i in range(num_devices)]
    
    trainpipes[0].build()
    valpipes[0].build()
    
    train_loader = DALIClassificationIterator(trainpipes, trainpipes[0].epoch_size("Reader"))
    val_loader = DALIClassificationIterator(valpipes, valpipes[0].epoch_size("Reader"))
    
    return train_loader, val_loader


def get_data_rec(batch_size, opt):

    rec_train = os.path.expanduser(opt.rec_train)
    rec_train_idx = os.path.expanduser(opt.rec_train_idx)
    rec_val = os.path.expanduser(opt.rec_val)
    rec_val_idx = os.path.expanduser(opt.rec_val_idx)
    input_size = opt.input_size
    num_workers = opt.num_workers
    jitter_param = 0.4
    lighting_param = 0.1
    resize = int(math.ceil(input_size / opt.crop_ratio))
    mean_rgb = [123.68, 116.779, 103.939]
    std_rgb = [58.393, 57.12, 57.375]

    train_data = mx.io.ImageRecordIter(
        path_imgrec         = rec_train,
        path_imgidx         = rec_train_idx,
        preprocess_threads  = num_workers,
        shuffle             = True,
        batch_size          = batch_size,

        data_shape          = (3, input_size, input_size),
        mean_r              = mean_rgb[0],
        mean_g              = mean_rgb[1],
        mean_b              = mean_rgb[2],
        std_r               = std_rgb[0],
        std_g               = std_rgb[1],
        std_b               = std_rgb[2],
        rand_mirror         = True,
        random_resized_crop = True,
        max_aspect_ratio    = 4. / 3.,
        min_aspect_ratio    = 3. / 4.,
        max_random_area     = 1,
        min_random_area     = 0.08,
        brightness          = jitter_param,
        saturation          = jitter_param,
        contrast            = jitter_param,
        pca_noise           = lighting_param,
    )
    
    val_data = mx.io.ImageRecordIter(
        path_imgrec         = rec_val,
        path_imgidx         = rec_val_idx,
        preprocess_threads  = num_workers,
        shuffle             = False,
        batch_size          = batch_size,

        resize              = resize,
        data_shape          = (3, input_size, input_size),
        mean_r              = mean_rgb[0],
        mean_g              = mean_rgb[1],
        mean_b              = mean_rgb[2],
        std_r               = std_rgb[0],
        std_g               = std_rgb[1],
        std_b               = std_rgb[2],
    )

    logging.info('data loader initialized.')
    
    return train_data, val_data


def validation(val_data, ctx, opt):
    val_data.reset()
    acc_top1 = mx.metric.Accuracy()
    acc_top5 = mx.metric.TopKAccuracy(5)
    for i, batch in enumerate(val_data):
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        outputs = [net(X.astype(opt.dtype, copy=False)) for X in data]
        acc_top1.update(label, outputs)
        acc_top5.update(label, outputs)

    _, top1 = acc_top1.get()
    _, top5 = acc_top5.get()
    return top1, top5


def train(net, train_data, val_data, ctx, opt):

    lr_scheduler = get_lr_scheduler(opt)
    optimizer_params = {'wd': opt.wd, 'momentum': opt.momentum, 'lr_scheduler': lr_scheduler}
    
    trainer = gluon.Trainer(net.collect_params(), opt.optimizer, optimizer_params)

    train_metric = mx.metric.Accuracy()
    ce_metric = mx.metric.Loss('CrossEntropy')
    creteria = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=True)

    if opt.hybridize:
        net.hybridize(static_alloc=True, static_shape=True)

    logging.info('starting traing from scratch...')
    for epoch in range(opt.num_epochs):
        tic = time.time()
        train_data.reset()
        ce_metric.reset()
        train_metric.reset()
        btic = time.time()

        for i, batch in enumerate(train_data):
            if opt.dali:
                data = [data_batch.data[0] for data_batch in batch]
                label = [data_batch.label[0] for data_batch in batch]
            else:
                data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
                label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)

            with autograd.record():
                outputs = [net(X.astype(opt.dtype, copy=False)) for X in data]
                losses = [creteria(yhat, y.astype(opt.dtype, copy=False)) for yhat, y in zip(outputs, label)]
            for l in losses:
                l.backward()
            
            trainer.step(batch_size)

            train_metric.update(label, outputs)
            ce_metric.update(0, [l * batch_size for l in losses])

            if i>0 and i%opt.log_interval==0:
                train_metric_name, train_metric_score = train_metric.get()
                _, ce_loss = ce_metric.get()
                logging.info('Epoch[%d] Batch [%d]\tSpeed: %.2f samples/sec\t%s=%f\tloss=%.2f\tlr=%f'%(
                            epoch, i, batch_size*opt.log_interval/(time.time()-btic),
                            train_metric_name, train_metric_score, ce_loss, trainer.learning_rate))
                btic = time.time()

        train_metric_name, train_metric_score = train_metric.get()
        throughput_speed = int(batch_size * i /(time.time() - tic))

        top1_val, top5_val = validation(val_data, ctx, opt)

        logging.info('[Epoch %d] training: %s=%f'%(epoch, train_metric_name, train_metric_score))
        logging.info('[Epoch %d] speed: %d samples/sec\ttime cost: %f'%(epoch, throughput_speed, time.time()-tic))
        logging.info('[Epoch %d] validation: acc-top1=%f acc-top5=%f'%(epoch, top1_val, top5_val))

        if (epoch+1) % opt.save_frequency == 0:
            datestamp = datetime.datetime.now().strftime('%Y%m%d')
            prefix = '{}_{}_{}'.format(opt.model, 'imagenet', datestamp) 
            save_prefix = os.path.expanduser(os.path.join(opt.save_dir, prefix))
            net.export(save_prefix, epoch+1)
            param_file = '{}-{}.params'.format(opt.save_prefix, str(epoch+1).zfill(4))
            logging.info('model parameters were saved at {}'.format(param_file))


if __name__ == '__main__':
    opt = parse_args()

    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    log_file = '{}_{}_train_{}.log'.format(opt.model, 'imagenet', timestamp) 
    opt.log_file = log_file
    log_path = os.path.expanduser(os.path.join(opt.save_dir, log_file))
    build_logger(log_path)

    logging.info(opt)
    
    ctx = [mx.gpu(int(i)) for i in opt.gpus.split(',')]
    num_gpus = len(ctx)
    
    net = build_net(ctx, opt)
    
    if opt.dali:
        batch_size = opt.batch_size
        train_data, val_data = get_dali_dataloder(batch_size, ctx, opt)
    else:
        batch_size = opt.batch_size*max(1, num_gpus)
        train_data, val_data = get_data_rec(batch_size, opt)

    train(net, train_data, val_data, ctx, opt)


