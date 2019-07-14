import os
import sys
import math
import argparse
import warnings
import mxnet as mx
from mxnet import gluon

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model for image classification.')
    parser.add_argument('--model-prefix', type=str, default='params',
                        help='directory of saved models')
    parser.add_argument('--epoch', type=int, default=0,
                        help='number of training epochs.')
    parser.add_argument('--rec-val', type=str, default='~/.mxnet/datasets/imagenet/rec/val.rec',
                        help='the validation data')
    parser.add_argument('--rec-val-idx', type=str, default='~/.mxnet/datasets/imagenet/rec/val.idx',
                        help='the index of validation data')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='training batch size per device (CPU/GPU).')
    parser.add_argument('--gpus', type=str, default='0,1,2,3',
                        help='number of gpus to use.')
    parser.add_argument('--num-workers', type=int, default=16,
                        help='number of preprocessing workers')
    parser.add_argument('--input-size', type=int, default=224,
                        help='size of the input image size. default is 224')
    parser.add_argument('--crop-ratio', type=float, default=0.875,
                        help='Crop ratio during validation. default is 0.875')
    parser.add_argument('--hybridize', action='store_true',
                        help='if hybridize the model.')
    parser.add_argument('--dtype', type=str, default='float32',
                        help='data type for training. default is float32')
    opt = parser.parse_args()
    return opt


def get_data_rec(batch_size, opt):

    rec_val = os.path.expanduser(opt.rec_val)
    rec_val_idx = os.path.expanduser(opt.rec_val_idx)
    input_size = opt.input_size
    num_workers = opt.num_workers
    jitter_param = 0.4
    lighting_param = 0.1
    resize = int(math.ceil(input_size / opt.crop_ratio))
    mean_rgb = [123.68, 116.779, 103.939]
    std_rgb = [58.393, 57.12, 57.375]

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

    print('data loader initialized.')
    
    return val_data


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

def build_network(ctx, opt):
    param_file = '{}-{}.params'.format(opt.model_prefix, str(opt.epoch).zfill(4))
    json_file = '{}-symbol.json'.format(opt.model_prefix)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net = gluon.nn.SymbolBlock.imports(json_file, ['data'], param_file, ctx=ctx)
        if opt.hybridize:
            net.hybridize()
        return net


if __name__ == '__main__':
    opt = parse_args()
    ctx = [mx.gpu(int(i)) for i in opt.gpus.split(',')]
    net = build_network(ctx, opt)
    num_gpus = len(ctx)
    batch_size = opt.batch_size*max(1, num_gpus)
    val_data = get_data_rec(batch_size, opt)
    top1_acc, top5_acc = validation(val_data, ctx, opt)
    print('Evaluation with model %s results: acc-top1=%f acc-top5=%f'%(opt.model_prefix, top1_acc, top5_acc))


