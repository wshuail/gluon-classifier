
cd /world/data-gpu-107/wangshuailong/lib/gluon_classifier/tools

python trainer.py \
  --rec-train /world/data-gpu-107/imagenet1k_mxnet_batch/imagenet1k-train_480_q90.rec \
  --rec-train-idx /world/data-gpu-107/imagenet1k_mxnet_batch/imagenet1k-train_480_q90.idx \
  --rec-val /world/data-gpu-107/imagenet1k_mxnet_batch/imagenet1k-val.rec \
  --rec-val-idx /world/data-gpu-107/imagenet1k_mxnet_batch/imagenet1k-val.idx \
  --model shufflenetv2_0.5 \
  --hybridize \
  --lr 0.4 \
  --lr-mode cosine \
  --wd 0.00005 \
  --num-epochs 120 \
  --batch-size 256 \
  --gpus '0,1,2,3' \
  --num-workers 32 \
  --warmup-epochs 5 \
  --save-frequency 1 \
  --save-dir /world/data-gpu-107/wangshuailong/experiments/gluon_classifier_output
