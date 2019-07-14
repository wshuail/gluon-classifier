
cd ~/gluon_classifier/tools

python trainer.py \
  --rec-train /world/data-gpu-107/imagenet1k_mxnet_batch/imagenet1k-train_480_q90.rec \
  --rec-train-idx /world/data-gpu-107/imagenet1k_mxnet_batch/imagenet1k-train_480_q90.idx \
  --rec-val /world/data-gpu-107/imagenet1k_mxnet_batch/imagenet1k-val.rec \
  --rec-val-idx /world/data-gpu-107/imagenet1k_mxnet_batch/imagenet1k-val.idx \
  --model resnet18_v1 \
  --hybridize \
  --lr 0.4 \
  --lr-mode cosine \
  --num-epochs 120 \
  --batch-size 256 \
  --gpus '0,1,2,3' \
  --num-workers 32 \
  --warmup-epochs 5 \
  --save-dir ~/gluon_classifier/output/
