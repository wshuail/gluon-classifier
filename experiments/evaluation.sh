
cd ~/gluon_classifier/tools

python evaluate.py \
  --model-prefix ~/gluon_classifier/output/shufflenet0.5_g3_imagenet_20190714 \
  --epoch 2 \
  --rec-val ~/gluon_classifier/dataset/imagenet/imagenet1k-val.rec \
  --rec-val-idx ~/gluon_classifier/dataset/imagenet/imagenet1k-val.idx \
  --hybridize \
  --batch-size 256 \
  --gpus '0,1,2,3'
