
cd ~/gluon_classifier/tools

python trainer.py \
  --rec-train ~/gluon_classifier/dataset/imagenet/imagenet1k-train_480_q90.rec \
  --rec-train-idx ~/gluon_classifier/dataset/imagenet/imagenet1k-train_480_q90.idx \
  --rec-val ~/gluon_classifier/dataset/imagenet/imagenet1k-val.rec \
  --rec-val-idx ~/gluon_classifier/dataset/imagenet/imagenet1k-val.idx \
  --model shufflenet0.5_g3 \
  --hybridize \
  --lr 0.5 \
  --lr-mode step \
  --wd 0.00004 \
  --num-epochs 120 \
  --batch-size 256 \
  --gpus '0,1,2,3' \
  --num-workers 32 \
  --warmup-epochs 5 \
  --save-frequency 1 \
  --save-dir ~/gluon_classifier/output/
