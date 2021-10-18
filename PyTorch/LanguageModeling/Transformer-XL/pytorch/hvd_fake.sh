#!/bin/bash
NUM_NODES=4
bucket_size=128

#for q in 2; do
#  export ADJUST_ALPHA=$alpha
#adapt="--adapt-compression --adapt-compression-reset-freq 10000 --adapt-compression-adjust-freq 1000"
adapt=""
root_dir="/nfs/scistore14/alistgrp/imarkov/repos/DeepLearningExamples/PyTorch/LanguageModeling/Transformer-XL/pytorch"
for q in 4; do
#  workdir="$root_dir/workdir_stats/embed_topk_0.01_other_q${q}_${bucket_size}"
  workdir="$root_dir/workdir_stats/test"
#  mkdir -p $workdir
  horovodrun -np $NUM_NODES --fusion-threshold-mb 64 --cycle-time-ms 1 \
      python train.py \
      --config_file wt103_base.yaml \
      --config rtx3090_fp16 \
      --hvd \
      --log_interval 50 \
      --work_dir $workdir \
      --dllog_file train_log1.json --txtlog_file train_log1.log \
      --compression-type maxmin --bucket-size $bucket_size --quantization-bits $q \
      --eval_interval 1000 --restart $workdir/checkpoint_last.pt $adapt \
      --batch_chunk 2 \
      "${@:2}"
done