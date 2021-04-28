#!/bin/bash
NUM_NODES=8
bucket_size=128

workdir="workdir_stats/3LC"
mkdir -p $workdir
horovodrun -np $NUM_NODES \
    python train.py \
    --config_file wt103_base.yaml \
    --config rtx3090_fp16 \
    --hvd \
    --log_interval 50 \
    --work_dir $workdir \
    --dllog_file train_log.json --txtlog_file train_log.log \
    --compression-type 3LC --bucket-size -1 \
    --eval_interval 1000 \
    "${@:2}" 2>&1


k=0.125
workdir="workdir_stats/topk-${k}"
mkdir -p $workdir
horovodrun -np $NUM_NODES \
    python train.py \
    --config_file wt103_base.yaml \
    --config rtx3090_fp16 \
    --hvd \
    --log_interval 50 \
    --work_dir $workdir \
    --dllog_file train_log.json --txtlog_file train_log.log \
    --compression-type topk --bucket-size -1 --topk-ratio $k \
    --eval_interval 1000 \
    "${@:2}" 2>&1