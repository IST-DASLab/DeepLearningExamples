#!/bin/bash
NUM_NODES=8
q=5
bucket_size=128

workdir="workdir_stats/linadapt_5bits-4"
mkdir -p $workdir
horovodrun -np $NUM_NODES \
    python train.py \
    --config_file wt103_base.yaml \
    --config rtx3090_fp16 \
    --hvd \
    --work_dir $workdir \
    --dllog_file train_log.json --txtlog_file train_log.log \
    --compression-type maxmin --quantization-bits $q --bucket-size $bucket_size \
    --eval_interval 1000 \
    "${@:2}" 2>&1