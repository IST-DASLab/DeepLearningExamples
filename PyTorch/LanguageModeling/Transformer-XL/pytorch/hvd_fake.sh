#!/bin/bash
NUM_NODES=8
q=8
bucket_size=1024

workdir="workdir_stats/linadapt_8bits-1"
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