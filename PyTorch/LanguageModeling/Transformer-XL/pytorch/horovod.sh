#!/bin/bash

# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
NUM_NODES=8
q=5
bucket_size=1024

compression_config_file="--compression-config-filename config_compress.yaml"
#compression_config_file=""
BATCH_SIZE=$(( 32 * $NUM_NODES ))
echo 'Run training...'
for type in none; do
#  workdir="workdir_quan/${type}_${q}_${bucket_size}"
  workdir="workdir_topk/test"
  mkdir -p $workdir
  horovodrun -np $NUM_NODES -q $q --compression-bucket-size ${bucket_size} \
      --compression-nccl-fake-ratio 1.0 \
      --reduction-type $type --communicator-type SHM --compression-type maxmin \
      --fusion-threshold-mb 128 --cache-capacity 2048 --no-hierarchical-allgather --cycle-time-ms 1 --no-hierarchical-allreduce \
      --compression-mode NonFused --compression-skip-incomplete-buckets $compression_config_file \
      python train.py \
      --config_file wt103_base.yaml \
      --config rtx3090_fp16 \
      --hvd \
      --batch_size $BATCH_SIZE \
      --work_dir $workdir \
      --dllog_file train_log.json --txtlog_file train_log.log \
      --max_step 200 --debug \
      "${@:2}" 2>&1
done