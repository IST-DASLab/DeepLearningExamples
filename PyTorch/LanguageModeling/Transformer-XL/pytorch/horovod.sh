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
q=8
bucket_size=1024

workdir="workdir_quan/sra_${q}_${bucket_size}"
#compression_config_file="--compression-config-filename config_compress.yaml"
compression_config_file=""

mkdir -p $workdir
echo 'Run training...'
for type in SRA; do
  horovodrun -np $NUM_NODES -q $q --compression-bucket-size ${bucket_size} \
      --compression-nccl-fake-ratio 0.5 \
      --reduction-type $type --communicator-type MPI --compression-type maxmin \
      --fusion-threshold-mb 4 --cache-capacity 2048 --no-hierarchical-allgather --cycle-time-ms 1 --no-hierarchical-allreduce \
      --compression-mode NonFused \
      python train.py \
      --config_file wt103_base.yaml \
      --config rtx2080 \
      --hvd \
      --work_dir $workdir \
      "${@:2}" 2>&1
done