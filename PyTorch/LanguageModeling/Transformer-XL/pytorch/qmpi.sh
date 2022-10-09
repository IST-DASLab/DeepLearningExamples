NUM_NODES=${1:-8}
#workdir="workdir_topk/test"
root_dir="/nfs/scistore14/alistgrp/imarkov/repos/DeepLearningExamples/PyTorch/LanguageModeling/Transformer-XL/pytorch"
workdir="$root_dir/workdir_pytorch_apex/test"
export FUSION_BUFFER_SIZE_MB=128
export NCCL_NTHREADS=64

BATCH_SIZE=$((32 * $NUM_NODES ))
mkdir -p $workdir

BACKEND=qmpi
#
COMP_TYPE=qsgd
comp_param=4

export DEBUG_ALL_TO_ALL_REDUCTION=0
export DEBUG_DUMMY_COMPRESSION=0
export INNER_COMMUNICATOR_TYPE=SHM
export INNER_REDUCTION_TYPE=SRA
#COMP_TYPE=topk
#comp_param=0.1


#BACKEND=nccl
#COMP_TYPE=none

#COMP_TYPE=psgd
#comp_param=32

#python -m torch.distributed.launch --nproc_per_node=$NUM_NODES \
#nsys profile --stats=true -t nvtx,cuda \
mpirun -np $NUM_NODES --tag-output \
-x INNER_COMMUNICATOR_TYPE -x DEBUG_ALL_TO_ALL_REDUCTION \
python train.py \
  --config_file wt103_base.yaml --config dgxa100_8gpu_tf32 \
  --batch_size $BATCH_SIZE \
  --work_dir $workdir --data /nvmedisk/Datasets/wikitext-103/ --warmup_step 10 \
  --dllog_file baseline.json --txtlog_file baseline.log --amp apex --max_step 780 --debug \
  --comp-type $COMP_TYPE --default-comp-param $comp_param --dist-backend $BACKEND --compression-schemes compression_schemes/$COMP_TYPE --load-comp-freq 40 \
  "${@:2}" | tee log
