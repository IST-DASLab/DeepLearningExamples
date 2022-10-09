NUM_NODES=${1:-8}
BATCH_SIZE=256
#raport_file="raport_`date +%H%M`.json"
raport_file="raport.json"
#dataset_path=/imagenet/ILSVRC/Data/CLS-LOC/
dataset_path=/nvmedisk/Datasets/ILSVRC/Data/CLS-LOC/

#dataset_path=$IMAGENET_PATH
workspace="./workspace_quan/imagenet_rn50/benchmark/"
mkdir -p $workspace
lr=`echo "print($BATCH_SIZE * $NUM_NODES * 0.001)" | python`

#export TORCH_DISTRIBUTED_DEBUG=INFO
#export NCCL_DEBUG=INFO
#export TORCH_SHOW_CPP_STACKTRACES=1
#python ./multiproc.py --nproc_per_node $NUM_NODES --master_port 1234 ./launch.py --model resnet50 --precision AMP --mode convergence \
#--platform DGX1V $dataset_path --no-checkpoints --mixup 0.0 --workspace $workspace --raport-file raport.json \
#--batch-size $BATCH_SIZE --epochs 1 --training-only --dist-backend nccl --prof 800 --comp-type none # --default-comp-param 3 --compression-schemes compression_schemes/psgd --load-comp-freq 100

BACKEND=qmpi
#
COMP_TYPE=qsgd
comp_param=3

export DEBUG_ALL_TO_ALL_REDUCTION=0
export DEBUG_DUMMY_COMPRESSION=0
export INNER_COMMUNICATOR_TYPE=SHM
export INNER_REDUCTION_TYPE=SRA
#COMP_TYPE=topk
#comp_param=0.01


#BACKEND=nccl
#COMP_TYPE=none

#COMP_TYPE=psgd
#comp_param=3


mpirun --np $NUM_NODES \
-x INNER_COMMUNICATOR_TYPE -x DEBUG_ALL_TO_ALL_REDUCTION \
python ./launch.py --model resnet50 --precision AMP --mode convergence \
--platform DGX1V $dataset_path  --mixup 0.0 --workspace $workspace --raport-file raport.json --no-checkpoints \
--batch-size $BATCH_SIZE --epochs 1 --training-only --prof 400 --comp-type $COMP_TYPE \
--default-comp-param $comp_param --dist-backend $BACKEND #--compression-schemes compression_schemes/$COMP_TYPE --load-comp-freq 20 2>&1 | tee log
