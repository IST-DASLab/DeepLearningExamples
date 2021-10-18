NUM_NODES=$1
BATCH_SIZE=32
raport_file="raport.json"
#dataset_path=/imagenet/ILSVRC/Data/CLS-LOC/
dataset_path=/nvmedisk/Datets/ILSVRC/Data/CLS-LOC/
#
#dataset_path=$IMAGENET_PATH
export COMPRESSION_QUANTIZATION_BITS=4
export COMPRESSION_BUCKET_SIZE=1024
#export COMPRESSION_MINIMAL_SIZE=1000
workspace="./workspace_quan/imagenet_rn50/test/"
mkdir -p $workspace
lr=`echo "print($BATCH_SIZE * $NUM_NODES * 0.001)" | python`
$MPI_HOME/bin/mpirun -np $NUM_NODES -mca pml ob1 -tag-output python \
./main.py  $dataset_path \
--data-backend dali-cpu --raport-file $raport_file -j8 -p 25 --lr $lr \
--optimizer-batch-size $(( $BATCH_SIZE  * $NUM_NODES )) --warmup 8 --arch resnet50 -c fanin \
--label-smoothing 0.1 --lr-schedule cosine --mom 0.875 --wd 3.0517578125e-05 --workspace ${workspace} \
-b $BATCH_SIZE --epochs 1 --static-loss-scale 128 --prof 200 --training-only --amp --opt-level O1 --no-checkpoints --backend qmpi
