NUM_NODES=8
raport_file="raport.json"
#dataset_path=/nfs/scistore14/alistgrp/imarkov/Datasets/imagenette
dataset_path=/home/imarkov/Datasets/imagenette
#BATCH_SIZE=256
BATCH_SIZE=32
rm -rf ~/.horovod
#lr=`echo "print($BATCH_SIZE * $NUM_NODES * 0.001)" | python`
#lr=0.01
lr=`echo "print($BATCH_SIZE * $NUM_NODES * 0.001)" | python`
momentum=0.9
wd=0.00005
bucket_size=128

q=2
workspace="./workspace_quan/imagenette_rn18/adapt-4"
mkdir -p $workspace
horovodrun -np $NUM_NODES --disable-cache \
python ./main.py $dataset_path --data-backend dali-cpu --raport-file raport.json -j8 -p 100 --lr $lr \
--optimizer-batch-size $(( BATCH_SIZE * NUM_NODES )) --warmup 8 --arch resnet18 -c fanin --label-smoothing 0.1 \
--lr-schedule cosine --mom $momentum --wd $wd --workspace $workspace -b $BATCH_SIZE \
--static-loss-scale 128 --epochs 90 --hvd --bucket-size $bucket_size --no-checkpoints --tensorboard-dir "$workspace/tb_logs" --quantization-bits $q \
--compression-type maxmin