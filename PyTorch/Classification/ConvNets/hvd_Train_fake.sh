NUM_NODES=4
raport_file="raport.json"
#dataset_path=/nvmedisk/Datasets/ILSVRC/Data/CLS-LOC/
#dataset_path=/nfs/scistore14/alistgrp/imarkov/Datasets/imagewoof
dataset_path=/nfs/scistore14/alistgrp/imarkov/Datasets/imagenet/
#BATCH_SIZE=256
BATCH_SIZE=32
rm -rf ~/.horovod
lr=`echo "print($BATCH_SIZE * $NUM_NODES * 0.001)" | python`
q=1
workspace="./workspace_stats/imagenet_rn50/linadapt-${q}"
mkdir -p $workspace
horovodrun -np $NUM_NODES --disable-cache \
python ./main.py $dataset_path --data-backend dali-cpu --raport-file raport.json -j8 -p 100 --lr $lr \
--optimizer-batch-size $(( BATCH_SIZE * NUM_NODES )) --warmup 8 --arch resnet50 -c fanin --label-smoothing 0.1 \
--lr-schedule cosine --mom 0.875 --wd 3.0517578125e-05 --workspace $workspace -b $BATCH_SIZE --amp \
--static-loss-scale 128 --epochs 90 --hvd --bucket-size 1024 --no-checkpoints --tensorboard-dir "$workspace/tb_logs" --quantization-bits $q \
--compression-type maxmin --error-feedback
