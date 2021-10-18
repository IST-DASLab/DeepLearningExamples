NUM_NODES=8
raport_file="raport.json"
dataset_path=/nvmedisk/Datasets/ILSVRC/Data/CLS-LOC/
#dataset_path=/home/imarkov/Datasets/imagewoof
BATCH_SIZE=256
#BATCH_SIZE=32
rm -rf ~/.horovod
lr=`echo "print($BATCH_SIZE * $NUM_NODES * 0.001)" | python`

q=3
bucket_size=1024
workspace="./workspace_stats/imagenet_rn18/adaptive_2_4_reset"
#workspace="./workspace_stats/imagenet_rn50/kmeans_1_4_magnitude_based_adaptive_bucket_size_${bucket_size}_2_no_sizes"
mkdir -p $workspace
horovodrun -np $NUM_NODES --disable-cache \
python ./main.py $dataset_path --data-backend dali-cpu --raport-file raport.json -j8 -p 100 --lr $lr \
--optimizer-batch-size $(( BATCH_SIZE * NUM_NODES )) --warmup 8 --arch resnet18 -c fanin --label-smoothing 0.1 \
--lr-schedule cosine --mom 0.875 --wd 3.0517578125e-05 --workspace $workspace -b $BATCH_SIZE --amp \
--static-loss-scale 128 --epochs 20 --resume ./workspace_stats/imagenet_rn18/adaptive_1_4/checkpoint.pth.tar --hvd --bucket-size $bucket_size --tensorboard-dir "$workspace/tb_logs" --quantization-bits $q \
--compression-type maxmin --no-checkpoints