NUM_NODES=$1
#raport_file="raport_`date +%H%M`.json"
#dataset_path=/imagenet/ILSVRC/Data/CLS-LOC/

#dataset_path=~/Datasets/imagewoof/
#BATCH_SIZE=64

dataset_path=$IMAGENET_PATH
#BATCH_SIZE=256
BATCH_SIZE=128

rm -rf ~/.horovod
BUCKET_SIZE=1024
q=4
incomplete="--compression-skip-incomplete-buckets"
#incomplete=""
#workspace="./workspace_quan/woof_rn18/allgather_exp_q${q}_${BUCKET_SIZE}_1/"
#workspace="./workspace_quan/imagenet_rn18/sra_q${q}_${BUCKET_SIZE}_skip/"
#workspace="./workspace_quan/imagenet_rn18/allgather_uni_ef_q${q}_${BUCKET_SIZE}_skip"
workspace="./workspace_quan/imagenet_rn18/test"
mkdir -p $workspace

compression_config_file=""
#compression_config_file="--compression-config-filename config_compress.yaml"
cp config_compress.yaml $workspace

lr=`echo "print($BATCH_SIZE * $NUM_NODES * 0.001)" | python`

horovodrun -np $NUM_NODES --reduction-type none --compression-nccl-fake-ratio 1.0 --communicator-type SHM --compression-type uni \
${compression_config_file} -q $q --compression-bucket-size ${BUCKET_SIZE} $incomplete \
python ./main.py $dataset_path --data-backend dali-cpu --raport-file raport.json -j 8 -p 25 --lr $lr \
--optimizer-batch-size $(( BATCH_SIZE * NUM_NODES )) --warmup 8 --arch resnet50 -c fanin --label-smoothing 0.1 \
--lr-schedule cosine --mom 0.875 --wd 3.0517578125e-05 --workspace $workspace -b $BATCH_SIZE --amp \
--static-loss-scale 128 --epochs 1 --prof 200 --training-only --hvd --tensorboard-dir "$workspace/tb_logs"