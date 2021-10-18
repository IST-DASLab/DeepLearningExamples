NUM_NODES=8
#raport_file="raport_`date +%H%M`.json"
#dataset_path=/imagenet/ILSVRC/Data/CLS-LOC/

#dataset_path=~/Datasets/imagewoof/
#BATCH_SIZE=64

#dataset_path=$IMAGENET_PATH
dataset_path=/nvmedisk/Datasets/ILSVRC/Data/CLS-LOC/
#BATCH_SIZE=256

#rm -rf ~/.horovod
#BUCKET_SIZE=1024
#q=4
#incomcplete="--compression-skip-incomplete-buckets"
#incomplete=""

#compression_config_file=""
compression_config_file="--compression-config-filename config_compress.yaml"

#lr=`echo "print($BATCH_SIZE * $NUM_NODES * 0.001)" | python`

#for type in SRA; do
#  if [[ $type == "none" ]]; then
#    name_exp="baseline"
#  else
#    name_exp="${type}_${q}"
#  fi
#  workspace="./workspace_quan/imagenet_rn101/${name_exp}"
#  mkdir -p $workspace
#  cp config_compress.yaml $workspace
#  horovodrun -np $NUM_NODES --reduction-type $type --compression-nccl-fake-ratio 1.0 --communicator-type MPI --compression-type maxmin \
#  ${compression_config_file} -q $q --compression-bucket-size ${BUCKET_SIZE} $incomplete \
#  python ./main.py $dataset_path --data-backend dali-cpu --raport-file raport.json -j 8 -p 100 --lr $lr \
#  --optimizer-batch-size $(( BATCH_SIZE * NUM_NODES )) --warmup 8 --arch resnet101 -c fanin --label-smoothing 0.1 \
#  --lr-schedule cosine --mom 0.875 --wd 3.0517578125e-05 --workspace $workspace -b $BATCH_SIZE --amp \
#  --static-loss-scale 128 --epochs 90 --hvd --resume $workspace/checkpoint.pth.tar  --raport-file raport2.json --tensorboard-dir "$workspace/tb_logs"
#done
BATCH_SIZE=64
lr=0.1
name_exp="baseline"
workspace="./workspace_quan/imagenet_rn50_small_batch/baseline"
mkdir -p $workspace
horovodrun -np $NUM_NODES \
python ./main.py $dataset_path --data-backend dali-cpu --raport-file raport.json -j 8 -p 100 --lr $lr \
--optimizer-batch-size $(( BATCH_SIZE * NUM_NODES )) --warmup 8 --arch resnet50 -c fanin --label-smoothing 0.0 \
--lr-schedule step --mom 0.9 --wd 1e-4 --workspace $workspace -b $BATCH_SIZE --amp \
--static-loss-scale 128 --epochs 90 --hvd  --raport-file raport.json --tensorboard-dir "$workspace/tb_logs"