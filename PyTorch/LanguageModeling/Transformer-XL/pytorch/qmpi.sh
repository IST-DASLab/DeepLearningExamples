#NUM_NODES=8
workdir="workdir_topk/test"
export FUSION_BUFFER_SIZE_MB=128
export COMPRESSION_TOPK=1.0
q=4
bucket_size=128
#export COMPRESSION_FAKE_RATIO=$fake_ratio

for NUM_NODES in 1 2 4 8; do
  BATCH_SIZE=$(( 32 * $NUM_NODES ))
  mpirun -np $NUM_NODES --tag-output \
    python train.py \
    --config_file wt103_base.yaml \
    --config rtx3090_fp16 \
    --batch_size $BATCH_SIZE \
    --work_dir $workdir \
    --dllog_file train_log.json --txtlog_file train_log.log \
    --max_step 200 --debug \
    --backend nccl --quantization-bits $q --bucket-size $bucket_size --amp apex \
    "${@:2}"
done
