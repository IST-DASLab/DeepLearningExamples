# Differential evolution search

To search for layer-wise bit width using differential evolution, run `main.py`. 

```
usage: main.py [-h] [--data-backend BACKEND] [--arch ARCH]
               [--model-config CONF] [--num-classes N] [-j N] [--epochs N]
               [--run-epochs N] [-b N] [--optimizer-batch-size N] [--lr LR]
               [--lr-schedule SCHEDULE] [--warmup E] [--label-smoothing S]
               [--mixup ALPHA] [--momentum M] [--weight-decay W]
               [--bn-weight-decay] [--nesterov] [--print-freq N]
               [--resume PATH] [--pretrained-weights PATH] [--fp16]
               [--static-loss-scale STATIC_LOSS_SCALE] [--dynamic-loss-scale]
               [--prof N] [--amp] [--seed SEED] [--gather-checkpoints]
               [--raport-file RAPORT_FILE] [--evaluate] [--training-only]
               [--no-checkpoints] [--checkpoint-filename CHECKPOINT_FILENAME]
               [--workspace DIR] [--memory-format {nchw,nhwc}]
               [--tensorboard-dir TENSORBOARD_DIR] [--opt-level {O0,O1,O2,O3}]
               [--hvd] [--bb-ratio BB_RATIO]
               [--bb-num-parallel-steps BB_NUM_PARALLEL_STEPS]
               [--compression-type {none,sanity,maxmin,exponential,1bit,norm_uniform,terngrad,topk,topk_rmsprop,svd,stats,quantile}]
               [--quantization-bits QUANTIZATION_BITS]
               [--bucket-size BUCKET_SIZE] [--error-feedback]
               [--efx-bits EFX_BITS] [--efx-randk EFX_RANDK]
               [--topk-ratio TOPK_RATIO] [--big-grad] [--dgc]
               [--compressor-warmup-steps COMPRESSOR_WARMUP_STEPS]
               [--local_rank LOCAL_RANK] [--powersgd-rank POWERSGD_RANK]
               [--search] [--search-alpha SEARCH_ALPHA]
               [--search-beta SEARCH_BETA] [--search-max-bits SEARCH_MAX_BITS]
               [--search-min-bits SEARCH_MIN_BITS] [--popsize POPSIZE]
               [--search-iterations SEARCH_ITERATIONS]
               DIR

PyTorch ImageNet Training

positional arguments:
  DIR                   path to dataset

optional arguments:
  -h, --help            show this help message and exit
  --data-backend BACKEND
                        data backend: pytorch | syntetic | dali-gpu | dali-cpu
                        (default: dali-cpu)
  --arch ARCH, -a ARCH  model architecture: resnet18 | resnet34 | resnet50 |
                        resnet101 | resnet152 | resnext101-32x4d | se-
                        resnext101-32x4d | wide_resnet50 | wide_resnet101 |
                        vgg16 | resnet18_convex (default: resnet50)
  --model-config CONF, -c CONF
                        model configs: classic | fanin | grp-fanin | grp-
                        fanout(default: classic)
  --num-classes N       number of classes in the dataset
  -j N, --workers N     number of data loading workers (default: 5)
  --epochs N            number of total epochs to run
  --run-epochs N        run only N epochs, used for checkpointing runs
  -b N, --batch-size N  mini-batch size (default: 256) per gpu
  --optimizer-batch-size N
                        size of a total batch size, for simulating bigger
                        batches using gradient accumulation
  --lr LR, --learning-rate LR
                        initial learning rate
  --lr-schedule SCHEDULE
                        Type of LR schedule: step, linear, cosine
  --warmup E            number of warmup epochs
  --label-smoothing S   label smoothing
  --mixup ALPHA         mixup alpha
  --momentum M          momentum
  --weight-decay W, --wd W
                        weight decay (default: 1e-4)
  --bn-weight-decay     use weight_decay on batch normalization learnable
                        parameters, (default: false)
  --nesterov            use nesterov momentum, (default: false)
  --print-freq N, -p N  print frequency (default: 10)
  --resume PATH         path to latest checkpoint (default: none)
  --pretrained-weights PATH
                        load weights from here
  --fp16                Run model fp16 mode.
  --static-loss-scale STATIC_LOSS_SCALE
                        Static loss scale, positive power of 2 values can
                        improve fp16 convergence.
  --dynamic-loss-scale  Use dynamic loss scaling. If supplied, this argument
                        supersedes --static-loss-scale.
  --prof N              Run only N iterations
  --amp                 Run model AMP (automatic mixed precision) mode.
  --seed SEED           random seed used for numpy and pytorch
  --gather-checkpoints  Gather checkpoints throughout the training, without
                        this flag only best and last checkpoints will be
                        stored
  --raport-file RAPORT_FILE
                        file in which to store JSON experiment raport
  --evaluate            evaluate checkpoint/model
  --training-only       do not evaluate
  --no-checkpoints      do not store any checkpoints, useful for benchmarking
  --checkpoint-filename CHECKPOINT_FILENAME
  --workspace DIR       path to directory where checkpoints will be stored
  --memory-format {nchw,nhwc}
                        memory layout, nchw or nhwc
  --tensorboard-dir TENSORBOARD_DIR
                        directory for tensorboard logs
  --opt-level {O0,O1,O2,O3}
                        AMP optimization levels
  --hvd                 use horovod
  --bb-ratio BB_RATIO   Ratio for broken barrier
  --bb-num-parallel-steps BB_NUM_PARALLEL_STEPS
                        Number of parallel steps
  --compression-type {none,sanity,maxmin,exponential,1bit,norm_uniform,terngrad,topk,topk_rmsprop,svd,stats,quantile}
                        Compression Type (default: none)
  --quantization-bits QUANTIZATION_BITS
                        Quantization bits (default: 4)
  --bucket-size BUCKET_SIZE
                        Quantization bucket size (default: 512)
  --error-feedback      enable error correction
  --efx-bits EFX_BITS   Quantization bits of error feedback (default - no efx)
  --efx-randk EFX_RANDK
                        Randk of error feedback compression (default - no efx)
  --topk-ratio TOPK_RATIO
                        topK selecting ratio (default: 0.01)
  --big-grad            stack all gradients into signle tensor before
                        compression and allreduce
  --dgc                 Do deep gradient compression
  --compressor-warmup-steps COMPRESSOR_WARMUP_STEPS
                        Warm up period for compressor
  --local_rank LOCAL_RANK
                        local rank
  --powersgd-rank POWERSGD_RANK
                        Rank of powersgd compression to run DDP with
  --search              Invoke differential evolution search
  --search-alpha SEARCH_ALPHA
                        Power compression error is raised to minimize
                        objective function
  --search-beta SEARCH_BETA
                        Power compression ratio is raised to minimize
                        objective function
  --search-max-bits SEARCH_MAX_BITS
                        Maximum bit-width to compress the gradients
  --search-min-bits SEARCH_MIN_BITS
                        Minimum bit-width to compress the gradients
  --popsize POPSIZE     Population size
  --search-iterations SEARCH_ITERATIONS
                        Number of iterations - Total iterations =
                        (search_iterations+1)*popsize*num_layers
```


## Search arguments

popsize = 50
search-iterations = 5 (total search iterations for resnet = (5+1) * 50 * 53)
