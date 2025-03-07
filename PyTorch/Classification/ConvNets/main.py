# Copyright (c) 2018-2019, NVIDIA CORPORATION
# Copyright (c) 2017-      Facebook, Inc
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import argparse
import os
import shutil
import time
import random

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import json

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp
except ImportError:
    raise ImportError(
        "Please install apex from https://www.github.com/nvidia/apex to run this example."
    )

try:
    import horovod.torch as hvd
except ImportError:
    print(
        "Horovod is not installed"
    )

import image_classification.resnet as models
import image_classification.logger as log

from image_classification.smoothing import LabelSmoothing
from image_classification.mixup import NLLMultiLabelSmooth, MixUpWrapper
from image_classification.dataloaders import *
from image_classification.training import *
from image_classification.utils import *

import dllogger
import sys

try:
    from gcomp_sim import CompressorManager, OTHERS, MaxMinQuantizer, TopKCompressor
    from gcomp_sim import DistributedOptimizer as gcomp_DistributedOptimizer
    from gcomp_sim import Adjuster, KMeanAdjuster
    grad_sim_available=True
except ModuleNotFoundError:
    grad_sim_available=False
    print('Gradient compression simulation is unavailable')


def add_parser_arguments(parser):
    model_names = list(models.resnet_versions.keys()) + ["wide_resnet50", "wide_resnet101", "vgg16", "resnet18_convex"]
    model_configs = models.resnet_configs.keys()

    parser.add_argument("data", metavar="DIR", help="path to dataset")
    parser.add_argument(
        "--data-backend",
        metavar="BACKEND",
        default="dali-cpu",
        choices=DATA_BACKEND_CHOICES,
        help="data backend: "
             + " | ".join(DATA_BACKEND_CHOICES)
             + " (default: dali-cpu)",
    )

    parser.add_argument(
        "--arch",
        "-a",
        metavar="ARCH",
        default="resnet50",
        choices=model_names,
        help="model architecture: " + " | ".join(model_names) + " (default: resnet50)",
    )

    parser.add_argument(
        "--model-config",
        "-c",
        metavar="CONF",
        default="classic",
        choices=model_configs,
        help="model configs: " + " | ".join(model_configs) + "(default: classic)",
    )

    parser.add_argument(
        "--num-classes",
        metavar="N",
        default=1000,
        type=int,
        help="number of classes in the dataset",
    )

    parser.add_argument(
        "-j",
        "--workers",
        default=5,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 5)",
    )
    parser.add_argument(
        "--epochs",
        default=90,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--run-epochs",
        default=-1,
        type=int,
        metavar="N",
        help="run only N epochs, used for checkpointing runs",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=256,
        type=int,
        metavar="N",
        help="mini-batch size (default: 256) per gpu",
    )

    parser.add_argument(
        "--optimizer-batch-size",
        default=-1,
        type=int,
        metavar="N",
        help="size of a total batch size, for simulating bigger batches using gradient accumulation",
    )

    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.1,
        type=float,
        metavar="LR",
        help="initial learning rate",
    )
    parser.add_argument(
        "--lr-schedule",
        default="step",
        type=str,
        metavar="SCHEDULE",
        choices=["step", "linear", "cosine"],
        help="Type of LR schedule: {}, {}, {}".format("step", "linear", "cosine"),
    )

    parser.add_argument(
        "--warmup", default=0, type=int, metavar="E", help="number of warmup epochs"
    )

    parser.add_argument(
        "--label-smoothing",
        default=0.0,
        type=float,
        metavar="S",
        help="label smoothing",
    )
    parser.add_argument(
        "--mixup", default=0.0, type=float, metavar="ALPHA", help="mixup alpha"
    )

    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum"
    )
    parser.add_argument(
        "--weight-decay",
        "--wd",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
    )
    parser.add_argument(
        "--bn-weight-decay",
        action="store_true",
        help="use weight_decay on batch normalization learnable parameters, (default: false)",
    )
    parser.add_argument(
        "--nesterov",
        action="store_true",
        help="use nesterov momentum, (default: false)",
    )

    parser.add_argument(
        "--print-freq",
        "-p",
        default=10,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--pretrained-weights",
        default="",
        type=str,
        metavar="PATH",
        help="load weights from here",
    )

    parser.add_argument("--fp16", action="store_true", help="Run model fp16 mode.")
    parser.add_argument(
        "--static-loss-scale",
        type=float,
        default=1,
        help="Static loss scale, positive power of 2 values can improve fp16 convergence.",
    )
    parser.add_argument(
        "--dynamic-loss-scale",
        action="store_true",
        help="Use dynamic loss scaling.  If supplied, this argument supersedes "
             + "--static-loss-scale.",
    )
    parser.add_argument(
        "--prof", type=int, default=-1, metavar="N", help="Run only N iterations"
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Run model AMP (automatic mixed precision) mode.",
    )

    parser.add_argument(
        "--seed", default=None, type=int, help="random seed used for numpy and pytorch"
    )

    parser.add_argument(
        "--gather-checkpoints",
        action="store_true",
        help="Gather checkpoints throughout the training, without this flag only best and last checkpoints will be stored",
    )

    parser.add_argument(
        "--raport-file",
        default="experiment_raport.json",
        type=str,
        help="file in which to store JSON experiment raport",
    )

    parser.add_argument(
        "--evaluate", action="store_true", help="evaluate checkpoint/model"
    )
    parser.add_argument("--training-only", action="store_true", help="do not evaluate")

    parser.add_argument(
        "--no-checkpoints",
        action="store_false",
        dest="save_checkpoints",
        help="do not store any checkpoints, useful for benchmarking",
    )

    parser.add_argument("--checkpoint-filename", default="checkpoint.pth.tar", type=str)

    parser.add_argument(
        "--workspace",
        type=str,
        default="./",
        metavar="DIR",
        help="path to directory where checkpoints will be stored",
    )
    parser.add_argument(
        "--memory-format",
        type=str,
        default="nchw",
        choices=["nchw", "nhwc"],
        help="memory layout, nchw or nhwc",
    )
    parser.add_argument(
        "--tensorboard-dir",
        default=None,
        type=str,
        help="directory for tensorboard logs",
    )
    parser.add_argument(
        "--opt-level",
        type=str,
        default="O1",
        choices=["O0", "O1", "O2", "O3"],
        help="AMP optimization levels",
    )
    parser.add_argument(
        "--hvd", action="store_true", help="use horovod"
    )
    parser.add_argument(
        "--bb-ratio", default=None, type=float, help="Ratio for broken barrier"
    )
    parser.add_argument(
        "--bb-num-parallel-steps", default=10, type=int, help="Number of parallel steps"
    )
    parser.add_argument("--compression-type", type=str, default="none", choices=["maxmin", "none"],
                        help="Compression Type (default: none)")
    parser.add_argument("--quantization-bits", type=int, default=4,
                        help="Quantization bits (default: 4)")
    parser.add_argument("--bucket-size", type=int, default=512,
                        help="Quantization bucket size (default: 512)")
    parser.add_argument('--error-feedback', action='store_true', default=False,
                        help='enable error correction')
    parser.add_argument("--efx-bits", type=int, default=None,
                        help="Quantization bits of error feedback (default - no efx)")
    parser.add_argument("--efx-randk", type=float, default=None,
                        help="Randk of error feedback compression (default - no efx)")
    parser.add_argument('--topk-ratio', type=float, default=0.01,
                        help='topK selecting ratio (default: 0.01)')
    parser.add_argument('--big-grad', action='store_true', default=False,
                        help='stack all gradients into signle tensor before compression and allreduce')
    parser.add_argument('--dgc', action='store_true', default=False,
                        help='Do deep gradient compression')
    parser.add_argument("--compressor-warmup-steps", type=int, default=0,
                        help="Warm up period for compressor")
    parser.add_argument("--local_rank", type=int, default=0,
                        help="local rank")
    parser.add_argument('--powersgd-rank', type=int, default=None,
                        help='Rank of powersgd compression to run DDP with')
    parser.add_argument('--adapt-compression', action='store_true', default=False,
                        help='Perform adaptive compression.')
    parser.add_argument('--adapt-compression-adjust-freq', type=int, default=1,
                        help='Adaptive compression. Frequency in epochs to perform bits adjustment.')
    parser.add_argument('--adapt-compression-reset-freq', type=int, default=10,
                        help='Adaptive compression. Frequency in epochs to reset stats.')
    parser.add_argument('--backend', choices=['qmpi', 'nccl', 'gloo'], default='nccl',
                        help='Backend for torch distributed.')



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(args):
    exp_start_time = time.time()
    global best_prec1
    best_prec1 = 0

    args.distributed = False
    if args.hvd:
        hvd.init()
        args.local_rank = hvd.local_rank()
    elif "WORLD_SIZE" in os.environ:
        args.distributed = int(os.environ["WORLD_SIZE"]) > 1
        args.local_rank = int(os.environ["LOCAL_RANK"])
    elif "OMPI_COMM_WORLD_SIZE" in os.environ:
        args.local_rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '4040'
        os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]
        os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]
        args.distributed = int(os.environ["WORLD_SIZE"]) > 1
    if args.backend == 'qmpi':
        import torch_qmpi
        assert "OMPI_COMM_WORLD_SIZE" in os.environ
        if 'COMPRESSION_QUANTIZATION_BITS' not in os.environ:
            os.environ['COMPRESSION_QUANTIZATION_BITS'] = str(args.quantization_bits)
        if 'COMPRESSION_BUCKET_SIZE' not in os.environ:
            os.environ['COMPRESSION_BUCKET_SIZE'] = str(args.quantization_bucket_size)


    args.gpu = 0
    args.world_size = 1
    if args.hvd:
        args.gpu = args.local_rank % hvd.local_size()
        torch.cuda.set_device(args.gpu)
        args.world_size = hvd.size()

    if args.distributed:
        args.gpu = args.local_rank % torch.cuda.device_count()
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend=args.backend, init_method="env://")
        args.world_size = torch.distributed.get_world_size()
        torch.distributed.barrier()

    if args.amp and args.fp16:
        print("Please use only one of the --fp16/--amp flags")
        exit(1)

    if args.seed is not None:
        print("Using seed = {}".format(args.seed))
        torch.manual_seed(args.seed + args.local_rank)
        torch.cuda.manual_seed(args.seed + args.local_rank)
        np.random.seed(seed=args.seed + args.local_rank)
        random.seed(args.seed + args.local_rank)

        def _worker_init_fn(id):
            np.random.seed(seed=args.seed + args.local_rank + id)
            random.seed(args.seed + args.local_rank + id)

    else:

        def _worker_init_fn(id):
            pass

    if args.fp16:
        assert (
            torch.backends.cudnn.enabled
        ), "fp16 mode requires cudnn backend to be enabled."

    if args.static_loss_scale != 1.0:
        if not args.fp16:
            print("Warning:  if --fp16 is not used, static_loss_scale will be ignored.")

    if args.optimizer_batch_size < 0:
        batch_size_multiplier = 1
    else:
        tbs = args.world_size * args.batch_size
        if args.optimizer_batch_size % tbs != 0:
            print(
                "Warning: simulated batch size {} is not divisible by actual batch size {}".format(
                    args.optimizer_batch_size, tbs
                )
            )
        batch_size_multiplier = int(args.optimizer_batch_size / tbs)
        print("BSM: {}".format(batch_size_multiplier))

    pretrained_weights = None
    if args.pretrained_weights:
        if os.path.isfile(args.pretrained_weights):
            print(
                "=> loading pretrained weights from '{}'".format(
                    args.pretrained_weights
                )
            )
            pretrained_weights = torch.load(args.pretrained_weights)
        else:
            print("=> no pretrained weights found at '{}'".format(args.resume))

    start_epoch = 0
    # optionally resume from a checkpoint
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(
                args.resume, map_location=lambda storage, loc: storage.cuda(args.gpu)
            )
            start_epoch = checkpoint["epoch"]
            best_prec1 = checkpoint["best_prec1"]
            model_state = checkpoint["state_dict"]
            optimizer_state = checkpoint["optimizer"]
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            model_state = None
            optimizer_state = None
    else:
        model_state = None
        optimizer_state = None

    loss = nn.CrossEntropyLoss
    if args.mixup > 0.0:
        loss = lambda: NLLMultiLabelSmooth(args.label_smoothing)
    elif args.label_smoothing > 0.0:
        loss = lambda: LabelSmoothing(args.label_smoothing)

    memory_format = (
        torch.channels_last if args.memory_format == "nhwc" else torch.contiguous_format
    )

    model_and_loss = ModelAndLoss(
        (args.arch, args.model_config, args.num_classes),
        loss,
        pretrained_weights=pretrained_weights,
        cuda=True,
        fp16=args.fp16,
        memory_format=memory_format,
    )

    # Create data loaders and optimizers as needed
    if args.data_backend == "pytorch":
        get_train_loader = get_pytorch_train_loader
        get_val_loader = get_pytorch_val_loader
    elif args.data_backend == "dali-gpu":
        get_train_loader = get_dali_train_loader(dali_cpu=False)
        get_val_loader = get_dali_val_loader()
    elif args.data_backend == "dali-cpu":
        get_train_loader = get_dali_train_loader(dali_cpu=True)
        get_val_loader = get_dali_val_loader()
    elif args.data_backend == "synthetic":
        get_val_loader = get_synthetic_loader
        get_train_loader = get_synthetic_loader

    train_loader, train_loader_len = get_train_loader(
        args.data,
        args.batch_size,
        args.num_classes,
        args.mixup > 0.0,
        start_epoch=start_epoch,
        workers=args.workers,
        fp16=args.fp16,
        memory_format=memory_format,
    )
    if args.mixup != 0.0:
        train_loader = MixUpWrapper(args.mixup, train_loader)

    val_loader, val_loader_len = get_val_loader(
        args.data,
        args.batch_size,
        args.num_classes,
        False,
        workers=args.workers,
        fp16=args.fp16,
        memory_format=memory_format,
    )
    is_root = (torch.distributed.is_initialized() and torch.distributed.get_rank() == 0) or (
            args.hvd and hvd.rank() == 0) or \
              (not args.hvd and not torch.distributed.is_initialized())
    if is_root:
        logger = log.Logger(
            args.print_freq,
            [
                dllogger.StdOutBackend(
                    dllogger.Verbosity.DEFAULT, step_format=log.format_step
                ),
                dllogger.JSONStreamBackend(
                    dllogger.Verbosity.VERBOSE,
                    os.path.join(args.workspace, args.raport_file),
                ),
            ],
            start_epoch=start_epoch - 1,
            tb_dir=args.tensorboard_dir,
            train_steps_per_epoch=train_loader_len,
            val_steps_per_epoch=val_loader_len,
        )
    else:
        logger = log.Logger(args.print_freq, [], start_epoch=start_epoch - 1)

    logger.log_parameter(args.__dict__, verbosity=dllogger.Verbosity.DEFAULT)
    optimizer = get_optimizer(
        list(model_and_loss.model.named_parameters()),
        args.fp16,
        args.lr,
        args.momentum,
        args.weight_decay,
        nesterov=args.nesterov,
        bn_weight_decay=args.bn_weight_decay,
        state=optimizer_state,
        static_loss_scale=args.static_loss_scale,
        dynamic_loss_scale=args.dynamic_loss_scale,
    )

    if args.lr_schedule == "step":
        lr_policy = lr_step_policy(
            args.lr, [30, 60, 80], 0.1, args.warmup, logger=logger
        )
        # lr_policy = lr_step_policy(
        #     args.lr, [3, 12, 20], 0.01, args.warmup, logger=logger
        # )
    elif args.lr_schedule == "cosine":
        lr_policy = lr_cosine_policy(args.lr, args.warmup, args.epochs, logger=logger)
    elif args.lr_schedule == "linear":
        lr_policy = lr_linear_policy(args.lr, args.warmup, args.epochs, logger=logger)

    bb_settings = None
    compression = None
    if args.hvd:
        quantizer = MaxMinQuantizer(args.quantization_bits, args.bucket_size, enable_error_correction=True, named_parameters=model_and_loss.model.named_parameters())
        adjuster = KMeanAdjuster(quantizer)
        quantizer.add_adjuster(adjuster)
        compression = CompressorManager({OTHERS: quantizer}, model_and_loss.model.named_parameters())
        optimizer = gcomp_DistributedOptimizer(optimizer,
                                             named_parameters=model_and_loss.model.named_parameters(),
                                             op=hvd.Average, compression=compression)

    if args.amp:
        model_and_loss, optimizer = amp.initialize(
            model_and_loss,
            optimizer,
            opt_level=args.opt_level,
            loss_scale="dynamic" if args.dynamic_loss_scale else args.static_loss_scale,
            verbosity=0,
        )

    if args.distributed:
        model_and_loss.distributed(powersgd_rank=args.powersgd_rank)
    if is_root:
        print("Model size {}".format(count_parameters(model_and_loss)))
    model_and_loss.load_model_state(model_state)
    if args.hvd and hvd.size() > 1:
        hvd.broadcast_parameters(model_and_loss.model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    if args.backend == 'qmpi':
        layers = [(name, p.numel()) for name, p in model_and_loss.model.named_parameters()]
        torch_qmpi.register_model(layers)
        torch_qmpi.exclude_layer("bn")
        torch_qmpi.exclude_layer("bias")

    train_loop(
        model_and_loss,
        optimizer,
        lr_policy,
        train_loader,
        val_loader,
        args.fp16,
        logger,
        should_backup_checkpoint(args),
        use_amp=args.amp,
        batch_size_multiplier=batch_size_multiplier,
        start_epoch=start_epoch,
        end_epoch=(start_epoch + args.run_epochs)
        if args.run_epochs != -1
        else args.epochs,
        best_prec1=best_prec1,
        prof=args.prof,
        skip_training=args.evaluate,
        skip_validation=args.training_only,
        save_checkpoints=args.save_checkpoints and not args.evaluate,
        checkpoint_dir=args.workspace,
        checkpoint_filename=args.checkpoint_filename,
        bb_settings=bb_settings,
        compression=compression,
        args=args
    )
    exp_duration = time.time() - exp_start_time
    if is_root:
        logger.end()
    print("Experiment ended, total time: {:.2f}".format(exp_duration))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")

    add_parser_arguments(parser)
    args = parser.parse_args()
    cudnn.benchmark = True

    main(args)
