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
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from . import logger as log
from . import resnet as models
from . import utils
import dllogger

try:
    import horovod.torch as hvd
except ImportError:
    print(
        "Horovod is not installed"
    )
import json

import sys
try:
    from gcomp_sim import Compressor, CompressorManager
    grad_sim_available=True
except ImportError:
    grad_sim_available=False
    pass


try:
    # from apex.parallel import DistributedDataParallel as DDP
    from torch.nn.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp
except ImportError:
    raise ImportError(
        "Please install apex from https://www.github.com/nvidia/apex to run this example."
    )
import torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook as powerSGD

ACC_METADATA = {"unit": "%", "format": ":.2f"}
IPS_METADATA = {"unit": "img/s", "format": ":.2f"}
TIME_METADATA = {"unit": "s", "format": ":.5f"}
LOSS_METADATA = {"format": ":.5f"}


class ModelAndLoss(nn.Module):
    def __init__(
            self,
            arch,
            loss,
            pretrained_weights=None,
            cuda=True,
            fp16=False,
            memory_format=torch.contiguous_format,
    ):
        super(ModelAndLoss, self).__init__()
        self.arch = arch

        print("=> creating model '{}'".format(arch))
        model = models.build_resnet(arch[0], arch[1], arch[2])
        if pretrained_weights is not None:
            print("=> using pre-trained model from a file '{}'".format(arch))
            model.load_state_dict(pretrained_weights)

        if cuda:
            model = model.cuda()  # .to(memory_format=memory_format)
        if fp16:
            model = network_to_half(model)

        # define loss function (criterion) and optimizer
        criterion = loss()

        if cuda:
            criterion = criterion.cuda()

        self.model = model
        self.loss = criterion

    def forward(self, data, target):
        output = self.model(data)
        loss = self.loss(output, target)

        return loss, output

    def distributed(self, powersgd_rank=None):
        # self.model = DDP(self.model)
        rank = torch.distributed.get_rank()
        self.model = DDP(self.model, device_ids=[rank], output_device=rank)
        if powersgd_rank:
            state = powerSGD.PowerSGDState(torch.distributed.group.WORLD,
                                           matrix_approximation_rank=powersgd_rank, warm_start=True, start_powerSGD_iter=2,
                                           use_error_feedback=True)
            self.model.register_comm_hook(state, powerSGD.powerSGD_hook)
            # self.model.register_comm_hook(torch.distributed.group.WORLD, self.encode_and_decode)

    # def distributed(self, fake_comp_ratio=1.0, num_allreduce_streams=1, hvd_dist=False):
    #     self.model = DDP(self.model, fake_comp_ratio=fake_comp_ratio, num_allreduce_streams=num_allreduce_streams,
    #                      hvd_dist=hvd_dist, allreduce_always_fp32=True)
    #     # self.model = DDP(self.model, hvd_dist=hvd_dist, allreduce_always_fp32=True)

    def load_model_state(self, state):
        if not state is None:
            self.model.load_state_dict(state)


def get_optimizer(
        parameters,
        fp16,
        lr,
        momentum,
        weight_decay,
        nesterov=False,
        state=None,
        static_loss_scale=1.0,
        dynamic_loss_scale=False,
        bn_weight_decay=False,
):
    if bn_weight_decay:
        print(" ! Weight decay applied to BN parameters ")
        optimizer = torch.optim.SGD(
            [v for n, v in parameters],
            lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
    else:
        print(" ! Weight decay NOT applied to BN parameters ")
        bn_params = [v for n, v in parameters if "bn" in n]
        rest_params = [v for n, v in parameters if not "bn" in n]
        print(len(bn_params))
        print(len(rest_params))
        optimizer = torch.optim.SGD(
            [
                {"params": bn_params, "weight_decay": 0},
                {"params": rest_params, "weight_decay": weight_decay},
            ],
            lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
    if fp16:
        optimizer = FP16_Optimizer(
            optimizer,
            static_loss_scale=static_loss_scale,
            dynamic_loss_scale=dynamic_loss_scale,
            verbose=False,
        )
    if not state is None:
        optimizer.load_state_dict(state)

    return optimizer


def lr_policy(lr_fn, logger=None):
    if logger is not None:
        logger.register_metric(
            "lr", log.LR_METER(), verbosity=dllogger.Verbosity.VERBOSE
        )

    def _alr(optimizer, iteration, epoch):
        lr = lr_fn(iteration, epoch)

        if logger is not None:
            logger.log_metric("lr", lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    return _alr


def lr_step_policy(base_lr, steps, decay_factor, warmup_length, logger=None):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            lr = base_lr
            for s in steps:
                if epoch >= s:
                    lr *= decay_factor
        return lr

    return lr_policy(_lr_fn, logger=logger)


def lr_linear_policy(base_lr, warmup_length, epochs, logger=None):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = base_lr * (1 - (e / es))
        return lr

    return lr_policy(_lr_fn, logger=logger)


def lr_cosine_policy(base_lr, warmup_length, epochs, logger=None):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        return lr

    return lr_policy(_lr_fn, logger=logger)


def lr_exponential_policy(
        base_lr, warmup_length, epochs, final_multiplier=0.001, logger=None
):
    es = epochs - warmup_length
    epoch_decay = np.power(2, np.log2(final_multiplier) / es)

    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            lr = base_lr * (epoch_decay ** e)
        return lr

    return lr_policy(_lr_fn, logger=logger)


def get_train_step(
        model_and_loss, optimizer, fp16, use_amp=False, batch_size_multiplier=1
):
    hvd_enabled = utils.horovod_enabled()

    def _step(input, target, optimizer_step=True, compression=None):
        # optimizer.zero_grad()
        input_var = Variable(input)
        target_var = Variable(target)
        loss, output = model_and_loss(input_var, target_var)
        if torch.distributed.is_initialized():
            reduced_loss = utils.reduce_tensor(loss.data)
        else:
            reduced_loss = loss.data
        if fp16:
            optimizer.backward(loss)
        elif use_amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
                if hvd_enabled:
                    optimizer.synchronize()
        else:
            loss.backward()
        if optimizer_step:
            opt = (
                optimizer.optimizer
                if isinstance(optimizer, FP16_Optimizer)
                else optimizer
            )
            for param_group in opt.param_groups:
                for param in param_group["params"]:
                    if param.requires_grad:
                        param.grad /= batch_size_multiplier

            if use_amp and hvd_enabled:
                with optimizer.skip_synchronize():
                    optimizer.step()
            else:
                optimizer.step()
            optimizer.zero_grad()

        torch.cuda.synchronize()

        return reduced_loss

    return _step


def train(
        train_loader,
        model_and_loss,
        optimizer,
        lr_scheduler,
        fp16,
        logger,
        epoch,
        use_amp=False,
        prof=-1,
        batch_size_multiplier=1,
        register_metrics=True,
        compression=None,
):
    if register_metrics and logger is not None:
        logger.register_metric(
            "train.loss",
            log.LOSS_METER(),
            verbosity=dllogger.Verbosity.DEFAULT,
            metadata=LOSS_METADATA,
            tb=True,
        )
        logger.register_metric(
            "train.compute_ips",
            log.PERF_METER(),
            verbosity=dllogger.Verbosity.VERBOSE,
            metadata=IPS_METADATA,
        )
        logger.register_metric(
            "train.total_ips",
            log.PERF_METER(),
            verbosity=dllogger.Verbosity.DEFAULT,
            metadata=IPS_METADATA,
            tb=True,
        )
        logger.register_metric(
            "train.data_time",
            log.PERF_METER(),
            verbosity=dllogger.Verbosity.DEFAULT,
            metadata=TIME_METADATA,
        )
        logger.register_metric(
            "train.compute_time",
            log.PERF_METER(),
            verbosity=dllogger.Verbosity.DEFAULT,
            metadata=TIME_METADATA,
        )

    step = get_train_step(
        model_and_loss,
        optimizer,
        fp16,
        use_amp=use_amp,
        batch_size_multiplier=batch_size_multiplier,
    )

    model_and_loss.train()
    end = time.time()

    optimizer.zero_grad()
    hvd_enabled = utils.horovod_enabled()
    data_iter = enumerate(train_loader)
    if logger is not None:
        data_iter = logger.iteration_generator_wrapper(data_iter)
    if prof > 0:
        data_iter = utils.first_n(prof, data_iter)
    for i, (input, target) in data_iter:
        bs = input.size(0)
        lr_scheduler(optimizer, i, epoch)
        # if hvd_enabled:
        #     torch.cuda.synchronize()
        #     # hvd.allreduce(torch.tensor([0], dtype=torch.int32, device=input.device), name="barrier")
        # elif torch.distributed.is_available() and torch.distributed.is_initialized():
        #     torch.cuda.synchronize()
        #     # torch.distributed.barrier()
        data_time = time.time() - end
        optimizer_step = ((i + 1) % batch_size_multiplier) == 0
        loss = step(input, target, optimizer_step=optimizer_step, compression=compression)

        it_time = time.time() - end

        if logger is not None:
            logger.log_metric("train.loss", to_python_float(loss), bs)
            logger.log_metric("train.compute_ips", calc_ips(bs, it_time - data_time))
            logger.log_metric("train.total_ips", calc_ips(bs, it_time))
            logger.log_metric("train.data_time", data_time)
            logger.log_metric("train.compute_time", it_time - data_time)

        end = time.time()


def get_val_step(model_and_loss):
    def _step(input, target):
        input_var = Variable(input)
        target_var = Variable(target)

        with torch.no_grad():
            loss, output = model_and_loss(input_var, target_var)

        prec1, prec5 = utils.accuracy(output.data, target, topk=(1, 5))

        if torch.distributed.is_initialized() or utils.horovod_enabled():
            reduced_loss = utils.reduce_tensor(loss.data)
            prec1 = utils.reduce_tensor(prec1)
            prec5 = utils.reduce_tensor(prec5)
        else:
            reduced_loss = loss.data

        torch.cuda.synchronize()

        return reduced_loss, prec1, prec5

    return _step


def validate(
        val_loader, model_and_loss, fp16, logger, epoch, prof=-1, register_metrics=True
):
    if register_metrics and logger is not None:
        logger.register_metric(
            "val.top1",
            log.ACC_METER(),
            verbosity=dllogger.Verbosity.DEFAULT,
            metadata=ACC_METADATA,
            tb=True,
        )
        logger.register_metric(
            "val.top5",
            log.ACC_METER(),
            verbosity=dllogger.Verbosity.DEFAULT,
            metadata=ACC_METADATA,
        )
        logger.register_metric(
            "val.loss",
            log.LOSS_METER(),
            verbosity=dllogger.Verbosity.DEFAULT,
            metadata=LOSS_METADATA,
            tb=True,
        )
        logger.register_metric(
            "val.compute_ips",
            log.PERF_METER(),
            verbosity=dllogger.Verbosity.VERBOSE,
            metadata=IPS_METADATA,
        )
        logger.register_metric(
            "val.total_ips",
            log.PERF_METER(),
            verbosity=dllogger.Verbosity.DEFAULT,
            metadata=IPS_METADATA,
        )
        logger.register_metric(
            "val.data_time",
            log.PERF_METER(),
            verbosity=dllogger.Verbosity.VERBOSE,
            metadata=TIME_METADATA,
        )
        logger.register_metric(
            "val.compute_latency",
            log.PERF_METER(),
            verbosity=dllogger.Verbosity.VERBOSE,
            metadata=TIME_METADATA,
        )
        logger.register_metric(
            "val.compute_latency_at100",
            log.LAT_100(),
            verbosity=dllogger.Verbosity.VERBOSE,
            metadata=TIME_METADATA,
        )
        logger.register_metric(
            "val.compute_latency_at99",
            log.LAT_99(),
            verbosity=dllogger.Verbosity.VERBOSE,
            metadata=TIME_METADATA,
        )
        logger.register_metric(
            "val.compute_latency_at95",
            log.LAT_95(),
            verbosity=dllogger.Verbosity.VERBOSE,
            metadata=TIME_METADATA,
        )
    step = get_val_step(model_and_loss)

    top1 = log.AverageMeter()
    # switch to evaluate mode
    model_and_loss.eval()

    end = time.time()

    data_iter = enumerate(val_loader)
    if not logger is None:
        data_iter = logger.iteration_generator_wrapper(data_iter, val=True)
    if prof > 0:
        data_iter = utils.first_n(prof, data_iter)

    for i, (input, target) in data_iter:
        bs = input.size(0)
        data_time = time.time() - end

        loss, prec1, prec5 = step(input, target)
        it_time = time.time() - end

        top1.record(to_python_float(prec1), bs)
        if logger is not None:
            logger.log_metric("val.top1", to_python_float(prec1), bs)
            logger.log_metric("val.top5", to_python_float(prec5), bs)
            logger.log_metric("val.loss", to_python_float(loss), bs)
            logger.log_metric("val.compute_ips", calc_ips(bs, it_time - data_time))
            logger.log_metric("val.total_ips", calc_ips(bs, it_time))
            logger.log_metric("val.data_time", data_time)
            logger.log_metric("val.compute_latency", it_time - data_time)
            logger.log_metric("val.compute_latency_at95", it_time - data_time)
            logger.log_metric("val.compute_latency_at99", it_time - data_time)
            logger.log_metric("val.compute_latency_at100", it_time - data_time)

        end = time.time()

    return top1.get_val()


# Train loop {{{
def calc_ips(batch_size, time):
    if torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
    elif utils.horovod_enabled():
        world_size = hvd.size()
    else:
        world_size = 1
    tbs = world_size * batch_size
    return tbs / time


def train_loop(
        model_and_loss,
        optimizer,
        lr_scheduler,
        train_loader,
        val_loader,
        fp16,
        logger,
        should_backup_checkpoint,
        use_amp=False,
        batch_size_multiplier=1,
        best_prec1=0,
        start_epoch=0,
        end_epoch=0,
        prof=-1,
        skip_training=False,
        skip_validation=False,
        save_checkpoints=True,
        checkpoint_dir="./",
        checkpoint_filename="checkpoint.pth.tar",
        bb_settings=None,
        compression=None,
        args=None
):
    prec1 = -1
    root = False
    try:
        if hvd.rank() == 0:
            root = True
    except:
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            root = True

    print(f"RUNNING EPOCHS FROM {start_epoch} TO {end_epoch}")
    for epoch in range(start_epoch, end_epoch):
        if logger is not None:
            logger.start_epoch()
        end = time.time()
        if not skip_training:
            train(
                train_loader,
                model_and_loss,
                optimizer,
                lr_scheduler,
                fp16,
                logger,
                epoch,
                use_amp=use_amp,
                prof=prof,
                register_metrics=epoch == start_epoch,
                batch_size_multiplier=batch_size_multiplier,
                compression=compression
            )
        if root:
            print("Time per epoch: ", time.time() - end)
        if not skip_validation:
            prec1, nimg = validate(
                val_loader,
                model_and_loss,
                fp16,
                logger,
                epoch,
                prof=prof,
                register_metrics=epoch == start_epoch,
            )
        if logger is not None:
            logger.end_epoch()

        # if issubclass(type(compression), Compressor):
        #     start_adjust = time.time()
        #     num_samples = 5
        #     data_iter = enumerate(train_loader)
        #     data_iter = list(utils.first_n(num_samples, data_iter))
        #     if root:
        #         compression.compute_eigen_values(model_and_loss.model, model_and_loss.loss, data_iter)

        # if epoch >= args.warmup and issubclass(type(compression), Compressor):
        if grad_sim_available and (issubclass(type(compression), Compressor) or isinstance(compression, CompressorManager)):
            start_adjust = time.time()
            e = epoch + 1
            if e % args.adapt_compression_adjust_freq == 0 and epoch >= args.warmup:
                compression.adjust_params()
                if root:
                    print("Time for adjusting: ", time.time() - start_adjust)
            if root:
                d = compression.get_all_metrics()["MaxMinQuantizer"]
                if d:
                    total = np.linalg.norm(np.array(list(d.values())), ord=2)
                    print("Total norm: ", total)
                    d["total"] = total
                    directory = os.path.join(checkpoint_dir, "adapt_logs".format(compression.bits))
                    os.makedirs(directory, exist_ok=True)
                    file = "epoch{}.json".format(epoch)
                    with open(os.path.join(directory, file), 'w') as f:
                        # d = {k: v / sum for k,v in d.items()}
                        print(d)
                        json.dump(d, f)
                    if compression.is_adaptive and e % args.adapt_compression_adjust_freq == 0 and epoch >= args.warmup:
                        d = compression.get_compression_scheme()["MaxMinQuantizer"]
                        print(d)
                        file = "compress_scheme_{}.json".format(epoch)
                        with open(os.path.join(directory, file), 'w') as f:
                            json.dump(d, f)
            compression.reset_metrics()

        if save_checkpoints and root:
            if not skip_validation:
                is_best = logger.metrics["val.top1"]["meter"].get_epoch() > best_prec1
                best_prec1 = max(
                    logger.metrics["val.top1"]["meter"].get_epoch(), best_prec1
                )
            else:
                is_best = False
                best_prec1 = 0

            if should_backup_checkpoint(epoch):
                backup_filename = "checkpoint-{}.pth.tar".format(epoch + 1)
            else:
                backup_filename = None
            utils.save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": model_and_loss.arch,
                    "state_dict": model_and_loss.model.state_dict(),
                    "best_prec1": best_prec1,
                    "optimizer": optimizer.state_dict(),
                },
                is_best,
                checkpoint_dir=checkpoint_dir,
                backup_filename=backup_filename,
                filename=checkpoint_filename,
            )

# }}}
