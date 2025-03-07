# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from contextlib import contextmanager
try:
    import horovod.torch as hvd
except ModuleNotFoundError:
    pass
import torch

hvd_initialized = False

def init_hvd():
    hvd.init()
    global hvd_initialized
    hvd_initialized = True


def init_distributed(cuda, backend='nccl'):
    """
    Initializes distributed backend.

    :param cuda: (bool) if True initializes nccl backend, if False initializes
        gloo backend
    """
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    distributed = (world_size > 1)
    if backend == 'qmpi':
       import torch_qmpi
    if distributed:
        # backend = 'nccl' if cuda else 'gloo'
        torch.distributed.init_process_group(backend=backend,
                                             init_method='env://')
        assert torch.distributed.is_initialized()
    return distributed


def barrier():
    """
    Call torch.distributed.barrier() if distritubed is in use
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
    elif hvd_initialized:
        hvd.allreduce(torch.tensor([0.0]), name="Barrier")
        # hvd.push_pull(torch.tensor([0.0]), name="Barrier")


def get_rank():
    """
    Gets distributed rank or returns zero if distributed is not initialized.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    elif hvd_initialized:
        rank = hvd.rank()
    else:
        rank = 0
    return rank


def get_world_size():
    """
    Gets total number of distributed workers or returns one if distributed is
    not initialized.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
    elif hvd_initialized:
        world_size = hvd.size()
    else:
        world_size = 1
    return world_size


def all_reduce_item(value, op='sum'):
    """
    All-reduces single scalar value if distributed is in use
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized() or hvd_initialized:
        if hvd_initialized:
            tensor = torch.tensor([value])
            if op == 'sum' or op == 'mean':
                hop = hvd.Sum if op == 'sum' else hvd.Average
                tensor = hvd.allreduce(tensor, op=hop)
                # tensor = hvd.push_pull(tensor, op=hop)
            elif op == 'min' or op == 'max' or op == 'product':
                tensor = hvd.allgather(tensor)
                if op == 'min':
                    tensor = torch.min(tensor)
                elif op == 'max':
                    tensor = torch.max(tensor)
                elif op == 'product':
                    tensor = torch.prod(tensor)
            else:
                raise RuntimeError('Unsupported reduce op')
            ret = tensor.item()
        else:
            if op == 'sum' or op == 'mean':
                dop = torch.distributed.ReduceOp.SUM
            elif op == 'min':
                dop = torch.distributed.ReduceOp.MIN
            elif op == 'max':
                dop = torch.distributed.ReduceOp.MAX
            elif op == 'product':
                dop = torch.distributed.ReduceOp.PRODUCT
            else:
                raise RuntimeError('Unsupported reduce op')

            backend = torch.distributed.get_backend()
            if backend == torch.distributed.Backend.NCCL or backend == 'qmpi':
                device = torch.device('cuda')
            elif backend == torch.distributed.Backend.GLOO:
                device = torch.device('cpu')
            else:
                raise RuntimeError('Unsupported distributed backend')

            tensor = torch.tensor(value, device=device)
            torch.distributed.all_reduce(tensor, dop)
            if op == 'mean':
                tensor /= get_world_size()
            ret = tensor.item()
    else:
        ret = value
    return ret


@contextmanager
def sync_workers():
    """
    Yields distributed rank and synchronizes all workers on exit.
    """
    rank = get_rank()
    yield rank
    barrier()
