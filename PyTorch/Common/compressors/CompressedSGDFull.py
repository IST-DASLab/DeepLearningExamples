import torch
from torch.optim import Optimizer
from contextlib import contextmanager
try:
    import horovod.torch as hvd
except ImportError:
    print(
        "Horovod is not installed"
    )


class _CompressedSGDBig(Optimizer):
    def __init__(self, params,
                 compressor):
        super(self.__class__, self).__init__(params)
        self.compressor = compressor
        self._step = 0

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        grads = []
        idx = 0
        offsets = {}
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grads.append(p.grad.data.view(-1))
                offsets[p] = idx
                idx += p.grad.numel()
        g_ = torch.cat(grads)
        self.compressor.compress(g_, self._step)
        # self.compressor.compress(g_)
        allreduce_(g_, name="topk", op=Average)
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                numel = d_p.numel()
                idx = offsets[p]
                d_p.copy_(g_[idx:idx + numel].view(*d_p.shape))
        self._step += 1
        return super(self.__class__, self).step(closure)

    def synchronize(self):
        pass

    @contextmanager
    def skip_synchronize(self):
        try:
            yield
        finally:
            pass

def CompressedSGDBig(optimizer, compression):
    cls = type(optimizer.__class__.__name__, (optimizer.__class__,),
               dict(_CompressedSGDBig.__dict__))
    return cls(optimizer.param_groups, compression)
