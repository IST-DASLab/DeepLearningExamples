import numpy as np
import horovod.torch as hvd
import pickle

try:
    from horovod.torch.compression import Compressor as hvd_Compressor
    class Compressor(hvd_Compressor):
        def __init__(self):
            super().__init__()

        def compress(self, param, step):
            raise NotImplementedError

        def decompress(self, tensor, ctx):
            return tensor
except ImportError:
    class Compressor:
        def compress(self, param, step):
            raise NotImplementedError


class StatisticsSGDBig(Compressor):
    def __init__(self, filename, flush_freq=10):
        super().__init__()
        self.filename = filename
        self.flush_freq = flush_freq
        self.saved_tensor = None
        self.qs = [x * 1./8 for x in range(1, 8)]

    def compress(self, grad, step):
        if step > 0 and hvd.rank() == 0:
            if step % self.flush_freq == 0:
                self.flush_quantile(grad)

            # for i in range(5):
            #     if (step - i) % self.flush_freq == 0:
            #         tensor = grad.detach().clone()
            #         if i == 0:
            #             self.saved_tensor = tensor
            #         elif self.saved_tensor is not None:
            #             self.flush(step, tensor.sub(self.saved_tensor))

        return grad, None

    def flush_hist(self, step, tensor):
        with open(self.filename + "_" + str(step), 'wb') as f:
            a = tensor.cpu().numpy()
            data = np.histogram(a, bins=500)
            pickle.dump(data, f)

    def flush_quantile(self, tensor):
        buckets = tensor.numel() // 512
        t_cpu = tensor.cpu()
        t = t_cpu[:512 * buckets].view(-1, 512)
        tail = tensor.cpu()[512 * buckets:]
        with open(self.filename, 'a') as f:
            norm = t.norm(p=2, dim=1)
            t.div_(norm[:, None])
            a = t.numpy()
            q = np.quantile(a, q=self.qs, axis=1)
            mean = q.mean(axis=1)
            std = q.std(axis=1)
            output = ','.join([str(f) for f in list(mean)] + [str(f) for f in list(std)]) + '\n'
            f.write(output)

        with open("quantiles_tail.csv", 'a') as f:
            norm = tail.norm(p=2)
            tail.div_(norm)
            a = tail.numpy()
            q = np.quantile(a, q=self.qs)
            output = ','.join([str(f) for f in list(q)]) + '\n'
            f.write(output)
