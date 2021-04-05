import torch
from .compressor import Compressor


class TopKCompressorBig(Compressor):
    def __init__(self, k_perc=1.0, enable_error_correction=None, warmup_steps=1000, ef_q=None):
        super().__init__()
        self.k_perc = k_perc
        self.enable_error_correction = enable_error_correction
        self.eps = 1e-10
        self.error_correction = None
        self.warmup_steps = warmup_steps
        self.ef_q = ef_q
        if self.ef_q is not None:
            for key in ["bits", "bucket_size", "rand_k"]:
                self.ef_q.setdefault(key, None)


    def compress(self, grad, step):
        # We get param as tensor not Variable here
        if self.error_correction is None and self.enable_error_correction:
            self.error_correction = torch.full_like(grad, 0.0, memory_format=torch.preserve_format)
        if step < self.warmup_steps:
            return
        e_c = self.error_correction
        if self.enable_error_correction:
            grad.add_(e_c)
            e_c.copy_(grad)

        self._zero(grad)

        if self.enable_error_correction:
            e_c.sub_(grad)
        return grad, None

    def _topk(self, grad):
        num_zero = int(grad.numel() * (1 - self.k_perc))
        # values, idx = grad.abs().topk(num_zero, sorted=False, largest=False)
        sort, indices = grad.abs().sort(descending=False)
        _, idx = sort[:num_zero], indices[:num_zero]
        if self.ef_q is not None and self.ef_q["rand_k"] is not None:
            zero_k = int(idx.numel() * (1 - self.ef_q["rand_k"]))
            idx = idx[torch.randperm(idx.numel())[:zero_k]]
        return idx

    def _zero(self, grad):
        idx = self._topk(grad)
        if self.ef_q is not None and self.ef_q["rand_k"] is None:
            values = grad.gather(0, idx)
            # self.quantize(values.view(-1))
            # g_.scatter_(g_.dim() - 1, indices, values)
            ef = torch.zeros_like(grad).scatter_(0, idx, values)
            self.quantize(ef)
            # g_.scatter_(g_.dim() - 1, indices, 0.0).add_(ef)
            values = ef.gather(0, idx)
            grad.scatter_(0, idx, values)
        else:
            grad.scatter_(0, idx, 0.0)

    def quantize(self, buf):
        q_bits = self.ef_q["bits"]
        if q_bits == 32 or q_bits <= 0:
            return buf
        levels = 1 << q_bits
        numel = buf.numel()
        bucket_size = self.ef_q["bucket_size"] if "bucket_size" in self.ef_q else numel
        main_chunk_size = (numel // bucket_size) * bucket_size
        tail_chunk_size = numel - main_chunk_size
        if main_chunk_size > 0:
            r_ = buf[:main_chunk_size].view((-1, bucket_size))
            buf[:main_chunk_size] = self.quantize_bucket(r_, levels).view(-1)
        # if tail_chunk_size > 0:
        #     r_ = buf[main_chunk_size:]
        #     buf[main_chunk_size:] = self.quantize_bucket(r_, levels)
        return buf

    # def quantize_bucket(self, a, levels):
    #     if a.dim() == 2:
    #         vnorm = torch.norm(a, p=2, dim=1)
    #         vnorm = vnorm[:, None]
    #         s = torch.Tensor([1e-11]).expand_as(vnorm).to(a.device)
    #     else:
    #         vnorm = torch.norm(a, p=2)
    #         s = torch.Tensor([1e-11]).to(a.device)
    #
    #     vnorm = torch.max(vnorm, s)
    #     sign = torch.sign(a)
    #     # sign.add_(1).div_(2)
    #     # sign.mul_(2).add_(-1)
    #     if levels <= 1:
    #         return vnorm * sign
    #     a = torch.abs(a / vnorm)
    #     logs = torch.log2(a)
    #     logs[logs == -float("inf")] = -32.0
    #
    #     max_pow = torch.zeros_like(logs) + torch.max(logs).int()
    #     min_pow = max_pow - levels + 2
    #     now = torch.max(min_pow, logs).float()
    #     l = torch.pow(2.0, now - 1)
    #     r = 2 * l
    #     a = torch.min(r, torch.max(a, l))
    #     rand = torch.rand(a.size(), device=a.device)
    #     c = (a - l) / (r - l)
    #     a = l * c.le(rand).float() + r * c.gt(rand).float()
    #     return a * vnorm * sign

    def quantize_bucket(self, a, levels):
        if a.dim() == 2:
            fmin = torch.min(a, dim=1)[0]
            fmax = torch.max(a, dim=1)[0]
            unit = (fmax - fmin) / (levels - 1)
            unit = unit[:, None]
            fmin = fmin[:, None]
            s = torch.Tensor([1e-11]).expand_as(unit).to(a.device)
        else:
            fmin = torch.min(a)
            fmax = torch.max(a)
            unit = (fmax - fmin) / (levels - 1)
            s = torch.Tensor([1e-11]).to(a.device)

        unit = torch.max(unit, s)
        a -= fmin
        a /= unit
        a += torch.empty(a.size(), device=a.device).uniform_(0, 1)
        a.floor_()
        a *= unit
        a += fmin
        return a



class TopKCompressor(Compressor):
    def __init__(self, k_perc=1.0, bucket_size=512, enable_error_correction=None, warmup_steps=1000, ef_q=None):
        super().__init__()
        self.k_perc = k_perc
        self.enable_error_correction = enable_error_correction
        self.eps = 1e-10
        self.bucket_size = bucket_size
        self.state = {}
        self.warm_up = warmup_steps
        self.ef_q = ef_q
        if self.ef_q is not None:
            for key in ["bits", "bucket_size", "rand_k"]:
                self.ef_q.setdefault(key, None)

    def compress(self, p, step):
        if not p.requires_grad:
            return p, None
        grad = p.grad.data
        if self.warm_up > step:
            return grad, None
        if p not in self.state:
            self.state[p] = {}
            if self.enable_error_correction:
                self.state[p]["error_correction"] = torch.zeros_like(grad)
        state = self.state[p]

        if self.enable_error_correction:
            e_c = state["error_correction"]
            # add error correction
            grad.add_(e_c)
            # update error correction before subtraction
            e_c.copy_(grad)

        grad_ = grad.view(-1)
        self._zero(grad_)

        if self.enable_error_correction:
            e_c = state["error_correction"]
            e_c.sub_(grad)
        return p.grad, None

    def _topk(self, grad):
        num_zero = int(grad.numel() * (1 - self.k_perc))
        # values, idx = grad.abs().topk(num_zero, sorted=False, largest=False)
        sort, indices = grad.abs().sort(descending=False)
        _, idx = sort[:num_zero], indices[:num_zero]
        return idx

    def _zero(self, grad):
        idx = self._topk(grad)
        if self.ef_q is not None and self.ef_q["rand_k"] is None:
            values = grad.gather(0, idx)
            # self.quantize(values.view(-1))
            # g_.scatter_(g_.dim() - 1, indices, values)
            ef = torch.zeros_like(grad).scatter_(0, idx, values)
            self.quantize(ef)
            # g_.scatter_(g_.dim() - 1, indices, 0.0).add_(ef)
            values = ef.gather(0, idx)
            grad.scatter_(0, idx, values)
        else:
            grad.scatter_(0, idx, 0.0)

    def quantize(self, buf):
        q_bits = self.ef_q["bits"]
        if q_bits == 32 or q_bits <= 0:
            return buf
        numel = buf.numel()
        bucket_size = self.ef_q["bucket_size"] if "bucket_size" in self.ef_q else numel
        main_chunk_size = (numel // bucket_size) * bucket_size
        tail_chunk_size = numel - main_chunk_size
        if main_chunk_size > 0:
            r_ = buf[:main_chunk_size].view((-1, bucket_size))
            self.quantize_bucket(r_, q_bits)
        # if tail_chunk_size > 0:
        #     r_ = buf[main_chunk_size:]
        #     self.quantize_bucket(r_, q_bits)
        return buf

    def quantize_bucket(self, a, q_bits):
        levels = 1 << q_bits
        if a.dim() == 2:
            fmin = torch.min(a, dim=1)[0]
            fmax = torch.max(a, dim=1)[0]
            unit = (fmax - fmin) / (levels - 1)
            unit = unit[:, None]
            fmin = fmin[:, None]
            s = torch.Tensor([1e-11]).expand_as(unit).to(a.device)
        else:
            fmin = torch.min(a)
            fmax = torch.max(a)
            unit = (fmax - fmin) / (levels - 1)
            s = torch.Tensor([1e-11]).to(a.device)

        unit = torch.max(unit, s)
        a -= fmin
        a /= unit
        a += torch.empty(a.size(), device=a.device).uniform_(0, 1)
        a.floor_()
        a *= unit
        a += fmin
        return a


class TopKRMSProp(TopKCompressor):
    def __init__(self, k_perc, bucket_size, enable_error_correction, enable_stats=False, warmup_steps=1000, mask_update_freq=1):
        super().__init__(k_perc, bucket_size, enable_error_correction=enable_error_correction, warmup_steps=warmup_steps, mask_update_freq=mask_update_freq)
        self.sum = {}
        self.stats = {}
        self.enable_stats = enable_stats
        self.alpha = 0.99

    def get_stats(self, params):
        result = []
        for p in params:
            result.append(self.state[p]["stats"])
        return torch.cat(result)

    def compress(self, p, step):
        if not p.requires_grad:
            return p, None
        grad = p.grad.data
        if p not in self.state:
            self.state[p] = {}
            self.state[p]["sum"] = torch.full_like(grad, 0.0, memory_format=torch.preserve_format)
            self.state[p]["error_correction"] = torch.full_like(grad, 0.0, memory_format=torch.preserve_format)
            self.state[p]["stats"] = torch.full_like(grad.view(-1), 0, memory_format=torch.preserve_format)
        state = self.state[p]
        sum = state["sum"]
        e_c = state["error_correction"]
        stats = state["stats"]
        if self.enable_error_correction and step > self.warm_up:
            # add error correction
            grad.add_(e_c)
            # update error correction before subtraction
            e_c.copy_(grad)

        sum.mul_(self.alpha).addcmul_(grad, grad, value=1 - self.alpha)
        # grad / (sqrt(sum) + eps)
        std = sum.sqrt().add_(self.eps).pow_(-1).mul_(grad).abs_().view(-1)
        if torch.max(std) - torch.min(std) > self.eps:
            g_ = grad.view(-1)
            numel = g_.numel()
            bucket_size = self.bucket_size if self.bucket_size else numel
            for i in range((numel + bucket_size - 1) // bucket_size):
                start_idx = i * bucket_size
                end_idx = min(numel, (i + 1) * bucket_size)
                if step % self.mask_update_freq == 0 or p not in self.masks or i >= len(self.masks[p]):
                    self._compute_mask(std[start_idx:end_idx], p, i)
                if step > self.warm_up:
                    self.compress_bucket(g_, stats, start_idx, end_idx, self.masks[p][i])
        if self.enable_error_correction and step > self.warm_up:
            e_c.sub_(grad)
        return p.grad, None

    def _compute_mask(self, std_, p, bucket_id):
        zeroed_num = std_.numel() - int(self.k_perc * std_.numel())
        _, idc = torch.topk(std_, zeroed_num, largest=False)
        if p in self.masks:
            if bucket_id < len(self.masks[p]):
                self.masks[p][bucket_id] = idc
            else:
                self.masks[p].append(idc)
        else:
            self.masks[p] = [idc]

    def compress_bucket(self, grad, stats, start_idx, end_idx, mask):
        g_ = grad[start_idx:end_idx]
        stats_ = stats[start_idx:end_idx]
        g_[mask] = 0.0
        if self.enable_stats:
            stats_ += 1
            stats_[mask] -= 1
