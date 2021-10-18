import torch
from .compressor import Compressor
from .adjuster import LinearAdjuster, BayesianAdjuster, BestPerLayer, KMeanAdjuster

from .pyhessian import hessian
import copy
from apex import amp

try:
    import horovod.torch as hvd
except ImportError:
    print(
        "Horovod is not installed"
    )

class Quantizer(Compressor):
    def __init__(self, bits, bucket_size, enable_error_correction=False, named_parameters=None, adaptive=False):
        super().__init__()
        self.bits = bits
        self.bits_min = 2
        self.bits_max = 4
        self.bits_default = bits
        self.num_levels = 1 << bits
        self.bucket_size = bucket_size
        self.bucket_size_default = bucket_size
        self.buckets_bits_mapping = {
            1: 32,
            2: 128,
            3: min(256, self.bucket_size_default),
            4: self.bucket_size_default
        }

        self.states = {}
        self.save_error_correction = enable_error_correction
        if adaptive:
            self.apply_error_correction = False
        else:
            self.apply_error_correction = enable_error_correction
        self.momentum_acc = 0.8
        self.excluded_layer_names = ["layer_norm", "bias", "bn"]
        self.is_adaptive = adaptive
        self.adjuster = KMeanAdjuster(self)
        # self.adjuster = LinearAdjuster(self)
        # self.adjuster = BayesianAdjuster(self)
        # self.adjuster = BestPerLayer(self)
        if named_parameters:
            named_parameters = list(named_parameters)
            self.named_parameters = {p: name for name, p in named_parameters if p.requires_grad}
            self.parameters_sizes = {name: p.numel() for name, p in named_parameters if p.requires_grad}
            for name, p in named_parameters:
                if not p.requires_grad:
                    continue
                skip_layer = False
                for excluded_name in self.excluded_layer_names:
                    if excluded_name in name:
                        skip_layer = True
                        break
                if skip_layer:
                    continue
                self.states[p] = {}
                self.states[p]["bits"] = self.bits_default
                self.states[p]["name"] = self.named_parameters[p]
                if self.named_parameters[p] == "word_emb.emb_layers.0.weight":
                    self.states[p]["bits"] = 3

        # self.exact_match = {name: False for name in named_parameters}
            # for i in range(len(named_parameters)):
            #     for j in range(i + 1, len(named_parameters)):
            #         if named_parameters[i] in named_parameters[j]:
            #             self.exact_match[named_parameters[j]] = True

        else:
            self.named_parameters = None
            self.parameters_sizes = None

    def get_states(self):
        return self.states

    def reset_metrics(self):
        for p, state in self.states.items():
            if "momentum" in state:
                state["momentum"].fill_(0.0)
            # state["bits"] = self.bits_default

    def update_metric_stats(self, parameters):
        if not self.is_adaptive:
            return
        for p in parameters:
            if not p.requires_grad or p not in self.states:
                continue
            state = self.states[p]
            d_p = p.grad.data
            if p.grad.data.dtype == torch.float16:
                d_p = d_p.float()
            if "momentum" not in state:
                state["momentum"] = torch.zeros_like(d_p)
            state["momentum"].add_(d_p, alpha=self.momentum_acc)

    def get_metric(self, p, compute=True):
        state = self.states[p]
        buf = state["momentum"]
        # return torch.norm(buf, p=2).item()
        estimate_num = 100
        values, _ = buf.abs().view(-1).topk(min(estimate_num, buf.numel()))
        return torch.norm(values, p=2).item()

    # def get_metric(self, p, compute=True):
    #     state = self.states[p]
    #     return state["hess_eigen_value"]

    # def get_metric(self, p, compute=True):
    #     state = self.states[p]
    #     if compute:
    #         buf = state["momentum"]
    #         self.set_num_levels(p)
    #         if torch.isinf(buf).sum() > 0:
    #             state["compression_error"] = float("inf")
    #         else:
    #             state["compression_error"] = (buf - self._compress(buf, inplace=False)).norm(p=2).item()
    #     return state["compression_error"]
        # return state["momentum"].norm(p=2).item()
        # return state["error_correction"].norm(p=2).item()

    def adjust_bits(self):
        if not self.states or not self.is_adaptive:
            return
        if hvd.rank() == 0:
            best_bits = self.adjuster.fit_predict(self.states, self.bits_min, self.bits_max)
            hvd.broadcast_object(best_bits, root_rank=0)
        else:
            best_bits = []
            best_bits = hvd.broadcast_object(best_bits, root_rank=0)
        self.set_bits_states(best_bits)
        for p, state in self.states.items():
            state["momentum"].mul_(self.momentum_acc)

    def get_metrics_magnitudes(self):
        d = {}
        sum = 0.0
        if self.named_parameters and self.is_adaptive:
            for p, name in self.named_parameters.items():
                if p in self.states:
                    d[name] = self.get_metric(p)
                    sum += d[name]
        return d, sum

    def get_compression_scheme(self):
        if not self.is_adaptive:
            return
        d = {32: self.excluded_layer_names.copy()}
        for p, state in self.states.items():
            name = self.named_parameters[p]
            bits = state["bits"]
            if bits in d:
                d[bits].append(name)
            else:
                d[bits] = [name]
        return d

    def compute_eigen_values(self, model, criterion, dataloader):
        model_copy = copy.deepcopy(model)
        h = hessian(model_copy, criterion, dataloader=dataloader)
        eigenvalues, _ = h.eigenvalues()
        eigenvalues = eigenvalues[0]
        for i, p in enumerate(model.parameters()):
            if p not in self.states:
                continue
            self.states[p]["hess_eigen_value"] = eigenvalues[i]

    def set_bits_states(self, bits):
        assert len(bits) == len(self.states)
        for bit, (p, state) in zip(bits, self.states.items()):
            state["bits"] = bit

    def set_num_levels(self, p):
        bits = self.states[p]["bits"]
        self.num_levels = 1 << bits
        if bits in self.buckets_bits_mapping:
            self.bucket_size = self.buckets_bits_mapping[bits]
        else:
            self.bucket_size = self.bucket_size_default
        # self.bucket_size = self.bucket_size_default // (1 << (self.bits_max - bits))

    def quantize_bucket(self, a):
        raise NotImplementedError

    def update_stats(self, a):
        pass

    def compress(self, p, step=None):
        if not p.requires_grad:
            return p, None
        if self.named_parameters:
            for skip_name in self.excluded_layer_names:
                if skip_name in self.named_parameters[p]:
                    return p.grad, None
        d_p = p.grad.data
        if p.grad.data.dtype == torch.float16:
            self.adjuster.set_original_bits(16)
            loss_scale = amp.state_dict()["loss_scaler0"]['loss_scale']
            d_p = d_p.float().div_(loss_scale)

        state = self.states[p]
        if self.save_error_correction:
            if "error_correction" not in state:
                state["error_correction"] = torch.zeros_like(d_p)
            e_c = state["error_correction"]
            # update error correction before subtraction
            if self.momentum_acc:
                e_c.mul_(self.momentum_acc)
            e_c.add_(d_p)
            # add error correction
            if self.apply_error_correction:
                d_p.copy_(e_c)
        self.set_num_levels(p)
        self._compress(d_p)
        if self.save_error_correction:
            e_c.sub_(d_p)
            # grad_copy.sub_(d_p)
        if p.grad.data.dtype == torch.float16:
            p.grad.data.copy_(d_p.mul_(loss_scale).half())
        return p.grad, None

    def _compress(self, tensor, inplace=True):
        if not inplace:
            tensor = tensor.clone()
        a = tensor.view(-1)
        numel = a.numel()
        if self.bucket_size == -1:
            a[:] = self.quantize_bucket(a)
        else:
            main_chunk_size = (numel // self.bucket_size) * self.bucket_size
            if main_chunk_size > 0:
                a[:main_chunk_size] = self.quantize_bucket(a[:main_chunk_size].view((-1, self.bucket_size))).view(-1)
            # if numel - main_chunk_size > 0:
            #     a[main_chunk_size:] = self.quantize_bucket(a[main_chunk_size:])
        return tensor

    @staticmethod
    def count_unique(buf):
        sum = 0
        for b in buf:
            sum += torch.unique(b).numel()
        return sum

    def compress_buffer(self, buf, step=None):
        if self.num_levels == 1 << 32:
            return buf
        numel = buf.numel()
        main_chunk_size = (numel // self.bucket_size) * self.bucket_size
        tail_chunk_size = numel - main_chunk_size
        if main_chunk_size > 0:
            r_ = buf[:main_chunk_size].view((-1, self.bucket_size))
            r_[:] = self.quantize_bucket(r_)
            # print("2d Unique: {} vs expected {}".format(self.count_unique(r_),
            #                                             (numel // self.bucket_size) * (1 << self.bits)))
        if tail_chunk_size > 0:
            r_ = buf[main_chunk_size:]
            r_[:] = self.quantize_bucket(r_)
            # print("Unique: {} vs expected {}".format(torch.unique(r_).numel(), (1 << self.bits)))

        return buf

    @staticmethod
    def quantize_1_dim_with_levels(buf, levels):
        rand = torch.rand(buf.size(), device=buf.device)
        res = torch.clamp(buf, levels[0], levels[-1])
        for l1, l2 in zip(levels[:-1], levels[1:]):
            l_l2 = buf.lt(l2)
            g_l1 = buf.ge(l1)
            res[l_l2 * g_l1] = l1
            b = buf + (l2 - l1) * rand
            # if exceeds after random assign l2
            g_l2 = b.ge(l2)
            res[l_l2 * g_l2] = l2
        return res

    @staticmethod
    def quantize_2_dim_with_levels(buf, levels):
        rand = torch.rand(buf.size(), device=buf.device)
        res = torch.max(buf, levels[0])
        res = torch.min(res, levels[-1])
        z = torch.zeros_like(buf)
        for l1, l2 in zip(levels[:-1], levels[1:]):
            l_l2 = buf < l2
            g_l1 = buf >= l1
            idx = l_l2 * g_l1
            # set indexed values to l1
            z[idx] = 1.0
            res[idx] = 0.0
            res.add_(z.mul_(l1))
            z.zero_()
            b = buf + (l2 - l1) * rand
            g_l2 = b >= l2
            idx = l_l2 * g_l2

            z[idx] = 1.0
            res[idx] = 0.0
            res.add_(z.mul_(l2))
            z.zero_()
        return res


class MaxMinQuantizer(Quantizer):
    def __init__(self, bits, bucket_size, enable_error_correction=False, named_parameters=None, adapt_compression=False):
        super().__init__(bits, bucket_size, enable_error_correction, named_parameters, adapt_compression)

    def quantize_bucket(self, a):
        if self.num_levels == 1 << 32 or torch.isinf(a).sum() > 0:
            return a
        if a.dim() == 2:
            fmin = torch.min(a, dim=1)[0]
            fmax = torch.max(a, dim=1)[0]
            unit = (fmax - fmin) / (self.num_levels - 1)
            unit = unit[:, None]
            fmin = fmin[:, None]
            s = torch.Tensor([1e-11]).expand_as(unit).to(a.device)
        else:
            fmin = torch.min(a)
            fmax = torch.max(a)
            unit = (fmax - fmin) / (self.num_levels - 1)
            s = torch.Tensor([1e-11]).to(a.device)

        unit = torch.max(unit, s)
        a -= fmin
        a /= unit
        a += torch.empty(a.size(), device=a.device).uniform_(0, 1)
        torch.floor_(a)
        a *= unit
        a += fmin
        return a


class ExponentialQuantizer(Quantizer):
    def __init__(self, bits, bucket_size, enable_error_correction=False, named_parameters=None, adapt_compression=False):
        super().__init__(bits, bucket_size, enable_error_correction, named_parameters, adapt_compression)
        self.num_levels = self.num_levels // 2
        # self.norm_type = float("inf")
        self.norm_type = 2
        self.levels_1dim = torch.tensor([0.5 ** i for i in range(self.num_levels, 0, -1)])
        self.levels_2dim = self.levels_1dim[:, None]

    def quantize_bucket_new(self, buf):
        sign = buf.sign()
        if buf.dim() == 2:
            vnorm = buf.norm(p=self.norm_type, dim=1)
            vnorm = vnorm[:, None]
        else:
            vnorm = torch.norm(buf, p=self.norm_type)
        if self.bits == 1:
            return sign * vnorm
        a = buf.abs() / vnorm
        if buf.dim() == 2:
            self.levels_2dim = self.levels_2dim.to(buf.device)
            res = self.quantize_2_dim_with_levels(a, self.levels_2dim)
        else:
            self.levels_1dim = self.levels_1dim.to(buf.device)
            res = self.quantize_1_dim_with_levels(a, self.levels_1dim)
        return res * sign * vnorm

    def set_num_levels(self, p):
        bits = self.states[p]["bits"]
        self.num_levels = (1 << bits) // 2

    def quantize_bucket(self, a):
        if a.dim() == 2:
            vnorm = torch.norm(a, p=self.norm_type, dim=1)
            vnorm = vnorm[:, None]
            s = torch.Tensor([1e-11]).expand_as(vnorm).to(a.device)
        else:
            vnorm = torch.norm(a, p=self.norm_type)
            s = torch.Tensor([1e-11]).to(a.device)
        vnorm = torch.max(vnorm, s)
        sign = torch.sign(a)
        sign[sign == 0.0] = 1.0
        if self.num_levels <= 1:
            return vnorm * sign
        a = torch.abs(a / vnorm)
        logs = torch.log2(a)
        logs[logs == -float("inf")] = -32.0
        logs = logs.int()
        max_pow = torch.zeros_like(logs) + torch.max(logs).int()
        min_pow = max_pow - self.num_levels + 2
        now = torch.max(min_pow, logs).float()
        l = torch.pow(2.0, now - 1)
        r = 2 * l
        a = torch.min(r, torch.max(a, l))
        rand = torch.rand(a.size(), device=a.device)
        c = (a - l) / (r - l)
        a = l * c.le(rand).float() + r * c.gt(rand).float()
        return a * vnorm * sign


class NormUniformQuantizer(Quantizer):
    def __init__(self, bits, bucket_size, enable_error_correction=False, named_parameters=None, adapt_compression=False):
        super().__init__(bits, bucket_size, enable_error_correction, named_parameters, adapt_compression)
        self.num_levels = self.num_levels // 2
        self.levels_1dim = torch.tensor([i * 1.0 / (self.num_levels + 1) for i in range(1, self.num_levels + 1)])
        self.levels_2dim = self.levels_1dim[:, None]
        self.norm_type = float("inf")

    def set_num_levels(self, p):
        bits = self.states[p]["bits"]
        self.num_levels = (1 << bits) // 2

    def quantize_bucket_new(self, buf):
        sign = buf.sign()
        if buf.dim() == 2:
            vnorm = torch.norm(buf, p=self.norm_type, dim=1)
            vnorm = vnorm[:, None]
        else:
            vnorm = torch.norm(buf, p=self.norm_type)
        if self.bits == 1:
            return sign * vnorm
        a = buf.abs() / vnorm
        if buf.dim() == 2:
            self.levels_2dim = self.levels_2dim.to(buf.device)
            res = self.quantize_2_dim_with_levels(a, self.levels_2dim)
        else:
            self.levels_1dim = self.levels_1dim.to(buf.device)
            res = self.quantize_1_dim_with_levels(a, self.levels_1dim)
        return res * sign * vnorm

    def quantize_bucket(self, a):
        if a.dim() == 2:
            vnorm = torch.norm(a, p=float("inf"), dim=1)
            vnorm = vnorm[:, None]
            s = torch.Tensor([1e-11]).expand_as(vnorm).to(a.device)
        else:
            vnorm = torch.norm(a, p=float("inf"))
            s = torch.Tensor([1e-11]).to(a.device)
        vnorm = torch.max(vnorm, s)
        sign = torch.sign(a)
        # cast sign to 1 bit
        sign[sign == 0.0] = 1.0
        if self.num_levels > 1:
            q = torch.abs(a / vnorm)
            r = torch.rand(a.shape, device=a.device)
            q.mul_((self.num_levels - 1))
            q.add_(r)
            torch.floor_(q)
            self.update_stats(q)
            q.div_((self.num_levels - 1))
            return q * vnorm * sign
        else:
            return vnorm * sign


class QuantileQuantizer(Quantizer):
    def __init__(self, bits, bucket_size, enable_error_correction=False):
        super().__init__(bits, bucket_size, enable_error_correction)
        self.quantiles = torch.tensor([i / (self.num_levels + 1) for i in range(1, self.num_levels + 1)])

    def quantize_bucket(self, buf):
        self.quantiles = self.quantiles.to(buf.device).type(buf.dtype)
        if buf.dim() == 1:
            qs = torch.quantile(buf, self.quantiles)
            res = self.quantize_1_dim_with_levels(buf, qs)
        else:
            qs_1dim = torch.quantile(buf, self.quantiles, dim=1)
            qs_2dim = []
            for q in qs_1dim:
                qs_2dim.append(q[:, None])
            res = self.quantize_2_dim_with_levels(buf, qs_2dim)
        # if torch.unique(res).numel() <= qs.numel():
        #     print(self.num_levels, torch.unique(res).numel())
        #     raise ValueError("Num unique values are {}, quantiles: {}".format(
        #         torch.unique(res), qs))
        return res


class TernGrad(Quantizer):
    def __init__(self, bucket_size, enable_error_correction=False, named_parameters=None):
        super().__init__(2, bucket_size, enable_error_correction, named_parameters)
        self.clip_constant = 2.5

    def quantize_bucket(self, a):
        sign = torch.sign(a)
        sign[sign == 0.0] = 1.0
        a.abs_()
        sigma = torch.std(a) * self.clip_constant
        torch.min(a, sigma, out=a)
        if a.dim() == 2:
            vnorm = torch.norm(a, p=float("inf"), dim=1)
            vnorm = vnorm[:, None]
            s = torch.Tensor([1e-11]).expand_as(vnorm).to(a.device)
        else:
            s = torch.Tensor([1e-11]).to(a.device)
            vnorm = torch.norm(a, p=float("inf")).expand_as(s)
        torch.max(vnorm, s, out=vnorm)
        r = torch.rand(a.shape, device=a.device)
        a.div_(vnorm).add_(r)
        torch.floor_(a)
        a.mul_(sign).mul_(vnorm)
        return a

class ThreeLC(Quantizer):
    def __init__(self, bucket_size, named_parameters=None):
        super(ThreeLC, self).__init__(2, bucket_size, True, named_parameters)

    def quantize_bucket(self, a):
        if a.dim() == 2:
            vnorm = torch.norm(a, p=float("inf"), dim=1)
            vnorm = vnorm[:, None]
            s = torch.Tensor([1e-11]).expand_as(vnorm).to(a.device)
        else:
            s = torch.Tensor([1e-11]).to(a.device)
            vnorm = torch.norm(a, p=float("inf")).expand_as(s)
        torch.max(vnorm, s, out=vnorm)
        a.div_(vnorm)
        # do rounding
        a.add_(0.5)
        torch.floor_(a)
        return a.mul_(vnorm)


class OneBitQuantizer(Quantizer):
    def __init__(self, bucket_size, enable_error_correction=False):
        super().__init__(1, bucket_size, enable_error_correction)

    def quantize_bucket(self, a):
        if a.dim() == 2:
            vnorm = torch.norm(a, p=float("inf"), dim=0)
            vnorm = vnorm[None, :]
            s = torch.Tensor([1e-11]).expand_as(vnorm).to(a.device)
            vnorm = torch.max(vnorm, s)
        else:
            vnorm = torch.norm(a, p=float("inf"))
            vnorm = torch.max(vnorm, torch.tensor([1e-11], device=vnorm.device))
        sign = torch.sign(a)
        # cast sign to 1 bit and multiply by norm
        return sign.add_(1).div_(2).mul_(2).add_(-1).mul(vnorm)


class SanityQuantizer(Quantizer):
    def __init__(self, bits, bucket_size, enable_error_correction=False):
        super().__init__(bits, bucket_size, enable_error_correction)

    def quantize_bucket(self, a):
        return torch.zeros_like(a)
