import torch
from .compressor import Compressor


class Quantizer(Compressor):
    def __init__(self, bits, bucket_size, enable_error_correction=False, named_parameters=None):
        super().__init__()
        self.bits = bits
        self.num_levels = 1 << bits
        self.bucket_size = bucket_size
        self.stats = [0 for i in range(self.num_levels)]
        self.state = {}
        self.save_error_correction = enable_error_correction
        self.apply_error_correction = False
        if named_parameters:
            named_parameters = list(named_parameters)
            self.named_parameters = {p: name for name, p in named_parameters if p.requires_grad}
            self.parameters_sizes = {name: p.numel() for name, p in named_parameters if p.requires_grad}
        else:
            self.named_parameters = None
            self.parameters_sizes = None
        self.parameters_qbits = {32: ["bn"], 2: ['layer4.1.conv2.weight',
                                                 'layer4.1.conv1.weight',
                                                 'layer4.0.conv2.weight',
                                                 'layer4.0.conv1.weight',
                                                 'layer3.1.conv2.weight',
                                                 'layer3.1.conv1.weight',
                                                 'layer3.0.conv2.weight',
                                                 'layer3.0.conv1.weight',
                                                 'fc.weight']}

    def get_states(self):
        return self.state

    def get_ef_magnitudes(self):
        d = {}
        sum = 0.0
        if self.save_error_correction:
            for p, name in self.named_parameters.items():
                if p in self.state:
                    d[name] = self.state[p]["error_correction"].norm(p=2).item() / p.numel()
                    sum += d[name]
                else:
                    flag = False
                    for skip_name in self.parameters_qbits[32]:
                        if skip_name in name:
                            flag = True
                    if not flag:
                        print(name)
        return d, sum

    def set_num_levels(self, p):
        if self.named_parameters:
            for bits, names in self.parameters_qbits.items():
                for name in names:
                    if name in self.named_parameters[p]:
                        if name == "conv1.weight" and name != self.named_parameters[p]:
                            # TODO: generalize the case of full inclusion of layers names
                            # exact match with conv1.weight
                            continue
                        self.num_levels = 1 << bits
                        return
        self.num_levels = 1 << self.bits

    def quantize_bucket(self, a):
        raise NotImplementedError

    def update_stats(self, a):
        pass

    def compress(self, p, step=None):
        if not p.requires_grad:
            return p, None
        if self.named_parameters:
            for skip_name in self.parameters_qbits[32]:
                if skip_name in self.named_parameters[p]:
                    return p.grad, None
        d_p = p.grad.data
        if p not in self.state:
            self.state[p] = {}
            self.state[p]["error_correction"] = torch.zeros_like(d_p)
            self.state[p]["rel_error"] = torch.Tensor([0.0]).to(d_p.device)
        state = self.state[p]
        e_c = state["error_correction"]
        # rel_error = state["rel_error"]

        if self.save_error_correction:
            # update error correction before subtraction
            e_c.add_(d_p)
            # grad_copy = d_p.clone().detach()
            # add error correction
            if self.apply_error_correction:
                d_p.copy_(e_c)

        a = d_p.view(-1)
        numel = a.numel()
        self.set_num_levels(p)
        if self.bucket_size == -1:
            a[:] = self.quantize_bucket(a)
        else:
            main_chunk_size = (numel // self.bucket_size) * self.bucket_size
            if main_chunk_size > 0:
                a[:main_chunk_size] = self.quantize_bucket(a[:main_chunk_size].view((-1, self.bucket_size))).view(-1)
            # if numel - main_chunk_size > 0:
            #     a[main_chunk_size:] = self.quantize_bucket(a[main_chunk_size:])
        if self.save_error_correction:
            e_c.sub_(d_p)
            # grad_copy.sub_(d_p)
            # rel_error += grad_copy.norm(p=2) / d_p.norm(p=2).add(1e-10)
        return p.grad, None

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
    def __init__(self, bits, bucket_size, enable_error_correction=False, named_parameters=None):
        super().__init__(bits, bucket_size, enable_error_correction, named_parameters)

    def quantize_bucket(self, a):
        if self.num_levels == 1 << 32:
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
        # self.update_stats(a)
        a *= unit
        a += fmin
        return a


class ExponentialQuantizer(Quantizer):
    def __init__(self, bits, bucket_size, enable_error_correction=False):
        super().__init__(bits, bucket_size, enable_error_correction)
        self.num_levels = self.num_levels // 2
        self.norm_type = float("inf")
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
        sign.add_(1).div_(2)
        sign.mul_(2).add_(-1)
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
    def __init__(self, bits, bucket_size, enable_error_correction=False):
        super().__init__(bits, bucket_size, enable_error_correction)
        self.num_levels = self.num_levels // 2
        self.levels_1dim = torch.tensor([i * 1.0 / (self.num_levels + 1) for i in range(1, self.num_levels + 1)])
        self.levels_2dim = self.levels_1dim[:, None]
        self.norm_type = float("inf")

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
        sign.add_(1).div_(2)
        sign.mul_(2).add_(-1)
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
    def __init__(self, bucket_size, enable_error_correction=False):
        super().__init__(2, bucket_size, enable_error_correction)

    def quantize_bucket(self, a):
        if a.dim() == 2:
            vnorm = torch.norm(a, p=float("inf"), dim=0)
            vnorm = vnorm[None, :]
            s = torch.Tensor([1e-11]).expand_as(vnorm).to(a.device)
        else:
            vnorm = torch.norm(a, p=float("inf"))
            s = torch.Tensor([1e-11]).to(a.device)
        torch.max(vnorm, s, out=vnorm)
        sign = torch.sign(a)
        sign[sign == 0.0] = 1.0
        return vnorm * sign


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
