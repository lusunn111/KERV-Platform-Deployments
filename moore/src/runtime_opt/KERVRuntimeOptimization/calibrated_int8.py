"""Activation-calibrated W8 storage with frozen original-BF16 column groups.

Offline GPTQ-style error feedback runs in FP32; actual decoded BF16 rounding is
included in that feedback. Runtime uses one decoded weight and one native GEMM.
No activation/KV quantization or training is performed.
"""
import math
import time
import torch
from torch import nn
from torch.nn import functional as F


class CalibratedInt8Linear(nn.Module):
    def __init__(self, in_features, out_features, group_size=32, protected_count=0):
        super().__init__()
        self.in_features = int(in_features); self.out_features = int(out_features)
        self.group_size = int(group_size); self.forward_calls = 0
        if min(self.in_features, self.out_features, self.group_size) <= 0:
            raise ValueError('Positive dimensions/group size required')
        groups = math.ceil(self.in_features/self.group_size)
        if not 0 <= protected_count <= groups: raise ValueError('Invalid protection count')
        self.register_buffer('qweight', torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer('scales', torch.ones(out_features, groups, dtype=torch.float32))
        self.register_buffer('inverse_channel_scales', torch.ones(in_features, dtype=torch.float32))
        self.register_buffer('protected_groups', torch.empty(protected_count, dtype=torch.int32))
        self.register_buffer('protected_values', torch.zeros(out_features, protected_count, group_size, dtype=torch.bfloat16))
        self.register_buffer('bias', None)
        self.calibration = {}

    @property
    def storage_bytes(self):
        return sum(t.numel()*t.element_size() for t in self.buffers())

    def dequantize(self):
        if self.qweight.device.type in {'musa', 'cuda'}:
            from .calibrated_int8_kernels import decode
            return decode(self)
        columns = torch.arange(self.in_features)
        value = (self.qweight.float()*self.scales[:, columns//self.group_size]
                 *self.inverse_channel_scales).to(torch.bfloat16)
        for index, group in enumerate(self.protected_groups.tolist()):
            start = group*self.group_size; count = min(self.group_size, self.in_features-start)
            value[:, start:start+count] = self.protected_values[:, index, :count]
        return value

    def forward(self, value):
        if value.requires_grad: raise RuntimeError('Calibrated INT8 is inference-only')
        if value.dtype != torch.bfloat16 or value.device != self.qweight.device:
            raise ValueError('Expected colocated BF16 input')
        result = F.linear(value, self.dequantize(), self.bias)
        self.forward_calls += 1
        return result

    def payload(self):
        return {'shape': [self.out_features, self.in_features], 'group_size': self.group_size,
                'protected_count': self.protected_groups.numel(),
                'state_dict': {k: v.detach().cpu() for k,v in self.state_dict().items()},
                'calibration': self.calibration}

    @classmethod
    def from_payload(cls, data):
        out_features, in_features = data['shape']
        result = cls(in_features, out_features, data['group_size'], data['protected_count'])
        if 'bias' in data['state_dict']: result.bias = torch.empty_like(data['state_dict']['bias'])
        result.load_state_dict(data['state_dict'], strict=True)
        result.calibration = data['calibration']
        return result.eval()


def _upper_inverse_factor(hessian):
    """CPU FP64 Cholesky avoids reliance on unsupported MUSA linalg kernels."""
    hessian = hessian.detach().double().cpu()
    diagonal = hessian.diagonal()
    scale = float(diagonal.mean())
    if not math.isfinite(scale) or scale <= 0: raise ValueError('No calibration energy')
    for fraction in (.01, .03, .1):
        matrix = hessian.clone()
        matrix.diagonal().add_(scale*fraction)
        lower, info = torch.linalg.cholesky_ex(matrix)
        if int(info) != 0: continue
        inverse = torch.cholesky_inverse(lower)
        upper, info = torch.linalg.cholesky_ex(inverse, upper=True)
        if int(info) == 0 and torch.isfinite(upper).all(): return upper.float(), fraction
    raise RuntimeError('Damped Hessian factorization failed')


@torch.no_grad()
def calibrate_linear(linear, inputs, group_size=32, protected_fraction=.05, alpha=0., method='gptq',
                     block_size=128, execution_device=None, precision_bits=8, scale_kind='fp32'):
    if method not in {'rtn', 'gptq'} or group_size <= 0 or block_size % group_size:
        raise ValueError('Unsupported method/group/block size')
    if not 0 <= protected_fraction <= 1 or not 0 <= alpha <= 1:
        raise ValueError('Invalid calibration parameter')
    if precision_bits not in (8,10,12) or scale_kind not in ('fp32','pow2'):
        raise ValueError('Unsupported precision/scale kind')
    if precision_bits!=8 and alpha!=0:raise ValueError('Packed calibration currently requires alpha=0')
    limit=(1<<(precision_bits-1))-1
    integer_dtype=torch.int8 if precision_bits==8 else torch.int16
    def format_scale(scale):
        scale=torch.where(scale==0,torch.ones_like(scale),scale)
        if scale_kind=='pow2':
            exponent=torch.ceil(torch.log2(scale)).to(torch.int32).clamp(-126,127)
            return ((exponent+127)<<23).view(torch.float32)
        return scale
    if linear.weight.dtype != torch.bfloat16 or inputs.ndim != 2 or inputs.shape[1] != linear.in_features:
        raise ValueError('Expected BF16 Linear and a matching 2D activation matrix')
    if inputs.shape[0] < 2 or not torch.isfinite(inputs).all(): raise ValueError('Invalid activations')
    started = time.perf_counter()
    device = execution_device or linear.weight.device
    weight = linear.weight.detach().to(device=device, dtype=torch.float32)
    x = inputs.to(device=device, dtype=torch.float32)
    activity = x.abs().mean(0).clamp_min(1e-6)
    scaling = activity.pow(alpha)
    scaling = (scaling/scaling.log().mean().exp()).clamp(.125, 8.)
    inverse_scaling = scaling.reciprocal()
    transformed = weight*scaling
    o, d = weight.shape; groups = math.ceil(d/group_size)
    padded = F.pad(transformed, (0, groups*group_size-d)).reshape(o,groups,group_size)
    scales = format_scale(padded.abs().amax(-1)/limit)
    columns = torch.arange(d, device=device)
    initial_q = (transformed/scales[:,columns//group_size]).round().clamp(-limit,limit).to(integer_dtype)
    initial_decoded = (initial_q.float()*scales[:,columns//group_size]*inverse_scaling).to(torch.bfloat16).float()
    diagonal = x.square().mean(0)
    risks = (weight-initial_decoded).square().sum(0)*diagonal
    risks = F.pad(risks, (0, groups*group_size-d)).reshape(groups,group_size).sum(1)
    count = math.ceil(protected_fraction*groups)
    protected = torch.topk(risks,count).indices if count else columns[:0]
    # A channel unobserved in calibration is not assumed permanently dead.
    dead_groups = (F.pad((diagonal==0).to(torch.int32), (0,groups*group_size-d))
                   .reshape(groups,group_size).sum(1)>0).nonzero().flatten()
    protected = torch.unique(torch.cat((protected,dead_groups)), sorted=True)
    protected_mask = torch.zeros(groups,dtype=torch.bool,device=device)
    protected_mask[protected] = True
    unprotected = (~protected_mask[columns//group_size]).nonzero().flatten()
    if precision_bits==8:result = CalibratedInt8Linear(d,o,group_size,protected.numel())
    else:
        from .calibrated_packed import CalibrationIntNLinear
        result=CalibrationIntNLinear(d,o,group_size,protected.numel(),precision_bits)
    result.inverse_channel_scales.copy_(inverse_scaling.cpu())
    result.protected_groups.copy_(protected.to('cpu',torch.int32))
    for j, group in enumerate(protected.tolist()):
        start = group*group_size; n = min(group_size,d-start)
        result.protected_values[:,j,:n].copy_(linear.weight[:,start:start+n].detach().cpu())
    if linear.bias is not None: result.bias = linear.bias.detach().cpu().clone()
    qweight = initial_q.clone(); damping = None; factor_backend = None
    if method == 'gptq' and unprotected.numel():
        xu = x[:,unprotected]*inverse_scaling[unprotected]
        hessian = xu.T @ xu / len(xu)
        factor, damping = _upper_inverse_factor(hessian)
        factor_backend = 'CPU FP64 Cholesky; FP32 error feedback'
        factor = factor.to(device)
        work = transformed[:,unprotected].clone()
        inverse_u = inverse_scaling[unprotected]
        original_groups = unprotected//group_size
        original_group_ids = original_groups.tolist()
        unprotected_ids = unprotected.tolist()
        for begin in range(0,work.shape[1],block_size):
            end = min(begin+block_size,work.shape[1]); size = end-begin
            local = work[:,begin:end].clone(); errors = torch.zeros_like(local)
            triangular = factor[begin:end,begin:end]
            for offset in range(size):
                j = begin+offset
                if j==0 or original_group_ids[j] != original_group_ids[j-1]:
                    # Whole protected groups were removed; remaining groups stay contiguous.
                    width = min(group_size, size-offset, work.shape[1]-j)
                    scale = format_scale(local[:,offset:offset+width].abs().amax(1)/limit)
                    scales[:,original_group_ids[j]] = scale
                value = local[:,offset].clone()
                quant = (value/scale).round().clamp(-limit,limit).to(integer_dtype)
                deployed = (quant.float()*scale*inverse_u[j]).to(torch.bfloat16).float()/inverse_u[j]
                error = (value-deployed)/triangular[offset,offset]
                errors[:,offset] = error
                local[:,offset:] -= error[:,None]*triangular[offset,offset:][None,:]
                qweight[:,unprotected_ids[j]] = quant
            if end < work.shape[1]: work[:,end:] -= errors @ factor[begin:end,end:]
    if not torch.isfinite(scales).all(): raise RuntimeError('Non-finite calibrated scales')
    result.qweight.copy_(qweight.cpu()); result.scales.copy_(scales.cpu())
    result.calibration = {'method':method,'alpha':alpha,'group_size':group_size,'input_rows':len(inputs),
        'precision_bits':precision_bits,'scale_kind':scale_kind,
        'protected_fraction_requested':protected_fraction,'protected_groups':protected.numel(),
        'unobserved_channel_groups_protected':len(dead_groups),'damping_fraction':damping,
        'factorization':factor_backend,'rounding_target':'actual deployed BF16 weights',
        'seconds':time.perf_counter()-started,'storage_bytes':result.storage_bytes,
        'original_weight_bytes':linear.weight.numel()*linear.weight.element_size()}
    return result.eval()
