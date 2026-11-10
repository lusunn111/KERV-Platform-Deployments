"""Bounded near-zero quantization with otherwise exact BF16 exponent coding.

A 12-bit exponent-delta code reserves its two signed floor codes for zero. Exponents below the
per-group representable floor and its signed power-of-two boundary
code are mapped to zero. Every other BF16 bit is preserved. No escape scatter
or integer GEMM is used. |error| <= group_max_abs /32768 for finite weights.
"""
import hashlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .weight_only_int8 import group_for


def _digest(value):
    return hashlib.sha256(value.detach().contiguous().view(torch.int16).cpu().numpy().tobytes()).hexdigest()


class NearZeroBF16Linear(nn.Module):
    def __init__(self,linear,group_size=128):
        super().__init__()
        if linear.weight.device.type!='cpu' or linear.weight.dtype!=torch.bfloat16 or group_size<=0:
            raise ValueError('Expected finite CPU BF16 Linear and positive group size')
        self.in_features=linear.in_features;self.out_features=linear.out_features
        self.group_size=int(group_size);self.forward_calls=0
        weight=linear.weight.detach().contiguous();self.original_sha256=_digest(weight)
        bits=weight.view(torch.int16).numpy().view(np.uint16).reshape(-1)
        exponent=((bits>>7)&255).astype(np.uint8)
        if np.any(exponent==255):raise ValueError('Non-finite weight')
        n=len(bits);groups=(n+group_size-1)//group_size
        maxima=np.pad(exponent,(0,groups*group_size-n)).reshape(groups,group_size).max(1)
        bases=np.maximum(maxima.astype(np.int16)-15,0).astype(np.uint8)
        expanded=np.repeat(bases,group_size)[:n].astype(np.int16)
        offset=exponent.astype(np.int16)-expanded
        sm=((bits&127)|((bits>>8)&128)).astype(np.uint16)
        codes=(sm<<4)|np.clip(offset,0,15).astype(np.uint16)
        codes[offset<0]=((sm[offset<0]&128)<<4)
        padded=np.pad(codes,(0,n%2));even,odd=padded[::2],padded[1::2]
        packed=np.empty(len(even)*3,dtype=np.uint8)
        packed[0::3]=(even&255).astype(np.uint8)
        packed[1::3]=((even>>8)|((odd&15)<<4)).astype(np.uint8)
        packed[2::3]=(odd>>4).astype(np.uint8)
        self.register_buffer('packed',torch.from_numpy(packed))
        self.register_buffer('exponent_bases',torch.from_numpy(bases.copy()))
        self.register_buffer('bias',linear.bias.detach().clone() if linear.bias is not None else None)
        decoded=self.dequantize();self.decoded_sha256=_digest(decoded)
        original_fp32=(bits.astype(np.uint32)<<16).view(np.float32)
        actual_bits=decoded.view(torch.int16).numpy().view(np.uint16).reshape(-1)
        actual_fp32=(actual_bits.astype(np.uint32)<<16).view(np.float32)
        error=actual_fp32-original_fp32
        # With base0, code0 is already +zero and no lower exponent exists.
        floor_bits=(expanded.astype(np.uint32)<<23)
        bound=floor_bits.view(np.float32)
        if np.any(np.abs(error)>bound):raise RuntimeError('Near-zero quantization bound violated')
        if np.any(((codes&2047)!=0)&(actual_bits!=bits)):raise RuntimeError('Nonzero code changed original BF16 bits')
        signal=np.square(original_fp32.astype(np.float64)).sum()
        squared=np.square(error.astype(np.float64)).sum()
        self.quantization={'method':'bounded_near_zero_with_exact_BF16_body','group_size':group_size,
                           'relative_weight_rmse':float(np.sqrt(squared/signal)) if signal else 0.,
                           'original_energy':float(signal),'squared_error':float(squared),
                           'maximum_absolute_weight_error':float(np.abs(error).max()),
                           'changed_weight_bits':int(np.count_nonzero(actual_bits!=bits)),
                           'zeroed_nonzero_weights':int(np.count_nonzero((actual_fp32==0)&(original_fp32!=0))),
                           'weight_count':n,'bound':'abs(weight_error) <= group_max_abs/32768; all nonzero codes retain original BF16 bits'}

    @property
    def storage_bytes(self):return sum(t.numel()*t.element_size() for t in self.buffers())

    def dequantize(self):
        if self.packed.device.type=='musa':
            from .nearzero_bf16_kernels import decode
            return decode(self)
        if self.packed.device.type!='cpu':raise ValueError('Expected CPU or MUSA weights')
        p=self.packed.numpy();n=self.in_features*self.out_features
        codes=np.empty(len(p)//3*2,dtype=np.uint16)
        codes[0::2]=p[0::3].astype(np.uint16)|((p[1::3].astype(np.uint16)&15)<<8)
        codes[1::2]=(p[1::3].astype(np.uint16)>>4)|(p[2::3].astype(np.uint16)<<4)
        codes=codes[:n];sm=codes>>4
        exponent=np.repeat(self.exponent_bases.numpy(),self.group_size)[:n].astype(np.uint16)+(codes&15)
        bits=(sm&127)|(exponent<<7)|((sm&128)<<8)
        zero=(codes&2047)==0
        bits[zero]=((sm[zero]&128)<<8)
        return torch.from_numpy(bits.view(np.int16).copy()).view(torch.bfloat16).reshape(self.out_features,self.in_features)

    def forward(self,value):
        if value.requires_grad:raise RuntimeError('Near-zero codec is inference-only')
        if value.dtype!=torch.bfloat16 or value.device!=self.packed.device:
            raise ValueError('Expected colocated BF16 input')
        result=F.linear(value,self.dequantize(),self.bias);self.forward_calls+=1;return result


def compress_nearzero(model,group_size=128,groups=None):
    selected=set(groups or ())
    if not selected or not selected<={'verify_mlp'}:raise ValueError('Near-zero pilot is limited to verifier MLP')
    eligible=[(n,m) for n,m in model.named_modules() if isinstance(m,nn.Linear) and group_for(n) in selected]
    if not eligible or any(m.weight.device.type!='cpu' for _,m in eligible):raise ValueError('Prepare a CPU model')
    records=[]
    for name,linear in eligible:
        q=NearZeroBF16Linear(linear,group_size)
        if q.storage_bytes>=linear.weight.numel()*2:continue
        parent,_,child=name.rpartition('.');setattr(model.get_submodule(parent) if parent else model,child,q)
        records.append({'name':name,'shape':list(linear.weight.shape),'original_sha256':q.original_sha256,
                        'decoded_sha256':q.decoded_sha256,'before_bytes':linear.weight.numel()*2,
                        'storage_bytes':q.storage_bytes,'storage':'bf16_exp_delta12_bounded_near_zero','quantization':q.quantization})
    if not records:raise RuntimeError('No near-zero projection selected')
    return {'modules':records}
