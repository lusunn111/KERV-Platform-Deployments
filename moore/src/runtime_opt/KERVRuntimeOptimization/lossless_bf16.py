"""Lossless BF16 storage: sign/mantissa + 4-bit exponent delta, BF16 escapes.

This is a 12-bit storage codec with sparse exact exceptions, not lossy W12
quantization. All decoded values retain the original 16 BF16 bits.
"""
import hashlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .weight_only_int8 import group_for


def raw_hash(value):
    return hashlib.sha256(value.detach().contiguous().view(torch.int16).cpu().numpy().tobytes()).hexdigest()


class LosslessBF16Linear(nn.Module):
    def __init__(self, linear, group_size=128):
        super().__init__()
        if linear.weight.device.type!='cpu' or linear.weight.dtype!=torch.bfloat16:
            raise ValueError('Lossless storage is prepared from CPU BF16 weights')
        if group_size<=0:raise ValueError('group_size must be positive')
        self.in_features=linear.in_features;self.out_features=linear.out_features
        self.group_size=int(group_size);self.forward_calls=0
        original=linear.weight.detach().contiguous()
        self.original_sha256=raw_hash(original)
        bits=original.view(torch.int16).numpy().view(np.uint16).reshape(-1)
        n=len(bits);groups=(n+group_size-1)//group_size
        exponent=((bits>>7)&255).astype(np.uint8)
        padded=np.pad(exponent,(0,groups*group_size-n))
        bases=np.maximum(padded.reshape(groups,group_size).max(1).astype(np.int16)-15,0).astype(np.uint8)
        expanded=np.repeat(bases,group_size)[:n].astype(np.int16)
        offset=exponent.astype(np.int16)-expanded
        escapes=np.flatnonzero(offset<0).astype(np.int32)
        sign_mantissa=((bits&127)|((bits>>8)&128)).astype(np.uint16)
        codes=(sign_mantissa<<4)|np.clip(offset,0,15).astype(np.uint16)
        codes=np.pad(codes,(0,n%2))
        even,odd=codes[::2],codes[1::2]
        packed=np.empty(len(even)*3,dtype=np.uint8)
        packed[0::3]=(even&255).astype(np.uint8)
        packed[1::3]=((even>>8)|((odd&15)<<4)).astype(np.uint8)
        packed[2::3]=(odd>>4).astype(np.uint8)
        self.register_buffer('packed',torch.from_numpy(packed))
        self.register_buffer('exponent_bases',torch.from_numpy(bases.copy()))
        self.register_buffer('escape_indices',torch.from_numpy(escapes.copy()))
        self.register_buffer('escape_values',original.reshape(-1)[torch.from_numpy(escapes.astype(np.int64))].clone())
        self.register_buffer('bias',linear.bias.detach().clone() if linear.bias is not None else None)
        restored=self.dequantize()
        if not torch.equal(restored.view(torch.int16),original.view(torch.int16)):
            raise RuntimeError('BF16 storage codec failed CPU bit-exact roundtrip')

    @property
    def storage_bytes(self):
        return sum(t.numel()*t.element_size() for t in (self.packed,self.exponent_bases,self.escape_indices,self.escape_values))

    @property
    def weight(self):
        return self.dequantize()

    def dequantize(self):
        if self.packed.device.type in {'musa','cuda'}:
            from .lossless_bf16_kernels import decode
            return decode(self)
        p=self.packed.numpy();n=self.in_features*self.out_features
        codes=np.empty(len(p)//3*2,dtype=np.uint16)
        codes[0::2]=p[0::3].astype(np.uint16)|((p[1::3].astype(np.uint16)&15)<<8)
        codes[1::2]=(p[1::3].astype(np.uint16)>>4)|(p[2::3].astype(np.uint16)<<4)
        codes=codes[:n];sm=codes>>4
        exponents=np.repeat(self.exponent_bases.numpy(),self.group_size)[:n].astype(np.uint16)+(codes&15)
        bits=(sm&127)|(exponents<<7)|((sm&128)<<8)
        value=torch.from_numpy(bits.view(np.int16).copy()).view(torch.bfloat16)
        value[self.escape_indices.long()]=self.escape_values
        return value.reshape(self.out_features,self.in_features)

    def forward(self,value):
        if value.requires_grad:raise RuntimeError('Lossless compressed Linear is inference-only')
        if value.dtype!=torch.bfloat16 or value.device!=self.packed.device:
            raise ValueError('Expected BF16 activations colocated with the compressed weights')
        result=F.linear(value,self.dequantize(),self.bias)
        self.forward_calls+=1
        return result


def compress_model(model,group_size=128,groups=None):
    selected=set(groups) if groups is not None else {'verify_mlp','verify_attention'}
    if not selected or not selected <= {'verify_mlp','verify_attention'}:
        raise ValueError('Expected verifier MLP and/or attention groups')
    eligible=[(name,module) for name,module in model.named_modules() if isinstance(module,nn.Linear)
              and group_for(name) in selected]
    if not eligible or any(x.weight.device.type!='cpu' for _,x in eligible):
        raise ValueError('Expected a CPU model before any accelerator transfer')
    records=[];retained=[]
    for name,linear in eligible:
        # Keep q_proj native, including direct dtype/device accesses in KERV.
        if name.endswith('.q_proj'):
            retained.append(name);continue
        compressed=LosslessBF16Linear(linear,group_size)
        before=linear.weight.numel()*linear.weight.element_size()
        if compressed.storage_bytes>=before:
            retained.append(name);continue
        parent,_,child=name.rpartition('.')
        setattr(model.get_submodule(parent) if parent else model,child,compressed)
        records.append({'name':name,'shape':list(linear.weight.shape),'original_sha256':compressed.original_sha256,
                        'weight_bytes_before':before,'weight_bytes_after':compressed.storage_bytes,
                        'exceptions':compressed.escape_indices.numel()})
    if not records:raise RuntimeError('No projection benefits from the BF16 storage codec')
    return {'enabled':True,'fallback':False,'implementation':'musa_lossless_bf16_exp_delta12_native_bf16_gemm',
            'lossy_quantization':False,'weight_storage':'12-bit codes + uint8 exponent bases + sparse BF16 exceptions',
            'decoded_weight_dtype':'bfloat16','activation_dtype':'bfloat16','group_size':group_size,
            'module_count':len(records),'modules':records,'retained_bf16_modules':retained,
            'selected_groups':sorted(selected),
            'weight_bytes_before':sum(x['weight_bytes_before'] for x in records),
            'weight_bytes_after':sum(x['weight_bytes_after'] for x in records),
            'estimated_weight_storage_saving_bytes':sum(x['weight_bytes_before']-x['weight_bytes_after'] for x in records),
            'cpu_bit_exact_verified_modules':len(records),'gemm':'native BF16; per-layer exact decode, no full BF16 model cache'}
