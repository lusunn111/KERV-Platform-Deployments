"""Exact BF16 codec: zero-aware body and CTA-local sparse exception restore.

A CSR prefix indexes original BF16 exceptions for each1024-weight decode tile.
The same kernel restores those values before writing the tile; no separate
scatter launch. Actual zeros use signed zero codes without sparse entries.
"""
import numpy as np
import torch
from torch import nn
from .nearzero_bf16 import NearZeroBF16Linear,_digest
from .weight_only_int8 import group_for


class InlineExactBF16Linear(NearZeroBF16Linear):
    decode_block_size=1024

    def __init__(self,linear,group_size=128):
        super().__init__(linear,group_size)
        original=linear.weight.detach().contiguous().reshape(-1)
        body=super().dequantize().reshape(-1)
        different=(original.view(torch.int16).numpy()!=body.view(torch.int16).numpy())
        indices=np.flatnonzero(different)
        n=original.numel();blocks=(n+self.decode_block_size-1)//self.decode_block_size
        counts=np.bincount(indices//self.decode_block_size,minlength=blocks)
        prefix=np.concatenate((np.zeros(1,dtype=np.int64),np.cumsum(counts)))
        if len(indices)>np.iinfo(np.int32).max:raise ValueError('Too many exceptions')
        prefix_dtype=np.int16 if len(indices)<=32767 else np.int32
        self.register_buffer('exception_starts',torch.from_numpy(prefix.astype(prefix_dtype)))
        self.register_buffer('exception_offsets',torch.from_numpy((indices%self.decode_block_size).astype(np.int16)))
        self.register_buffer('exception_values',original[torch.from_numpy(indices.astype(np.int64))].clone())
        self.maximum_exceptions_per_block=int(counts.max())
        self._exact_ready=True
        if _digest(self.dequantize())!=self.original_sha256:raise RuntimeError('Inline exact codec changed original BF16 bits')
        self.decoded_sha256=self.original_sha256
        self.quantization.update(method='exact_zero_aware_BF16_inline_exceptions',relative_weight_rmse=0.,squared_error=0.,
                                 maximum_absolute_weight_error=0.,changed_weight_bits=0,zeroed_nonzero_weights=0,
                                 exact_exceptions=len(indices),decode_block_size=self.decode_block_size,
                                 maximum_exceptions_per_block=self.maximum_exceptions_per_block,
                                 bound='zero; all original BF16 bits restored')

    def dequantize(self):
        if not getattr(self,'_exact_ready',False):return super().dequantize()
        if self.packed.device.type=='musa':
            from .inline_exact_bf16_kernels import decode
            return decode(self)
        value=super().dequantize().reshape(-1)
        counts=np.diff(self.exception_starts.numpy().astype(np.int64))
        indices=np.repeat(np.arange(len(counts)),counts)*self.decode_block_size+self.exception_offsets.numpy().astype(np.int64)
        value[torch.from_numpy(indices)]=self.exception_values
        return value.reshape(self.out_features,self.in_features)


def compress_inline_exact(model,group_size=128,groups=None):
    selected=set(groups or ())
    if not selected or not selected<={'verify_mlp'}:raise ValueError('Inline exact pilot supports verifier MLP only')
    eligible=[(n,m) for n,m in model.named_modules() if isinstance(m,nn.Linear) and group_for(n) in selected]
    if not eligible or any(m.weight.device.type!='cpu' for _,m in eligible):raise ValueError('Prepare a CPU model')
    records=[]
    for name,linear in eligible:
        q=InlineExactBF16Linear(linear,group_size)
        if q.storage_bytes>=linear.weight.numel()*2:continue
        parent,_,child=name.rpartition('.');setattr(model.get_submodule(parent) if parent else model,child,q)
        records.append({'name':name,'shape':list(linear.weight.shape),'original_sha256':q.original_sha256,
                        'decoded_sha256':q.decoded_sha256,'before_bytes':linear.weight.numel()*2,
                        'storage_bytes':q.storage_bytes,'storage':'exact_bf16_exp_delta12_inline_exceptions',
                        'quantization':q.quantization})
    if not records:raise RuntimeError('No projection benefits from inline exact codec')
    return {'modules':records}
