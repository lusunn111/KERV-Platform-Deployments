"""Exact BF16 exponent codec with two direct, collision-checked exception tables.

The primary table stores one key per bucket. Only displaced keys enter a
secondary table whose multiplier is searched offline until collision-free.
Both GPU probes compare full indices, including queries for actual zeros.
"""
import math
import numpy as np
import torch
from torch import nn
from .nearzero_bf16 import NearZeroBF16Linear,_digest
from .weight_only_int8 import group_for


def _size(n):return 1<<max(1,int(n-1).bit_length())

def _hash(indices,multiplier,shift):
    return (((indices.astype(np.uint64)*multiplier)&np.uint64(0xffffffff))>>shift).astype(np.int64)


def _tables(indices,bits):
    count=len(indices);primary_size=_size(max(2,8*count));shift1=32-int(math.log2(primary_size))
    rng=np.random.default_rng(461);best=None
    for _ in range(16):
        multiplier=int(rng.integers(0,2**32,dtype=np.uint64))|1
        bucket=_hash(indices,multiplier,shift1)
        _,keep=np.unique(bucket,return_index=True)
        if best is None or len(keep)>len(best[2]):best=(multiplier,bucket,keep)
    a1,bucket,keep=best
    remaining=np.ones(count,dtype=bool);remaining[keep]=False;displaced=np.flatnonzero(remaining)
    secondary_size=_size(max(2,2*len(displaced)**2))
    while True:
        shift2=32-int(math.log2(secondary_size));success=False
        for _ in range(64):
            a2=int(rng.integers(0,2**32,dtype=np.uint64))|1
            h2=_hash(indices[displaced],a2,shift2)
            if len(np.unique(h2))==len(displaced):success=True;break
        if success:break
        secondary_size*=2
        if secondary_size>2**24:raise RuntimeError('Exception table construction exceeded budget')
    key1=np.full(primary_size,-1,dtype=np.int32);value1=np.zeros(primary_size,dtype=np.uint16)
    key2=np.full(secondary_size,-1,dtype=np.int32);value2=np.zeros(secondary_size,dtype=np.uint16)
    key1[bucket[keep]]=indices[keep];value1[bucket[keep]]=bits[keep]
    key2[h2]=indices[displaced];value2[h2]=bits[displaced]
    return (key1,value1,key2,value2),(a1,shift1,a2,shift2)


class HashExactBF16Linear(NearZeroBF16Linear):
    decode_block_size=1024
    decode_num_warps=4

    def __init__(self,linear,group_size=128):
        super().__init__(linear,group_size)
        original=linear.weight.detach().contiguous().view(torch.int16).numpy().view(np.uint16).reshape(-1)
        body=super().dequantize().view(torch.int16).numpy().view(np.uint16).reshape(-1)
        indices=np.flatnonzero(original!=body)
        tables,config=_tables(indices,original[indices])
        for name,data in zip(('exception_keys1','exception_values1','exception_keys2','exception_values2'),tables):
            tensor=torch.from_numpy(data if data.dtype==np.int32 else data.view(np.int16)).clone()
            if 'values' in name:tensor=tensor.view(torch.bfloat16)
            self.register_buffer(name,tensor)
        self.hash_a1,self.hash_shift1,self.hash_a2,self.hash_shift2=config
        self.exception_count=len(indices);self._exact_ready=True
        if _digest(self.dequantize())!=self.original_sha256:raise RuntimeError('Hash exact codec changed original BF16 bits')
        self.decoded_sha256=self.original_sha256
        self.quantization.update(method='exact_zero_aware_BF16_two_level_hash',relative_weight_rmse=0.,squared_error=0.,
                                 maximum_absolute_weight_error=0.,changed_weight_bits=0,zeroed_nonzero_weights=0,
                                 exact_exceptions=len(indices),primary_slots=len(tables[0]),secondary_slots=len(tables[2]),
                                 hash_multipliers_and_shifts=list(config),bound='zero; all original BF16 bits restored')

    def dequantize(self):
        if not getattr(self,'_exact_ready',False):return super().dequantize()
        if self.packed.device.type=='musa':
            from .hash_exact_bf16_kernels import decode
            return decode(self,self.decode_block_size,self.decode_num_warps)
        value=super().dequantize().reshape(-1)
        for keys,values in ((self.exception_keys1,self.exception_values1),(self.exception_keys2,self.exception_values2)):
            valid=keys>=0;value[keys[valid].long()]=values[valid]
        return value.reshape(self.out_features,self.in_features)


def compress_hash_exact(model,group_size=128,groups=None):
    selected=set(groups or ())
    if not selected or not selected<={'verify_mlp'}:raise ValueError('Hash exact pilot supports verifier MLP only')
    eligible=[(n,m) for n,m in model.named_modules() if isinstance(m,nn.Linear) and group_for(n) in selected]
    if not eligible or any(m.weight.device.type!='cpu' for _,m in eligible):raise ValueError('Prepare a CPU model')
    records=[]
    for name,linear in eligible:
        q=HashExactBF16Linear(linear,group_size)
        if q.storage_bytes>=linear.weight.numel()*2:continue
        parent,_,child=name.rpartition('.');setattr(model.get_submodule(parent) if parent else model,child,q)
        records.append({'name':name,'shape':list(linear.weight.shape),'original_sha256':q.original_sha256,
                        'decoded_sha256':q.decoded_sha256,'before_bytes':linear.weight.numel()*2,
                        'storage_bytes':q.storage_bytes,'storage':'exact_bf16_exp_delta12_hash_exceptions',
                        'quantization':q.quantization})
    if not records:raise RuntimeError('No projection benefits from hash exact codec')
    return {'modules':records}
