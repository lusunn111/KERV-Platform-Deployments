"""Bit-exact GPU unpacking of 12-bit exponent-delta BF16 storage."""
import torch
import triton
import triton.language as tl


@triton.jit
def _decode(P,B,O,N:tl.constexpr,GROUP:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    valid=i<N;pair=i//2
    a=tl.load(P+pair*3,mask=valid,other=0).to(tl.uint16)
    b=tl.load(P+pair*3+1,mask=valid,other=0).to(tl.uint16)
    c=tl.load(P+pair*3+2,mask=valid,other=0).to(tl.uint16)
    code=tl.where(i%2==0,a|((b&15)<<8),(b>>4)|(c<<4))
    sm=code>>4
    exponent=tl.load(B+i//GROUP,mask=valid,other=0).to(tl.uint16)+(code&15)
    bits=(sm&127)|(exponent<<7)|((sm&128)<<8)
    tl.store(O+i,bits.to(tl.uint16).to(tl.bfloat16,bitcast=True),mask=valid)


@triton.jit
def _patch(I,V,O,N:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    index=tl.load(I+i,mask=i<N,other=0)
    value=tl.load(V+i,mask=i<N,other=0)
    tl.store(O+index,value,mask=i<N)


def decode(module):
    n=module.in_features*module.out_features
    out=torch.empty((module.out_features,module.in_features),device=module.packed.device,dtype=torch.bfloat16)
    _decode[(triton.cdiv(n,1024),)](module.packed,module.exponent_bases,out,n,module.group_size,1024,num_warps=4)
    exceptions=module.escape_indices.numel()
    if exceptions:_patch[(triton.cdiv(exceptions,256),)](module.escape_indices,module.escape_values,out,exceptions,256,num_warps=4)
    return out
