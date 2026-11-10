"""Single-pass BF16 exponent decode with bounded zero-code quantization."""
import torch
import triton
import triton.language as tl


@triton.jit
def _decode(P,B,O,N:tl.constexpr,GROUP:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);valid=i<N;pair=i//2
    a=tl.load(P+pair*3,valid,other=0).to(tl.uint16)
    b=tl.load(P+pair*3+1,valid,other=0).to(tl.uint16)
    c=tl.load(P+pair*3+2,valid,other=0).to(tl.uint16)
    code=tl.where(i%2==0,a|((b&15)<<8),(b>>4)|(c<<4));sm=code>>4
    exponent=tl.load(B+i//GROUP,valid,other=0).to(tl.uint16)+(code&15)
    bits=(sm&127)|(exponent<<7)|((sm&128)<<8)
    bits=tl.where((code&2047)==0,(sm&128)<<8,bits).to(tl.uint16)
    tl.store(O+i,bits.to(tl.bfloat16,bitcast=True),valid)


def decode(module):
    n=module.in_features*module.out_features
    out=torch.empty((module.out_features,module.in_features),device=module.packed.device,dtype=torch.bfloat16)
    _decode[(triton.cdiv(n,1024),)](module.packed,module.exponent_bases,out,n,module.group_size,1024,num_warps=4)
    return out
