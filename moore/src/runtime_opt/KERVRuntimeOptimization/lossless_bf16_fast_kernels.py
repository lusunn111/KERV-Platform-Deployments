"""Two-byte extraction of each12-bit BF16 code; exact sparse patch unchanged."""
import torch
import triton
import triton.language as tl
from .lossless_bf16_kernels import _patch


@triton.jit
def _decode(P,B,O,N:tl.constexpr,GROUP:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);valid=i<N
    offset=(i*3)//2;shift=((i&1)*4).to(tl.uint16)
    low=tl.load(P+offset,valid,other=0).to(tl.uint16)
    high=tl.load(P+offset+1,valid,other=0).to(tl.uint16)
    code=(((low|(high<<8))>>shift)&4095).to(tl.uint16);sm=code>>4
    exponent=tl.load(B+i//GROUP,valid,other=0).to(tl.uint16)+(code&15)
    bits=(sm&127)|(exponent<<7)|((sm&128)<<8)
    tl.store(O+i,bits.to(tl.uint16).to(tl.bfloat16,bitcast=True),valid)


def decode(module,block=1024,num_warps=4):
    n=module.in_features*module.out_features
    out=torch.empty((module.out_features,module.in_features),device=module.packed.device,dtype=torch.bfloat16)
    _decode[(triton.cdiv(n,block),)](module.packed,module.exponent_bases,out,n,module.group_size,block,num_warps=num_warps)
    count=module.escape_indices.numel()
    if count:_patch[(triton.cdiv(count,256),)](module.escape_indices,module.escape_values,out,count,256,num_warps=4)
    return out
