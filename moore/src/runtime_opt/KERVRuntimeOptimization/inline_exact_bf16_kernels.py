"""Single-pass exact BF16 decode, including CSR exceptions inside each CTA."""
import torch
import triton
import triton.language as tl


@triton.jit
def _decode(P,B,S,I,V,O,N:tl.constexpr,GROUP:tl.constexpr,BLOCK:tl.constexpr,MAX_EXCEPTIONS:tl.constexpr):
    block=tl.program_id(0);local=tl.arange(0,BLOCK);i=block*BLOCK+local;valid=i<N;pair=i//2
    a=tl.load(P+pair*3,valid,other=0).to(tl.uint16)
    b=tl.load(P+pair*3+1,valid,other=0).to(tl.uint16)
    c=tl.load(P+pair*3+2,valid,other=0).to(tl.uint16)
    code=tl.where(i%2==0,a|((b&15)<<8),(b>>4)|(c<<4));sm=code>>4
    exponent=tl.load(B+i//GROUP,valid,other=0).to(tl.uint16)+(code&15)
    bits=(sm&127)|(exponent<<7)|((sm&128)<<8)
    bits=tl.where((code&2047)==0,(sm&128)<<8,bits).to(tl.uint16)
    first=tl.load(S+block).to(tl.int32);last=tl.load(S+block+1).to(tl.int32)
    if MAX_EXCEPTIONS<=4:
        for index in tl.static_range(MAX_EXCEPTIONS):
            position=first+index;present=position<last
            offset=tl.load(I+position,present,other=-1).to(tl.int32)
            original=tl.load(V+position,present,other=0).to(tl.uint16,bitcast=True)
            bits=tl.where(present & (local==offset),original,bits)
    else:
        for position in range(first,last):
            offset=tl.load(I+position).to(tl.int32)
            original=tl.load(V+position).to(tl.uint16,bitcast=True)
            bits=tl.where(local==offset,original,bits)
    tl.store(O+i,bits.to(tl.bfloat16,bitcast=True),valid)


def decode(module):
    n=module.in_features*module.out_features;block=module.decode_block_size
    out=torch.empty((module.out_features,module.in_features),device=module.packed.device,dtype=torch.bfloat16)
    _decode[(triton.cdiv(n,block),)](module.packed,module.exponent_bases,module.exception_starts,
        module.exception_offsets,module.exception_values,out,n,module.group_size,block,module.maximum_exceptions_per_block,num_warps=4)
    return out
