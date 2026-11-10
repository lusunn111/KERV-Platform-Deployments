"""Exact decode with at most two full-index-checked exception probes, no loop."""
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['A1','SHIFT1','A2','SHIFT2'])
def _decode(P,B,K1,V1,K2,V2,O,A1,SHIFT1,A2,SHIFT2,N:tl.constexpr,GROUP:tl.constexpr,
            BLOCK:tl.constexpr,HAS_EXCEPTIONS:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);valid=i<N;pair=i//2
    a=tl.load(P+pair*3,valid,other=0).to(tl.uint16)
    b=tl.load(P+pair*3+1,valid,other=0).to(tl.uint16)
    c=tl.load(P+pair*3+2,valid,other=0).to(tl.uint16)
    code=tl.where(i%2==0,a|((b&15)<<8),(b>>4)|(c<<4));sm=code>>4
    exponent=tl.load(B+i//GROUP,valid,other=0).to(tl.uint16)+(code&15)
    bits=(sm&127)|(exponent<<7)|((sm&128)<<8)
    zero=(code&2047)==0;bits=tl.where(zero,(sm&128)<<8,bits).to(tl.uint16)
    if HAS_EXCEPTIONS:
        h1=(i.to(tl.uint32)*A1.to(tl.uint32))>>SHIFT1
        key1=tl.load(K1+h1,valid & zero,other=-1)
        found1=valid & zero & (key1==i)
        original1=tl.load(V1+h1,found1,other=0).to(tl.uint16,bitcast=True)
        h2=(i.to(tl.uint32)*A2.to(tl.uint32))>>SHIFT2
        key2=tl.load(K2+h2,valid & zero & ~found1,other=-1)
        found2=valid & zero & ~found1 & (key2==i)
        original2=tl.load(V2+h2,found2,other=0).to(tl.uint16,bitcast=True)
        bits=tl.where(found1,original1,tl.where(found2,original2,bits))
    tl.store(O+i,bits.to(tl.bfloat16,bitcast=True),valid)


def decode(module,block=1024,num_warps=4):
    n=module.in_features*module.out_features
    out=torch.empty((module.out_features,module.in_features),device=module.packed.device,dtype=torch.bfloat16)
    _decode[(triton.cdiv(n,block),)](module.packed,module.exponent_bases,module.exception_keys1,module.exception_values1,
        module.exception_keys2,module.exception_values2,out,module.hash_a1,module.hash_shift1,module.hash_a2,module.hash_shift2,
        n,module.group_size,block,bool(module.exception_count),num_warps=num_warps)
    return out
