"""Packed W10/W12 decode using an exact FP32 power-of-two bit construction."""
import torch
import triton
import triton.language as tl


@triton.jit
def _decode(P,E,MAP,V,O,K:tl.constexpr,G:tl.constexpr,NG:tl.constexpr,
            NP:tl.constexpr,BITS:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(1);col=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    valid=col<K;bit=(row*K+col)*BITS;offset=bit//8;shift=bit%8
    low=tl.load(P+offset,valid,other=0).to(tl.int32)
    high=tl.load(P+offset+1,valid,other=0).to(tl.int32)
    code=((low | (high<<8))>>shift)&((1<<BITS)-1)
    q=(code-(1<<(BITS-1))).to(tl.float32)
    exponent=tl.load(E+row*NG+col//G,valid,other=0).to(tl.int32)
    scale=((exponent+127)<<23).to(tl.float32,bitcast=True)
    w=(q*scale).to(tl.bfloat16)
    if NP:
        slot=tl.load(MAP+col//G,valid,other=-1)
        original=tl.load(V+(row*NP+slot)*G+col%G,valid & (slot>=0),other=0)
        w=tl.where(slot>=0,original,w)
    tl.store(O+row*K+col,w,valid)


def decode(module):
    out=torch.empty((module.out_features,module.in_features),device=module.packed.device,dtype=torch.bfloat16)
    _decode[(triton.cdiv(module.in_features,1024),module.out_features)](
        module.packed,module.scales,module.group_map,module.protected_values,out,
        module.in_features,module.group_size,module.scales.shape[1],module.protected_groups.numel(),
        module.bits,1024,num_warps=4,enable_fp_fusion=False)
    return out
