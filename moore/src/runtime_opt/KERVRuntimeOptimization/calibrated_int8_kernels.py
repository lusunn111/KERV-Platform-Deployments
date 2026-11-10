"""One BF16 weight allocation; fuse dequantization/scale inversion, then patch."""
import torch
import triton
import triton.language as tl


@triton.jit
def _decode(Q,S,I,O,K:tl.constexpr,N:tl.constexpr,G:tl.constexpr,NG:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK); valid=i<N; column=i%K
    q=tl.load(Q+i,mask=valid,other=0).to(tl.float32)
    scale=tl.load(S+(i//K)*NG+column//G,mask=valid,other=0)
    inv=tl.load(I+column,mask=valid,other=0)
    tl.store(O+i,(q*scale)*inv,mask=valid)


@triton.jit
def _patch(P,V,O,K:tl.constexpr,N:tl.constexpr,G:tl.constexpr,NP:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);valid=i<N
    row=i//(NP*G);position=(i//G)%NP;offset=i%G
    group=tl.load(P+position,mask=valid,other=0);column=group*G+offset
    value=tl.load(V+i,mask=valid,other=0)
    tl.store(O+row*K+column,value,mask=valid & (column<K))


def decode(module):
    out=torch.empty(module.qweight.shape,device=module.qweight.device,dtype=torch.bfloat16)
    _decode[(triton.cdiv(out.numel(),1024),)](module.qweight,module.scales,module.inverse_channel_scales,
        out,module.in_features,out.numel(),module.group_size,module.scales.shape[1],1024,num_warps=4)
    np=module.protected_groups.numel()
    if np:
        count=module.protected_values.numel()
        _patch[(triton.cdiv(count,1024),)](module.protected_groups,module.protected_values,out,
            module.in_features,count,module.group_size,np,1024,num_warps=4)
    return out
