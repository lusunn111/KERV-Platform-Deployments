"""Full-vocabulary MetaX Verify-Accept; no action-vocabulary truncation."""
import torch
import triton
import triton.language as tl


@triton.jit
def _matches(L,C,M, S0:tl.constexpr,S1:tl.constexpr,C0:tl.constexpr,
             NPOS:tl.constexpr,VOCAB:tl.constexpr,OFFSET:tl.constexpr,threshold,
             BLOCK:tl.constexpr):
    path=tl.program_id(0); pos=tl.program_id(1)
    offsets=tl.arange(0,BLOCK)
    logits=tl.load(L+path*S0+pos*S1+offsets,offsets<VOCAB,other=-float('inf'))
    # Triton's first-index argmax preserves native torch.argmax ties.
    predicted=tl.argmax(logits,axis=0,tie_break_left=True)+OFFSET
    candidate=tl.load(C+path*C0+pos+1)
    tl.store(M+path*NPOS+pos,tl.abs(candidate-predicted)<=threshold)


@triton.jit
def _reduce(M,BEST,LENGTH,PATHS:tl.constexpr,NPOS:tl.constexpr,
            BP:tl.constexpr,BS:tl.constexpr):
    paths=tl.arange(0,BP);positions=tl.arange(0,BS)
    matches=tl.load(M+paths[:,None]*NPOS+positions[None,:],
        (paths[:,None]<PATHS)&(positions[None,:]<NPOS),other=0).to(tl.int32)
    prefixes=tl.cumprod(matches,axis=1)
    lengths=tl.sum(prefixes,axis=1)
    lengths=tl.where(paths<PATHS,lengths,-1)
    tl.store(BEST,tl.argmax(lengths,axis=0,tie_break_left=True).to(tl.int64))
    tl.store(LENGTH,tl.max(lengths,axis=0).to(tl.int64))


def verify_accept(logits,candidates,threshold,offset=0):
    assert getattr(torch.version,'maca',None) and logits.is_cuda
    assert logits.ndim==3 and logits.shape[:2]==candidates.shape
    assert logits.stride(-1)==1 and candidates.stride(-1)==1
    paths,sequence,vocab=logits.shape
    matches=torch.empty((paths,sequence-1),device=logits.device,dtype=torch.int32)
    best=torch.empty((),device=logits.device,dtype=torch.int64)
    length=torch.empty_like(best)
    _matches[(paths,sequence-1)](logits,candidates,matches,logits.stride(0),logits.stride(1),
        candidates.stride(0),sequence-1,vocab,offset,0.0 if threshold is None else float(threshold),
        triton.next_power_of_2(vocab),num_warps=8)
    _reduce[(1,)](matches,best,length,paths,sequence-1,triton.next_power_of_2(paths),
        triton.next_power_of_2(sequence-1),num_warps=4)
    return best,length
