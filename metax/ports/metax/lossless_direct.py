"""Exact inline BF16 weight decoding with short per-block exception lists.
Replaces the expensive per-weight prefix scan of the failed fused prototype.
GEMM accumulation parity is evaluated independently from lossless weight bits.
"""
import numpy as np,torch,triton
import triton.language as tl
from ports.metax.lossless import encode as original_encode

def encode(weight):
 e=original_encode(weight);indices=e.pop('indices').numpy();values=e.pop('values').numpy().view(np.uint16)
 n=weight.numel()//128;counts=np.bincount(indices//128,minlength=n).astype(np.uint32)
 offsets=(np.cumsum(counts)-counts).astype(np.uint32)
 assert counts.max(initial=0)<256 and len(indices)<2**24
 e['info']=torch.from_numpy((offsets|(counts<<24)).view(np.int32))
 e['exceptions']=torch.from_numpy((((indices.astype(np.uint32)%128)<<8)|(values>>8)).astype(np.uint16).view(np.int16))
 return e

@triton.jit
def _restore(L,H,D,INFO,E,W,N:tl.constexpr,B:tl.constexpr):
 i=tl.program_id(0)*B+tl.arange(0,B);valid=i<N
 low=tl.load(L+i,valid,other=0).to(tl.uint16)
 pair=tl.load(H+i//2,valid,other=0).to(tl.uint32);code=(pair>>(4*(i%2)))&15
 high=tl.load(D+code).to(tl.uint16)
 info=tl.load(INFO+i//128,valid,other=0).to(tl.uint32);count=info>>24;offset=info&16777215
 for j in range(tl.max(count,0)):
  extra=tl.load(E+offset+j,(j<count)&valid,other=0).to(tl.uint16)
  high=tl.where(((i%128)==(extra>>8))&(j<count),extra&255,high)
 tl.store(W+i,(high<<8)|low,valid)
@triton.jit(do_not_specialize=['M'])
def _matmul(X,L,H,D,INFO,E,Y,M,N:tl.constexpr,K:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
 mi=tl.program_id(0)*BM+tl.arange(0,BM);ni=tl.program_id(1)*BN+tl.arange(0,BN);ki=tl.arange(0,BK)
 d=tl.cast(D,tl.pointer_type(tl.uint32));d0=tl.load(d);d1=tl.load(d+1);d2=tl.load(d+2);d3=tl.load(d+3)
 acc=tl.full((BM,BN),0,tl.float32)
 for block in range(triton.cdiv(K,BK)):
  kk=block*BK+ki;idx=ni[:,None]*K+kk[None,:];valid=(ni[:,None]<N)&(kk[None,:]<K)
  low=tl.load(L+idx,valid,other=0).to(tl.uint16)
  pair=tl.load(H+idx//2,valid,other=0).to(tl.uint32);code=(pair>>(4*(idx%2)))&15
  word=tl.where(code<4,d0,tl.where(code<8,d1,tl.where(code<12,d2,d3)))
  high=((word>>(8*(code%4)))&255).to(tl.uint16)
  info=tl.load(INFO+ni*(K//BK)+block,ni<N,other=0).to(tl.uint32);count=info>>24;offset=info&16777215
  for j in range(tl.max(count,0)):
   extra=tl.load(E+offset+j,(j<count)&(ni<N),other=0).to(tl.uint16)
   position=extra>>8
   high=tl.where((ki[None,:]==position[:,None])&(j<count[:,None]),(extra&255)[:,None],high)
  w=((high<<8)|low).to(tl.bfloat16,bitcast=True)
  x=tl.load(X+mi[:,None]*K+kk[None,:],(mi[:,None]<M)&(kk[None,:]<K),other=0)
  acc=tl.dot(x,tl.trans(w),acc)
 tl.store(Y+mi[:,None]*N+ni[None,:],acc,(mi[:,None]<M)&(ni[None,:]<N))
def matmul(x,e,shape):
 n,k=shape;m=x.numel()//k;assert x.is_contiguous() and x.dtype==torch.bfloat16 and k%128==0
 y=torch.empty((*x.shape[:-1],n),dtype=x.dtype,device=x.device)
 _matmul[(triton.cdiv(m,64),triton.cdiv(n,64))](x,e['low'],e['high'],e['dictionary'],e['info'],e['exceptions'],y,m,n,k,64,64,128,num_warps=4,num_stages=1)
 return y
