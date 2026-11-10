"""RMS repair candidates: native moments, explicit original BF16 boundary.
No layer/test-action dependent selection. Used only for verifier handwritten RMS.
"""
import torch,triton
import triton.language as tl

@triton.jit
def _tail(X,W,V,I,Y,D:tl.constexpr,EPS:tl.constexpr,NATIVE_INV:tl.constexpr,BF16_MUL:tl.constexpr):
 row=tl.program_id(0);d=tl.arange(0,D)
 x=tl.load(X+row*D+d).to(tl.float32)
 if NATIVE_INV:inverse=tl.load(I+row)
 else:inverse=tl.rsqrt(tl.load(V+row)+EPS)
 normalized=(x*inverse).to(tl.bfloat16)
 weight=tl.load(W+d)
 if BF16_MUL:result=normalized*weight
 else:result=normalized.to(tl.float32)*weight.to(tl.float32)
 tl.store(Y+row*D+d,result)

def rms_native(x,weight,eps,*,native_inverse=True,bf16_multiply=True):
 if not (x.device.type=='cuda' and getattr(torch.version,'maca',None) and x.dtype==weight.dtype==torch.bfloat16 and x.shape[-1]==4096 and x.is_contiguous() and weight.is_contiguous()):
  f=x.float();return weight*(f*torch.rsqrt(f.pow(2).mean(-1,keepdim=True)+eps)).to(x.dtype)
 # Retain the real vendor aten mean, with its actual reduction order.
 values=x.float();variance=values.pow(2).mean(-1,keepdim=True)
 inverse=torch.rsqrt(variance+eps) if native_inverse else variance
 y=torch.empty_like(x)
 _tail[(x.numel()//4096,)](x,weight,variance,inverse,y,4096,float(eps),native_inverse,bf16_multiply,num_warps=4,enable_fp_fusion=False)
 return y

@triton.jit
def _stages(X,W,SQ,MEAN,ADDED,SQRT,RECIP,RSQRT,NORM,CAST,Y,D:tl.constexpr,EPS:tl.constexpr,MODE:tl.constexpr):
 row=tl.program_id(0);i=tl.arange(0,D);v=tl.load(X+row*D+i).to(tl.float32)
 square=v*v;mean=tl.sum(square,0)/D;added=mean+EPS
 sqrt=tl.sqrt(added);recip=1./sqrt;rsqrt=tl.rsqrt(added)
 inv=recip if MODE==0 else rsqrt
 normalized=v*inv;cast=normalized.to(tl.bfloat16)
 out=cast.to(tl.float32)*tl.load(W+i).to(tl.float32)
 tl.store(SQ+row*D+i,square);tl.store(MEAN+row,mean);tl.store(ADDED+row,added)
 tl.store(SQRT+row,sqrt);tl.store(RECIP+row,recip);tl.store(RSQRT+row,rsqrt)
 tl.store(NORM+row*D+i,normalized);tl.store(CAST+row*D+i,cast);tl.store(Y+row*D+i,out)

def stages(x,weight,eps,*,mode=0,warps=8,fusion=True):
 out={k:torch.empty_like(x,dtype=torch.float32) for k in ['square','normalized']}
 out.update({k:torch.empty((x.numel()//4096,),device=x.device,dtype=torch.float32) for k in ['mean','added','sqrt','reciprocal','rsqrt']})
 out.update({k:torch.empty_like(x) for k in ['bf16','output']})
 _stages[(x.numel()//4096,)](x,weight,out['square'],out['mean'],out['added'],out['sqrt'],out['reciprocal'],out['rsqrt'],out['normalized'],out['bf16'],out['output'],4096,float(eps),mode,num_warps=warps,enable_fp_fusion=fusion)
 return out


@triton.jit
def _inverse(V,A,S,R,I,N:tl.constexpr,EPS:tl.constexpr,B:tl.constexpr):
 i=tl.program_id(0)*B+tl.arange(0,B);valid=i<N
 added=tl.load(V+i,valid,other=0)+EPS
 sq=tl.sqrt(added)
 tl.store(A+i,added,valid);tl.store(S+i,sq,valid)
 tl.store(R+i,1./sq,valid);tl.store(I+i,tl.rsqrt(added),valid)
def inverse_stages(variance,eps):
 out={k:torch.empty_like(variance) for k in ['added','sqrt','reciprocal','rsqrt']}
 _inverse[(triton.cdiv(variance.numel(),256),)](variance,out['added'],out['sqrt'],out['reciprocal'],out['rsqrt'],variance.numel(),float(eps),256,num_warps=4,enable_fp_fusion=False)
 return out
