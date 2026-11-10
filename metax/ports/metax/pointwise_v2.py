"""MetaX RoPE with explicit BF16 products; shape/stride preserving, no precision change."""
import torch,triton
import triton.language as tl

@triton.jit
def _rms_rsqrt(X,W,Y,D:tl.constexpr,EPS:tl.constexpr):
 row=tl.program_id(0);i=tl.arange(0,D)
 x=tl.load(X+row*D+i).to(tl.float32)
 variance=tl.sum(x*x,0)/D
 normalized=(x*tl.rsqrt(variance+EPS)).to(tl.bfloat16).to(tl.float32)
 weight=tl.load(W+i).to(tl.float32)
 tl.store(Y+row*D+i,normalized*weight)

def rms_rsqrt(x,weight,eps):
 assert x.is_contiguous() and x.shape[-1]==4096 and x.dtype==torch.bfloat16
 y=torch.empty_like(x)
 _rms_rsqrt[(x.numel()//4096,)](x,weight,y,4096,eps,num_warps=4,enable_fp_fusion=False)
 return y
@triton.jit(do_not_specialize=['N','T'])
def _rope(Q,K,C,S,P,OQ,OK,N,H:tl.constexpr,T,D:tl.constexpr,
 QS0:tl.constexpr,QS1:tl.constexpr,QS2:tl.constexpr,KS0:tl.constexpr,KS1:tl.constexpr,KS2:tl.constexpr,
 OS0:tl.constexpr,OS1:tl.constexpr,OS2:tl.constexpr,KT0:tl.constexpr,KT1:tl.constexpr,KT2:tl.constexpr,
 INDEXED:tl.constexpr,BLOCK:tl.constexpr):
 i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);valid=i<N
 d=i%D;h=(i//D)%H;t=i//(D*H);b=0
 r=(d+D//2)%D;sign=tl.where(d<D//2,-1.,1.)
 pos=tl.load(P+b*T+t,valid,other=0) if INDEXED else b*T+t
 c=tl.load(C+pos*D+d,valid,other=0).to(tl.float32);s=tl.load(S+pos*D+d,valid,other=0).to(tl.float32)
 q=tl.load(Q+b*QS0+h*QS1+t*QS2+d,valid,other=0).to(tl.float32)
 qr=tl.load(Q+b*QS0+h*QS1+t*QS2+r,valid,other=0).to(tl.float32)*sign
 k=tl.load(K+b*KS0+h*KS1+t*KS2+d,valid,other=0).to(tl.float32)
 kr=tl.load(K+b*KS0+h*KS1+t*KS2+r,valid,other=0).to(tl.float32)*sign
 # Each product is separately rounded to BF16 in the original eager implementation.
 a=(q*c).to(tl.bfloat16).to(tl.float32);bb=(qr*s).to(tl.bfloat16).to(tl.float32)
 ka=(k*c).to(tl.bfloat16).to(tl.float32);kb=(kr*s).to(tl.bfloat16).to(tl.float32)
 tl.store(OQ+b*OS0+h*OS1+t*OS2+d,a+bb,valid)
 tl.store(OK+b*KT0+h*KT1+t*KT2+d,ka+kb,valid)
def rope(q,k,cos,sin,positions=None):
 assert q.shape==k.shape and q.shape[0]==1 and q.dtype==k.dtype==torch.bfloat16 and q.shape[-1]==128
 indexed=cos.ndim==4
 if indexed:
  cos=cos.squeeze(0).squeeze(0);sin=sin.squeeze(0).squeeze(0)
  # Eager advanced indexing accepts CPU indices; Triton pointers must be device resident.
  positions=positions.to(device=q.device,non_blocking=False).contiguous()
 else:assert cos.shape==(q.shape[0],q.shape[2],q.shape[3])
 cos=cos.contiguous();sin=sin.contiguous()
 oq=torch.empty_like(q);ok=torch.empty_like(k)
 _rope[(triton.cdiv(q.numel(),256),)](q,k,cos,sin,positions if indexed else q,oq,ok,q.numel(),q.shape[1],q.shape[2],128,
  0,*q.stride()[1:3],0,*k.stride()[1:3],0,*oq.stride()[1:3],0,*ok.stride()[1:3],indexed,256,num_warps=4,enable_fp_fusion=False)
 return oq,ok
