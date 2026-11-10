"""Control for compressed-dot regression: plain BF16 GEMM and device copy."""
import json,torch,triton
import triton.language as tl
from pathlib import Path
from ports.metax.real_linear_inputs import weight,activation
@triton.jit
def plain(X,W,Y,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
 mi=tl.program_id(0)*BM+tl.arange(0,BM);ni=tl.program_id(1)*BN+tl.arange(0,BN);ki=tl.arange(0,BK)
 acc=tl.full((BM,BN),0,tl.float32)
 for start in range(triton.cdiv(K,BK)):
  kk=start*BK+ki
  x=tl.load(X+mi[:,None]*K+kk[None,:],(mi[:,None]<M)&(kk[None,:]<K),other=0)
  w=tl.load(W+ni[:,None]*K+kk[None,:],(ni[:,None]<N)&(kk[None,:]<K),other=0)
  acc=tl.dot(x,tl.trans(w),acc)
 tl.store(Y+mi[:,None]*N+ni[None,:],acc,(mi[:,None]<M)&(ni[None,:]<N))
def bench(fn):
 for _ in range(3):fn()
 a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
 for _ in range(10):fn()
 b.record();b.synchronize();return a.elapsed_time(b)/10
w=weight('language_model.model.layers.0.mlp.gate_proj.weight').cuda();x,source=activation('mlp.gate_proj.weight',279)
m=279;n,k=w.shape;y=torch.empty((m,n),device='cuda',dtype=torch.bfloat16);ref=torch.nn.functional.linear(x,w)
rows=[]
for bm in [32,64]:
 for stages in [1,2]:
  try:
   def run():plain[(triton.cdiv(m,bm),triton.cdiv(n,64))](x,w,y,m,n,k,bm,64,128,num_warps=4,num_stages=stages)
   ms=bench(run);rows.append(dict(bm=bm,bn=64,bk=128,num_stages=stages,plain_ms=ms,exact=torch.equal(y.reshape_as(ref),ref)))
  except Exception as exc:rows.append(dict(bm=bm,num_stages=stages,error=repr(exc)))
  print(rows[-1],flush=True)
buf=torch.empty_like(w);copy_ms=bench(lambda:buf.copy_(w))
report=dict(activation_source=source,shape=list(w.shape),M=m,native_ms=bench(lambda:torch.nn.functional.linear(x,w)),copy_ms=copy_ms,copy_read_plus_write_GB_s=w.numel()*4/copy_ms/1e6,plain=rows)
Path('/workspace/outputs/metax_v2_20261104/plain-matmul-control.json').write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True)
