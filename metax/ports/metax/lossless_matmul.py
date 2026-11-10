"""Experimental exact-weight decoding inside BF16 matmul tiles.
Weight bits are lossless. GEMM accumulation order must be compared separately.
No TF32/INT8 arithmetic and no materialized BF16 weight workspace.
"""
import torch,triton
import triton.language as tl
@triton.jit
def _matmul(X,L,H,D,BASE,E,Y,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
 BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
 mi=tl.program_id(0)*BM+tl.arange(0,BM)
 ni=tl.program_id(1)*BN+tl.arange(0,BN)
 kk=tl.arange(0,BK)
 ptr=tl.cast(D,tl.pointer_type(tl.uint32))
 d0=tl.load(ptr);d1=tl.load(ptr+1);d2=tl.load(ptr+2);d3=tl.load(ptr+3)
 acc=tl.full((BM,BN),0,tl.float32)
 for block in range(triton.cdiv(K,BK)):
  ki=block*BK+kk
  x=tl.load(X+mi[:,None]*K+ki[None,:],(mi[:,None]<M)&(ki[None,:]<K),other=0)
  idx=ni[:,None]*K+ki[None,:];valid=(ni[:,None]<N)&(ki[None,:]<K)
  low=tl.load(L+idx,valid,other=0).to(tl.uint16)
  pair=tl.load(H+idx//2,valid,other=0).to(tl.uint32);code=(pair>>(4*(idx%2)))&15
  word=tl.where(code<4,d0,tl.where(code<8,d1,tl.where(code<12,d2,d3)))
  high=((word>>(8*(code%4)))&255).to(tl.uint16)
  exception=(code==15)&valid
  rank=tl.cumsum(exception.to(tl.int32),1)-1
  base=tl.load(BASE+ni*(K//BK)+block,ni<N,other=0)
  extra=tl.load(E+base[:,None]+rank,exception,other=0).to(tl.uint16)
  high=tl.where(exception,extra,high)
  bits=((high<<8)|low).to(tl.uint16)
  w=bits.to(tl.bfloat16,bitcast=True)
  acc=tl.dot(x,tl.trans(w),acc)
 tl.store(Y+mi[:,None]*N+ni[None,:],acc,(mi[:,None]<M)&(ni[None,:]<N))
def matmul(x,encoding,shape,bm=32,bn=64):
 n,k=shape;assert x.dtype==torch.bfloat16 and x.is_contiguous() and k%128==0
 m=x.numel()//k;y=torch.empty((*x.shape[:-1],n),dtype=x.dtype,device=x.device)
 _matmul[(triton.cdiv(m,bm),triton.cdiv(n,bn))](x,encoding['low'],encoding['high'],encoding['dictionary'],encoding['bases'],encoding['exceptions'],y,m,n,k,bm,bn,128,num_warps=4,num_stages=2)
 return y

class FusedLinear(torch.nn.Module):
 def __init__(self,original,storage,offsets,counts,manifest,bm=32,bn=64):
  super().__init__();self.out_features,self.in_features=original.weight.shape
  object.__setattr__(self,'storage',storage);object.__setattr__(self,'manifest',manifest)
  self.offsets=offsets;self.counts=counts;self.bm=bm;self.bn=bn
 def forward(self,x):
  pointers=tuple(getattr(self.storage,key).data_ptr() for key in self.offsets)
  if getattr(self,'_pointers',None)!=pointers:
   object.__setattr__(self,'_views',{key:getattr(self.storage,key).narrow(0,self.offsets[key],self.counts[key]) for key in self.offsets});self._pointers=pointers
  self.manifest['forward_calls']+=1
  return matmul(x,self._views,(self.out_features,self.in_features),self.bm,self.bn)

def apply_cpu_fused(model):
 from ports.metax.lossless_v2 import encode,Storage
 from torch import nn
 # MLP-only policy avoids changing attention GEMM accumulation. No BF16 cache.
 selected=[(name,module) for name,module in model.named_modules() if name.startswith('base_model.language_model.model.layers.') and '.mlp.' in name and isinstance(module,nn.Linear)]
 assert len(selected)==96
 encodings=[];items=[]
 for name,module in selected:
  e=encode(module.weight,128);encodings.append(e)
  items.append(dict(name=name,shape=list(module.weight.shape),bf16_bytes=module.weight.numel()*2,compressed_bytes=sum(v.numel()*v.element_size() for v in e.values())))
 manifest=dict(profile='metax_bf16_lossless_fused_mlp',lossless=True,activation_dtype='bfloat16',restored_weight_dtype='bfloat16',forward_calls=0,modules=items,workspace_bytes=0,cached_weight_bytes=0,storage='low8 + dictionary15/escape + block128 prefix + exception high8',execution='MACA Triton exact weight bit decode inside BF16 dot, FP32 accumulator; accumulation parity evaluated separately')
 for begin in range(0,len(selected),16):
  storage=Storage(encodings[begin:begin+16]);model.add_module(f'_metax_fused_storage_{begin//16}',storage);offsets={k:0 for k in encodings[0]}
  for (name,module),e in zip(selected[begin:begin+16],encodings[begin:begin+16]):
   replacement=FusedLinear(module,storage,offsets.copy(),{k:v.numel() for k,v in e.items()},manifest)
   parent,child=name.rsplit('.',1);model.get_submodule(parent)._modules[child]=replacement
   for k in offsets:offsets[k]+=e[k].numel()
 manifest['storage_saving_minus_workspace_bytes']=sum(m['bf16_bytes']-m['compressed_bytes'] for m in items)
 model._metax_quant_manifest=manifest
 print('[MetaX fused lossless weights]',len(items),'MLP matrices; net theoretical bytes',manifest['storage_saving_minus_workspace_bytes'],flush=True)
 return manifest
