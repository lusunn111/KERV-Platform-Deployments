"""Lossless two-/four-word restore with pre-expanded high-byte pair tables."""
import numpy as np,torch,triton
from torch import nn
import triton.language as tl
from ports.metax.lossless import encode as original_encode,_exceptions,LosslessLinear
from ports.metax.weight_only import Workspace
@triton.jit
def _restore2(L,H,T,W,N:tl.constexpr,B:tl.constexpr):
 i=tl.program_id(0)*B+tl.arange(0,B);mask=i<N//2
 low=tl.load(tl.cast(L,tl.pointer_type(tl.uint16))+i,mask,other=0).to(tl.uint32)
 codes=tl.load(H+i,mask,other=0).to(tl.int32)
 high=tl.load(T+codes).to(tl.uint32)
 value=(low&255)|((low&65280)<<8)|high
 tl.store(tl.cast(W,tl.pointer_type(tl.uint32))+i,value,mask)
@triton.jit
def _restore4(L,H,T,W,N:tl.constexpr,B:tl.constexpr):
 i=tl.program_id(0)*B+tl.arange(0,B);mask=i<N//4
 low=tl.load(tl.cast(L,tl.pointer_type(tl.uint32))+i,mask,other=0)
 codes=tl.load(tl.cast(H,tl.pointer_type(tl.uint16))+i,mask,other=0).to(tl.uint32)
 high0=tl.load(T+(codes&255)).to(tl.uint32);high1=tl.load(T+(codes>>8)).to(tl.uint32)
 word0=(low&255)|((low&65280)<<8)|high0
 word1=((low>>16)&255)|((low>>24)<<16)|high1
 value=word0.to(tl.uint64)|(word1.to(tl.uint64)<<32)
 tl.store(tl.cast(W,tl.pointer_type(tl.uint64))+i,value,mask)
def encode(weight):
 e=original_encode(weight);dictionary=e.pop('dictionary').numpy().astype(np.uint32);code=np.arange(256,dtype=np.uint32)
 e['table']=torch.from_numpy(((dictionary[code&15]<<8)|(dictionary[code>>4]<<24)).view(np.int32))
 return e
class Storage(nn.Module):
 def __init__(self,encodings):
  super().__init__()
  for key in encodings[0]:self.register_buffer(key,torch.cat([e[key] for e in encodings]))
class Linear(LosslessLinear):
 def forward(self,x):
  assert x.dtype==torch.bfloat16 and x.device.type=='cuda' and getattr(torch.version,'maca',None)
  key=tuple(getattr(self.storage,k).data_ptr() for k in self.offsets)+(self.workspace.weight.data_ptr(),)
  count=self.in_features*self.out_features
  if getattr(self,'_vector_key',None)!=key:
   object.__setattr__(self,'views',{k:getattr(self.storage,k).narrow(0,self.offsets[k],self.counts[k]) for k in self.offsets})
   object.__setattr__(self,'weight_view',self.workspace.weight[:count].view(self.out_features,self.in_features))
   object.__setattr__(self,'bits',self.weight_view.view(torch.int16));self._vector_key=key
  v=self.views
  _restore4[(triton.cdiv(count//4,1024),)](v['low'],v['high'],v['table'],self.bits,count,1024,num_warps=4)
  n=self.counts['indices']
  if n:_exceptions[(triton.cdiv(n,512),)](v['indices'],v['values'],self.bits,n,512)
  self.manifest['forward_calls']+=1
  return nn.functional.linear(x,self.weight_view)
def apply_cpu_vector(model):
 selected=[]
 for name,module in model.named_modules():
  if not name.startswith('base_model.language_model.model.layers.') or not isinstance(module,nn.Linear):continue
  layer=int(name.split('.')[4])
  if '.mlp.' in name or ('.self_attn.o_proj' in name and 7<=layer<15):selected.append((name,module))
 assert len(selected)==104
 encodings=[];items=[]
 for name,module in selected:
  e=encode(module.weight);encodings.append(e)
  items.append(dict(name=name,shape=list(module.weight.shape),bf16_bytes=module.weight.numel()*2,compressed_bytes=sum(v.numel()*v.element_size() for v in e.values())))
 workspace=Workspace(max(m.weight.numel() for _,m in selected));model.add_module('_metax_lossless_workspace',workspace)
 manifest=dict(profile='metax_bf16_lossless_vector4',lossless=True,activation_dtype='bfloat16',restored_weight_dtype='bfloat16',forward_calls=0,modules=items,workspace_bytes=workspace.weight.numel()*2,cached_weight_bytes=0,execution='Exact packed four-word decode + sparse exception scatter + vendor BF16 GEMM')
 for begin in range(0,len(selected),16):
  storage=Storage(encodings[begin:begin+16]);model.add_module(f'_metax_lossless_storage_{begin//16}',storage);offsets={k:0 for k in encodings[0]}
  for (name,module),e in zip(selected[begin:begin+16],encodings[begin:begin+16]):
   replacement=Linear(module,storage,workspace,offsets.copy(),{k:v.numel() for k,v in e.items()},manifest)
   parent,child=name.rsplit('.',1);model.get_submodule(parent)._modules[child]=replacement
   for k in offsets:offsets[k]+=e[k].numel()
 manifest['storage_saving_minus_workspace_bytes']=sum(m['bf16_bytes']-m['compressed_bytes'] for m in items)-manifest['workspace_bytes']
 model._metax_quant_manifest=manifest;print('[MetaX vector lossless]',manifest['storage_saving_minus_workspace_bytes'],'theoretical saving bytes',flush=True)
 return manifest
