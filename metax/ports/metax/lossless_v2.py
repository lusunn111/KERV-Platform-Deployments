"""Experimental one-pass exact BF16 decoding, block-local sparse high bytes.
No rounded weights, no persistent BF16 shadows; all buffers count in peak memory.
"""
import numpy as np
import torch,triton
from torch import nn
import triton.language as tl
from ports.metax.weight_only import Workspace
from ports.metax.lossless import LosslessLinear
BLOCK=1024
@triton.jit
def _restore(L,H,D,BASE,E,W,N:tl.constexpr,B:tl.constexpr,REG:tl.constexpr=False):
 block=tl.program_id(0);r=tl.arange(0,B);i=block*B+r;valid=i<N
 low=tl.load(L+i,valid,other=0).to(tl.uint16)
 pair=tl.load(H+i//2,valid,other=0).to(tl.uint32);code=(pair>>(4*(i%2)))&15
 exception=(code==15)&valid
 if REG:
  ptr=tl.cast(D,tl.pointer_type(tl.uint32))
  d0=tl.load(ptr);d1=tl.load(ptr+1);d2=tl.load(ptr+2);d3=tl.load(ptr+3)
  word=tl.where(code<4,d0,tl.where(code<8,d1,tl.where(code<12,d2,d3)))
  high=((word>>(8*(code%4)))&255).to(tl.uint16)
 else:
  high=tl.load(D+tl.minimum(code,14)).to(tl.uint16)
 if tl.sum(exception.to(tl.int32),0)>0:
  rank=tl.cumsum(exception.to(tl.int32),0)-1
  base=tl.load(BASE+block)
  other=tl.load(E+base+rank,exception,other=0).to(tl.uint16)
  high=tl.where(exception,other,high)
 tl.store(W+i,(high<<8)|low,valid)
def encode(weight,block=BLOCK):
 bits=weight.detach().contiguous().view(torch.int16).numpy().reshape(-1).view(np.uint16)
 assert len(bits)%block==0
 high=(bits>>8).astype(np.uint8);counts=np.bincount(high,minlength=256)
 dictionary=np.argsort(-counts,kind='stable')[:15].astype(np.uint8)
 lookup=np.full(256,15,dtype=np.uint8);lookup[dictionary]=np.arange(15,dtype=np.uint8)
 code=lookup[high];exception=code==15;block_counts=exception.reshape(-1,block).sum(-1)
 bases=(np.cumsum(block_counts)-block_counts).astype(np.int32)
 return dict(low=torch.from_numpy((bits&255).astype(np.uint8)),high=torch.from_numpy(code[0::2]|(code[1::2]<<4)),dictionary=torch.from_numpy(np.pad(dictionary,(0,1))),bases=torch.from_numpy(bases),exceptions=torch.from_numpy(high[exception].copy()))
class Storage(nn.Module):
 def __init__(self,encodings):
  super().__init__()
  for key in encodings[0]:self.register_buffer(key,torch.cat([e[key] for e in encodings]))
class Linear(LosslessLinear):
 def forward(self,x):
  assert x.dtype==torch.bfloat16 and x.device.type=='cuda' and getattr(torch.version,'maca',None)
  pointer_key=tuple(getattr(self.storage,key).data_ptr() for key in self.offsets)+(self.workspace.weight.data_ptr(),)
  count=self.in_features*self.out_features
  if getattr(self,'_pointer_key',None)!=pointer_key:
   object.__setattr__(self,'_views',{key:getattr(self.storage,key).narrow(0,self.offsets[key],self.counts[key]) for key in self.offsets})
   object.__setattr__(self,'_weight',self.workspace.weight[:count].view(self.out_features,self.in_features))
   object.__setattr__(self,'_bits',self._weight.view(torch.int16));self._pointer_key=pointer_key
  v=self._views
  _restore[(triton.cdiv(count,BLOCK),)](v['low'],v['high'],v['dictionary'],v['bases'],v['exceptions'],self._bits,count,BLOCK,num_warps=4)
  self.manifest['forward_calls']+=1
  return nn.functional.linear(x,self._weight)
def install(model,selected):
 encodings=[];items=[]
 for name,linear in selected:
  e=encode(linear.weight);encodings.append(e)
  items.append(dict(name=name,shape=list(linear.weight.shape),bf16_bytes=linear.weight.numel()*2,compressed_bytes=sum(x.numel()*x.element_size() for x in e.values()),exception_count=e['exceptions'].numel(),weight_count=linear.weight.numel()))
 workspace=Workspace(max(linear.weight.numel() for _,linear in selected));model.add_module('_metax_lossless_workspace',workspace)
 manifest=dict(profile='metax_bf16_lossless_block15_v2',lossless=True,activation_dtype='bfloat16',restored_weight_dtype='bfloat16',storage='low8 + high4 (15-value dictionary, escape=15) + block1024 prefix + exception high8',forward_calls=0,modules=items,workspace_bytes=workspace.weight.numel()*2,execution='one MACA Triton exact restore + vendor BF16 GEMM',cached_weight_bytes=0)
 for begin in range(0,len(selected),16):
  storage=Storage(encodings[begin:begin+16]);model.add_module(f'_metax_lossless_storage_{begin//16}',storage);offsets={key:0 for key in encodings[0]}
  for (name,linear),e in zip(selected[begin:begin+16],encodings[begin:begin+16]):
   replacement=Linear(linear,storage,workspace,offsets.copy(),{key:v.numel() for key,v in e.items()},manifest)
   parent,child=name.rsplit('.',1);model.get_submodule(parent)._modules[child]=replacement
   for key in offsets:offsets[key]+=e[key].numel()
 manifest['storage_saving_minus_workspace_bytes']=sum(m['bf16_bytes']-m['compressed_bytes'] for m in items)-manifest['workspace_bytes']
 model._metax_quant_manifest=manifest
 print('[MetaX lossless v2]',len(items),'matrices; theoretical net bytes',manifest['storage_saving_minus_workspace_bytes'],flush=True)
 return manifest

def apply_cpu_lossless(model):
 selected=[]
 for name,module in model.named_modules():
  if not name.startswith('base_model.language_model.model.layers.') or not isinstance(module,nn.Linear):continue
  layer=int(name.split('.')[4])
  if '.mlp.' in name or ('.self_attn.o_proj' in name and 7<=layer<15):selected.append((name,module))
 assert len(selected)==104
 return install(model,selected)
