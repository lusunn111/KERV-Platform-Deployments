"""Two-buffer lossless decode/GEMM overlap, all memory counted in model buffers.
Old exact byte-plane codec and vendor BF16 GEMM; no GEMM precision change.
"""
import torch,triton
from torch import nn
from ports.metax.lossless import LosslessLinear,_restore,_exceptions,apply_cpu_lossless
class Pipeline(nn.Module):
 def __init__(self,capacity):
  super().__init__();self.register_buffer('weight',torch.empty((2,capacity),dtype=torch.bfloat16))
  self.ready=None;self.expected=0;self.current_buffer=0
 def initialize(self):
  self.stream=torch.cuda.Stream();self.ready=[torch.cuda.Event(),torch.cuda.Event()]
  self.used=[torch.cuda.Event(),torch.cuda.Event()];self.used_valid=[False,False]
  # These allocations also live on a side stream; allocator lifetime tracking
  # remains correct when a caller releases a model without a device-wide wait.
  self.weight.record_stream(self.stream);seen=set()
  for module in self.linear_refs:
   if id(module.storage) in seen:continue
   seen.add(id(module.storage))
   for tensor in module.storage.buffers():tensor.record_stream(self.stream)
 def run(self,module,x):
  if self.ready is None:
   self.initialize();assert module.index==0
   module.decode(0);self.ready[0].record()
  if module.index!=self.expected:raise RuntimeError(f'Lossless prefetch order changed: expected {self.expected}, got {module.index}')
  current=torch.cuda.current_stream();b=self.current_buffer;current.wait_event(self.ready[b])
  result=nn.functional.linear(x,module.weights[b])
  self.used[b].record(current);self.used_valid[b]=True
  following=(module.index+1)%len(self.linear_refs);other=1-b
  with torch.cuda.stream(self.stream):
   if self.used_valid[other]:self.stream.wait_event(self.used[other])
   self.linear_refs[following].decode(other)
   self.ready[other].record(self.stream)
  self.expected=following;self.current_buffer=other
  module.manifest['forward_calls']+=1
  return result
class PrefetchLinear(LosslessLinear):
 def prepare(self):
  key=tuple(getattr(self.storage,k).data_ptr() for k in self.offsets)+(self.workspace.weight.data_ptr(),)
  if getattr(self,'_prefetch_key',None)!=key:
   object.__setattr__(self,'views',{k:getattr(self.storage,k).narrow(0,self.offsets[k],self.counts[k]) for k in self.offsets})
   count=self.in_features*self.out_features
   object.__setattr__(self,'weights',tuple(self.workspace.weight[b,:count].view(self.out_features,self.in_features) for b in range(2)))
   object.__setattr__(self,'bits',tuple(w.view(torch.int16) for w in self.weights));self._prefetch_key=key
 def decode(self,b):
  self.prepare();v=self.views;count=self.in_features*self.out_features
  _restore[(triton.cdiv(count,1024),)](v['low'],v['high'],v['dictionary'],self.bits[b],count,1024)
  n=self.counts['indices']
  if n:_exceptions[(triton.cdiv(n,512),)](v['indices'],v['values'],self.bits[b],n,512)
  self.manifest['decode_calls']+=1
 def forward(self,x):
  assert x.dtype==torch.bfloat16 and x.device.type=='cuda' and getattr(torch.version,'maca',None)
  return self.workspace.run(self,x)
def apply_cpu_prefetch(model):
 manifest=apply_cpu_lossless(model);old_workspace=model._metax_lossless_workspace
 pipeline=Pipeline(old_workspace.weight.numel());model._modules['_metax_lossless_workspace']=pipeline
 refs=[]
 for module in model.modules():
  if isinstance(module,LosslessLinear):
   module.__class__=PrefetchLinear;object.__setattr__(module,'workspace',pipeline);module.index=len(refs);refs.append(module)
 object.__setattr__(pipeline,'linear_refs',tuple(refs));assert len(refs)==104
 manifest.update(profile='metax_bf16_lossless_double_buffer',workspace_bytes=pipeline.weight.numel()*2,cached_weight_bytes=pipeline.weight.numel()*2,decode_calls=0,execution='Original exact byte-plane codec + two shared BF16 workspaces + asynchronous one-projection lookahead + vendor BF16 GEMM')
 manifest['storage_saving_minus_workspace_bytes']-=old_workspace.weight.numel()*2
 print('[MetaX lossless prefetch] buffers bytes',manifest['workspace_bytes'],'theoretical net saving',manifest['storage_saving_minus_workspace_bytes'],flush=True)
 return manifest
