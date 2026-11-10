"""Real-checkpoint restore/GEMM screening; must run alone under the device lock."""
import json,time,torch,triton
from pathlib import Path
from safetensors import safe_open
from ports.metax.real_linear_inputs import activation
from ports.metax import lossless as old,lossless_v2 as new
root=Path('/workspace/checkpoints/openvla-libero-goal');index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
def weight(key):
 with safe_open(root/index[key],framework='pt',device='cpu') as f:return f.get_tensor(key)
def bench(fn):
 for _ in range(5):fn()
 a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
 for _ in range(30):fn()
 b.record();b.synchronize();return a.elapsed_time(b)/30
rows=[]
for suffix in ['mlp.gate_proj.weight','mlp.down_proj.weight','self_attn.o_proj.weight']:
 key='language_model.model.layers.'+('7.' if suffix.startswith('self_attn') else '0.')+suffix;cpu=weight(key);w=cpu.cuda();buf=torch.empty_like(w);bits=buf.view(torch.int16);N=w.numel()
 a={k:v.cuda() for k,v in old.encode(cpu).items()};b={k:v.cuda() for k,v in new.encode(cpu).items()}
 def restore_old():
  old._restore[(triton.cdiv(N,1024),)](a['low'],a['high'],a['dictionary'],bits,N,1024)
  if a['indices'].numel():old._exceptions[(triton.cdiv(a['indices'].numel(),512),)](a['indices'],a['values'],bits,a['indices'].numel(),512)
 def restore_new():new._restore[(triton.cdiv(N,1024),)](b['low'],b['high'],b['dictionary'],b['bases'],b['exceptions'],bits,N,1024,num_warps=4)
 def restore_regs():new._restore[(triton.cdiv(N,1024),)](b['low'],b['high'],b['dictionary'],b['bases'],b['exceptions'],bits,N,1024,True,num_warps=4)
 restore_new();torch.cuda.synchronize();assert torch.equal(w.view(torch.int16),bits),'v2 decode bits differ'
 restore_regs();torch.cuda.synchronize();assert torch.equal(w.view(torch.int16),bits),'register decode bits differ'
 for M in [208,279,301]:
  x,activation_source=activation(suffix,M)
  ref=torch.nn.functional.linear(x,w)
  def combined(fn):
   def run():fn();return torch.nn.functional.linear(x,buf)
   return run
  assert torch.equal(ref,combined(restore_new)()),'v2 BF16 GEMM differs'
  rows.append(dict(name=key,activation_source=activation_source,shape=list(w.shape),M=M,bits_exact=True,gemm_exact=True,bf16_bytes=w.numel()*2,old_storage_bytes=sum(v.numel()*v.element_size() for v in a.values()),new_storage_bytes=sum(v.numel()*v.element_size() for v in b.values()),old_restore_ms=bench(restore_old),new_restore_ms=bench(restore_new),register_restore_ms=bench(restore_regs),gemm_ms=bench(lambda:torch.nn.functional.linear(x,w)),old_combined_ms=bench(combined(restore_old)),new_combined_ms=bench(combined(restore_new)),register_combined_ms=bench(combined(restore_regs))))
 del cpu,w,buf,bits,a,b,x,ref;torch.cuda.empty_cache()
out=Path('/workspace/outputs/metax_v2_20261104/lossless-screen.json');out.write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
