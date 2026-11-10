"""Packed BF16 word restore on real checkpoints/MLP tensors; standalone GPU screen."""
import json,torch,triton
from pathlib import Path
from ports.metax import lossless as old,lossless_vector as new
from ports.metax.real_linear_inputs import weight,activation
rows=[]
def bench(fn):
 for _ in range(5):fn()
 a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
 for _ in range(30):fn()
 b.record();b.synchronize();return a.elapsed_time(b)/30
for suffix in ['mlp.gate_proj.weight','mlp.down_proj.weight','self_attn.o_proj.weight']:
 key='language_model.model.layers.'+('7.' if suffix.startswith('self_attn') else '0.')+suffix
 cpu=weight(key);w=cpu.cuda();N=w.numel();buf=torch.empty_like(w);bits=buf.view(torch.int16)
 a={k:v.cuda() for k,v in old.encode(cpu).items()};b={k:v.cuda() for k,v in new.encode(cpu).items()};x,source=activation(suffix,279)
 def exceptions():
  n=b['indices'].numel()
  if n:old._exceptions[(triton.cdiv(n,512),)](b['indices'],b['values'],bits,n,512)
 def original():
  old._restore[(triton.cdiv(N,1024),)](a['low'],a['high'],a['dictionary'],bits,N,1024)
  n=a['indices'].numel()
  if n:old._exceptions[(triton.cdiv(n,512),)](a['indices'],a['values'],bits,n,512)
 def combined(fn):
  def run():fn();return torch.nn.functional.linear(x,buf)
  return run
 original_ms=bench(original);original_combined=bench(combined(original));ref=torch.nn.functional.linear(x,w)
 for width,fn in [(2,new._restore2),(4,new._restore4)]:
  for B in [512,1024,2048]:
   def restore():
    fn[(triton.cdiv(N//width,B),)](b['low'],b['high'],b['table'],bits,N,B,num_warps=4);exceptions()
   restore();assert torch.equal(w.view(torch.int16),bits),'Packed restore bit pattern changed'
   assert torch.equal(ref,combined(restore)()),'Vendor BF16 GEMM changed'
   row=dict(name=key,activation_source=source,width=width,block=B,bits_exact=True,gemm_exact=True,old_restore_ms=original_ms,new_restore_ms=bench(restore),old_combined_ms=original_combined,new_combined_ms=bench(combined(restore)))
   rows.append(row);print(row,flush=True)
 del cpu,w,buf,bits,a,b,x,ref;torch.cuda.empty_cache()
Path('/workspace/outputs/metax_v2_20261104/vector-screen.json').write_text(json.dumps(rows,indent=2)+'\n')
