"""Focused repair screen: no prefix scan; real tensors, fixed proven plain-dot tile."""
import json,torch,triton
from pathlib import Path
from ports.metax.lossless_direct import encode,matmul,_restore
from ports.metax.real_linear_inputs import weight,activation
rows=[]
def bench(fn):
 for _ in range(3):fn()
 a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
 for _ in range(10):fn()
 b.record();b.synchronize();return a.elapsed_time(b)/10
for suffix in ['mlp.gate_proj.weight','mlp.down_proj.weight','self_attn.o_proj.weight']:
 key='language_model.model.layers.'+('7.' if suffix.startswith('self_attn') else '0.')+suffix;cpu=weight(key);w=cpu.cuda();encoded=encode(cpu);e={k:v.cuda() for k,v in encoded.items()}
 restored=torch.empty_like(w);N=w.numel()
 _restore[(triton.cdiv(N,1024),)](e['low'],e['high'],e['dictionary'],e['info'],e['exceptions'],restored.view(torch.int16),N,1024,num_warps=4)
 assert torch.equal(restored.view(torch.int16),w.view(torch.int16)),'Direct metadata bit restore failed'
 del restored
 info=encoded['info'].numpy().view('uint32');counts=info>>24;max_exceptions=int(counts.max());nonempty=float((counts>0).mean())
 for m in [208,279,301]:
  x,source=activation(suffix,m);ref=torch.nn.functional.linear(x,w)
  try:
   fn=lambda:matmul(x,e,w.shape);out=fn();diff=out.float()-ref.float()
   row=dict(name=key,M=m,bits_exact=True,max_exceptions_per_block=max_exceptions,nonempty_block_fraction=nonempty,activation_source=source,exact=torch.equal(out,ref),different=int((out!=ref).sum()),max_abs=diff.abs().max().item(),relative_l2=(diff.norm()/ref.float().norm()).item(),native_ms=bench(lambda:torch.nn.functional.linear(x,w)),direct_ms=bench(fn),bf16_bytes=w.numel()*2,storage_bytes=sum(v.numel()*v.element_size() for v in e.values()))
  except Exception as exc:row=dict(name=key,M=m,error=repr(exc))
  rows.append(row);print(row,flush=True)
 del cpu,w,e,x,ref;torch.cuda.empty_cache()
Path('/workspace/outputs/metax_v2_20261104/direct-screen.json').write_text(json.dumps(rows,indent=2)+'\n')
