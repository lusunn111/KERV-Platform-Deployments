"""Real weight/input shape screening for decode+BF16 GEMM fusion, not a quality claim."""
import json,torch
from pathlib import Path
from safetensors import safe_open
from ports.metax.real_linear_inputs import activation
from ports.metax.lossless_v2 import encode,_restore
import triton
from ports.metax.lossless_matmul import matmul
root=Path('/workspace/checkpoints/openvla-libero-goal');index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
def bench(fn):
 for _ in range(3):fn()
 a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
 for _ in range(10):fn()
 b.record();b.synchronize();return a.elapsed_time(b)/10
rows=[]
for suffix in ['mlp.gate_proj.weight','mlp.down_proj.weight','self_attn.o_proj.weight']:
 key='language_model.model.layers.'+('7.' if suffix.startswith('self_attn') else '0.')+suffix
 with safe_open(root/index[key],framework='pt',device='cpu') as f:cpu=f.get_tensor(key)
 enc={k:v.cuda() for k,v in encode(cpu,128).items()};w=cpu.cuda()
 restored=torch.empty_like(w);count=w.numel()
 _restore[(triton.cdiv(count,128),)](enc['low'],enc['high'],enc['dictionary'],enc['bases'],enc['exceptions'],restored.view(torch.int16),count,128,True,num_warps=4)
 assert torch.equal(restored.view(torch.int16),w.view(torch.int16)),'block128 bit restoration failed'
 del restored
 for M in [208,279,301]:
  x,activation_source=activation(suffix,M);reference=torch.nn.functional.linear(x,w)
  for bm,bn in [(32,64),(32,128),(64,64)]:
   try:
    fn=lambda:matmul(x,enc,w.shape,bm,bn);result=fn();diff=result.float()-reference.float()
    item=dict(name=key,activation_source=activation_source,M=M,bm=bm,bn=bn,bf16_bytes=w.numel()*2,storage_bytes=sum(v.numel()*v.element_size() for v in enc.values()),exact=torch.equal(result,reference),different=int((result!=reference).sum()),max_abs=diff.abs().max().item(),relative_l2=(diff.norm()/reference.float().norm()).item(),native_gemm_ms=bench(lambda:torch.nn.functional.linear(x,w)),fused_ms=bench(fn))
   except Exception as exc:item=dict(name=key,M=M,bm=bm,bn=bn,error=repr(exc))
   rows.append(item);print(item,flush=True)
 del cpu,w,enc,x,reference;torch.cuda.empty_cache()
Path('/workspace/outputs/metax_v2_20261104/lossless-matmul-screen.json').write_text(json.dumps(rows,indent=2)+'\n')
