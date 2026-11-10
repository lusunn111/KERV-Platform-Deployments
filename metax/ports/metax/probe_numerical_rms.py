"""Throughput screen using real all-layer-diagnostic inputs, never random data."""
import json,torch
from pathlib import Path
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from ports.metax.numerical_rms import rms_native
root=Path('/workspace/outputs/metax_numerical_20261105');rows=[]
def bench(fn):
 for _ in range(10):fn()
 a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
 for _ in range(100):fn()
 b.record();b.synchronize();return a.elapsed_time(b)/100
for path in [root/'all_layer_diag02/first-real-divergence.pt',root/'all_layer_diag02/first-tail-input.pt']:
 d=torch.load(path,weights_only=True);x=d['x'].cuda();weight=d['weight'].cuda();eps=d['eps']
 module=LlamaRMSNorm(4096,eps).cuda().to(torch.bfloat16);module.weight=torch.nn.Parameter(weight)
 with torch.inference_mode():
  ref=lambda:module(x)
  old=ref();result=dict(input=str(path),shape=list(x.shape),native_ms=bench(ref))
  for name,native_inv in [('native_inverse',True),('native_mean',False)]:
   fn=lambda:rms_native(x,weight,eps,native_inverse=native_inv)
   actual=fn();assert torch.equal(actual.view(torch.int16),old.view(torch.int16))
   result[name]=dict(ms=bench(fn),bits_exact=True)
  rows.append(result);print(result,flush=True)
(root/'rms-throughput.json').write_text(json.dumps(rows,indent=2)+'\n')
