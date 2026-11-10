import json,torch
from pathlib import Path
from safetensors import safe_open
from ports.metax.pointwise_v2 import rope
from KERVRuntimeOptimization.adaptive_rms_norm import rms_norm_inference
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
root=Path('/workspace/checkpoints/openvla-libero-goal');index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
def weight(name):
 key='language_model.model.layers.0.'+name
 with safe_open(root/index[key],framework='pt',device='cpu') as f:return f.get_tensor(key).cuda()
qweight=weight('self_attn.q_proj.weight');kweight=weight('self_attn.k_proj.weight');norm=weight('input_layernorm.weight')
from transformers import LlamaConfig
eps=LlamaConfig(**json.loads((root/'config.json').read_text())['text_config']).rms_norm_eps
inputs=torch.load('/workspace/outputs/metax_v2_20261104/probe01/actual_inputs.pt',weights_only=True)
def bench(fn):
 for _ in range(5):fn()
 torch.cuda.synchronize();a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
 for _ in range(50):fn()
 b.record();b.synchronize();return a.elapsed_time(b)/50
rows=[]
for name,cpu in inputs.items():
 x=cpu.cuda();M=x.shape[1]
 with torch.inference_mode():
  if name.startswith('rms_'):
   def ref():
    f=x.float();return norm*(f*torch.rsqrt(f.square().mean(-1,keepdim=True)+eps)).to(x.dtype)
   def new():return rms_norm_inference(x,(4096,),norm,eps,round_before_weight=True)
   a=ref();b=new();rows.append(dict(kind='rms',rows=M,old_ms=bench(ref),new_ms=bench(new),exact=torch.equal(a,b),different=int((a!=b).sum()),max_abs=(a.float()-b.float()).abs().max().item()))
  if name.startswith('qkv_'):
   q=torch.nn.functional.linear(x,qweight).view(1,M,32,128).transpose(1,2);k=torch.nn.functional.linear(x,kweight).view(1,M,32,128).transpose(1,2)
   inv=1/(10000**(torch.arange(0,128,2,device='cuda').float()/128));angle=torch.arange(M,device='cuda').float()[:,None]*inv[None]
   angles=torch.cat([angle,angle],-1)[None];cos=angles.cos().to(torch.bfloat16);sin=angles.sin().to(torch.bfloat16)
   ref=lambda:apply_rotary_pos_emb(q,k,cos,sin);new=lambda:rope(q,k,cos,sin)
   a=ref();b=new();rows.append(dict(kind='rope',rows=M,old_ms=bench(ref),new_ms=bench(new),exact=all(torch.equal(x,y) for x,y in zip(a,b)),strides_equal=[x.stride()==y.stride() for x,y in zip(a,b)],different=sum(int((x!=y).sum()) for x,y in zip(a,b))))
Path('/workspace/outputs/metax_v2_20261104/pointwise-screen.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
