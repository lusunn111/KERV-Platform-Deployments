"""Real-input operator screening; diagnostic time is never rollout performance."""
import json,os,time
from pathlib import Path
from types import SimpleNamespace
import torch,numpy as np
from experiments.robot.openvla_utils import get_vla,get_processor,get_vla_action
from experiments.robot.flagos_ops import enable_embodied_ops,enable_tree_builder
from experiments.robot.tool_utils import compute_dynamic_threshold
from ports.metax.verify import verify_accept
out=Path('/workspace/outputs/metax_v2_20261104/probe01');out.mkdir(parents=True,exist_ok=False)
cfg=SimpleNamespace(pretrained_checkpoint='/workspace/checkpoints/openvla-libero-goal',spec_checkpoint='/workspace/checkpoints/kerv-drafter',use_spec=True,parallel_draft=False,accept_threshold=9,accept_threshold_start=15,load_in_8bit=False,load_in_4bit=False)
torch.manual_seed(7);model=get_vla(cfg).eval();processor=get_processor(cfg)
enable_embodied_ops(model=model,enabled=True,include='kerv_silu_mul,kerv_add_rms_norm,kerv_verify_accept_control,kerv_static_tree_pack,kerv_tree_embed_pack,kerv_kv_commit,kerv_kv_accept_commit',backend='native',record=False,record_once=True,log_path=None,manifest_path=None)
enable_tree_builder(model=model,enabled=True,static_runtime=True,verify_max_paths=0,node_buckets='',operatorized=True,draft_tree_operatorized=True,kalman_batch=True,kv_commit_batched=True,fused_control_transfer=True,compile_verifier=False,persistent_runtime=False,resident_kv_cache=False,static_tree_attention='off',manifest_path=None)
model._flagos_system_optimization_enabled=True;model._flagos_persistent_control_buffer=True;model._flagos_persistent_decode_workspace=True;model._flagos_inference_mode_enabled=True
model.ea_layer._flagos_persistent_draft_workspace=True;model.ea_layer._flagos_persistent_draft_kv_cache=True
captures={};logits_captures=[]
def capture(name):
 def hook(m,args):
  x=args[0];key=(name,tuple(x.shape))
  if key not in captures:captures[key]=x.detach().clone()
 return hook
layer=model.base_model.language_model.model.layers[0]
for name,module in [('qkv',layer.self_attn.q_proj),('gate_up',layer.mlp.gate_proj),('rms',layer.input_layernorm)]:module.register_forward_pre_hook(capture(name))
from openvla.prismatic.extern.hf import modeling_speculation as ms
original=ms.tree_decoding
def observed(*args,**kwargs):
 result=original(*args,**kwargs)
 if len(logits_captures)<4:logits_captures.append((result[0].detach().clone(),args[2][0,args[6]].detach().clone() if False else None))
 return result
# Capture unique logits at the output of the LM (before path expansion).
def lm_hook(m,args,output):
 if hasattr(output,'logits') and output.logits.shape[1]<100 and len(logits_captures)<4:logits_captures.append(output.logits.detach().clone())
model.base_model.language_model.register_forward_hook(lm_hook)
model.start_rollout(max_steps=220,kalman_cfg={'enabled':True,'tree_enabled':True},store_history=True)
history=[]
for i,file in enumerate(sorted(Path('/workspace/outputs/metax_optimized_smoke_01/observations').glob('*.npz'))[:4]):
 with np.load(file) as data:obs={'full_image':data['image'].copy()};task=str(data['task'])
 value=get_vla_action(model,processor,cfg.pretrained_checkpoint,obs,task,'libero_goal',center_crop=True,generate_mode='speculative',history=history,step_idx=i,max_steps=220,dynamic_threshold=compute_dynamic_threshold(i,220,15,3),use_kalman=True,kalman_tree_enabled=True)
 history.append(value.copy())
print('real captures',[(k,tuple(v.shape)) for k,v in captures.items()],flush=True)
def bench(fn,n=30):
 for _ in range(5):fn()
 torch.cuda.synchronize();start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record()
 for _ in range(n):fn()
 end.record();end.synchronize();return start.elapsed_time(end)/n
rows=[]
for (kind,shape),x in captures.items():
 if kind not in ['qkv','gate_up']:continue
 mods=[layer.self_attn.q_proj,layer.self_attn.k_proj,layer.self_attn.v_proj] if kind=='qkv' else [layer.mlp.gate_proj,layer.mlp.up_proj]
 packed=torch.cat([m.weight.detach() for m in mods],0)
 with torch.inference_mode():
  baseline=[m(x) for m in mods];actual=torch.nn.functional.linear(x,packed).split([m.out_features for m in mods],-1)
  old=bench(lambda:[m(x) for m in mods]);new=bench(lambda:torch.nn.functional.linear(x,packed))
  rows.append(dict(kind=kind,shape=list(shape),separate_ms=old,merged_ms=new,speedup=old/new,exact=all(torch.equal(a,b) for a,b in zip(baseline,actual)),max_abs=max((a.float()-b.float()).abs().max().item() for a,b in zip(baseline,actual))))
 torch.cuda.empty_cache()
(out/'screen.json').write_text(json.dumps(rows,indent=2)+'\n')
torch.save({f'{kind}_{"x".join(map(str,shape))}':v.cpu() for (kind,shape),v in captures.items()},out/'actual_inputs.pt')
torch.save([x.cpu() for x in logits_captures],out/'unique_logits.pt')
print(json.dumps(rows,indent=2),flush=True)
