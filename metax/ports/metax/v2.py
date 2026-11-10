"""Opt-in v2 experiments, isolated from first-round native/optimized profiles."""
import os,types
import torch

def install(model):
 features=set(filter(None,os.environ.get('KERV_METAX_V2','').split(',')))
 if not features:return
 if features-{'compact','vision_graph','qkv','gate_up','rms','rms_rsqrt','rms_native','rms_native_mean','rope','prefill_graph'}:raise ValueError(features)
 if not getattr(torch.version,'maca',None):raise RuntimeError('MetaX vendor runtime required')
 model._metax_v2_features=sorted(features);model._metax_v2_stats={}
 model._metax_compact_verify='compact' in features
 model._metax_global_restores=[]
 model._metax_rms_restores=[]
 rms_features=features & {'rms','rms_rsqrt','rms_native','rms_native_mean'}
 if len(rms_features)>1:raise ValueError('Choose exactly one RMS implementation')
 if rms_features:
  from KERVRuntimeOptimization.adaptive_rms_norm import rms_norm_inference
  for module in model.base_model.language_model.modules():
   if type(module).__name__=='LlamaRMSNorm':
    original_rms=module.forward
    model._metax_rms_restores.append((module,original_rms))
    def rms_forward(this,x,original_rms=original_rms):
     stats=model._metax_v2_stats
     if features & {'rms_native','rms_native_mean'}:
      supported=(x.device.type=='cuda' and x.dtype==this.weight.dtype==torch.bfloat16 and x.shape[-1]==4096 and x.is_contiguous() and this.weight.is_contiguous())
      if not supported:
       stats['rms_native_fallbacks']=stats.get('rms_native_fallbacks',0)+1
       return original_rms(x)
      from ports.metax.numerical_rms import rms_native
      stats['rms_native_hits']=stats.get('rms_native_hits',0)+1
      return rms_native(x,this.weight,this.variance_epsilon,native_inverse='rms_native' in features)
     stats['rms_hits']=stats.get('rms_hits',0)+1
     if 'rms_rsqrt' in features:
      from ports.metax.pointwise_v2 import rms_rsqrt
      return rms_rsqrt(x,this.weight,this.variance_epsilon)
     return rms_norm_inference(x,(x.shape[-1],),this.weight,this.variance_epsilon,round_before_weight=True)
    module.forward=types.MethodType(rms_forward,module)
 if 'rope' in features:
  from ports.metax.pointwise_v2 import rope
  from transformers.models.llama import modeling_llama as hf
  from openvla.specdecoding.model import cnets
  original_hf=hf.apply_rotary_pos_emb;original_draft=cnets.apply_rotary_pos_emb
  def hf_rope(q,k,cos,sin,position_ids=None,unsqueeze_dim=1):
   if q.dtype!=torch.bfloat16 or q.shape!=k.shape or q.shape[0]!=1 or q.shape[-1]!=128 or cos.dtype!=q.dtype or sin.dtype!=q.dtype or unsqueeze_dim!=1:return original_hf(q,k,cos,sin,position_ids,unsqueeze_dim)
   stats=model._metax_v2_stats;stats['verifier_rope_hits']=stats.get('verifier_rope_hits',0)+1
   return rope(q,k,cos,sin)
  def draft_rope(q,k,cos,sin,position_ids):
   if q.dtype!=torch.bfloat16 or q.shape!=k.shape or q.shape[0]!=1 or q.shape[-1]!=128 or cos.dtype!=q.dtype or sin.dtype!=q.dtype:return original_draft(q,k,cos,sin,position_ids)
   stats=model._metax_v2_stats;stats['draft_rope_hits']=stats.get('draft_rope_hits',0)+1
   return rope(q,k,cos,sin,position_ids)
  model._metax_global_restores=[(hf,'apply_rotary_pos_emb',original_hf),(cnets,'apply_rotary_pos_emb',original_draft)]
  hf.apply_rotary_pos_emb=hf_rope;cnets.apply_rotary_pos_emb=draft_rope

 for group in getattr(model,'_metax_v2_groups',[]):
  original_group=group.forward
  def recorded(index,value,original_group=original_group,group=group):
   if index==0 and value.numel()//value.shape[-1] in group.fused_rows:
    key=group.kind+'_hits';model._metax_v2_stats[key]=model._metax_v2_stats.get(key,0)+1
   return original_group(index,value)
  group.forward=recorded
 if 'vision_graph' in features:
  module=model.base_model.vision_backbone;original=module.forward
  state={}
  model._metax_vision_restore=(module,original,state)
  def forward(this,pixel):
   stats=model._metax_v2_stats
   if not state:
    static=torch.empty_like(pixel);static.copy_(pixel)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
     for _ in range(2):original(static)
    stream.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):result=original(static)
    torch.cuda.current_stream().wait_stream(stream)
    graph.replay();torch.cuda.synchronize()
    expected=original(pixel)
    if not torch.equal(result,expected):raise RuntimeError('Vision CUDA graph changed BF16 features')
    state.update(input=static,output=result,graph=graph,shape=tuple(pixel.shape))
    stats['vision_captures']=1
   if tuple(pixel.shape)!=state['shape']:raise ValueError('Vision graph shape mismatch')
   state['input'].copy_(pixel);state['graph'].replay()
   stats['vision_replays']=stats.get('vision_replays',0)+1
   return state['output']
  module.forward=types.MethodType(forward,module)
 if 'prefill_graph' in features:
  from ports.metax.prefill_graph import install as install_prefill
  install_prefill(model)
 print('[MetaX v2] enabled',sorted(features),flush=True)

def prepare_cpu(model):
 features=set(filter(None,os.environ.get('KERV_METAX_V2','').split(',')))
 groups=[]
 if features & {'qkv','gate_up'}:
  if os.environ.get('KERV_METAX_QUANT','none')!='none':raise ValueError('Projection fusion and storage compression require separate ownership validation')
  from KERVRuntimeOptimization.adaptive_linear_fusion import _replace_group
  for layer in model.base_model.language_model.model.layers:
   if 'qkv' in features:
    group=_replace_group(layer.self_attn,('q_proj','k_proj','v_proj'),'metax_qkv',range(256,321));groups.append(group)
   if 'gate_up' in features:
    group=_replace_group(layer.mlp,('gate_proj','up_proj'),'metax_gate_up',range(256,321));groups.append(group)
  model._metax_v2_groups=groups
 return groups


def release(model):
 from ports.metax.prefill_graph import release as release_prefill
 release_prefill(model)
 rms_restores=getattr(model,'_metax_rms_restores',[])
 for module,original in rms_restores:module.forward=original
 globals_restores=getattr(model,'_metax_global_restores',[])
 for module,name,original in globals_restores:setattr(module,name,original)
 model._metax_restore_audit=dict(rms_modules=len(rms_restores),rms_restored=all(module.forward==original for module,original in rms_restores),global_modules=len(globals_restores),globals_restored=all(getattr(module,name) is original for module,name,original in globals_restores))
 rms_restores.clear()
 if hasattr(model,'_metax_vision_restore'):
  module,original,state=model._metax_vision_restore;module.forward=original;state.clear()
  del model._metax_vision_restore
