"""Real all-layer diagnostics, native trajectory retained; never a speed run."""
import functools,hashlib,inspect,json,sys
from pathlib import Path
import torch
from transformers.models.llama import modeling_llama as hf
from openvla.specdecoding.model import cnets
from KERVRuntimeOptimization.adaptive_rms_norm import rms_norm_inference
from ports.metax.pointwise_v2 import rms_rsqrt
from ports.metax.numerical_rms import rms_native,stages,inverse_stages
from ports.metax import benchmark

root=Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=False)
input_dir=Path('/workspace/outputs/metax_v2_20261104/representative_capture/observations')
frozen=[json.loads(x) for x in (input_dir.parent/'raw/steps.jsonl').read_text().splitlines()]
ctx={'active':False,'step':-1};records=[];state_rows=[];first=[];norm_calls=0
selected=set(map(int,sys.argv[2].split(','))) if len(sys.argv)>2 else set(range(8))|{31,63,127,191,219}
native_hf=hf.apply_rotary_pos_emb;native_draft=cnets.apply_rotary_pos_emb
load_original=benchmark.get_vla;action_original=benchmark.get_vla_action

def bits(x):return x.contiguous().view(torch.int16 if x.dtype==torch.bfloat16 else torch.int32).reshape(-1)
def count(a,b):return (bits(a)!=bits(b)).count_nonzero()
def equal(a,b):return bool(torch.equal(bits(a),bits(b)))
def native_stages(x,weight,eps):
 def op(label,fn):
  with torch.profiler.record_function('NumericalNative::'+label):return fn()
 f=op('float',lambda:x.float());square=op('square',lambda:f.pow(2))
 mean=op('mean',lambda:square.mean(-1,keepdim=True));added=op('epsilon',lambda:mean+eps)
 inv=op('rsqrt',lambda:torch.rsqrt(added));norm=op('normalize',lambda:f*inv)
 cast=op('bf16',lambda:norm.to(x.dtype));y=op('weight',lambda:weight*cast)
 return dict(square=square,mean=mean,added=added,sqrt=torch.sqrt(added),reciprocal=torch.reciprocal(torch.sqrt(added)),rsqrt=inv,normalized=norm,bf16=cast,output=y)
def audit_norm(name,module,original,x):
 global norm_calls
 out=original(x)
 if not ctx['active']:return out
 norm_calls+=1
 if norm_calls==1:
  with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as prof:
   ref=native_stages(x,module.weight,module.variance_epsilon);torch.cuda.synchronize()
  prof.export_chrome_trace(str(root/'native-rms-trace.json'))
  (root/'native-rms-kernels.json').write_text(json.dumps([dict(name=i.key,count=i.count) for i in prof.key_averages()],indent=2)+'\n')
 else:ref=native_stages(x,module.weight,module.variance_epsilon)
 assert equal(out,ref['output']),f'Explicit native stages failed to reproduce actual {name}'
 variants={'old_fma':stages(x,module.weight,module.variance_epsilon,mode=0,warps=8,fusion=True),
           'old_no_fma':stages(x,module.weight,module.variance_epsilon,mode=0,warps=8,fusion=False),
           'rsqrt_no_fma':stages(x,module.weight,module.variance_epsilon,mode=1,warps=4,fusion=False)}
 actual={'old_actual':rms_norm_inference(x,(4096,),module.weight,module.variance_epsilon,True),
         'rsqrt_actual':rms_rsqrt(x,module.weight,module.variance_epsilon),
         'promoted_native_inv_tail':rms_native(x,module.weight,module.variance_epsilon,native_inverse=True,bf16_multiply=False),
         'native_inv_tail':rms_native(x,module.weight,module.variance_epsilon,native_inverse=True),
         'native_mean_tail':rms_native(x,module.weight,module.variance_epsilon,native_inverse=False)}
 inv=inverse_stages(ref['mean'],module.variance_epsilon)
 targets={};maxes={}
 for key,value in inv.items():
  targets['native_mean_inverse/'+key]=count(ref[key],value)
  maxes['native_mean_inverse/'+key]=(ref[key]-value).abs().max()
 for mode,values in variants.items():
  for key,value in values.items():
   targets[mode+'/'+key]=count(ref[key],value);maxes[mode+'/'+key]=(ref[key].float().reshape(-1)-value.float().reshape(-1)).abs().max()
 for key,value in actual.items():
  targets[key]=count(out,value);maxes[key]=(out.float()-value.float()).abs().max()
 targets['old_actual_vs_instrumented']=count(actual['old_actual'],variants['old_fma']['output'])
 targets['rsqrt_actual_vs_instrumented']=count(actual['rsqrt_actual'],variants['rsqrt_no_fma']['output'])
 for key in variants['old_fma']:
  targets['fma_stage/'+key]=count(variants['old_fma'][key],variants['old_no_fma'][key])
 targets['fma_output_difference']=count(variants['old_fma']['output'],variants['old_no_fma']['output'])
 labels=list(targets);counts=torch.stack([targets[k] for k in labels]).cpu().tolist()
 max_labels=list(maxes);max_values=torch.stack([maxes[k] for k in max_labels]).cpu().tolist()
 row=dict(step=ctx['step'],call=norm_calls,layer=name,shape=list(x.shape),stride=list(x.stride()),dtype=str(x.dtype),eps=module.variance_epsilon,different=dict(zip(labels,counts)),max_abs=dict(zip(max_labels,max_values)))
 records.append(row)
 if row['different']['promoted_native_inv_tail'] and not (root/'first-tail-divergence.json').exists():
  idx=(bits(out)!=bits(actual['promoted_native_inv_tail'])).nonzero().flatten()[:12]
  examples=[]
  for at in idx.cpu().tolist():
   examples.append(dict(index=at,x=float(x.reshape(-1)[at]),weight=float(module.weight[at%4096]),native_bf16_bits=int(bits(out)[at].item())&65535,promoted_bf16_bits=int(bits(actual['promoted_native_inv_tail'])[at].item())&65535,typed_bf16_bits=int(bits(actual['native_inv_tail'])[at].item())&65535))
  (root/'first-tail-divergence.json').write_text(json.dumps(dict(step=ctx['step'],layer=name,shape=list(x.shape),examples=examples),indent=2)+'\n')
  torch.save(dict(x=x.detach().cpu(),weight=module.weight.detach().cpu(),eps=module.variance_epsilon),root/'first-tail-input.pt')
 if not first and row['different']['old_actual']:
  where=(bits(out)!=bits(actual['old_actual'])).nonzero()[0].item();flat=out.reshape(-1)
  first.append(dict(**row,first_flat_index=where,reference=float(flat[where]),old=float(actual['old_actual'].reshape(-1)[where])))
  torch.save(dict(x=x.detach().cpu(),weight=module.weight.detach().cpu(),eps=module.variance_epsilon),root/'first-real-divergence.pt')
  (root/'first-divergence.json').write_text(json.dumps(first[0],indent=2)+'\n')
 return out

def load(cfg):
 assert hf.apply_rotary_pos_emb is native_hf and cnets.apply_rotary_pos_emb is native_draft,'Global patch leaked from previous model'
 model=load_original(cfg);model._numerical_calls=0;names=[]
 for name,module in model.base_model.language_model.named_modules():
  if isinstance(module,hf.LlamaRMSNorm):
   assert module.forward.__func__ is hf.LlamaRMSNorm.forward,('Unexpected RMS patch',name)
   original=module.forward;names.append(name)
   def forward(x,name=name,module=module,original=original):return audit_norm(name,module,original,x)
   module.forward=forward
 assert len(names)==65,len(names)
 # Capture native drafter path as a separate semantic contract, never apply HF-tail to it.
 draft_names=[]
 for name,module in model.ea_layer.named_modules():
  if isinstance(module,cnets.LlamaRMSNorm):
   original=module.forward;draft_names.append(name)
   def draft_forward(x,name=name,module=module,original=original):
    result=original(x)
    if ctx['active']:
     expected=torch.nn.functional.rms_norm(x,(x.shape[-1],),module.weight,module.variance_epsilon)
     state_rows.append(dict(step=ctx['step'],kind='draft_rms',layer=name,shape=list(x.shape),different=int(count(result,expected).item())))
    return result
   module.forward=draft_forward
 patched_hf=hf.apply_rotary_pos_emb;patched_draft=cnets.apply_rotary_pos_emb
 def rotary(kind,patched,reference,*args,**kwargs):
  result=patched(*args,**kwargs)
  if ctx['active']:
   expected=reference(*args,**kwargs)
   state_rows.append(dict(step=ctx['step'],kind=kind,shape=list(args[0].shape),different=[int(count(a,b).item()) for a,b in zip(result,expected)],strides_equal=[a.stride()==b.stride() for a,b in zip(result,expected)],actual_strides=[list(a.stride()) for a in result],native_strides=[list(a.stride()) for a in expected]))
  return result
 hf.apply_rotary_pos_emb=functools.partial(rotary,'verifier_rope',patched_hf,native_hf)
 cnets.apply_rotary_pos_emb=functools.partial(rotary,'draft_rope',patched_draft,native_draft)
 vision=model.base_model.vision_backbone;patched_vision=vision.forward;native_vision=model._metax_vision_restore[1]
 def vision_forward(pixel):
  result=patched_vision(pixel)
  if ctx['active']:
   snapshot=result.clone();expected=native_vision(pixel)
   state_rows.append(dict(step=ctx['step'],kind='vision',different=int(count(snapshot,expected).item()),shape=list(result.shape)))
  return result
 vision.forward=vision_forward
 (root/'paths.json').write_text(json.dumps(dict(torch=torch.__version__,maca=torch.version.maca,native_rms_file=inspect.getfile(hf.LlamaRMSNorm),native_rms_source=inspect.getsource(hf.LlamaRMSNorm),drafter_rms_source=inspect.getsource(cnets.LlamaRMSNorm),layers=names,drafter_layers=draft_names,selected_steps=sorted(selected)),indent=2)+'\n')
 return model

def action(model,*args,**kwargs):
 i=model._numerical_calls;model._numerical_calls+=1;ctx.update(step=i%220,active=i in selected)
 result=action_original(model,*args,**kwargs)
 assert model.get_history()[-1]==frozen[i%220]['action_tokens'],f'Diagnostic native trajectory changed at {i}'
 if ctx['active']:
  (root/'rms-stages.json').write_text(json.dumps(dict(diagnostic_only=True,samples=records),indent=2)+'\n')
  (root/'state-checks.json').write_text(json.dumps(state_rows,indent=2)+'\n')
  print('all-layer diagnostic step',i,'norm calls',norm_calls,flush=True)
 ctx['active']=False
 return result
benchmark.get_vla=load;benchmark.get_vla_action=action
sys.argv=['benchmark.py','--comparison','numerical_diagnostic','--v2-features','rope,vision_graph','--frozen-history','--samples','220','--repeats','1','--input-dir',str(input_dir),'--output',str(root/'diagnostic-replay')]
benchmark.main()
assert hf.apply_rotary_pos_emb is native_hf and cnets.apply_rotary_pos_emb is native_draft,'Restore failed'
(root/'completed.json').write_text(json.dumps(dict(diagnostic_only=True,normalization_calls=norm_calls,global_restore_exact=True,first_divergence=first[0] if first else None),indent=2)+'\n')
