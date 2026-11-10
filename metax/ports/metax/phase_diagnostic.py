"""Separate asynchronous phase diagnosis; never use these timings as speed scores.
Event intervals include dispatch gaps; nested intervals must not be added together.
"""
import json,sys,time,functools
from pathlib import Path
import torch
from ports.metax import benchmark
from openvla.prismatic.extern.hf import modeling_speculation as ms,utils
output=Path(sys.argv[1]);features=sys.argv[2];output.mkdir(parents=True,exist_ok=False)
original_load=benchmark.get_vla;original_action=benchmark.get_vla_action
rows=[];pending=[];state={'active':False};restores=[]
def wrap(owner,name,label):
 original=getattr(owner,name)
 @functools.wraps(original)
 def call(*args,**kwargs):
  if not state['active']:return original(*args,**kwargs)
  a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
  a.record();t=time.perf_counter();result=original(*args,**kwargs);wall=time.perf_counter()-t;b.record()
  shape=next((list(x.shape) for x in args if isinstance(x,torch.Tensor)),None)
  pending.append(dict(label=label,shape=shape,cpu_enqueue_ms=wall*1000,start=a,end=b))
  return result
 setattr(owner,name,call);restores.append((owner,name,original))
def load(cfg):
 # Previous model wrappers must be restored before installing this model's v2 globals.
 for owner,name,original in restores:setattr(owner,name,original)
 restores.clear()
 model=original_load(cfg)
 wrap(model.base_model.vision_backbone,'forward','vision')
 wrap(model.base_model.language_model,'forward','verifier_all')
 wrap(model.ea_layer,'forward','drafter_forward')
 wrap(ms,'evaluate_posterior','posterior')
 wrap(ms,'update_inference_inputs','kv_commit_and_next_draft_nested')
 wrap(utils,'_commit_past_key_values','kv_copy_only')
 model._metax_diag_action=0
 return model
def action(model,*args,**kwargs):
 step=model._metax_diag_action;model._metax_diag_action+=1
 state['active']=step in {8,11,15};pending.clear()
 result=original_action(model,*args,**kwargs)
 if state['active']:
  torch.cuda.synchronize()
  for item in pending:
   a=item.pop('start');b=item.pop('end');item.update(event_interval_ms=a.elapsed_time(b),profile='v2' if getattr(model,'_metax_v2_features',[]) else 'runtime',step=step%8)
   rows.append(item)
  (output/'phases.json').write_text(json.dumps(dict(diagnostic_only=True,note='Nested GPU event spans and CPU enqueue times overlap; do not sum or use as throughput',samples=rows),indent=2)+'\n')
 state['active']=False
 return result
benchmark.get_vla=load;benchmark.get_vla_action=action
sys.argv=['benchmark.py','--comparison','runtime_v2','--v2-features',features,'--allow-action-change','--samples','8','--repeats','1','--input-dir','/workspace/outputs/metax_optimized_smoke_01/observations','--output',str(output/'replay')]
benchmark.main()
for owner,name,original in restores:setattr(owner,name,original)
