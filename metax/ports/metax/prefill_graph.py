"""Opt-in stateless verifier prefill graph, independent of vision and tree verify.
Capture original BF16 math, compare every output tensor once, bound graph count.
"""
import types,torch
from openvla.prismatic.extern.hf.flagos_stage_graph import StageGraphEntry,_signature

def equal(a,b):
 if torch.is_tensor(a):return torch.is_tensor(b) and a.shape==b.shape and a.dtype==b.dtype and torch.equal(a,b)
 if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return a==b

def install(model):
 module=model.base_model.language_model;original=module.forward;entries={};disabled=set()
 model._metax_prefill_restore=(module,original,entries,disabled)
 def forward(this,*args,**kwargs):
  if args or kwargs.get('past_key_values') is not None or this.tree_mask is not None or kwargs.get('inputs_embeds') is None:
   return original(*args,**kwargs)
  key=_signature(kwargs);stats=model._metax_v2_stats
  if key in disabled or (key not in entries and len(entries)>=3):
   stats['prefill_fallbacks']=stats.get('prefill_fallbacks',0)+1
   return original(**kwargs)
  if key not in entries:
   expected=original(**kwargs)
   try:
    entry=StageGraphEntry.capture(original,kwargs)
    result=entry.replay(kwargs);torch.cuda.synchronize()
   except Exception as exc:
    disabled.add(key);stats.setdefault('prefill_errors',[]).append(repr(exc));stats['prefill_fallbacks']=stats.get('prefill_fallbacks',0)+1
    print('[MetaX prefill graph] capture failed',repr(exc),flush=True)
    return original(**kwargs)
   if not equal(result,expected):raise RuntimeError('Prefill graph changed logits/hidden states/KV tensors')
   entries[key]=entry;stats['prefill_captures']=stats.get('prefill_captures',0)+1
  result=entries[key].replay(kwargs);stats['prefill_replays']=stats.get('prefill_replays',0)+1
  return result
 module.forward=types.MethodType(forward,module)

def release(model):
 if hasattr(model,'_metax_prefill_restore'):
  module,original,entries,disabled=model._metax_prefill_restore
  module.forward=original;entries.clear();disabled.clear();del model._metax_prefill_restore
