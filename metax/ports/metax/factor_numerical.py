"""Controlled factor substitution on the first real discrepancy, not a broad proof."""
import json,torch
from pathlib import Path
from ports.metax.numerical_rms import stages,rms_native,_tail
from KERVRuntimeOptimization.adaptive_rms_norm import rms_norm_inference
from ports.metax.pointwise_v2 import rms_rsqrt
root=Path('/workspace/outputs/metax_numerical_20261105');rows=[]
def different(a,b):return int((a.contiguous().view(torch.int16)!=b.contiguous().view(torch.int16)).count_nonzero().item())
for name in ['first-real-divergence.pt','first-tail-input.pt']:
 data=torch.load(root/'all_layer_diag02'/name,weights_only=True);x=data['x'].cuda();w=data['weight'].cuda();eps=data['eps']
 with torch.inference_mode():
  f=x.float();v=f.pow(2).mean(-1,keepdim=True);inv=torch.rsqrt(v+eps)
  ref=w*(f*inv).to(x.dtype)
  s8=stages(x,w,eps,warps=8,mode=0);s4=stages(x,w,eps,warps=4,mode=1,fusion=False)
  def output(var,kind):
   inverse=torch.rsqrt(var+eps) if kind=='rsqrt' else torch.reciprocal(torch.sqrt(var+eps))
   return w*(f*inverse).to(x.dtype)
  variants={'only_inverse_formula':output(v,'reciprocal'),
   'only_reduce_8_warps':output(s8['mean'].reshape_as(v),'rsqrt'),
   'reduce8_and_inverse_formula':output(s8['mean'].reshape_as(v),'reciprocal'),
   'only_reduce_4_warps':output(s4['mean'].reshape_as(v),'rsqrt'),
   'old_actual':rms_norm_inference(x,(4096,),w,eps,True),
   'rsqrt_actual':rms_rsqrt(x,w,eps),
   'native_inverse_typed':rms_native(x,w,eps,native_inverse=True),
   'native_mean_typed':rms_native(x,w,eps,native_inverse=False),
   'native_inverse_promoted':rms_native(x,w,eps,native_inverse=True,bf16_multiply=False),
   'torch_fp32_weight_multiply':((f*inv).to(x.dtype).float()*w.float()).to(x.dtype)}
  result=dict(input=name,different={k:different(ref,y) for k,y in variants.items()},mean8_different=int((v.view(torch.int32).flatten()!=s8['mean'].view(torch.int32)).count_nonzero().item()),mean4_different=int((v.view(torch.int32).flatten()!=s4['mean'].view(torch.int32)).count_nonzero().item()),old_actual_vs_substituted=different(variants['old_actual'],variants['reduce8_and_inverse_formula']))
  rows.append(result);print(result,flush=True)
  if name=='first-tail-input.pt':
   for promote in [False,True]:
    y=torch.empty_like(x);kernel=_tail[(x.numel()//4096,)](x,w,v,inv,y,4096,float(eps),True,not promote,num_warps=4,enable_fp_fusion=False)
    for key,text in kernel.asm.items():
     if key in ['ttir','ttgir','llir','ptx','amdgcn'] and isinstance(text,str):(root/f'tail-{promote}.{key}').write_text(text)
(root/'root-factor-substitution.json').write_text(json.dumps(rows,indent=2)+'\n')
