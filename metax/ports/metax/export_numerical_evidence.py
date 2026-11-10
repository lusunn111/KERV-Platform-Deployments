"""Persist complete intermediate tensors for the first real RMS discrepancy."""
import hashlib,json,torch
from pathlib import Path
from ports.metax.numerical_rms import stages,rms_native
r=Path('/workspace/outputs/metax_numerical_20261105');d=torch.load(r/'all_layer_diag02/first-real-divergence.pt',weights_only=True)
x=d['x'].cuda();w=d['weight'].cuda();eps=d['eps']
with torch.inference_mode():
 f=x.float();square=f.pow(2);mean=square.mean(-1,keepdim=True);added=mean+eps;inv=torch.rsqrt(added);normalized=f*inv;bf16=normalized.to(x.dtype)
 native=dict(square=square,mean=mean,added=added,sqrt=torch.sqrt(added),reciprocal=torch.reciprocal(torch.sqrt(added)),rsqrt=inv,normalized=normalized,bf16=bf16,output=w*bf16)
 old=stages(x,w,eps,mode=0,warps=8,fusion=True);rsqrt=stages(x,w,eps,mode=1,warps=4,fusion=False)
 fixed=rms_native(x,w,eps,native_inverse=False)
 saved={name:{k:v.detach().cpu() for k,v in values.items()} for name,values in [('native',native),('old_8warp_reciprocal',old),('rsqrt_4warp',rsqrt)]}
 saved.update(x=x.cpu(),weight=w.cpu(),eps=eps,fixed_output=fixed.cpu())
 out=r/'first-rms-stage-tensors.pt';torch.save(saved,out)
 examples=[]
 for row,col in [(22,80),(0,80)]:
  entry=dict(row=row,column=col)
  for name,values in [('native',native),('old_8warp_reciprocal',old),('rsqrt_4warp',rsqrt)]:
   entry[name]={}
   for key,v in values.items():
    at=row if key in {'mean','added','sqrt','reciprocal','rsqrt'} else row*4096+col
    item=v.reshape(-1)[at];integer=item.view(torch.int16 if v.dtype==torch.bfloat16 else torch.int32).item()
    entry[name][key]=dict(value=float(item),bits=integer&((1<<(16 if v.dtype==torch.bfloat16 else 32))-1))
  examples.append(entry)
 report=dict(torch=torch.__version__,maca=torch.version.maca,input_shape=list(x.shape),complete_stage_tensor_file=str(out),sha256=hashlib.sha256(out.read_bytes()).hexdigest(),examples=examples)
 (r/'first-rms-stage-values.json').write_text(json.dumps(report,indent=2)+'\n')
 print(out,out.stat().st_size,flush=True)
