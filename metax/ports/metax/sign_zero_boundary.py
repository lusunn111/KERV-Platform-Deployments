"""Localize the observed signed-zero boundary on a captured real RMS input."""
import json,torch
from pathlib import Path
r=Path('/workspace/outputs/metax_numerical_20261105');d=torch.load(r/'all_layer_diag02/first-tail-input.pt',weights_only=True)
x=d['x'].cuda();w=d['weight'].cuda();f=x.float();inv=torch.rsqrt(f.pow(2).mean(-1,keepdim=True)+d['eps']);normalized=(f*inv).to(x.dtype)
bf=w*normalized;fp=w.float()*normalized.float();cast=fp.to(torch.bfloat16)
indices=(bf.view(torch.int16).flatten()!=cast.view(torch.int16).flatten()).nonzero().flatten()[:12].cpu().tolist();examples=[]
for i in indices:
 examples.append(dict(index=i,x_bits=int(x.view(torch.int16).flatten()[i].item())&65535,normalized_bits=int(normalized.view(torch.int16).flatten()[i].item())&65535,weight_bits=int(w.view(torch.int16)[i%4096].item())&65535,bf16_mul_bits=int(bf.view(torch.int16).flatten()[i].item())&65535,fp32_mul_bits=int(fp.view(torch.int32).flatten()[i].item())&4294967295,post_cast_bits=int(cast.view(torch.int16).flatten()[i].item())&65535))
negzero=torch.tensor([-2147483648],device='cuda',dtype=torch.int32).view(torch.float32)
report=dict(torch=torch.__version__,examples=examples,constructed_negative_zero_to_bf16=int(negzero.to(torch.bfloat16).view(torch.int16).item())&65535)
(r/'signed-zero-boundary.json').write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True)
