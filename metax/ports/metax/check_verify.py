"""Synthetic boundary/tie tests at the real full-vocabulary shape, separate timing."""
import json
import time
import torch
from ports.metax.verify import verify_accept

torch.manual_seed(19)
results=[]
for threshold in [None,0,3.0,3.49,3.99,15.0]:
    logits=torch.randn(49,8,32064,device='cuda',dtype=torch.bfloat16)
    pred=logits[:,:-1].argmax(-1)
    candidate=torch.cat([torch.zeros((49,1),device='cuda',dtype=torch.long),pred+4],1)
    candidate[1,1:]=pred[1]+3
    candidate[2,1:]=pred[2]+3  # First of equally accepted paths must win.
    a,b=verify_accept(logits,candidate,threshold)
    lengths=(torch.abs(candidate[:,1:]-pred)<=(0 if threshold is None else threshold)).int().cumprod(1).sum(1)
    torch.testing.assert_close(a,lengths.argmax(),rtol=0,atol=0)
    torch.testing.assert_close(b,lengths.max(),rtol=0,atol=0)
    results.append(dict(threshold=threshold,best=a.item(),accepted=b.item()))
print(json.dumps(dict(shape=[49,8,32064],exact=True,tests=results),indent=2))
