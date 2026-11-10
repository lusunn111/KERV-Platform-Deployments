import json,torch
from ports.metax.compact_verify import compact_accept
rows=[]
for threshold in [None,0,3.49,3.99,15.0]:
 torch.manual_seed(21);L=torch.randn(1,301,32064,device='cuda',dtype=torch.float32)
 R=torch.randint(0,301,(49,8),device='cuda');R[0,-2:]=-1
 E=L[0,R];pred=E[:,:-1].argmax(-1)
 C=torch.cat([torch.full((49,1),31800,device='cuda'),pred+4],1);C[1,1:]=pred[1]+3;C[2]=C[1];R[2]=R[1]
 E=L[0,R];pred=E[:,:-1].argmax(-1)
 n=(abs(C[:,1:]-pred)<=(0 if threshold is None else threshold)).int().cumprod(1).sum(1)
 b,l,s,bonus=compact_accept(L,R,C,threshold)
 assert b.item()==n.argmax().item() and l.item()==n.max().item()
 assert torch.equal(s,E[b,l]) and bonus.item()==s.argmax().item()
 rows.append(dict(threshold=threshold,best=b.item(),length=l.item()))
print(json.dumps(dict(real_shape=[301,32064,49,8],exact=True,cases=rows),indent=2))
