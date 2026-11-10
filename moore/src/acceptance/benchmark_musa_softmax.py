"""Reproduce native Softmax corruption from saved same-input diagnostic tensors."""
import argparse
import json
from pathlib import Path
import torch
import torch_musa
from KERVRuntimeOptimization.musa_safe_softmax import softmax, guarded_softmax


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repeats',type=int,default=1000)
    p.add_argument('--poison-reuse',action='store_true',help='Fill and free an equal-sized buffer before each Softmax')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    data=torch.load(a.input,weights_only=False)
    xcpu=data['old'][0][1].float()
    assert 'softmax.input' in data['old'][0][0]
    assert torch.equal(xcpu,data['new'][0][1])
    reference=xcpu.softmax(-1)
    x=xcpu.to('musa');masked=x< -1e30
    repairs=torch.zeros((),device=x.device,dtype=torch.int32)
    report={'input':str(a.input),'shape':list(x.shape),'repeats':a.repeats,'poison_reuse':a.poison_reuse,'backends':{}}
    for label,fn in [('native',lambda v:torch.nn.functional.softmax(v,dim=-1,dtype=torch.float32)),
                     ('triton',lambda v:softmax(v,dim=-1,dtype=torch.float32)),
                     ('guarded',lambda v:guarded_softmax(v,dim=-1,dtype=torch.float32,repaired_rows=repairs))]:
        failures=[];max_error=0.;drift=0;old=None
        for i in range(a.repeats):
            if a.poison_reuse:
                poison=torch.full_like(x,.5)
                del poison
            y=fn(x)
            # One synchronization after a batch of device-side invariant checks.
            invalid=(y.masked_select(masked)!=0).any() | ((y.sum(-1)-1).abs()>1e-5).any()
            if bool(invalid.cpu()):
                cpu=y.cpu();failures.append(i)
                if len(failures)==1:torch.save({'input':xcpu,'output':cpu,'reference':reference},a.output/(label+'_failure.pt'))
            if i<3 or failures and failures[-1]==i:
                cpu=y.cpu();max_error=max(max_error,float((cpu-reference).abs().max()))
            if old is not None:drift+=int(not torch.equal(y,old))
            old=y
        report['backends'][label]={'invalid_iterations':failures,'nonrepeatable_iterations':drift,'max_abs_error_checked':max_error}
        print(label,report['backends'][label],flush=True)
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    report['guarded_repaired_rows']=int(repairs.cpu())
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
