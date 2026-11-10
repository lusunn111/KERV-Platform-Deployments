"""Rotate all policy projections to avoid a one-weight hot-cache benchmark.

The multi-GB working set is recorded, not claimed to force every cache miss.
This remains a Linear-only benchmark, separate from full-model results.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
import torch
import torch_musa
from KERVRuntimeOptimization.calibrated_int8 import CalibratedInt8Linear
from KERVRuntimeOptimization import calibrated_int8_kernels as old
from KERVRuntimeOptimization import calibrated_w8a16_ops as new


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy',type=Path,required=True)
    p.add_argument('--activations',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    policy=json.loads(a.policy.read_text());modules=[];torch.manual_seed(351)
    for entry in policy['modules']:
        path=a.policy.parent/entry['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']
        q=CalibratedInt8Linear.from_payload(torch.load(path,map_location='cpu',weights_only=True)).to('musa')
        new.prepare(q);assert torch.equal(old.decode(q),new.decode_rows(q))
        modules.append((entry['name'],q))
    print('LOADED',len(modules),flush=True)
    captured=torch.load(a.activations,map_location='cpu',weights_only=True)['inputs']['validation']
    working_set=sum(q.storage_bytes for _,q in modules)
    report={'scope':'resident all75 projections rotated, fixed activation matrix per feature size; no SR/full-model claim',
            'working_set_bytes':working_set,'module_count':len(modules),'cases':[]}
    for m in [1,279,303]:
        xs={4096:captured[:m].to('musa'),11008:torch.randn(m,11008,dtype=torch.bfloat16,device='musa')}
        def one_pass(mode):
            for _,q in modules:
                w=old.decode(q) if mode=='reference' else new.decode_rows(q,1024)
                torch.nn.functional.linear(xs[q.in_features],w,q.bias)
        for mode in ['reference','rows']:one_pass(mode)
        torch.musa.synchronize();samples={l:[] for l in ['reference','rows']}
        for repeat in range(8):
            order=['reference','rows'] if repeat%2==0 else ['rows','reference']
            for mode in order:
                start=time.perf_counter();one_pass(mode);torch.musa.synchronize()
                samples[mode].append((time.perf_counter()-start)*1000)
        result={'rows':m,'samples_75projection_pass_ms':samples,
                'median_ms':{k:statistics.median(v) for k,v in samples.items()}}
        result['speedup']=result['median_ms']['reference']/result['median_ms']['rows']
        report['cases'].append(result)
        (a.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
        print(m,result,flush=True)


if __name__=='__main__':main()
