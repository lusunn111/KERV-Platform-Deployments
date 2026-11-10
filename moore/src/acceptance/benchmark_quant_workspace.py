"""All 75 frozen projections: allocation versus one stream-local workspace."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
import torch
import torch_musa
from KERVRuntimeOptimization.calibrated_int8 import CalibratedInt8Linear
from KERVRuntimeOptimization import calibrated_w8a16_ops as ops
from KERVRuntimeOptimization.calibrated_prefetch import decode_into
from KERVRuntimeOptimization import calibrated_native_decode as native


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    policy=json.loads(a.policy.read_text());named=[]
    for entry in policy['modules']:
        path=a.policy.parent/entry['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']
        q=CalibratedInt8Linear.from_payload(torch.load(path,weights_only=True)).to('musa')
        ops.prepare(q);named.append((entry['name'],q))
    pool=torch.empty(max(q.qweight.numel() for _,q in named),device='musa',dtype=torch.bfloat16)
    views=[pool[:q.qweight.numel()].view(q.qweight.shape) for _,q in named]
    report={'scope':'75 real frozen projections; fixed activations; no end-to-end claim',
            'workspace_bytes':pool.numel()*2,'build':native.load_library(a.output/'build'),'cases':[]}
    modes=['rows','workspace','native_workspace']
    torch.manual_seed(353)
    for m in [1,279,303]:
        xs={k:torch.randn(m,k,device='musa',dtype=torch.bfloat16) for k in [4096,11008]}
        def run(mode,retain=False):
            outputs=[]
            for i,(_,q) in enumerate(named):
                if mode=='rows': w=ops.decode_rows(q)
                else:
                    w=views[i]
                    if mode=='workspace':decode_into(q,w)
                    else:native.decode_into(q,w,64 if q.in_features==4096 else 256)
                y=torch.nn.functional.linear(xs[q.in_features],w,q.bias)
                if retain:outputs.append(y)
            return outputs
        reference=run('rows',True)
        for mode in modes[1:]:
            actual=run(mode,True)
            assert all(torch.equal(x,y) for x,y in zip(reference,actual)),mode
        del reference,actual
        for mode in modes:run(mode)
        torch.musa.synchronize();samples={mode:[] for mode in modes}
        for i in range(9):
            order=modes[i%3:]+modes[:i%3]
            for mode in order:
                start=time.perf_counter();run(mode);torch.musa.synchronize()
                samples[mode].append((time.perf_counter()-start)*1000)
        median={k:statistics.median(v) for k,v in samples.items()}
        item={'rows':m,'outputs_bit_exact':True,'samples_ms':samples,'median_ms':median,
              'speedup':{k:median['rows']/v for k,v in median.items()}}
        report['cases'].append(item)
        (a.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
        print(item,flush=True)


if __name__=='__main__':main()
