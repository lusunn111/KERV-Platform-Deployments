"""Compare row decode and two-slot prefetch on all actual quantized weights."""
import argparse
import json
from pathlib import Path
import statistics
import time
import torch
import torch_musa
from KERVRuntimeOptimization.calibrated_int8 import CalibratedInt8Linear
from KERVRuntimeOptimization.calibrated_prefetch import OrderedWeightPrefetch
from KERVRuntimeOptimization import calibrated_w8a16_ops as ops
from experiments.robot.libero.run_calibrated_libero import digest_tensor

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    policy=json.loads(a.policy.read_text());named=[]
    for item in policy['modules']:
        q=CalibratedInt8Linear.from_payload(torch.load(a.policy.parent/item['file'],weights_only=True)).to('musa')
        named.append((item['name'],q))
    pipeline=OrderedWeightPrefetch(named);proof=pipeline.audit(digest_tensor)
    assert {x['name']:x['decoded_sha256'] for x in proof}=={x['name']:x['decoded_weight_sha256'] for x in policy['modules']}
    report={'scope':'75 real projections, resident buffers, fixed random activations; no full-model claim',
        'workspace_bytes':pipeline.workspace_bytes,'all_decoded_weights_exact':True,'cases':[]}
    torch.manual_seed(353)
    for m in [1,279,303]:
        xs={k:torch.randn(m,k,device='musa',dtype=torch.bfloat16) for k in [4096,11008]}
        def run(mode,retain=False):
            if mode=='prefetch':pipeline.begin()
            ys=[]
            for i,(_,q) in enumerate(named):
                x=xs[q.in_features]
                y=(pipeline.linear(i,x) if mode=='prefetch' else torch.nn.functional.linear(x,ops.decode_rows(q),q.bias))
                if retain:ys.append(y)
            if mode=='prefetch':pipeline.end()
            return ys
        left=run('rows',True);right=run('prefetch',True)
        assert all(torch.equal(x,y) for x,y in zip(left,right))
        del left,right
        torch.musa.synchronize();samples={'rows':[],'prefetch':[]}
        for i in range(8):
            for mode in (['rows','prefetch'] if i%2==0 else ['prefetch','rows']):
                start=time.perf_counter();run(mode);torch.musa.synchronize()
                samples[mode].append((time.perf_counter()-start)*1000)
        median={k:statistics.median(v) for k,v in samples.items()}
        result={'rows':m,'samples_ms':samples,'median_ms':median,'speedup':median['rows']/median['prefetch'],'outputs_bit_exact':True}
        report['cases'].append(result);(a.output/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(result,flush=True)

if __name__=='__main__':main()
