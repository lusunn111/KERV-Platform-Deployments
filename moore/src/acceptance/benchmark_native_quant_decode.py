"""Compare exact native vector decode against the frozen Triton row decoder."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
import torch_musa
from KERVRuntimeOptimization.calibrated_int8 import CalibratedInt8Linear
from KERVRuntimeOptimization import calibrated_w8a16_ops as reference
from KERVRuntimeOptimization import calibrated_native_decode as candidate
from .benchmark_paired_quant_ops import timed


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--all-modules', action='store_true')
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=False)
    policy = json.loads(a.policy.read_text())
    result = {'policy_sha256':hashlib.sha256(a.policy.read_bytes()).hexdigest(),
              'build':candidate.load_library(a.output/'build'), 'modules':[],
              'scope':'decode only; frozen mixed policy; no task quality claim'}
    modules = []
    def save(): (a.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    for entry in policy['modules']:
        if not a.all_modules and '.layers.15.' not in entry['name']: continue
        path = a.policy.parent/entry['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256']
        q = CalibratedInt8Linear.from_payload(torch.load(path,weights_only=True)).to('musa')
        reference.prepare(q); modules.append(q)
        oracle = reference.decode_rows(q)
        row = {'name':entry['name'], 'baseline':timed(lambda:reference.decode_rows(q)),
               'candidates':[]}
        for threads in (64,128,256):
            for _ in range(3):
                assert torch.equal(candidate.decode(q,threads),oracle), (entry['name'],threads)
            item = {'threads':threads,'bit_exact':True,
                    **timed(lambda:candidate.decode(q,threads))}
            item['speedup'] = row['baseline']['median_ms']/item['median_ms']
            row['candidates'].append(item)
            print(entry['name'],threads,item['median_ms'],item['speedup'],flush=True)
        result['modules'].append(row);save()
    if a.all_modules:
        def rotation(fn):
            for q in modules: fn(q)
        baseline = timed(lambda:rotation(reference.decode_rows),rounds=5,batch=2)
        result['rotation'] = {'baseline':baseline,'candidates':[]}
        for threads in (64,128,256):
            item = {'threads':threads,**timed(lambda:rotation(
                lambda q:candidate.decode(q,threads)),rounds=5,batch=2)}
            item['speedup'] = baseline['median_ms']/item['median_ms']
            result['rotation']['candidates'].append(item)
            print('rotation',item,flush=True)
        save()


if __name__ == '__main__': main()
