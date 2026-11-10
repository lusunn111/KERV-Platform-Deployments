"""Benchmark tiled decode and paired gate/up using the frozen real payloads."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch
import torch_musa
from KERVRuntimeOptimization.calibrated_int8 import CalibratedInt8Linear
from KERVRuntimeOptimization import calibrated_w8a16_ops as old
from KERVRuntimeOptimization import calibrated_paired_ops as new


def timed(fn, rounds=5, batch=12):
    for _ in range(4): fn()
    torch.musa.synchronize()
    samples = []
    for _ in range(rounds):
        start = time.perf_counter()
        for _ in range(batch): fn()
        torch.musa.synchronize()
        samples.append((time.perf_counter()-start)*1000/batch)
    return {'median_ms':statistics.median(samples), 'samples_ms':samples}


def errors(value, reference):
    d = value.float()-reference.float()
    return {'bit_exact':torch.equal(value, reference),
            'max_abs':float(d.abs().max()),
            'relative_l2':float(d.norm()/reference.float().norm().clamp_min(1e-20)),
            'finite':bool(value.isfinite().all())}


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy', type=Path, required=True)
    p.add_argument('--activations', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--rows', type=int, nargs='+', default=[1, 8, 279, 303])
    p.add_argument('--layer', type=int, default=15)
    p.add_argument('--pair-rows-only', action='store_true')
    a = p.parse_args();a.output.mkdir(parents=True, exist_ok=False)
    policy = json.loads(a.policy.read_text());modules = {}
    for projection in ['gate_proj', 'up_proj', 'down_proj']:
        entry = next(e for e in policy['modules']
            if e['name'].endswith(f'.layers.{a.layer}.mlp.{projection}'))
        path = a.policy.parent/entry['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256']
        q = CalibratedInt8Linear.from_payload(torch.load(path, weights_only=True)).to('musa')
        old.prepare(q);modules[projection] = q
    captures = torch.load(a.activations, weights_only=True)['inputs']['validation']
    torch.manual_seed(387)
    result = {'scope':'real frozen layer payloads, hot resident microbenchmark; no model/quality claim',
              'policy_sha256':hashlib.sha256(a.policy.read_bytes()).hexdigest(),
              'decode':[], 'paired':[]}
    configs = [(1,1024,4),(2,512,4),(4,256,4),(4,512,4),(8,256,4),(8,512,4),(16,256,4)]
    def save(): (a.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    for name,q in ([] if a.pair_rows_only else modules.items()):
        oracle = old.decode_rows(q)
        baseline = timed(lambda:old.decode_rows(q))
        row = {'projection':name, 'baseline':baseline, 'candidates':[]}
        for config in configs:
            actual = new.decode_tiled(q,config)
            assert torch.equal(actual,oracle), (name,config,'decoded weights changed')
            item = {'config':config, 'bit_exact_weights':True,
                    **timed(lambda:new.decode_tiled(q,config))}
            item['speedup'] = baseline['median_ms']/item['median_ms']
            row['candidates'].append(item)
            print('decode',name,config,round(item['median_ms'],4),round(item['speedup'],3),flush=True)
        result['decode'].append(row);save()
    gate,up = modules['gate_proj'],modules['up_proj']
    wg,wu = old.decode_rows(gate),old.decode_rows(up)
    pair_configs = ([512,1024,2048,4096] if a.pair_rows_only else
        sorted(configs,key=lambda cfg:next(i['median_ms'] for i in
        result['decode'][0]['candidates'] if tuple(i['config'])==cfg))[:3])
    for config in pair_configs:
        weight = (new.decode_pair_rows(gate,up,config) if a.pair_rows_only
                  else new.decode_pair(gate,up,config))
        assert torch.equal(weight[:gate.out_features],wg)
        assert torch.equal(weight[gate.out_features:],wu)
    for m in a.rows:
        assert captures.shape[0] >= m
        x = captures[:m].to('musa')
        reference_g,reference_u = torch.nn.functional.linear(x,wg),torch.nn.functional.linear(x,wu)
        reference = torch.nn.functional.silu(reference_g)*reference_u
        def baseline():
            g = torch.nn.functional.linear(x,old.decode_rows(gate))
            u = torch.nn.functional.linear(x,old.decode_rows(up))
            return torch.nn.functional.silu(g)*u
        row = {'rows':m, 'baseline':timed(baseline), 'candidates':[]}
        for config in pair_configs:
            def candidate():
                g,u = (new.paired_linear(gate,up,x,row_block=config) if a.pair_rows_only
                       else new.paired_linear(gate,up,x,config))
                return torch.nn.functional.silu(g)*u
            actual = candidate()
            g,u = (new.paired_linear(gate,up,x,row_block=config) if a.pair_rows_only
                   else new.paired_linear(gate,up,x,config))
            item = {'config':config, **errors(actual,reference),
                    'gate':errors(g,reference_g), 'up':errors(u,reference_u),
                    **timed(candidate)}
            item['speedup'] = row['baseline']['median_ms']/item['median_ms']
            row['candidates'].append(item)
            print('paired',m,config,round(item['median_ms'],4),round(item['speedup'],3),
                  'rel_l2',item['relative_l2'],'exact',item['bit_exact'],flush=True)
        result['paired'].append(row);save()


if __name__ == '__main__': main()
