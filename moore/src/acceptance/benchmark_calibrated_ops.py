"""Sequential MUSA microbenchmark; reference is the SAME decoded weight.

Repeated buffers are resident and hot. This does not measure end-to-end speed
or claim that the calibrated model passed quality acceptance.
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
from KERVRuntimeOptimization import calibrated_w8a16_ops as ops


def measure(fn, repeats, batch):
    for _ in range(4): fn()
    torch.musa.synchronize()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(batch): fn()
        torch.musa.synchronize()
        samples.append((time.perf_counter()-start)*1000/batch)
    return {'median_ms': statistics.median(samples), 'samples_ms': samples}


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--rows', nargs='+', type=int, default=[1, 4, 8, 49, 279, 303, 320])
    p.add_argument('--projections', nargs='+', default=['gate_proj', 'down_proj'])
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--batch', type=int, default=10)
    p.add_argument('--include-gemm', action='store_true', help='Include experimental dot path even if rejected')
    p.add_argument('--rows-decode-only', action='store_true')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(342)
    result = {'scope': 'hot resident microbenchmark; no model/SR claim',
              'reference': str(a.reference), 'baseline': 'existing decode+patch+native BF16 GEMM',
              'timing': 'wall clock, synchronized batch boundary; compilation and four warmups excluded',
              'repeats': a.repeats, 'batch': a.batch, 'cases': []}
    captures = torch.load(a.reference/'block_pilot_v2/activations.pt', weights_only=True)['inputs']['validation']
    for projection in a.projections:
        path = a.reference/'model_quant'/('layer15_'+projection+'.pt')
        data = torch.load(path, map_location='cpu', weights_only=True)
        module = CalibratedInt8Linear.from_payload(data).to('musa'); ops.prepare(module)
        w = module.dequantize(); decoded = ops.decode(module)
        assert torch.equal(w, decoded), 'Fused decode changed weights'
        for m in a.rows:
            x = (captures[:m].clone() if module.in_features == captures.shape[1]
                 else torch.randn(m, module.in_features, dtype=torch.bfloat16))
            x = x.to('musa')
            oracle = torch.nn.functional.linear(x, w, module.bias)
            candidates = [('original', lambda: module(x)),
                          ('native_gemm_only', lambda: torch.nn.functional.linear(x, w, module.bias)),
                          ('decode', lambda: ops.linear(module, x, kind='decode')),
                          ('groups_16', lambda: ops.linear(module, x, kind='decode_groups', config=(16,))),
                          ('groups_32', lambda: ops.linear(module, x, kind='decode_groups', config=(32,))),
                          ('groups_64', lambda: ops.linear(module, x, kind='decode_groups', config=(64,)))]
            if a.include_gemm:
                candidates += [('gemm_16_64_64', lambda: ops.linear(module, x, config=(16,64,64))),
                               ('gemm_32_64_128', lambda: ops.linear(module, x, config=(32,64,128)))]
            if m <= 8:
                candidates += [('gemv_4_1024', lambda: ops.linear(module, x, kind='gemv', config=(4,1024))),
                               ('gemv_8_512', lambda: ops.linear(module, x, kind='gemv', config=(8,512)))]
            if a.rows_decode_only:
                candidates = [('original',lambda:module(x)),('decode',lambda:ops.linear(module,x,kind='decode')),
                    ('groups_64',lambda:ops.linear(module,x,kind='decode_groups',config=(64,)))]
                for block in [512,1024,2048,4096]:
                    candidates.append(('rows_'+str(block), lambda block=block:ops.linear(module,x,kind='decode_rows',config=(block,))))
            case = {'projection': projection, 'rows': m, 'shape': list(w.shape),
                    'payload_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                    'decoded_weights_bit_exact': True, 'candidates': {}}
            # Alternate timing order by row count; retain all samples.
            if m % 2 == 0: candidates.reverse()
            for name, fn in candidates:
                value = fn(); torch.musa.synchronize()
                error = value.float()-oracle.float()
                item = measure(fn, a.repeats, a.batch)
                item.update(bit_exact=bool(torch.equal(value, oracle)),
                            relative_l2=float(error.norm()/oracle.float().norm().clamp_min(1e-20)),
                            max_abs=float(error.abs().max()), finite=bool(torch.isfinite(value).all()))
                item['numeric_gate_rel_l2_lt_0_002'] = item['finite'] and item['relative_l2'] < .002
                case['candidates'][name] = item
                print(projection, m, name, round(item['median_ms'],4),
                      'rel', round(item['relative_l2'],7), flush=True)
            original = case['candidates']['original']['median_ms']
            for item in case['candidates'].values(): item['speedup_vs_original'] = original/item['median_ms']
            result['cases'].append(case)
            (a.output/'results.json').write_text(json.dumps(result, indent=2)+'\n')
        del w, decoded, module
        torch.musa.empty_cache()


if __name__ == '__main__': main()
