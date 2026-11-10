"""Calibrate native W8A8 from the existing BF16 development trajectories."""
import argparse
import gc
import hashlib
import json
from pathlib import Path

from safetensors import safe_open
import torch
import torch_npu

from experiments.robot.ascend_quantization import weight_sha256
from experiments.robot.ascend_dynamic_quantization import NPUDynamicInt8Linear


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--fuse-gate-up', action='store_true')
    p.add_argument('--direct-mlp', action='store_true')
    p.add_argument('--fused-swiglu-quant', action='store_true')
    p.add_argument('--preserve-down', type=int, default=0,
                   help='Keep this many down projections in BF16, ranked by calibration error')
    a = p.parse_args()
    if (a.direct_mlp or a.fused_swiglu_quant) and not a.fuse_gate_up:
        p.error('Direct MLP and fused SwiGLU quantization require --fuse-gate-up')
    if not 0 <= a.preserve_down <= 18:
        p.error('--preserve-down must be between 0 and 18')
    a.output.mkdir(parents=True, exist_ok=False)
    inputs = torch.load(a.inputs, map_location='cpu', weights_only=True)
    index = json.loads((a.checkpoint / 'model.safetensors.index.json').read_text())['weight_map']
    modules = []
    records = []
    with torch.inference_mode():
        for name, cpu_x in sorted(inputs.items()):
            if a.fuse_gate_up and name.endswith('up_proj'):
                continue
            key = name.removeprefix('base_model.') + '.weight'
            with safe_open(a.checkpoint / index[key], framework='pt', device='cpu') as file:
                weight = file.get_tensor(key).bfloat16()
            identities = [(name, weight_sha256(weight))]
            if a.fuse_gate_up and name.endswith('gate_proj'):
                up_name = name.removesuffix('gate_proj') + 'up_proj'
                assert torch.equal(cpu_x, inputs[up_name]), 'Gate/up calibration input mismatch'
                up_key = up_name.removeprefix('base_model.') + '.weight'
                with safe_open(a.checkpoint / index[up_key], framework='pt', device='cpu') as file:
                    up_weight = file.get_tensor(up_key).bfloat16()
                identities.append((up_name, weight_sha256(up_weight)))
                weight = torch.cat([weight, up_weight], dim=0)
                del up_weight
            n, k = weight.shape
            linear = torch.nn.Linear(k, n, bias=False, dtype=torch.bfloat16)
            linear.weight.data = weight
            x = cpu_x.npu()
            reference = torch.nn.functional.linear(x, weight.npu())
            actmax = cpu_x.float().abs().amax(0).clamp_min(1e-5)
            wmax = weight.float().abs().amax(0).clamp_min(1e-5)
            candidates = []
            for alpha in [0.0, 0.25, 0.5, 0.75]:
                smooth = (wmax.pow(alpha) / actmax.pow(alpha)).clamp(1e-4, 1e4).bfloat16()
                quant = NPUDynamicInt8Linear(linear, smooth, {'forward_calls': 0}, name).npu()
                output = quant(x)
                assert bool(output.isfinite().all())
                error = float((output.float() - reference.float()).square().sum() / reference.float().square().sum())
                candidates.append(dict(alpha=alpha, relative_l2=error ** 0.5, smooth=smooth))
                del quant, output
            best = min(candidates, key=lambda entry: entry['relative_l2'])
            modules.extend(dict(name=member, weight_sha256=identity, **best) for member, identity in identities)
            records.append(dict(name=name, chosen_alpha=best['alpha'], relative_l2=best['relative_l2'],
                                candidates=[{k: v for k, v in entry.items() if k != 'smooth'} for entry in candidates]))
            (a.output / 'progress.json').write_text(json.dumps(records, indent=2) + '\n')
            print(name, 'alpha', best['alpha'], 'relative_l2', best['relative_l2'], flush=True)
            del linear, weight, x, reference, candidates, actmax, wmax
            gc.collect()
            torch.npu.empty_cache()
    preserved = {r['name'] for r in sorted([r for r in records if r['name'].endswith('down_proj')],
                                         key=lambda r:-r['relative_l2'])[:a.preserve_down]}
    for item in modules:
        if item['name'] in preserved:
            item['precision'] = 'bf16'
    torch.save(dict(version=1, implementation='ascend_dynamic_w8a8', modules=modules, fused_gate_up=a.fuse_gate_up,
                    direct_mlp=a.direct_mlp or a.fused_swiglu_quant, fused_swiglu_quant=a.fused_swiglu_quant,
                    bf16_selection=dict(criterion='largest down projection relative L2 on init0 calibration', names=sorted(preserved)),
                    input_sha256=hashlib.sha256(a.inputs.read_bytes()).hexdigest()), a.output / 'policy.pt')
    print('COMPLETE', len(modules), flush=True)


if __name__ == '__main__':
    main()
