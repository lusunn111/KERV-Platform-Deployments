"""Screen native SwiGLU+quant fusion using captured real layer-7 tensors."""
import argparse
import json
from pathlib import Path
import statistics
import time

import torch
import torch_npu
from experiments.robot.ascend_dynamic_quantization import NPUDynamicInt8Linear


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--policy', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    data = torch.load(a.inputs, weights_only=False, map_location='cpu')
    settings = {x['name'].rsplit('.', 1)[-1]: x for x in torch.load(a.policy, weights_only=True)['modules']
                if '.layers.7.' in x['name']}
    x = next(x for (name, shape), x in data['inputs'].items() if name == 'gate_proj' and shape[-2] == 279).npu()
    with torch.inference_mode():
        layers = {}
        for name, weight in [('gate_proj', torch.cat([data['weights']['gate_proj'], data['weights']['up_proj']])),
                             ('down_proj', data['weights']['down_proj'])]:
            linear = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False, dtype=torch.bfloat16)
            linear.weight.data = weight
            layers[name] = NPUDynamicInt8Linear(linear, settings[name]['smooth'], {'forward_calls': 0}, name).npu()
        gate, down = layers['gate_proj'], layers['down_proj']
        smooth = down.smooth.float().reshape(1, -1)

        def ordinary():
            value = gate(x)
            half = value.shape[-1] // 2
            return down(torch.nn.functional.silu(value[..., :half]) * value[..., half:])

        def fused():
            value = gate(x).reshape(-1, gate.out_features)
            quant, scales = torch_npu.npu_dequant_swiglu_quant(
                value, quant_scale=smooth, activate_left=True, quant_mode=1)
            return torch_npu.npu_quant_matmul(quant, down.qweight, down.scales,
                                            pertoken_scale=scales, output_dtype=torch.bfloat16).reshape(*x.shape[:-1], down.out_features)

        reference = ordinary()
        functions = {'ordinary': ordinary, 'native_swiglu_quant': fused}
        results = {}; failures = {}
        for name, function in functions.items():
            try:
                actual = function()
                torch.npu.synchronize()
                results[name] = dict(relative_l2_vs_unfused=float((actual.float()-reference.float()).norm()/reference.float().norm()),
                                     finite=bool(actual.isfinite().all()), samples_ms=[])
            except Exception as exc:
                failures[name] = str(exc)
        for repeat in range(6):
            order = list(results) if repeat % 2 == 0 else list(reversed(results))
            for name in order:
                fn = functions[name]
                for _ in range(20): fn()
                torch.npu.synchronize()
                start = time.perf_counter()
                for _ in range(100): fn()
                torch.npu.synchronize()
                results[name]['samples_ms'].append(1000*(time.perf_counter()-start)/100)
        for result in results.values():
            result['median_ms'] = statistics.median(result['samples_ms'])
        a.output.write_text(json.dumps(dict(results=results, failures=failures), indent=2) + '\n')
        print(results, failures, flush=True)


if __name__ == '__main__':
    main()
