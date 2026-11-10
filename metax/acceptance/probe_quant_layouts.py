"""Real-input NPU layout and dynamic INT8 probes, including quantization cost."""
import argparse
import gc
import json
from pathlib import Path
import statistics

import torch
import torch_npu

from diagnose_quantization import bench


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--internal-format', action='store_true')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    if a.internal_format:
        torch.npu.config.allow_internal_format = True
    payload = torch.load(a.inputs, weights_only=False, map_location='cpu')
    rows = []
    with torch.inference_mode():
        for projection in ['gate_proj', 'down_proj']:
            weight = payload['weights'][projection]
            n, k = weight.shape
            candidates = [(s, x) for (name, s), x in payload['inputs'].items() if name == projection]
            shape, cpu_x = next(((s, x) for s, x in candidates if s[-2] == 279), candidates[0])
            x = cpu_x.reshape(-1, k).npu()
            w = weight.npu()
            reference = torch.nn.functional.linear(x, w)
            functions = {'bf16': lambda: torch.nn.functional.linear(x, w)}
            failures = {}
            layouts = {}
            buffers = []
            wf = weight.float().reshape(n, -1, 32)
            scale = (wf.abs().amax(-1) / 127).clamp_min(1e-8).bfloat16()
            q = (wf / scale.float().unsqueeze(-1)).round().clamp(-127, 127).to(torch.int8).reshape(n, k).npu()
            scales = scale.npu()
            buffers.extend([q, scales])
            for layout in ['transpose', 'contiguous', 'nz_transpose', 'nz_contiguous']:
                try:
                    if layout == 'transpose':
                        qw, qs = q.t(), scales.t()
                    elif layout == 'contiguous':
                        qw, qs = q.t().contiguous(), scales.t().contiguous()
                    elif layout == 'nz_transpose':
                        qw, qs = torch_npu.npu_format_cast(q, 29).t(), scales.t().contiguous()
                    else:
                        qw, qs = torch_npu.npu_format_cast(q.t().contiguous(), 29), scales.t().contiguous()
                    buffers.extend([qw, qs])
                    layouts['w8a16_' + layout] = dict(format=torch_npu.get_npu_format(qw), stride=list(qw.stride()))
                    functions['w8a16_' + layout] = lambda qw=qw, qs=qs: torch_npu.npu_weight_quant_batchmatmul(
                        x, qw, qs, antiquant_group_size=32)
                except Exception as exc:
                    failures['w8a16_' + layout] = str(exc)
            # Each call includes per-token dynamic activation quantization.
            # SmoothQuant scale is estimated here only for kernel screening.
            for alpha in [0.0, 0.5]:
                if alpha:
                    actmax = cpu_x.reshape(-1, k).float().abs().amax(0).clamp_min(1e-5)
                    wmax = weight.float().abs().amax(0).clamp_min(1e-5)
                    smooth = (wmax.pow(alpha) / actmax.pow(alpha)).bfloat16()
                else:
                    smooth = torch.ones(k, dtype=torch.bfloat16)
                wsmooth = weight.float() / smooth.float()
                s8 = (wsmooth.abs().amax(-1) / 127).clamp_min(1e-8)
                q8 = (wsmooth / s8[:, None]).round().clamp(-127, 127).to(torch.int8).npu()
                s8 = s8.npu()
                smooth = smooth.npu() if alpha else None
                buffers.extend([q8, s8, smooth])
                for layout in ['transpose', 'contiguous', 'nz_contiguous']:
                    mode = f'w8a8_a{alpha}_{layout}'
                    try:
                        qw = q8.t() if layout == 'transpose' else q8.t().contiguous()
                        if layout == 'nz_contiguous':
                            qw = torch_npu.npu_format_cast(qw, 29)
                        buffers.append(qw)
                        layouts[mode] = dict(format=torch_npu.get_npu_format(qw), stride=list(qw.stride()))

                        def call(qw=qw, scale=s8, smooth=smooth):
                            qx, sx = torch_npu.npu_dynamic_quant(x, smooth_scales=smooth)
                            return torch_npu.npu_quant_matmul(qx, qw, scale, pertoken_scale=sx,
                                                            output_dtype=torch.bfloat16)

                        functions[mode] = call
                    except Exception as exc:
                        failures[mode] = str(exc)
            measurements = {}
            for mode, function in functions.items():
                try:
                    actual = function()
                    torch.npu.synchronize()
                    measurements[mode] = dict(
                        relative_l2=float((actual.float() - reference.float()).norm() / reference.float().norm()),
                        finite=bool(actual.isfinite().all()), samples=[])
                except Exception as exc:
                    failures[mode] = str(exc)
            live = [name for name in functions if name in measurements]
            for repeat in range(4):
                for mode in live if repeat % 2 == 0 else list(reversed(live)):
                    measurements[mode]['samples'].append(bench(functions[mode]))
            for value in measurements.values():
                value['median_device_ms'] = statistics.median(x['device_ms'] for x in value['samples'])
                value['median_batch_wall_ms'] = statistics.median(x['synchronized_batch_wall_ms'] for x in value['samples'])
            rows.append(dict(projection=projection, shape=list(shape), measurements=measurements, failures=failures,
                             actual_weight_layouts=layouts, internal_format=a.internal_format))
            (a.output / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
            print(projection, {key: {k: v for k, v in value.items() if k != 'samples'}
                               for key, value in measurements.items()}, 'failed', list(failures), flush=True)
            del functions, buffers, x, w, reference, q, scales, q8, qw, qs, s8, smooth, actual
            gc.collect()
            torch.npu.empty_cache()


if __name__ == '__main__':
    main()
