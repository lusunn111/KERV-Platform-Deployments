"""Same-chip replay and real-input kernel diagnosis; separate timing and tracing.

Run inside the Ascend container, with the Moore port on PYTHONPATH. This does
not require quantized actions to equal BF16 actions. Each profile must repeat
its own actions exactly. Kernel inputs and weights are identical across dtypes.
"""
import argparse
from collections import Counter
import gc
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import torch
import torch_npu

from experiments.robot.device_runtime import get_device, synchronize
from experiments.robot.openvla_utils import get_vla, get_processor, get_vla_action
from experiments.robot.flagos_ops import enable_embodied_ops, enable_tree_builder
from experiments.robot.tool_utils import compute_dynamic_threshold
from experiments.robot.ascend_quantization import NPUWeightOnlyLinear


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


class VerifyCounter:
    def __init__(self):
        self.events = []

    def verify(self, proposed, accepted, threshold, forced, shape):
        self.events.append(dict(proposed=int(proposed), accepted=int(accepted),
                                forced=bool(forced), shape=list(shape)))


def configure(model, output, label):
    include = 'kerv_silu_mul,kerv_add_rms_norm,kerv_verify_accept_control,kerv_static_tree_pack,kerv_tree_embed_pack,kerv_kv_commit,kerv_kv_accept_commit'
    enable_embodied_ops(model=model, enabled=True, include=include, backend='native',
        record=True, record_once=True, log_path=str(output/f'{label}_ops.jsonl'),
        manifest_path=str(output/f'{label}_manifest.json'))
    enable_tree_builder(model=model, enabled=True, static_runtime=True,
        verify_max_paths=0, node_buckets='', operatorized=True,
        draft_tree_operatorized=True, kalman_batch=True, kv_commit_batched=True,
        fused_control_transfer=False, compile_verifier=False, persistent_runtime=False,
        resident_kv_cache=False, static_tree_attention='off', manifest_path=None)
    model._flagos_system_optimization_enabled = True
    model._flagos_persistent_control_buffer = True
    model._flagos_persistent_decode_workspace = True
    model._flagos_inference_mode_enabled = True
    model.ea_layer._flagos_persistent_draft_workspace = True
    model.ea_layer._flagos_persistent_draft_kv_cache = True
    model._flagos_suppress_runtime_stats = True


def replay(args, output, observations):
    project = Path(__file__).resolve().parents[1]
    reports = {}
    captured = {}
    kernel_weights = {}
    device = get_device()
    for label in args.order:
        assert label not in reports, 'Use a fresh process for the reverse order'
        os.environ['KERV_NPU_QUANT'] = 'none' if label == 'bf16' else 'mlp_int8'
        os.environ['KERV_NPU_QUANT_LAYERS'] = '7:25'
        os.environ['KERV_NPU_QUANT_GROUP_SIZE'] = '128'
        torch.manual_seed(7); np.random.seed(7)
        cfg = SimpleNamespace(pretrained_checkpoint=str(project/'checkpoints/openvla-libero-goal'),
            spec_checkpoint=str(project/'checkpoints/kerv-drafter'), use_spec=True,
            parallel_draft=False, accept_threshold=9, accept_threshold_start=15,
            load_in_8bit=False, load_in_4bit=False)
        model = get_vla(cfg).eval(); processor = get_processor(cfg)
        configure(model, output, label)
        processor._flagos_prompt_token_cache_enabled = False
        counter = VerifyCounter(); model._acceptance_recorder = counter
        step_shapes = []
        capture_enabled = False
        hooks = []
        for projection in ['gate_proj', 'up_proj', 'down_proj']:
            layer = model.base_model.language_model.model.layers[7].mlp.get_submodule(projection)
            def hook(module, inputs, name=projection):
                x = inputs[0]
                step_shapes.append({'projection':name, 'shape':list(x.shape)})
                key = (name, tuple(x.shape))
                if capture_enabled and key not in captured:
                    captured[key] = x.detach().cpu()
            hooks.append(layer.register_forward_pre_hook(hook))
        rows = []; all_actions = []
        for repeat in range(args.repeats+1):
            model.start_rollout(max_steps=220, kalman_cfg={'enabled':True,'tree_enabled':True}, store_history=True)
            history = []; sequence = []
            for index, (obs, task) in enumerate(observations):
                # D2H diagnostic capture occurs only during the unmeasured warmup.
                capture_enabled = label == 'bf16' and repeat == 0 and index < 4
                counter.events = []; step_shapes.clear()
                synchronize(device); start = time.perf_counter()
                values, (end_model, start_model) = get_vla_action(model, processor,
                    cfg.pretrained_checkpoint, obs, task, 'libero_goal', return_time=True,
                    center_crop=True, generate_mode='speculative', history=history,
                    step_idx=index, max_steps=220,
                    dynamic_threshold=compute_dynamic_threshold(index,220,15,3),
                    use_kalman=True, kalman_tree_enabled=True)
                synchronize(device); pipeline_ms = 1000*(time.perf_counter()-start)
                assert values.shape == (7,) and np.isfinite(values).all()
                history.append(values.copy()); sequence.append(values.tolist())
                if repeat:
                    rows.append(dict(repeat=repeat, index=index, model_ms=1000*(end_model-start_model),
                        pipeline_ms=pipeline_ms, verify=list(counter.events),
                        layer7_calls=list(step_shapes), action=values.tolist()))
                if (index+1)%16 == 0:
                    print(label, 'sequence', repeat, 'action', index+1, flush=True)
            if repeat: all_actions.append(sequence)
        for hook in hooks: hook.remove()
        assert all(np.array_equal(all_actions[0], x) for x in all_actions), 'Within-profile action drift'
        if label == 'bf16':
            for name in ['gate_proj', 'up_proj', 'down_proj']:
                kernel_weights[name] = model.base_model.language_model.model.layers[7].mlp.get_submodule(name).weight.detach().cpu()
        events = [e for row in rows for e in row['verify']]
        shapes = Counter(tuple(call['shape']) for row in rows for call in row['layer7_calls'] if call['projection']=='gate_proj')
        reports[label] = dict(steps=len(rows), model_mean_ms=float(np.mean([r['model_ms'] for r in rows])),
            pipeline_mean_ms=float(np.mean([r['pipeline_ms'] for r in rows])),
            verify_cycles=len(events), mean_raw_acceptance=float(np.mean([e['accepted'] for e in events])),
            zero_accepts=sum(e['accepted']==0 for e in events),
            verifier_forward_shapes={str(k):v for k,v in shapes.items()}, rows=rows,
            quantization=getattr(model,'_flagos_npu_quant_manifest',None),
            allocated_peak_bytes=int(torch.npu.max_memory_allocated()),
            within_profile_actions_exact=True)
        write(output/'replay_partial.json', reports)
        print(label, {k:v for k,v in reports[label].items() if k not in ['rows','quantization']}, flush=True)
        model._acceptance_recorder = None
        del model, processor, layer, hook; gc.collect(); torch.npu.empty_cache()
        torch.npu.reset_peak_memory_stats()
    torch.save({'inputs':captured, 'weights':kernel_weights}, output/'kernel_inputs.pt')
    return reports


def bench(function, repeats=100):
    for _ in range(15): function()
    torch.npu.synchronize()
    begin = torch.npu.Event(enable_timing=True); end = torch.npu.Event(enable_timing=True)
    start = time.perf_counter(); begin.record()
    for _ in range(repeats): function()
    end.record(); end.synchronize()
    return dict(device_ms=begin.elapsed_time(end)/repeats,
                synchronized_batch_wall_ms=1000*(time.perf_counter()-start)/repeats)


def kernels(output, trace=False):
    payload = torch.load(output/'kernel_inputs.pt', weights_only=False, map_location='cpu')
    rows = []
    with torch.inference_mode():
        for name, weight in payload['weights'].items():
            original = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False, dtype=torch.bfloat16)
            original.weight.data.copy_(weight)
            quant = NPUWeightOnlyLinear(original, {'forward_calls':0}, name, 128).npu()
            reference = original.npu()
            for (projection, shape), cpu_input in payload['inputs'].items():
                if projection != name: continue
                x = cpu_input.npu()
                bf16 = lambda: reference(x)
                int8 = lambda: quant(x)
                actual = int8(); ref = bf16()
                rel = float((actual.float()-ref.float()).norm()/ref.float().norm())
                samples = {'bf16':[], 'int8':[]}
                for repeat in range(4):
                    for mode in (['bf16','int8'] if repeat%2==0 else ['int8','bf16']):
                        samples[mode].append(bench(bf16 if mode=='bf16' else int8))
                row = dict(projection=name, shape=list(shape), weight_shape=list(weight.shape),
                    relative_l2=rel, alternating_batch_samples=samples,
                    bf16_median_ms=float(np.median([s['device_ms'] for s in samples['bf16']])),
                    int8_median_ms=float(np.median([s['device_ms'] for s in samples['int8']])))
                row['int8_over_bf16_latency'] = row['int8_median_ms']/row['bf16_median_ms']
                rows.append(row); write(output/'kernels.json', rows); print('kernel', row, flush=True)
                if trace:
                    trace_dir = output/'device_trace'/f'{name}_M{int(np.prod(shape[:-1]))}'
                    with torch_npu.profiler.profile(
                        activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],
                        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(trace_dir)),
                        record_shapes=True, with_stack=False, profile_memory=False):
                        for _ in range(3):
                            with torch.profiler.record_function('DIAG::BF16'): bf16()
                            with torch.profiler.record_function('DIAG::INT8'): int8()
                        torch.npu.synchronize()
                del x, actual, ref
            del original, reference, quant; gc.collect(); torch.npu.empty_cache()
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--input-dir', type=Path, required=True)
    p.add_argument('--samples', type=int, default=64)
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--order', nargs=2, choices=['bf16','int8'], default=['bf16','int8'])
    p.add_argument('--trace', action='store_true')
    p.add_argument('--kernels-only', action='store_true')
    args = p.parse_args(); output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if args.kernels_only:
        kernels(output, args.trace); return
    assert not (output/'plan.json').exists(), 'Preserve earlier evidence; use a new directory'
    files = sorted(args.input_dir.glob('*.npz'))[:args.samples]
    assert len(files) == args.samples
    observations = []
    for file in files:
        with np.load(file, allow_pickle=False) as data:
            observations.append(({'full_image':data['image'].copy()}, str(data['task'])))
    plan = dict(samples=args.samples, repeats=args.repeats, warmup_sequences=1, order=args.order,
        physical_device=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'), seed=7, planning_horizon=220,
        quant_layers='7:25', group_size=128, action_protocol=os.environ.get('KERV_ACTION_PROTOCOL'),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
        scope='same observations; each precision generates its own history and speculative tree; not forced identical work')
    write(output/'plan.json', plan)
    reports = replay(args, output, observations)
    write(output/'replay.json', dict(plan=plan, profiles=reports,
        bf16_over_int8_model_ratio=reports['bf16']['model_mean_ms']/reports['int8']['model_mean_ms']))
    kernels(output, args.trace)


if __name__ == '__main__': main()
