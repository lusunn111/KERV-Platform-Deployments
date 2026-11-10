#!/usr/bin/env python3
"""FlagScale benchmark: replay identical LIBERO observations on one NPU."""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf

from experiments.robot.device_utils import get_device, synchronize
from experiments.robot.openvla_utils import get_processor, get_vla, get_vla_action
from experiments.robot.flagos_ops import enable_embodied_ops, enable_tree_builder, enable_action_head, enable_linear_fusion
from experiments.robot.tool_utils import compute_dynamic_threshold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", required=True)
    args = parser.parse_args()
    runtime = OmegaConf.load(args.config_path).runtime
    output = Path(runtime.work_dir)
    output.mkdir(parents=True, exist_ok=True)
    observation_dir = Path(runtime.get("benchmark_observations", output.parent / "npu_smoke" / "observations"))
    files = sorted(observation_dir.glob("*.npz"))[:int(runtime.get("benchmark_samples", 8))]
    if not files:
        raise RuntimeError(f"No captured observations: {observation_dir}")
    observations = []
    for file in files:
        with np.load(file, allow_pickle=False) as data:
            observations.append(({"full_image": data["image"].copy()}, str(data["task"])))
    repeats = int(runtime.get("benchmark_repeats", 3))
    warmup = int(runtime.get("benchmark_warmup", 2))
    planning_horizon = int(runtime.max_planning_steps)
    device = get_device()
    if device.type != "npu":
        raise RuntimeError(f"Benchmark requires NPU, selected {device}")
    torch.manual_seed(7)
    np.random.seed(7)
    results = {}
    all_actions = {}
    profiles = str(runtime.get("benchmark_profiles", "ar,naive-sd,specvla,kerv-native,kerv-ops")).split(",")
    if repeats < 1 or warmup < 1:
        raise ValueError("Benchmark repeats and complete-sequence warmups must be positive")
    if set(profiles) - {"ar", "ar-ops", "naive-sd", "specvla", "kerv-native", "kerv-ops"}:
        raise ValueError(f"Unknown benchmark profiles: {profiles}")
    if not {"kerv-native", "kerv-ops"}.issubset(profiles):
        raise ValueError("Include kerv-native and kerv-ops to check action parity")
    reference_path = runtime.get("benchmark_reference_report")
    reference = None
    reused_profiles = []
    if reference_path:
        reference = json.loads(Path(str(reference_path)).read_text())
        expected = {
            "observation_sha256": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
            "physical_device": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
            "torch": torch.__version__, "dtype": "bfloat16", "seed": 7,
            "warmup_sequences": warmup, "repeats": repeats, "planning_horizon": planning_horizon,
            "kalman_interval": int(runtime.get("kalman_interval", 4)),
            "verifier": str(runtime.pretrained_checkpoint), "drafter": str(runtime.spec_checkpoint),
            "preprocessing": "original TensorFlow JPEG/Lanczos resize and fractional center crop",
        }
        for key, value in expected.items():
            if reference.get(key) != value:
                raise ValueError(f"Cached baseline metadata mismatch: {key}")
    for profile in profiles:
        # Reuse only independently completed deterministic baselines. The
        # optimized candidate is always measured anew, even if a prior report
        # failed its optimized-action gate. Record this reuse in the report.
        if reference is not None and profile != "kerv-ops" and profile in reference["profiles"]:
            previous = np.asarray(reference["actions"][profile])
            if (previous.shape != (repeats, len(files), 7)
                    or not np.isfinite(previous).all()
                    or not np.array_equal(previous, np.broadcast_to(previous[0], previous.shape))
                    or reference["profiles"][profile]["count"] != repeats * len(files)):
                raise ValueError(f"Incomplete or non-deterministic cached baseline: {profile}")
            results[profile] = reference["profiles"][profile]
            all_actions[profile] = reference["actions"][profile]
            reused_profiles.append(profile)
            print(f"REUSE BASELINE {profile}: {reference_path}", flush=True)
            continue
        cfg = SimpleNamespace(pretrained_checkpoint=str(runtime.pretrained_checkpoint),
                              spec_checkpoint=str(runtime.spec_checkpoint), use_spec=profile not in {"ar", "ar-ops"},
                              parallel_draft=False, accept_threshold=9, accept_threshold_start=15,
                              load_in_8bit=False, load_in_4bit=False)
        model = get_vla(cfg).eval()
        model._kalman_interval = int(runtime.get("kalman_interval", 4))
        if model._kalman_interval < 1:
            raise ValueError("kalman_interval must be positive")
        processor = get_processor(cfg)
        model._flagos_inference_mode_enabled = True
        model._flagos_prompt_token_cache_enabled = False
        include = str(runtime.embodied_ops.include)
        ops_backend = str(runtime.get("benchmark_ops_backend", "native"))
        model._flagos_npu_attention_backend = str(runtime.get("ascend_attention_backend", "native"))
        enable_embodied_ops(enabled=profile.endswith("ops"), include=include, backend=ops_backend,
                            record=True, record_once=True, log_path=str(output / f"{profile}_ops.jsonl"),
                            manifest_path=str(output / f"{profile}_manifest.json"), model=model)
        if cfg.use_spec:
            enable_action_head(model=model,
                enabled=profile.endswith("ops") and bool(runtime.get("ascend_action_head", False)),
                action_vocab_size=256, record=False, record_once=True, log_path=None,
                manifest_path=str(output / f"{profile}_action_head.json"))
        if profile.endswith("ops") and bool(runtime.get("ascend_linear_fusion", False)):
            enable_linear_fusion(model=model, enabled=True,
                qkv_rows=str(runtime.get("ascend_qkv_rows", "8,224,240,248,256,264,279,280,320")),
                gate_up_rows=str(runtime.get("ascend_gate_up_rows", "8,224,240,248,256,264,279,280,320")), swiglu_rows="",
                qkv_input_sizes="4096,1152,1024", gate_up_input_sizes="4096,1152,1024",
                swiglu_input_sizes="4096", swiglu_backend="native_inplace", record=False,
                add_rms_norm_rows="", add_rms_norm_input_sizes="4096", add_rms_norm_backend="native_inplace",
                record_once=True, log_path=None, manifest_path=str(output / f"{profile}_linear_fusion.json"))
            model._flagos_linear_fusion_enabled = True
        if cfg.use_spec:
            kerv_runtime = profile.startswith("kerv")
            enable_tree_builder(model=model, enabled=kerv_runtime, manifest_path=None, static_runtime=kerv_runtime,
                                verify_max_paths=48 if kerv_runtime else 0,
                                cache_topology=kerv_runtime, operatorized=False, compile_verifier=False,
                                persistent_runtime=False, static_tree_attention="off")
        kalman = profile.startswith("kerv")
        if cfg.use_spec and profile.endswith("ops") and bool(runtime.get("ascend_graphs", False)):
            from experiments.robot.ascend_ops import install_npu_stage_graphs
            install_npu_stage_graphs(model)

        def reset():
            if cfg.use_spec:
                model.start_rollout(max_steps=planning_horizon, kalman_cfg={"enabled": kalman, "tree_enabled": kalman}, store_history=True)

        def action(index, history):
            obs, task = observations[index]
            threshold = compute_dynamic_threshold(index, planning_horizon, 15, 3) if kalman else (0 if profile == "naive-sd" else 9)
            return get_vla_action(model, processor, cfg.pretrained_checkpoint, obs, task, "libero_goal",
                                  return_time=True, center_crop=True, generate_mode="speculative",
                                  history=history, step_idx=index, max_steps=planning_horizon,
                                  dynamic_threshold=threshold, use_kalman=kalman, kalman_tree_enabled=kalman)

        for _ in range(warmup):
            reset()
            history = []
            for i in range(len(observations)):
                values, _ = action(i, history)
                if not kalman and getattr(model, "_flagos_last_forced_kalman", False):
                    raise RuntimeError(f"Kalman completion unexpectedly enabled for {profile}")
                history.append(values.copy())
            print(f"WARMUP {profile}: complete sequence of {len(observations)} observations", flush=True)
        synchronize(device)
        model_times, pipeline_times, measured_actions = [], [], []
        # Recording is restricted to warmup. Disk writes must not affect latency.
        from KERVRuntimeOptimization.embodied_ops.kerv_ops import configure_recording
        configure_recording(None, False)
        for repeat in range(repeats):
            reset()
            history, repeat_actions = [], []
            for i in range(len(observations)):
                synchronize(device)
                start = time.perf_counter()
                values, (end_model, start_model) = action(i, history)
                synchronize(device)
                if not kalman and getattr(model, "_flagos_last_forced_kalman", False):
                    raise RuntimeError(f"Kalman completion unexpectedly enabled for {profile}")
                pipeline_times.append((time.perf_counter() - start) * 1000)
                model_times.append((end_model - start_model) * 1000)
                if values.shape != (7,) or not np.isfinite(values).all():
                    raise RuntimeError(f"Invalid action for {profile}: {values}")
                history.append(values.copy())
                repeat_actions.append(values.tolist())
                if (i + 1) % 32 == 0:
                    print(f"PROGRESS {profile}: repeat {repeat+1}/{repeats}, action {i+1}/{len(observations)}", flush=True)
            measured_actions.append(repeat_actions)
        if not np.allclose(measured_actions, measured_actions[0], rtol=0, atol=0):
            raise RuntimeError(f"Non-deterministic repeated actions: {profile}")
        results[profile] = {
            "count": len(model_times), "warmup_actions": warmup * len(observations),
            "model_mean_ms": float(np.mean(model_times)), "model_median_ms": float(np.median(model_times)),
            "model_p95_ms": float(np.percentile(model_times, 95)),
            "pipeline_mean_ms": float(np.mean(pipeline_times)),
            "raw_model_ms": model_times, "raw_pipeline_ms": pipeline_times,
            "runtime_optimizations": {
                "npu_graphs_enabled": bool(getattr(model, "_flagos_npu_graphs_enabled", False)),
                "stage_graph_captures": int(getattr(model, "_flagos_stage_graph_captures", 0)),
                "prompt_graph_replays": int(getattr(model, "_flagos_prompt_graph_replays", 0)),
                "draft_graph_replays": int(getattr(model, "_flagos_draft_graph_replays", 0)),
                "verify_graph_replays": int(getattr(model, "_flagos_verify_graph_replays", 0)),
                "stage_graph_fallbacks": int(getattr(model, "_flagos_stage_graph_fallbacks", 0)),
                "gems_attention_hits": sum(getattr(m, "_kerv_gems_attention_hits", 0) for m in model.modules()),
                "cann_attention_hits": sum(getattr(m, "_kerv_cann_attention_hits", 0) for m in model.modules()),
                "action_head_enabled": bool(getattr(model, "_flagos_action_head_enabled", False)),
                "linear_fusion_enabled": bool(getattr(model, "_flagos_linear_fusion_enabled", False)),
            },
        }
        all_actions[profile] = measured_actions
        (output / "benchmark.partial.json").write_text(json.dumps(
            {"complete": False, "profiles": results, "actions": all_actions}, indent=2) + "\n")
        print(profile, json.dumps({k:v for k,v in results[profile].items() if not k.startswith("raw_")}), flush=True)
        del model, processor
        gc.collect()
        torch.npu.empty_cache()
    metric = "pipeline_mean_ms"
    speedups = {key.replace("-", "_") + "_over_kerv_ops": results[key][metric] / results["kerv-ops"][metric]
                for key in ("ar", "ar-ops", "naive-sd", "specvla", "kerv-native") if key in results}
    native_actions = np.asarray(all_actions["kerv-native"])
    ops_actions = np.asarray(all_actions["kerv-ops"])
    report = {"device": str(device), "physical_device": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
              "torch": torch.__version__, "dtype": "bfloat16", "observation_files": [str(f) for f in files],
              "observation_sha256": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
              "seed": 7, "warmup_sequences": warmup,
              "verifier": str(runtime.pretrained_checkpoint), "drafter": str(runtime.spec_checkpoint),
              "repeats": repeats, "planning_horizon": planning_horizon,
              "kalman_interval": int(runtime.get("kalman_interval", 4)),
              "ops_backend": ops_backend,
              "reused_baseline_report": str(reference_path) if reference_path else None,
              "reused_baseline_profiles": reused_profiles,
              "preprocessing": "original TensorFlow JPEG/Lanczos resize and fractional center crop",
              "baseline_definition": "naive-sd: threshold=0, no Kalman, no tree runtime optimizations; ar is a separate comparison",
              "timing": "synchronized perf_counter; model excludes processor; pipeline includes processor; simulator excluded",
              "profiles": results, "speedups_pipeline": speedups,
              "kerv_ops_actions_exact": bool(np.array_equal(native_actions, ops_actions)),
              "kerv_ops_max_action_difference": float(np.max(np.abs(native_actions - ops_actions))),
              "actions": all_actions}
    (output / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    print("BENCHMARK",json.dumps(speedups),flush=True)
    if not report["kerv_ops_actions_exact"]:
        raise RuntimeError("Operator action parity failed; benchmark.json is diagnostic, not a validated result")


if __name__ == "__main__":
    main()
