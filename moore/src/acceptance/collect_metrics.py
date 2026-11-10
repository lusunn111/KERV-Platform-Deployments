"""Opt-in live collection. Diagnostic transfer tracing is never a timing benchmark.

KERV_ACCEPTANCE_CONTEXT names a JSON file created by run_matrix.py. Without it,
the existing inference path has no collector or dispatch mode installed.
"""
import atexit
from contextlib import nullcontext, contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

from acceptance.common import dump


def now():
    return datetime.now(timezone.utc).isoformat()


class Collector:
    @classmethod
    def from_environment(cls, cfg):
        path = os.environ.get("KERV_ACCEPTANCE_CONTEXT")
        return cls(path, cfg) if path else None

    def __init__(self, context, cfg):
        import torch
        from experiments.robot.device_runtime import get_device, get_accelerator
        self.torch = torch
        self.context = json.loads(Path(context).read_text())
        self.root = Path(context).parent
        self.diagnostic = bool(self.context.get("diagnostic", False))
        self.device = get_device()
        self.backend = get_accelerator(self.device)
        self.start = time.perf_counter()
        self.timestamp = now()
        self.pending = {}
        self.step = None
        self.episode = None
        self.closed = False
        self.observer = False
        self.mode = None
        self.unsupported = []
        self.hooks = []
        self.cfg = {key: str(value) if isinstance(value, Path) or isinstance(value, float) and not math.isfinite(value) else value
                    for key, value in vars(cfg).items()}
        self.emit("phases", {"phase": "before_model_load"})
        self.reset_peak("whole_run_initialization")
        self.memory("before_model_load")
        atexit.register(self.close)

    def emit(self, name, record):
        ids = {k: self.context.get(k) for k in ("campaign_id", "run_id", "profile", "model_id", "platform_id", "chip_id")}
        if self.step:
            ids.update({k: self.step[k] for k in ("episode_id", "step_id")})
        self.pending.setdefault(name, []).append({**ids, "timestamp": now(), **record})

    def flush(self):
        for name, records in self.pending.items():
            if records:
                with (self.root / (name + ".jsonl")).open("a") as stream:
                    for record in records:
                        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                records.clear()

    def reset_peak(self, phase):
        try:
            self.backend.reset_peak_memory_stats(self.device)
            self.emit("phases", {"phase": "memory_peak_reset", "window": phase})
        except (AttributeError, RuntimeError) as error:
            self.unsupported.append("reset_peak_memory_stats: " + str(error))

    def memory(self, phase):
        result = {"phase": phase, "monitor_method": "framework allocator", "sample_interval_seconds": None,
                  "process_device_bytes": None, "device_total_used_bytes": None}
        for field, method in (("allocated_bytes", "memory_allocated"), ("reserved_bytes", "memory_reserved"),
                              ("peak_allocated_bytes", "max_memory_allocated"), ("peak_reserved_bytes", "max_memory_reserved")):
            try:
                result[field] = int(getattr(self.backend, method)(self.device))
            except (AttributeError, RuntimeError) as error:
                result[field] = None
                self.unsupported.append(method + ": " + str(error))
        try:
            result["cpu_rss_bytes"] = int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError):
            result["cpu_rss_bytes"] = None
        self.emit("memory", result)

    def loaded(self, model):
        self.emit("phases", {"phase": "model_loaded", "elapsed_since_collector_start_seconds": time.perf_counter() - self.start})
        self.memory("after_model_load")
        model._acceptance_collector = self
        self.emit("placement", {"module": "all_model_parameters", "devices": sorted({str(p.device) for p in model.parameters()}),
                                "dtypes": sorted({str(p.dtype) for p in model.parameters()}), "method": "actual loaded parameters"})
        self.quantized_modules = []
        if getattr(model, "_flagos_w8a16_enabled", False):
            from KERVRuntimeOptimization.weight_only_int8 import Int8WeightOnlyLinear
            self.quantized_modules = [(name, module) for name, module in model.named_modules()
                                      if isinstance(module, Int8WeightOnlyLinear)]
            self.emit("quantization", {"phase": "after_model_load", "manifest": model._flagos_w8a16_manifest,
                "retained_bf16_execution_weights": [
                    {"name": name, "dtype": str(model.get_submodule(name).weight.dtype),
                     "device": str(model.get_submodule(name).weight.device),
                     "shape": list(model.get_submodule(name).weight.shape)}
                    for name in model._flagos_w8a16_manifest.get('retained_bf16_modules', [])],
                "execution_buffers": [{"name": name, "weight_device": str(module.qweight.device),
                                       "weight_dtype": str(module.qweight.dtype), "scale_dtype": str(module.scales.dtype),
                                       "weight_bytes": module.qweight.numel(), "scale_bytes": module.scales.numel() * 4}
                                      for name, module in self.quantized_modules]})
        if self.diagnostic:
            # Record each module's first real input, then remove its hook.
            def install(name, module):
                handle = None
                def before(mod, args):
                    tensors = []
                    def visit(value):
                        if isinstance(value, self.torch.Tensor):
                            tensors.append({"device": str(value.device), "shape": list(value.shape), "dtype": str(value.dtype)})
                        elif isinstance(value, (tuple, list)):
                            for v in value:
                                visit(v)
                    visit(args)
                    self.emit("placement", {"module": name, "function": type(mod).__name__, "inputs": tensors[:4],
                                            "host_thread": __import__("threading").get_ident(), "method": "first real forward pre-hook"})
                    handle.remove()
                handle = module.register_forward_pre_hook(before)
                self.hooks.append(handle)
            for name, module in model.named_modules():
                if name and (name.endswith("q_proj") or name.endswith("fc") or "vision_backbone" == name.split(".")[-1]):
                    install(name, module)
        self.flush()

    def begin_step(self, task_id, episode, step_id, observation):
        wall = time.perf_counter()
        # Hash actual incoming RGB and proprioception, before measured policy work.
        h = hashlib.sha256()
        for key in ("agentview_image", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"):
            value = observation.get(key)
            if value is not None:
                h.update(key.encode())
                h.update(str(value.shape).encode())
                h.update(str(value.dtype).encode())
                h.update(value.tobytes())
        self.step = {"episode_id": "%s_%s" % (task_id, episode), "step_id": step_id,
                     "task_id": task_id, "observation_hash": h.hexdigest(), "wall_start": wall,
                     "pipeline_start": time.perf_counter(), "cpu_start": time.process_time(), "verify_count": 0}
        if self.episode is None:
            self.episode = {"episode_id": self.step["episode_id"], "start": wall, "steps": 0}

    def transfer_context(self):
        if not self.diagnostic:
            return nullcontext()
        if self.mode is None:
            self.mode = make_transfer_mode(self)
        return self.diagnostic_context()

    @contextmanager
    def diagnostic_context(self):
        profiler = None
        if self.step and self.step["step_id"] == 0:
            try:
                activity = getattr(self.torch.profiler.ProfilerActivity, "MUSA")
                profiler = self.torch.profiler.profile(
                    activities=[self.torch.profiler.ProfilerActivity.CPU, activity], record_shapes=True)
                profiler.start()
            except (AttributeError, RuntimeError) as error:
                profiler = None
                self.unsupported.append("diagnostic profiler start: " + str(error))
        try:
            with self.mode:
                yield
        finally:
            if profiler is not None:
                try:
                    profiler.stop()
                    profiler.export_chrome_trace(str(self.root / "diagnostic_trace.json"))
                except (AttributeError, RuntimeError) as error:
                    self.unsupported.append("diagnostic profiler export: " + str(error))

    def verify(self, candidates, best_index, accepted, threshold, compensation_triggered, tokens_before):
        if self.step is None:
            return None
        proposed = None
        selected = None
        if self.diagnostic:
            self.observer = True
            try:
                selected = candidates[best_index].detach().cpu().tolist()
                proposed = sum(int(x) >= 0 for x in selected[1:])
            finally:
                self.observer = False
        event = {"verify_id": self.step["verify_count"], "proposed_length": proposed,
                 "proposed_capacity_including_padding": int(candidates.shape[1]) - 1,
                 "accepted_prefix_length": int(accepted), "selected_path": int(best_index),
                 "selected_candidates": selected, "root_length": 1, "bonus_length": 0,
                 "compensation_length": 0, "threshold": None if threshold is None else float(threshold),
                 "compensation_triggered": bool(compensation_triggered), "tokens_before": int(tokens_before),
                 "fallback_reason": "direct_kalman_completion" if compensation_triggered else None,
                 "acceptance_semantics": "posterior-selected candidate prefix excluding root, bonus, compensation and padding",
                 "proposed_length_note": "selected real path excluding negative padding" if self.diagnostic else "not sampled to avoid an extra device-to-host synchronization"}
        self.step["verify_count"] += 1
        # Store the same object so completion can append actual committed lengths.
        self.emit("verify", event)
        stored = self.pending["verify"][-1]
        return stored

    def policy_finished(self):
        self.step["pipeline_seconds"] = time.perf_counter() - self.step["pipeline_start"]
        self.step["env_start"] = time.perf_counter()

    def finish_step(self, raw, executed, done, model_seconds, tokens):
        end = time.perf_counter()
        s = self.step
        self.emit("steps", {k: s[k] for k in ("episode_id", "step_id", "task_id", "observation_hash") } | {
            "feedback_source": "history of model-generated action tokens/raw actions; not measured joint feedback",
            "action_tokens": [int(x) for x in tokens], "raw_action": raw.tolist(), "executed_action": executed.tolist(),
            "done": bool(done), "model_seconds": float(model_seconds), "pipeline_seconds": s["pipeline_seconds"],
            "env_seconds": end - s["env_start"], "wall_seconds": end - s["wall_start"],
            "cpu_seconds": time.process_time() - s["cpu_start"], "verify_events": s["verify_count"],
            "h2d_bytes": None, "d2h_bytes": None, "copy_seconds": None,
            "transfer_note": "diagnostic transfer events are separate; absent in low-interference runs"})
        self.memory("first_action_end" if s["step_id"] == 0 else "action_end")
        if s["step_id"] == 0:
            self.reset_peak("after_first_action_inference_steady_window")
        self.episode["steps"] += 1
        self.flush()
        self.step = None

    def finish_episode(self, done, limit):
        if self.episode is None:
            return
        self.emit("episodes", {"episode_id": self.episode["episode_id"], "steps": self.episode["steps"],
                               "task_success": bool(done), "termination_reason": "task_success" if done else "step_limit",
                               "max_steps": limit, "closed_loop_wall_seconds": time.perf_counter() - self.episode["start"],
                               "scope": "first incoming observation through terminal env.step and step logging; excludes model load, settling and video"})
        self.episode = None
        self.flush()

    def error(self, error):
        self.emit("errors", {"error_type": type(error).__name__, "error": str(error), "episode_incomplete": True})
        self.flush()

    def close(self):
        if self.closed:
            return
        self.closed = True
        if getattr(self, "quantized_modules", None):
            self.emit("quantization", {"phase": "process_exit", "module_forward_calls":
                {name: module.forward_calls for name, module in self.quantized_modules}})
        self.memory("process_exit")
        self.flush()
        dump(self.root / "collection.json", {"start_timestamp": self.timestamp, "end_timestamp": now(),
             "collector_seconds": time.perf_counter() - self.start, "diagnostic": self.diagnostic,
             "resolved_generate_config": self.cfg, "unsupported": sorted(set(self.unsupported)),
             "incomplete_episode": self.episode is not None,
             "transfer_coverage": "TorchDispatch copy/_to_copy/_local_scalar_dense logical payloads only; custom C++/Triton/driver internal transfers and profiler cross-check not certified" if self.diagnostic else "not collected in timing run",
             "physical_dma_bytes_measured": False, "transfer_threshold_eligible": False,
             "memory_note": "allocator peak reset before load and after first action; take maximum across both windows for initialization-inclusive peak"})


def make_transfer_mode(collector):
    from torch.utils._python_dispatch import TorchDispatchMode
    torch = collector.torch

    class Transfers(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            name = str(func)
            source = None
            destination = None
            if name in ("aten._to_copy.default", "aten._local_scalar_dense.default") and args:
                source = args[0]
                destination = torch.device("cpu") if "_local_scalar" in name else torch.device(kwargs.get("device", source.device))
            elif name == "aten.copy_.default" and len(args) > 1:
                source, destination = args[1], args[0].device
            crossing = isinstance(source, torch.Tensor) and destination is not None and source.device.type != destination.type
            if not crossing:
                return func(*args, **kwargs)
            # Diagnostic-only synchronization measures completed copies, and changes scheduling.
            collector.backend.synchronize(collector.device)
            begin = time.perf_counter()
            result = func(*args, **kwargs)
            collector.backend.synchronize(collector.device)
            duration = time.perf_counter() - begin
            src_bytes = source.numel() * source.element_size()
            dst_bytes = result.numel() * result.element_size() if isinstance(result, torch.Tensor) else src_bytes
            direction = "H2D" if source.device.type == "cpu" else ("D2H" if destination.type == "cpu" else "D2D")
            collector.emit("transfers", {"op": name, "direction": direction, "source_device": str(source.device),
                "destination_device": str(destination), "source_shape": list(source.shape),
                "source_logical_bytes": src_bytes, "destination_logical_bytes": dst_bytes,
                "payload_upper_of_logical_bytes": max(src_bytes, dst_bytes),
                "copy_sync_wall_seconds": duration, "observer_generated": collector.observer,
                "physical_dma_bytes": None, "timing_scope": "synchronized dispatch copy plus backend wait; diagnostic only"})
            return result

    return Transfers()
