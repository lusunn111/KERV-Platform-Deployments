"""Measure the actual KERV shapes on MUSA instead of inheriting A100 gates."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import flag_gems

from openvla.experiments.robot.device_runtime import get_device, get_accelerator
from KERVRuntimeOptimization.adaptive_rms_norm import rms_norm_inference, rms_norm_aten_inference
from KERVRuntimeOptimization.embodied_ops.kerv_ops import _silu_mul_flat_kernel
import triton


def measure(call, device, iterations=20):
    backend = get_accelerator(device)
    begin, end = backend.Event(enable_timing=True), backend.Event(enable_timing=True)
    backend.synchronize(device)
    wall = time.perf_counter()
    begin.record()
    for _ in range(iterations):
        call()
    end.record()
    end.synchronize()
    return {"device_ms": begin.elapsed_time(end) / iterations,
            "wall_ms": (time.perf_counter() - wall) * 1000 / iterations}


def compare(native, fused, device):
    for _ in range(6):
        native()
        fused()
    trials = {"native": [], "fused": []}
    for trial in range(3):
        order = (("native", native), ("fused", fused))
        if trial % 2:
            order = tuple(reversed(order))
        for name, call in order:
            trials[name].append(measure(call, device))
    result = {name: {key: statistics.median(row[key] for row in rows)
                     for key in ("device_ms", "wall_ms")} for name, rows in trials.items()}
    result["wall_speedup"] = result["native"]["wall_ms"] / result["fused"]["wall_ms"]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("/workspace/outputs/musa_operator_benchmark.json"))
    args = parser.parse_args()
    device = get_device()
    assert device.type == "musa"
    torch.manual_seed(7)
    weights = [torch.randn(4096, 4096, device=device, dtype=torch.bfloat16) for _ in range(3)]
    qkv = torch.cat(weights)
    gate, up = [torch.randn(11008, 4096, device=device, dtype=torch.bfloat16) for _ in range(2)]
    gate_up = torch.cat((gate, up))
    norm_weight = torch.ones(4096, device=device, dtype=torch.bfloat16)
    weighted_norm = torch.randn_like(norm_weight)
    report = {"device": str(device), "name": torch.musa.get_device_name(0),
              "flag_gems_vendor": flag_gems.vendor_name, "shapes": {}}
    with torch.inference_mode():
        for rows in (4, 8, 192, 224, 232, 240, 248, 256, 264, 279, 280, 292, 320):
            x = torch.randn(rows, 4096, device=device, dtype=torch.bfloat16)
            activation = torch.randn(rows, 11008, device=device, dtype=torch.bfloat16)
            factor = torch.randn_like(activation)
            # Repeated in-place calls must not grow/overflow their input.
            # A zero buffer stays finite while executing the same kernels.
            inplace_gate = torch.zeros_like(activation)
            def native_norm():
                x32 = x.float()
                return (x32 * torch.rsqrt(x32.square().mean(-1, keepdim=True) + 1e-6)).bfloat16() * norm_weight
            results = {
                "qkv": compare(lambda: [F.linear(x, w) for w in weights], lambda: F.linear(x, qkv), device),
                "gate_up": compare(lambda: (F.linear(x, gate), F.linear(x, up)), lambda: F.linear(x, gate_up), device),
                "rms_norm": compare(native_norm, lambda: rms_norm_inference(x, (4096,), norm_weight), device),
                "swiglu": compare(lambda: F.silu(activation) * factor,
                                  lambda: flag_gems.silu_and_mul(activation, factor), device),
                "swiglu_flat_inplace": compare(lambda: F.silu(inplace_gate, inplace=True).mul_(factor),
                    lambda: _silu_mul_flat_kernel[(triton.cdiv(inplace_gate.numel(), 1024),)](
                        inplace_gate, factor, inplace_gate.numel(), BLOCK=1024), device),
            }
            # Record the native ATen contract separately from handwritten
            # Llama RMSNorm. A fast kernel can still fail the action gate.
            native_aten = F.rms_norm(x, (4096,), weighted_norm, 1e-6)
            fused_aten = rms_norm_aten_inference(x, (4096,), weighted_norm, 1e-6)
            results["aten_rms_norm_precision"] = {
                "equal_element_fraction": float((native_aten == fused_aten).float().mean().cpu()),
                "maximum_absolute_difference": float((native_aten - fused_aten).abs().max().cpu()),
            }
            report["shapes"][str(rows)] = results
            print(rows, {name: round(values["wall_speedup"], 3) for name, values in results.items()
                         if "wall_speedup" in values}, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
