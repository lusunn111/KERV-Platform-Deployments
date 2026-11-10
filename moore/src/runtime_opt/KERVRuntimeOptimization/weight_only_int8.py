"""Inference-only groupwise W8A16 storage for CPU/MUSA/CUDA.

Weights remain INT8 between calls. Each Linear materializes only its own BF16
weight, calls the native floating-point GEMM and releases the temporary. This
is a memory-oriented path, not a claim of native INT8 GEMM acceleration.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class Int8WeightOnlyLinear(nn.Module):
    def __init__(self, in_features, out_features, group_size=128):
        super().__init__()
        if group_size <= 0:
            raise ValueError("group_size must be positive")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.group_size = int(group_size)
        groups = (in_features + group_size - 1) // group_size
        self.register_buffer("qweight", torch.empty(out_features, in_features, dtype=torch.int8))
        self.register_buffer("scales", torch.empty(out_features, groups, dtype=torch.float32))
        self.register_buffer("bias", None)
        self.forward_calls = 0

    @classmethod
    @torch.no_grad()
    def from_linear(cls, linear, group_size=128):
        if linear.weight.device.type != "cpu":
            raise ValueError("Quantize on CPU before transferring the model to the accelerator")
        if not linear.weight.is_floating_point() or linear.weight.is_meta:
            raise ValueError("Expected materialized floating-point Linear weights")
        result = cls(linear.in_features, linear.out_features, group_size)
        groups = result.scales.shape[1]
        # Bound temporary CPU FP32 storage, including the largest MLP projection.
        for start in range(0, linear.out_features, 128):
            end = min(start + 128, linear.out_features)
            value = linear.weight[start:end].detach().float()
            value = F.pad(value, (0, groups * group_size - linear.in_features))
            value = value.reshape(end - start, groups, group_size)
            scale = value.abs().amax(-1) / 127.0
            scale = torch.where(scale == 0, torch.ones_like(scale), scale)
            quantized = (value / scale.unsqueeze(-1)).round().clamp(-127, 127).to(torch.int8)
            result.qweight[start:end].copy_(quantized.reshape(end - start, -1)[:, :linear.in_features])
            result.scales[start:end].copy_(scale)
        if linear.bias is not None:
            result.bias = linear.bias.detach().clone()
        return result.eval()

    def dequantize(self, dtype):
        if self.qweight.device.type in {"musa", "cuda"}:
            from .weight_only_int8_kernels import dequantize
            return dequantize(self.qweight, self.scales, self.group_size, dtype)
        columns = torch.arange(self.in_features, device=self.qweight.device) // self.group_size
        return (self.qweight.float() * self.scales[:, columns]).to(dtype)

    def forward(self, value):
        if value.requires_grad:
            raise RuntimeError("Weight-only INT8 is inference-only")
        if value.device != self.qweight.device:
            raise ValueError("Input and quantized weights must be on the same device")
        if value.dtype not in {torch.bfloat16, torch.float16, torch.float32}:
            raise ValueError("W8A16 requires floating-point activations")
        weight = self.dequantize(value.dtype)
        result = F.linear(value, weight, self.bias)
        self.forward_calls += 1
        return result


def group_for(name):
    name = name.lower()
    if "ea_layer" in name or "drafter" in name:
        return "draft_linear"
    if "vision" in name:
        return "vision_mlp" if ".mlp." in name else None
    if ".mlp." in name:
        return "verify_mlp"
    if "self_attn" in name or "attention" in name:
        return "verify_attention"
    return None


def quantize_linears(model, groups, group_size=128, policy=None):
    allowed = {"verify_mlp", "verify_attention", "vision_mlp", "draft_linear"}
    unknown = set(groups) - allowed
    if unknown:
        raise ValueError("Unknown quantization groups: " + str(sorted(unknown)))
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    candidates = [(name, module) for name, module in model.named_modules()
                  if isinstance(module, nn.Linear) and group_for(name) in groups]
    if not candidates:
        raise ValueError("No compatible Linear modules matched the requested groups")
    if any(module.weight.device.type != 'cpu' for _, module in candidates):
        raise ValueError("All selected weights must be on CPU before quantization")
    eligible = list(candidates)
    if policy is not None:
        if not isinstance(policy, dict):
            raise ValueError('Mixed policy must be a JSON object')
        if policy.get('schema') != 'kerv_mixed_w8_bf16_v1' or policy.get('group_size') != group_size:
            raise ValueError('Mixed policy schema/group_size does not match this quantizer')
        selected = policy.get('int8_modules')
        if not isinstance(selected, list) or not selected or len(set(selected)) != len(selected):
            raise ValueError('Mixed policy needs a nonempty, unique int8_modules list')
        known = dict(candidates)
        if set(selected) - set(known):
            raise ValueError('Mixed policy references unknown/ineligible modules')
        expected_shapes = policy.get('weight_shapes', {})
        for name in selected:
            if expected_shapes.get(name) != list(known[name].weight.shape):
                raise ValueError('Mixed policy weight shape mismatch: ' + name)
        candidates = [(name, module) for name, module in candidates if name in set(selected)]
    # All validation precedes mutation; failures are fatal for the strict profile.
    prepared = [(name, module, Int8WeightOnlyLinear.from_linear(module, group_size))
                for name, module in candidates]
    records = []
    for name, original, replacement in prepared:
        parent_path, _, child_name = name.rpartition('.')
        parent = model.get_submodule(parent_path) if parent_path else model
        setattr(parent, child_name, replacement)
        before = original.weight.numel() * original.weight.element_size()
        after = replacement.qweight.numel() + replacement.scales.numel() * 4
        records.append({"name": name, "group": group_for(name),
                        "shape": list(original.weight.shape), "weight_bytes_before": before,
                        "weight_bytes_after": after, "bias_preserved": original.bias is not None})
    selected_names = {x['name'] for x in records}
    retained = [(name, module) for name, module in eligible if name not in selected_names]
    return {"selected_groups": sorted({x['group'] for x in records}), "modules": records,
            "module_count": len(records), "group_size": group_size,
            "mixed_precision": policy is not None,
            "retained_bf16_modules": [name for name, _ in retained],
            "retained_bf16_module_count": len(retained),
            "retained_bf16_weight_bytes": sum(module.weight.numel() * module.weight.element_size() for _, module in retained),
            "weight_bytes_before": sum(x['weight_bytes_before'] for x in records),
            "weight_bytes_after": sum(x['weight_bytes_after'] for x in records),
            "estimated_weight_storage_saving_bytes": sum(x['weight_bytes_before'] - x['weight_bytes_after'] for x in records),
            "weight_storage": "int8", "scale_dtype": "float32", "activation_dtype": "unchanged",
            "gemm": "native BF16/FP16 after per-layer dequantization; no full-model BF16 cache",
            "preparation_device": "cpu", "accuracy": "lossy; closed-loop quality must be measured"}
