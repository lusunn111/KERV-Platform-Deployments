"""Opt-in calibrated W8A8 MLPs using native Ascend dynamic quantization."""
import hashlib
from pathlib import Path
from types import MethodType

import torch
from torch import nn

from experiments.robot.ascend_quantization import NPUWeightOnlyLinear, storage_bytes, weight_sha256


class NPUPrepackedWeightOnlyLinear(NPUWeightOnlyLinear):
    """Prepare K,N weights and group,N scales once, before NPU transfer."""
    def __init__(self, linear, manifest, name, group_size=32):
        super().__init__(linear, manifest, name, group_size)
        self.qweight = self.qweight.t().contiguous()
        self.antiquant_scale = self.antiquant_scale.t().contiguous()

    def forward(self, x):
        import torch_npu
        if x.device.type != 'npu' or x.dtype != torch.bfloat16:
            raise ValueError('Packed W8A16 requires NPU BF16 input')
        result = torch_npu.npu_weight_quant_batchmatmul(
            x.reshape(-1, self.in_features), self.qweight, self.antiquant_scale,
            antiquant_group_size=self.group_size)
        self._manifest['forward_calls'] += 1
        return result.reshape(*x.shape[:-1], self.out_features)


class NPUDynamicInt8Linear(nn.Module):
    def __init__(self, linear, smooth, manifest, name):
        super().__init__()
        if linear.weight.device.type != 'cpu' or linear.weight.dtype != torch.bfloat16:
            raise ValueError('Dynamic INT8 conversion requires original CPU BF16 weights')
        if linear.bias is not None:
            raise ValueError('Only bias-free verifier projections are supported')
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        if smooth.shape != (self.in_features,) or smooth.dtype != torch.bfloat16:
            raise ValueError('Invalid smooth scale shape or dtype')
        if not bool(torch.isfinite(smooth).all() and (smooth > 0).all()):
            raise ValueError('Smooth scales must be finite and positive')
        weight = linear.weight.detach().float() / smooth.float()
        scales = (weight.abs().amax(-1) / 127).clamp_min(1e-8)
        quant = (weight / scales[:, None]).round().clamp(-127, 127).to(torch.int8)
        # Measured layouts: gate/up favor K,N contiguous, down favors a
        # transposed N,K allocation. Conversion and views happen before loading.
        qweight = quant.t() if name.endswith('down_proj') else quant.t().contiguous()
        self.register_buffer('qweight', qweight)
        self.register_buffer('scales', scales)
        self.register_buffer('smooth', smooth.detach().clone())
        self._manifest = manifest

    def forward(self, x):
        import torch_npu
        if x.device.type != 'npu' or x.dtype != torch.bfloat16:
            raise ValueError('Dynamic INT8 requires NPU BF16 input')
        shape = x.shape
        quant, scale = torch_npu.npu_dynamic_quant(x.reshape(-1, self.in_features), smooth_scales=self.smooth)
        result = torch_npu.npu_quant_matmul(quant, self.qweight, self.scales,
                                          pertoken_scale=scale, output_dtype=torch.bfloat16)
        self._manifest['forward_calls'] += 1
        return result.reshape(*shape[:-1], self.out_features)


class DynamicGateUpSlice(nn.Module):
    """Preserve the existing MLP call interface while sharing one projection.

    The usual gate(x), up(x) sequence consumes and clears the temporary output.
    Unexpected isolated calls compute a fresh output rather than using stale data.
    """
    def __init__(self, group, index):
        super().__init__()
        object.__setattr__(self, '_group', group)
        self.index = index
        self.in_features = group.in_features
        self.out_features = group.out_features // 2

    def forward(self, x):
        group = self._group
        if self.index == 1 and getattr(group, '_slice_input', None) is x:
            result = group._slice_output
            group._slice_input = None
            group._slice_output = None
            return result
        group._slice_input = None
        group._slice_output = None
        combined = group(x)
        if self.index == 0:
            group._slice_input = x
            group._slice_output = combined[..., self.out_features:]
            return combined[..., :self.out_features]
        return combined[..., self.out_features:]


def direct_dynamic_mlp_forward(self, x):
    combined = self._dynamic_gate_up(x)
    if getattr(self, '_dynamic_fused_swiglu_quant', False):
        import torch_npu
        down = self.down_proj
        if not isinstance(down, NPUDynamicInt8Linear):
            rounded = getattr(self, '_dynamic_rounded_swiglu_fn', None)
            if rounded is not None:
                return down(rounded(combined))
            size = combined.shape[-1] // 2
            return down(self.act_fn(combined[..., :size]) * combined[..., size:])
        quant, scales = torch_npu.npu_dequant_swiglu_quant(
            combined.reshape(-1, combined.shape[-1]), quant_scale=down.fused_smooth_scale,
            activate_left=True, quant_mode=1)
        result = torch_npu.npu_quant_matmul(quant, down.qweight, down.scales,
                                          pertoken_scale=scales, output_dtype=torch.bfloat16)
        down._manifest['forward_calls'] += 1
        return result.reshape(*x.shape[:-1], down.out_features)
    size = combined.shape[-1] // 2
    gate, up = combined[..., :size], combined[..., size:]
    return self.down_proj(self.act_fn(gate) * up)


def apply_cpu_mlp_dynamic_int8(model, policy_path):
    if not policy_path:
        raise ValueError('Calibrated dynamic INT8 requires KERV_NPU_QUANT_POLICY')
    path = Path(policy_path)
    policy = torch.load(path, map_location='cpu', weights_only=True)
    if policy.get('version') != 1 or policy.get('implementation') != 'ascend_dynamic_w8a8':
        raise ValueError('Unsupported dynamic INT8 policy')
    entries = policy['modules']
    if not entries or len({item['name'] for item in entries}) != len(entries):
        raise ValueError('Empty or duplicate dynamic INT8 policy')
    # Validate the complete policy before mutating modules.
    settings = {item['name']: item for item in entries}
    fused = bool(policy.get('fused_gate_up', False))
    if getattr(model.base_model.language_model.config, 'pretraining_tp', 1) != 1:
        raise ValueError('Dynamic INT8 requires pretraining_tp=1')
    if fused:
        for item in entries:
            if item['name'].endswith('up_proj') and item['name'].removesuffix('up_proj') + 'gate_proj' not in settings:
                raise ValueError('Fused gate/up policy is missing gate projection')
            if item['name'].endswith('gate_proj'):
                partner = settings.get(item['name'].removesuffix('gate_proj') + 'up_proj')
                if partner is None or partner.get('precision','dynamic_int8') != item.get('precision','dynamic_int8'):
                    raise ValueError('Fused gate/up requires matching precision')
                if item.get('precision') == 'weight_only_int8':
                    if partner.get('group_size',32) != item.get('group_size',32):
                        raise ValueError('Fused weight-only gate/up requires matching groups')
                elif not torch.equal(partner['smooth'], item['smooth']):
                    raise ValueError('Fused gate/up requires identical calibrated smoothing')
    silu_backend = policy.get('silu_backend','native')
    if silu_backend not in {'native','triton_rounded'}:
        raise ValueError('Unsupported SwiGLU backend')
    if silu_backend == 'triton_rounded':
        if not policy.get('direct_mlp') or not policy.get('fused_swiglu_quant'):
            raise ValueError('Rounded SwiGLU requires the direct fused MLP route')
        if any(item['name'].endswith('down_proj') and item.get('precision','dynamic_int8')=='dynamic_int8' for item in entries):
            raise ValueError('Rounded SwiGLU requires BF16 or W8A16 down input')
    if policy.get('fused_swiglu_quant', False):
        for item in entries:
            if item['name'].endswith('gate_proj'):
                parent = item['name'].rsplit('.', 1)[0]
                if parent + '.down_proj' not in settings:
                    raise ValueError('Fused SwiGLU quantization requires a quantized down projection')
                activation = type(model.get_submodule(parent).act_fn).__name__.lower()
                if 'silu' not in activation and 'swish' not in activation:
                    raise ValueError('Native fused activation requires a SiLU MLP')
    for item in entries:
        name = item['name']
        precision = item.get('precision', 'dynamic_int8')
        if precision not in {'bf16', 'dynamic_int8', 'weight_only_int8'}:
            raise ValueError('Unsupported projection precision')
        if precision == 'bf16' and not name.endswith('down_proj'):
            raise ValueError('BF16 preservation currently supports down projections')
        if precision == 'weight_only_int8' and not name.endswith('down_proj') and not fused:
            raise ValueError('Weight-only gate/up requires fused projection pairs')
        if precision == 'weight_only_int8' and item.get('group_size', 32) not in {32, 64, 128}:
            raise ValueError('Invalid weight-only group size')
        if not name.startswith('base_model.language_model.model.layers.') or '.mlp.' not in name:
            raise ValueError('Policy targets a non-verifier MLP')
        if name.rsplit('.', 1)[-1] not in {'gate_proj', 'up_proj', 'down_proj'}:
            raise ValueError('Policy targets an unsupported projection')
        layer = model.get_submodule(name)
        if not isinstance(layer, nn.Linear) or weight_sha256(layer.weight) != item['weight_sha256']:
            raise ValueError(f'Dynamic quantization weight identity mismatch: {name}')
    before = storage_bytes(model)
    manifest = dict(enabled=True, implementation='torch_npu.npu_dynamic_quant + torch_npu.npu_quant_matmul',
                    profile='verifier_mlp_calibrated_w8a8', weight_dtype='int8', activation_dtype='dynamic int8',
                    group_size=None, layer_indices=sorted({int(item['name'].split('.layers.')[1].split('.')[0]) for item in entries}),
                    output_dtype='bfloat16', converted_on='cpu before NPU transfer', fallback=False,
                    policy_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), modules=[], forward_calls=0,
                    storage_bytes_before_by_dtype=before)
    manifest['fused_gate_up'] = fused
    manifest['silu_backend'] = silu_backend
    manifest['preserved_bf16_modules'] = []
    manifest['precision_counts'] = {kind:sum(item.get('precision','dynamic_int8')==kind for item in entries)
                                    for kind in ['dynamic_int8','weight_only_int8','bf16']}
    manifest['direct_mlp'] = bool(policy.get('direct_mlp', False))
    manifest['fused_swiglu_quant'] = bool(policy.get('fused_swiglu_quant', False))
    if manifest['direct_mlp'] and not fused:
        raise ValueError('Direct MLP requires a fused gate/up policy')
    if manifest['fused_swiglu_quant'] and not manifest['direct_mlp']:
        raise ValueError('Fused SwiGLU quantization requires the direct MLP route')
    for item in entries:
        name = item['name']
        if item.get('precision') == 'bf16':
            manifest['preserved_bf16_modules'].append(name)
            continue
        if fused and name.endswith('up_proj'):
            continue
        original = model.get_submodule(name)
        parent, child = name.rsplit('.', 1)
        parent_module = model.get_submodule(parent)
        if fused and name.endswith('gate_proj'):
            up = parent_module.up_proj
            combined = nn.Linear(original.in_features, original.out_features + up.out_features,
                                 bias=False, dtype=torch.bfloat16)
            combined.weight.data = torch.cat([original.weight.detach(), up.weight.detach()], dim=0)
            if item.get('precision') == 'weight_only_int8':
                quantized = NPUPrepackedWeightOnlyLinear(combined, manifest, name, item.get('group_size',32))
            else:
                quantized = NPUDynamicInt8Linear(combined, item['smooth'], manifest, name)
            parent_module.add_module('_dynamic_gate_up', quantized)
            parent_module.gate_proj = DynamicGateUpSlice(quantized, 0)
            parent_module.up_proj = DynamicGateUpSlice(quantized, 1)
        elif item.get('precision') == 'weight_only_int8':
            quantized = NPUPrepackedWeightOnlyLinear(original, manifest, name, item.get('group_size',32))
            parent_module._modules[child] = quantized
        else:
            quantized = NPUDynamicInt8Linear(original, item['smooth'], manifest, name)
            parent_module._modules[child] = quantized
        manifest['modules'].append(dict(name=name, alpha=item['alpha'],
                                       in_features=original.in_features, out_features=original.out_features,
                                       weight_stride=list(quantized.qweight.stride()),
                                       precision=item.get('precision','dynamic_int8'),
                                       calibration_relative_l2=item['relative_l2']))
    if manifest['direct_mlp']:
        for item in entries:
            if item['name'].endswith('gate_proj'):
                mlp = model.get_submodule(item['name'].rsplit('.', 1)[0])
                mlp.forward = MethodType(direct_dynamic_mlp_forward, mlp)
                mlp._dynamic_direct_mlp = True
                if manifest['fused_swiglu_quant']:
                    if isinstance(mlp.down_proj, NPUDynamicInt8Linear):
                        mlp.down_proj.register_buffer('fused_smooth_scale', mlp.down_proj.smooth.float().reshape(1, -1))
                    mlp._dynamic_fused_swiglu_quant = True
                    if silu_backend == 'triton_rounded':
                        from experiments.robot.ascend_mlp_ops import rounded_swiglu
                        mlp._dynamic_rounded_swiglu_fn = rounded_swiglu
    manifest['native_swiglu_quant_projection_count'] = sum(
        item['name'].endswith('down_proj') and item.get('precision','dynamic_int8')=='dynamic_int8'
        for item in entries) if manifest['fused_swiglu_quant'] else 0
    if manifest['native_swiglu_quant_projection_count']:
        manifest['implementation'] += ' + torch_npu.npu_dequant_swiglu_quant'
        manifest['activation_rounding'] = 'native fused SwiGLU+quant; not bitwise identical to separate BF16 activation/mul'
    else:
        manifest['activation_rounding'] = 'separate BF16 SiLU/mul before down projection'
    if manifest['precision_counts']['weight_only_int8']:
        manifest['implementation'] += ' + prepacked torch_npu.npu_weight_quant_batchmatmul'
        if manifest['precision_counts']['dynamic_int8']:
            manifest['activation_dtype'] = 'mixed: dynamic int8 gate/up, BF16 weight-only down'
        else:
            manifest['activation_dtype'] = 'bfloat16'
            manifest['profile'] = 'verifier_mlp_packed_weight_only_int8'
            manifest['implementation'] = 'prepacked torch_npu.npu_weight_quant_batchmatmul'
    if silu_backend == 'triton_rounded':
        manifest['implementation'] += ' + Ascend Triton rounded SwiGLU'
        manifest['activation_rounding'] = 'explicit BF16 rounding after SiLU and multiplication'
    after = storage_bytes(model)
    manifest.update(storage_bytes_after_by_dtype=after, module_count=len(manifest['modules']),
                    replaced_projection_count=len(entries)-len(manifest['preserved_bf16_modules']),
                    unique_storage_saving_bytes=sum(before.values()) - sum(after.values()))
    model._flagos_npu_quant_manifest = manifest
    model._flagos_w8a8_manifest = manifest
    model._flagos_w8a16_enabled = True  # Existing bridge uses this flag to exclude incompatible Linear fusion.
    print(f"[KERV/NPU W8A8] converted {manifest['replaced_projection_count']} projections; "
          f"preserved {len(manifest['preserved_bf16_modules'])} BF16 projections; "
          f"saving {manifest['unique_storage_saving_bytes']} bytes", flush=True)
    return manifest
