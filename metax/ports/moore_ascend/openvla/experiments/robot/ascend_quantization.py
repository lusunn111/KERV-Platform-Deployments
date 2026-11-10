"""Explicit NPU W8A16 MLP profile, quantized on CPU before NPU residency.

Activations remain BF16; symmetric scales are BF16 per output row and K group.
Original checkpoints are read-only. No BF16 shadow copy is retained on device.
"""
import hashlib
import json
from pathlib import Path
import torch
from torch import nn


class NPUWeightOnlyLinear(nn.Module):
    def __init__(self, linear, manifest, name, group_size=128, clip_ratio=1.0):
        super().__init__()
        if linear.weight.device.type!='cpu':
            raise RuntimeError('Quantize on CPU before device transfer; late conversion would hide load-time peaks')
        if linear.weight.dtype!=torch.bfloat16:
            raise ValueError('This profile requires original BF16 weights')
        self.in_features=linear.in_features;self.out_features=linear.out_features
        if group_size not in [32,64,128] or group_size>=self.in_features or self.in_features%group_size:
            raise ValueError('Group size must be 32/64/128, divide K, and be smaller than K')
        self.group_size=group_size
        if not 0<clip_ratio<=1:raise ValueError('clip_ratio must be in (0,1]')
        weights=linear.weight.detach().float().reshape(self.out_features,-1,group_size)
        scales=(weights.abs().amax(-1)*clip_ratio/127).clamp_min(1e-8).to(torch.bfloat16)
        q=(weights/scales.float().unsqueeze(-1)).round().clamp(-127,127).to(torch.int8)
        self.register_buffer('qweight',q.reshape(self.out_features,self.in_features).contiguous())
        self.register_buffer('antiquant_scale',scales.contiguous())
        self.register_buffer('bias',None if linear.bias is None else linear.bias.detach().clone())
        self._manifest=manifest;self._module_name=name

    def forward(self,x):
        import torch_npu
        if x.device.type!='npu' or x.dtype!=self.antiquant_scale.dtype:
            raise RuntimeError(f'W8A16 expects NPU BF16 activations, got {x.device}/{x.dtype}')
        shape=x.shape
        result=torch_npu.npu_weight_quant_batchmatmul(x.reshape(-1,self.in_features),
            self.qweight.t(),self.antiquant_scale.t(),bias=self.bias,
            antiquant_group_size=self.group_size)
        self._manifest['forward_calls']+=1
        return result.reshape(*shape[:-1],self.out_features)


def storage_bytes(model):
    seen=set();values={}
    for _,tensor in list(model.named_parameters())+list(model.named_buffers()):
        storage=tensor.untyped_storage();key=(str(tensor.device),storage.data_ptr())
        if key in seen:continue
        seen.add(key);dtype=str(tensor.dtype)
        values[dtype]=values.get(dtype,0)+storage.nbytes()
    return values


def weight_sha256(weight):
    return hashlib.sha256(weight.detach().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def apply_cpu_mlp_int8(model,group_size=128,layer_range='all',policy_path=None):
    if group_size not in [32,64,128]:raise ValueError('Supported group sizes: 32,64,128')
    config=model.base_model.language_model.config
    if layer_range=='all':
        selected_layers=list(range(config.num_hidden_layers))
    else:
        start,end=map(int,layer_range.split(':'))
        if not 0<=start<end<=config.num_hidden_layers:raise ValueError('Invalid quantized layer range')
        selected_layers=list(range(start,end))
    if getattr(config,'pretraining_tp',1)!=1:raise ValueError('pretraining_tp must be 1')
    before=storage_bytes(model)
    manifest={'enabled':True,'implementation':'torch_npu.npu_weight_quant_batchmatmul',
        'profile':f'verifier_mlp_w8a16_g{group_size}','weight_dtype':'int8','activation_dtype':'bfloat16',
        'scale_dtype':'bfloat16','scheme':'symmetric per-output-row, per-K-group',
        'group_size':group_size,'layer_indices':selected_layers,'converted_on':'cpu before NPU transfer','fallback':False,
        'requested_groups':['verify_mlp'],'selected_groups':['verify_mlp'],
        'precision_gate':'task quality, not bitwise parity against BF16; user approved quantization 2026-10-26',
        'modules':[],'forward_calls':0,'storage_bytes_before_by_dtype':before}
    prefix='base_model.language_model.model.layers.'
    candidates=[(name,layer) for name,layer in model.named_modules() if isinstance(layer,nn.Linear)
        and name.startswith(prefix) and '.mlp.' in name and name.rsplit('.',1)[-1] in ['gate_proj','up_proj','down_proj']]
    policy=None;settings={}
    if policy_path:
        policy=json.loads(Path(policy_path).read_text())
        if policy.get('version')!=1:raise ValueError('Unsupported quantization policy version')
        for item in policy['modules']:
            if item['name'] in settings:raise ValueError('Duplicate module in quantization policy')
            if item['group_size'] not in [32,64,128]:raise ValueError('Unsupported policy group size')
            settings[item['name']]=item
        if not settings or not set(settings)<={name for name,_ in candidates}:raise ValueError('Invalid or empty policy modules')
        candidates=[(name,layer) for name,layer in candidates if name in settings]
        selected_layers=sorted({int(name[len(prefix):].split('.')[0]) for name,_ in candidates})
        manifest.update(profile='verifier_mlp_w8a16_calibrated_policy',layer_indices=selected_layers,
            policy_sha256=hashlib.sha256(Path(policy_path).read_bytes()).hexdigest(),
            calibration=policy.get('calibration'),group_size=None)
    else:
        candidates=[(name,layer) for name,layer in candidates if int(name[len(prefix):].split('.')[0]) in selected_layers]
    expected=len(settings) if policy else len(selected_layers)*3
    if len(candidates)!=expected:raise RuntimeError(f'Expected {expected} verifier MLP linears, found {len(candidates)}')
    for name,linear in candidates:
        setting=settings.get(name,{})
        if policy and weight_sha256(linear.weight)!=setting.get('weight_sha256'):
            raise ValueError(f'Quantization policy weight hash mismatch: {name}')
        selected_group=setting.get('group_size',group_size);clip=setting.get('clip_ratio',1.0)
        quantized=NPUWeightOnlyLinear(linear,manifest,name,selected_group,clip)
        parent_path,child=name.rsplit('.',1)
        model.get_submodule(parent_path)._modules[child]=quantized
        manifest['modules'].append({'name':name,'in_features':linear.in_features,'out_features':linear.out_features,
            'group_size':selected_group,'clip_ratio':clip,
            'bf16_weight_bytes':linear.weight.numel()*linear.weight.element_size(),
            'int8_weight_bytes':quantized.qweight.numel(),
            'scale_bytes':quantized.antiquant_scale.numel()*quantized.antiquant_scale.element_size()})
    after=storage_bytes(model)
    manifest['storage_bytes_after_by_dtype']=after
    manifest['unique_storage_saving_bytes']=sum(before.values())-sum(after.values())
    manifest['module_count']=len(candidates)
    manifest['group_counts']={str(g):sum(x['group_size']==g for x in manifest['modules']) for g in [32,64,128]}
    model._flagos_npu_quant_manifest=manifest
    model._flagos_w8a16_manifest=manifest
    model._flagos_w8a16_enabled=True
    print(f"[KERV/NPU INT8] converted {len(candidates)} MLP linears on CPU; storage saving {manifest['unique_storage_saving_bytes']} bytes",flush=True)
    return manifest
