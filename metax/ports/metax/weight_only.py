"""Experimental MetaX W8A16 storage, per-row K-group32, native BF16 GEMM.

Only inference CPU-loaded verifier MLP weights are converted. One reusable
BF16 decode workspace is shared sequentially; no full BF16 device shadow.
This is lossy quantization, never labelled lossless BF16 compression.
"""
import torch
from torch import nn
import triton
import triton.language as tl


@triton.jit
def _decode(Q, S, W, N: tl.constexpr, GROUP: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    q = tl.load(Q + index, index < N, other=0).to(tl.float32)
    scale = tl.load(S + index // GROUP, index < N, other=0).to(tl.float32)
    tl.store(W + index, q * scale, index < N)


class Workspace(nn.Module):
    def __init__(self, capacity):
        super().__init__()
        self.register_buffer('weight', torch.empty(capacity, dtype=torch.bfloat16))


class WeightOnlyLinear(nn.Module):
    def __init__(self, original, workspace, manifest, group=32):
        super().__init__()
        assert original.weight.device.type == 'cpu'
        assert original.weight.dtype == torch.bfloat16 and original.bias is None
        self.out_features,self.in_features=original.weight.shape
        assert self.in_features % group == 0
        self.group=group
        source=original.weight.detach().float().reshape(self.out_features,-1,group)
        # BF16 scales are the actual stored reconstruction scales.
        scales=(source.abs().amax(-1)/127).clamp_min(1e-12).to(torch.bfloat16)
        quant=(source/scales.float().unsqueeze(-1)).round().clamp(-127,127).to(torch.int8)
        self.register_buffer('qweight',quant.reshape(self.out_features,self.in_features).contiguous())
        self.register_buffer('scales',scales.contiguous())
        # Shared workspace belongs to the parent; avoid registering duplicate ownership.
        object.__setattr__(self,'workspace',workspace)
        object.__setattr__(self,'manifest',manifest)

    def forward(self,x):
        if x.device.type != 'cuda' or not getattr(torch.version,'maca',None) or x.dtype != torch.bfloat16:
            raise ValueError('MetaX BF16 activation required')
        if hasattr(self,'packed_storage'):
            storage_key=(self.packed_storage.qweight.data_ptr(),self.packed_storage.scales.data_ptr())
            if getattr(self,'_view_storage_key',None)!=storage_key:
                object.__setattr__(self,'_qview',self.packed_storage.qweight.narrow(0,self.q_offset,self.weight_count).view(self.out_features,self.in_features))
                object.__setattr__(self,'_sview',self.packed_storage.scales.narrow(0,self.scale_offset,self.scale_count))
                self._view_storage_key=storage_key
            qweight=self._qview;scales=self._sview
        else:
            qweight=self.qweight;scales=self.scales
        count=qweight.numel()
        if getattr(self,'_workspace_ptr',None)!=self.workspace.weight.data_ptr():
            object.__setattr__(self,'_weight_view',self.workspace.weight[:count].view(self.out_features,self.in_features))
            self._workspace_ptr=self.workspace.weight.data_ptr()
        weight=self._weight_view
        _decode[(triton.cdiv(count,1024),)](qweight,scales,weight,count,self.group,1024)
        self.manifest['forward_calls']+=1
        return nn.functional.linear(x,weight)


class PackedStorage(nn.Module):
    def __init__(self,replacements):
        super().__init__()
        self.register_buffer('qweight',torch.cat([x.qweight.reshape(-1) for x in replacements]))
        self.register_buffer('scales',torch.cat([x.scales.reshape(-1) for x in replacements]))


def pack_cpu_storage(model,replacements):
    packed=PackedStorage(replacements)
    model.add_module('_metax_packed_storage',packed)
    q_offset=scale_offset=0
    for module in replacements:
        count=module.qweight.numel();scales=module.scales.numel()
        module.weight_count=count;module.scale_count=scales
        module.q_offset=q_offset;module.scale_offset=scale_offset
        # Views are resolved from the sole parent-owned device allocations after .to().
        del module._buffers['qweight'];del module._buffers['scales']
        object.__setattr__(module,'packed_storage',packed)
        q_offset+=count;scale_offset+=scales
    return packed


def apply_cpu_weight_only(model):
    # Independent starting policy. Quality must be measured on MetaX.
    layers=model.base_model.language_model.model.layers
    selected=[]
    for i in range(7,25):
        for name in ['gate_proj','up_proj','down_proj']:
            if name=='down_proj' and i in {15,16,19,20}:continue
            selected.append((i,name,getattr(layers[i].mlp,name)))
    manifest=dict(profile='metax_w8a16_group32',activation_dtype='bfloat16',
        weight_dtype='int8',scale_dtype='bfloat16',group_size=32,forward_calls=0,
        execution='MACA Triton weight reconstruction + vendor BF16 linear',modules=[],
        quality='experimental lossy; no quality claim before paired closed loop')
    workspace=Workspace(max(linear.weight.numel() for _,_,linear in selected))
    model.add_module('_metax_weight_workspace',workspace)
    replacements=[]
    for i,name,linear in selected:
        replacement=WeightOnlyLinear(linear,workspace,manifest)
        setattr(layers[i].mlp,name,replacement)
        replacements.append(replacement)
        manifest['modules'].append(dict(layer=i,projection=name,
            bf16_bytes=linear.weight.numel()*2,quantized_bytes=replacement.qweight.numel()+replacement.scales.numel()*2))
    pack_cpu_storage(model,replacements)
    manifest['packed_storage']=True
    manifest['workspace_bytes']=workspace.weight.numel()*2
    manifest['storage_saving_minus_workspace_bytes']=sum(m['bf16_bytes']-m['quantized_bytes'] for m in manifest['modules'])-manifest['workspace_bytes']
    model._metax_quant_manifest=manifest
    print(f"[MetaX W8A16] {len(selected)} verifier MLP matrices; packed INT8/scales; BF16 GEMM; experimental lossy profile",flush=True)
    return manifest
