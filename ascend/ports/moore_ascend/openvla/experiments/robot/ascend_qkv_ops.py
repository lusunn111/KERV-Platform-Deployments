"""CPU prepacking for verifier QKV groups, plus an undeployed dispatch probe."""
import torch


def verifier_qkv_layer_ids(model, selection="all"):
    """Resolve a strict zero-based layer list, or half-open start:stop range."""
    count = len(model.base_model.language_model.model.layers)
    if selection == "all":
        return list(range(count))
    if ":" in selection:
        parts = selection.split(":")
        if len(parts) != 2 or not all(x.isdigit() for x in parts):
            raise ValueError("QKV layers must be all, start:stop, or comma-separated indices")
        start, stop = map(int, parts)
        if not 0 <= start < stop <= count:
            raise ValueError("QKV layer range is empty or out of bounds")
        indices = list(range(start, stop))
    else:
        if not selection or not all(x.isdigit() for x in selection.split(",")):
            raise ValueError("QKV layer list must be nonempty zero-based indices")
        indices = [int(x) for x in selection.split(",")]
        if len(indices) != len(set(indices)) or any(x >= count for x in indices):
            raise ValueError("QKV layer list contains duplicates or out-of-bounds indices")
    return sorted(indices)


def verifier_qkv_scope(model, selection="all"):
    """Expose only selected verifier attention modules to the generic fuser.

    This temporary view shares modules; it never clones or transfers weights.
    Unselected verifier layers and the drafter cannot be fused through this view.
    """
    indices = verifier_qkv_layer_ids(model, selection)
    if selection == "all":
        return model.base_model.language_model
    layers = model.base_model.language_model.model.layers
    return torch.nn.ModuleDict({f"layer_{i}": layers[i].self_attn for i in indices})


def pack_cpu_verifier_qkv(model, rows, selection="all", backend="concat"):
    """Pack before device placement; original Q/K/V never acquire NPU storage."""
    from KERVRuntimeOptimization.adaptive_linear_fusion import _replace_group, _parse_rows
    if backend not in {"concat", "bmm"}:
        raise ValueError("Unsupported QKV projection backend")
    all_layers=model.base_model.language_model.model.layers
    indices=verifier_qkv_layer_ids(model,selection)
    layers=[all_layers[i] for i in indices]
    for layer in layers:
        for name in ['q_proj','k_proj','v_proj']:
            projection=getattr(layer.self_attn,name)
            if not isinstance(projection,torch.nn.Linear) or projection.weight.device.type!='cpu':
                raise ValueError('CPU QKV packing requires original CPU Linear weights')
            if projection.weight.dtype!=torch.bfloat16 or projection.bias is not None:
                raise ValueError('CPU QKV profile supports bias-free BF16 verifier projections')
        if backend.startswith("bmm") and len({getattr(layer.self_attn,n).out_features for n in ['q_proj','k_proj','v_proj']}) != 1:
            raise ValueError("Batched QKV requires equal projection output widths")
    parsed=_parse_rows(rows)
    if not parsed:raise ValueError('CPU QKV requires a nonempty row allowlist')
    if backend.startswith("bmm") and any(not 257 <= m <= 320 for m in parsed):
        raise ValueError("Batched QKV profile is validated only for rows 257..320")
    groups=[]
    for index,layer in zip(indices,layers):
        group=_replace_group(layer.self_attn,('q_proj','k_proj','v_proj'),'qkv_linear',parsed)
        group.prepacked_on_cpu=True
        group.projection_backend=backend
        group.verifier_layer_index=index
        groups.append(group)
    return groups


def transfer_packed_qkv(groups,device):
    # FlagScale's inference-only groups own tensors outside nn.Parameters.
    for group in groups:
        group.weight=group.weight.to(device)
        if group.bias is not None:group.bias=group.bias.to(device)
        # Views share the sole packed allocation. Create them after the final
        # transfer so fallback projections never dispatch narrow per action.
        group.cached_weight_slices=tuple(group.weight.narrow(0,offset,width)
            for offset,width in zip(group.offsets,group.output_sizes))
        if group.bias is not None:
            group.cached_bias_slices=tuple(group.bias.narrow(0,offset,width)
                for offset,width in zip(group.offsets,group.output_sizes))
        if getattr(group,'projection_backend','concat').startswith('bmm'):
            group.batched_weight=group.weight.view(3,group.output_sizes[0],group.in_features).transpose(1,2)
        if getattr(group,'projection_backend','concat')=='bmm_scripted':
            group.batched_projection=native_batched_qkv


@torch.jit.script
def native_qkv(input:torch.Tensor,q:torch.Tensor,k:torch.Tensor,v:torch.Tensor):
    return (torch.nn.functional.linear(input,q),
            torch.nn.functional.linear(input,k),
            torch.nn.functional.linear(input,v))


@torch.jit.script
def native_batched_qkv(value:torch.Tensor,batched_weight:torch.Tensor):
    """Batch=3 GEMM with views dispatched inside TorchScript, no weight copy."""
    if value.dim()==3 and value.size(0)==1:
        combined=torch.bmm(value.expand(3,-1,-1),batched_weight).unsqueeze(1)
    else:
        flat=value.reshape(-1,value.size(-1))
        result=torch.bmm(flat.unsqueeze(0).expand(3,-1,-1),batched_weight)
        shape=[3]+list(value.size())
        shape[-1]=batched_weight.size(-1)
        combined=result.reshape(shape)
    outputs=combined.unbind(0)
    return outputs[0],outputs[1],outputs[2]
