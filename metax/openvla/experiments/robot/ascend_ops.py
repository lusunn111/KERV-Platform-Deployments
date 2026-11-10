"""Ascend model hooks for the public KERV operator namespace."""

from types import MethodType
import torch


def install_npu_stage_graphs(model):
    """Reuse KERV's bounded stage cache with the native NPUGraph API."""
    if not hasattr(model, "ea_layer") or next(model.parameters()).device.type != "npu":
        raise ValueError("NPU stage graphs require a speculative NPU model")
    from openvla.prismatic.extern.hf.flagos_stage_graph import StageGraphCache
    model._flagos_shared_graph_pool = False
    model._flagos_prompt_stage_graph = StageGraphCache(model, "auto", "prompt", max_entries=2)
    model._flagos_draft_stage_graph = StageGraphCache(model, "auto", "draft", max_entries=8)
    model._flagos_npu_verifier_stage_graph = StageGraphCache(model, "auto", "verify", max_entries=16)
    model.ea_layer._flagos_draft_stage_graph_runner = model._run_draft_stage_graph
    model._flagos_npu_graphs_enabled = True


def _gems_attention_forward(self, hidden_states, attention_mask=None, position_ids=None,
                            past_key_value=None, output_attentions=False, use_cache=False,
                            cache_position=None, **kwargs):
    if output_attentions or self.config.pretraining_tp != 1:
        return self._kerv_npu_original_forward(hidden_states, attention_mask=attention_mask,
                position_ids=position_ids, past_key_value=past_key_value,
                output_attentions=output_attentions, use_cache=use_cache,
                cache_position=cache_position, **kwargs)
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
    bsz, count, _ = hidden_states.shape
    query = self.q_proj(hidden_states).view(bsz, count, self.num_heads, self.head_dim).transpose(1, 2)
    key = self.k_proj(hidden_states).view(bsz, count, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value = self.v_proj(hidden_states).view(bsz, count, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    cos, sin = self.rotary_emb(value, position_ids)
    query, key = apply_rotary_pos_emb(query, key, cos, sin)
    past_key_value = getattr(self, "past_key_value", past_key_value)
    if past_key_value is not None:
        key, value = past_key_value.update(key, value, self.layer_idx,
                            {"sin": sin, "cos": cos, "cache_position": cache_position})
    key = repeat_kv(key, self.num_key_value_groups)
    value = repeat_kv(value, self.num_key_value_groups)
    mask = None if attention_mask is None else attention_mask[:, :, :, :key.shape[-2]]
    if getattr(self, "_kerv_npu_attention_backend", "flag_gems") == "cann":
        import torch_npu
        blocked = None
        if mask is not None:
            blocked = ~mask if mask.dtype == torch.bool else mask < 0
            blocked = blocked.contiguous()
        elif count > 1:
            blocked = torch.ones(count, key.shape[-2], dtype=torch.bool, device=query.device).triu(1)
        out = torch_npu.npu_fusion_attention(query.contiguous(), key.contiguous(), value.contiguous(),
                    self.num_heads, "BNSD", atten_mask=blocked, scale=self.head_dim**-0.5,
                    keep_prob=1.0)[0]
        self._kerv_cann_attention_hits = getattr(self, "_kerv_cann_attention_hits", 0) + 1
        return self.o_proj(out.transpose(1, 2).contiguous().view(bsz, count, self.hidden_size)), None, past_key_value
    import flag_gems
    if mask is not None and mask.dtype == torch.bool:
        # FlagGems Ascend boolean masks mark blocked elements. Use additive
        # masks to preserve PyTorch's True=allowed convention unambiguously.
        mask = torch.zeros_like(mask, dtype=torch.float32).masked_fill(~mask, float("-inf"))
    if mask is not None:
        # The Ascend FlagGems kernel directly uses the head stride; broadcast
        # explicitly so a shared HF [B,1,Q,K] mask has head stride zero.
        mask = mask.expand(bsz, self.num_heads, count, key.shape[-2])
    out = flag_gems.scaled_dot_product_attention(query, key, value, attn_mask=mask,
                            dropout_p=0.0, is_causal=mask is None and count > 1)
    self._kerv_gems_attention_hits = getattr(self, "_kerv_gems_attention_hits", 0) + 1
    return self.o_proj(out.transpose(1, 2).contiguous().view(bsz, count, self.hidden_size)), None, past_key_value


def install_npu_model_ops(model, enabled=True):
    """Patch verifier RMSNorm and verifier/drafter SwiGLU; preserve methods."""
    include = getattr(model, "_flagos_embodied_ops_include", set())
    patched = []
    for name, module in model.named_modules():
        original = getattr(module, "_kerv_npu_original_forward", None)
        if not enabled:
            if original is not None:
                module.forward = original
            continue
        weight = getattr(module, "weight", None)
        if module.__class__.__name__.endswith("RMSNorm") and isinstance(weight, torch.Tensor):
            if weight.device.type != "npu" or "kerv_rms_norm" not in include:
                continue
            # Preserve the drafter's original SpecVLA normalization method;
            # its BF16 rounding boundary is part of speculative acceptance.
            if module.__class__.__module__ != "transformers.models.llama.modeling_llama":
                continue

            def norm_forward(self, hidden):
                return torch.ops.flagos_embodied.kerv_rms_norm(hidden, self.weight, float(self.variance_epsilon))

            forward = norm_forward
        elif module.__class__.__name__ == "LlamaMLP" and all(hasattr(module, n) for n in ("gate_proj", "up_proj", "down_proj")):
            if module.gate_proj.weight.device.type != "npu" or "kerv_silu_mul" not in include:
                continue
            if getattr(getattr(module, "config", None), "pretraining_tp", 1) != 1:
                continue

            def mlp_forward(self, hidden):
                gate = self.gate_proj(hidden)
                up = self.up_proj(hidden)
                return self.down_proj(torch.ops.flagos_embodied.kerv_silu_mul(gate, up))

            forward = mlp_forward
        elif (module.__class__.__module__ == "transformers.models.llama.modeling_llama"
              and module.__class__.__name__ in {"LlamaAttention", "LlamaSdpaAttention"}
              and getattr(model, "_flagos_npu_attention_backend", "native") in {"flag_gems", "cann"}):
            if module.q_proj.weight.device.type != "npu":
                continue
            module._kerv_npu_attention_backend = model._flagos_npu_attention_backend
            forward = _gems_attention_forward
        else:
            continue
        if original is None:
            module._kerv_npu_original_forward = module.forward
        module.forward = MethodType(forward, module)
        patched.append(name)
    model._kerv_npu_patched_modules = patched
    return patched
