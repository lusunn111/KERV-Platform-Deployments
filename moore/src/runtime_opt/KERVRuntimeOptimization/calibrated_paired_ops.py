"""Tiled mixed-weight decode and paired gate/up projection on MUSA.

Decode preserves the frozen policy's BF16 values. Combining native GEMMs may
change their reduction order; callers must validate output and model behavior.
No decoded weight is cached between calls.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _tile(Q, S, I, P, V, rows, cols, K: tl.constexpr, N: tl.constexpr,
          G: tl.constexpr, NG: tl.constexpr, NP: tl.constexpr,
          IDENTITY: tl.constexpr):
    valid = (rows[:, None] < N) & (cols[None, :] < K)
    q = tl.load(Q+rows[:, None]*K+cols[None, :], valid, other=0).to(tl.float32)
    scale = tl.load(S+rows[:, None]*NG+cols[None, :]//G, valid, other=0)
    w = q*scale
    if not IDENTITY:
        w = w*tl.load(I+cols, cols < K, other=0)[None, :]
    w = w.to(tl.bfloat16)
    if NP:
        slot = tl.load(P+cols//G, cols < K, other=-1)
        original = tl.load(V+(rows[:, None]*NP+slot[None, :])*G+cols[None, :] % G,
                           valid & (slot[None, :] >= 0), other=0)
        w = tl.where(slot[None, :] >= 0, original, w)
    return w


@triton.jit
def _decode_tiled(Q, S, I, P, V, O, K: tl.constexpr, N: tl.constexpr,
                  G: tl.constexpr, NG: tl.constexpr, NP: tl.constexpr,
                  IDENTITY: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr):
    rows = tl.program_id(1)*BR+tl.arange(0, BR)
    cols = tl.program_id(0)*BC+tl.arange(0, BC)
    w = _tile(Q, S, I, P, V, rows, cols, K, N, G, NG, NP, IDENTITY)
    tl.store(O+rows[:, None]*K+cols[None, :], w,
             (rows[:, None] < N) & (cols[None, :] < K))


@triton.jit
def _decode_pair(Q0, S0, I0, P0, V0, Q1, S1, I1, P1, V1, O,
                 K: tl.constexpr, N: tl.constexpr, G: tl.constexpr,
                 NG: tl.constexpr, NP0: tl.constexpr, NP1: tl.constexpr,
                 ID0: tl.constexpr, ID1: tl.constexpr,
                 BR: tl.constexpr, BC: tl.constexpr):
    rows = tl.program_id(1)*BR+tl.arange(0, BR)
    cols = tl.program_id(0)*BC+tl.arange(0, BC)
    if tl.program_id(2) == 0:
        w = _tile(Q0, S0, I0, P0, V0, rows, cols, K, N, G, NG, NP0, ID0)
        offset = 0
    else:
        w = _tile(Q1, S1, I1, P1, V1, rows, cols, K, N, G, NG, NP1, ID1)
        offset = N*K
    tl.store(O+offset+rows[:, None]*K+cols[None, :], w,
             (rows[:, None] < N) & (cols[None, :] < K))


def _buffers(module):
    if not hasattr(module, '_w8_group_map'):
        raise RuntimeError('Call calibrated_w8a16_ops.prepare before decoding')
    return (module.qweight, module.scales, module.inverse_channel_scales,
            module._w8_group_map, module.protected_values)


def decode_tiled(module, config=(4, 256, 4)):
    br, bc, warps = config
    out = torch.empty_like(module.qweight, dtype=torch.bfloat16)
    _decode_tiled[(triton.cdiv(module.in_features, bc),
                   triton.cdiv(module.out_features, br))](
        *_buffers(module), out, module.in_features, module.out_features,
        module.group_size, module.scales.shape[1], module.protected_groups.numel(),
        module._w8_identity_inverse, br, bc, num_warps=warps, enable_fp_fusion=False)
    return out


def decode_pair(gate, up, config=(4, 256, 4)):
    if (gate.in_features, gate.out_features, gate.group_size, gate.qweight.device) != (
            up.in_features, up.out_features, up.group_size, up.qweight.device):
        raise ValueError('Paired projections must share shape, group size and device')
    br, bc, warps = config
    k, n = gate.in_features, gate.out_features
    out = torch.empty((2*n, k), dtype=torch.bfloat16, device=gate.qweight.device)
    _decode_pair[(triton.cdiv(k, bc), triton.cdiv(n, br), 2)](
        *_buffers(gate), *_buffers(up), out, k, n, gate.group_size,
        gate.scales.shape[1], gate.protected_groups.numel(), up.protected_groups.numel(),
        gate._w8_identity_inverse, up._w8_identity_inverse, br, bc,
        num_warps=warps, enable_fp_fusion=False)
    return out


@triton.jit
def _row(Q, S, I, P, V, row, col, K: tl.constexpr, G: tl.constexpr,
         NG: tl.constexpr, NP: tl.constexpr, IDENTITY: tl.constexpr):
    valid = col < K
    q = tl.load(Q+row*K+col, valid, other=0).to(tl.float32)
    scale = tl.load(S+row*NG+col//G, valid, other=0)
    w = q*scale
    if not IDENTITY:
        w = w*tl.load(I+col, valid, other=0)
    w = w.to(tl.bfloat16)
    if NP:
        slot = tl.load(P+col//G, valid, other=-1)
        original = tl.load(V+(row*NP+slot)*G+col % G, valid & (slot >= 0), other=0)
        w = tl.where(slot >= 0, original, w)
    return w


@triton.jit
def _decode_pair_rows(Q0, S0, I0, P0, V0, Q1, S1, I1, P1, V1, O,
                      K: tl.constexpr, N: tl.constexpr, G: tl.constexpr,
                      NG: tl.constexpr, NP0: tl.constexpr, NP1: tl.constexpr,
                      ID0: tl.constexpr, ID1: tl.constexpr, BC: tl.constexpr):
    row = tl.program_id(1)
    col = tl.program_id(0)*BC+tl.arange(0, BC)
    if tl.program_id(2) == 0:
        w = _row(Q0, S0, I0, P0, V0, row, col, K, G, NG, NP0, ID0)
        offset = 0
    else:
        w = _row(Q1, S1, I1, P1, V1, row, col, K, G, NG, NP1, ID1)
        offset = N*K
    tl.store(O+offset+row*K+col, w, col < K)


def decode_pair_rows(gate, up, block=1024):
    if (gate.in_features, gate.out_features, gate.group_size, gate.qweight.device) != (
            up.in_features, up.out_features, up.group_size, up.qweight.device):
        raise ValueError('Paired projections must share shape, group size and device')
    k, n = gate.in_features, gate.out_features
    out = torch.empty((2*n, k), dtype=torch.bfloat16, device=gate.qweight.device)
    _decode_pair_rows[(triton.cdiv(k,block), n, 2)](
        *_buffers(gate), *_buffers(up), out, k, n, gate.group_size,
        gate.scales.shape[1], gate.protected_groups.numel(), up.protected_groups.numel(),
        gate._w8_identity_inverse, up._w8_identity_inverse, block,
        num_warps=4, enable_fp_fusion=False)
    return out


def paired_linear(gate, up, value, config=(4, 256, 4), row_block=None):
    if value.requires_grad:
        raise RuntimeError('Paired quantized projection is inference-only')
    if value.dtype != torch.bfloat16 or value.device != gate.qweight.device:
        raise ValueError('Expected matching BF16 input')
    if value.shape[-1] != gate.in_features:
        raise ValueError('Input feature mismatch')
    if gate.bias is not None or up.bias is not None:
        raise ValueError('Paired deployment currently requires bias-free projections')
    weight = (decode_pair_rows(gate, up, row_block) if row_block is not None
              else decode_pair(gate, up, config))
    combined = torch.nn.functional.linear(value, weight)
    return combined.split(gate.out_features, dim=-1)
