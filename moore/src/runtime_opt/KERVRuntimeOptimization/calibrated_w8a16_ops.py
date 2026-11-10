"""Experimental MUSA W8A16 operators with original-BF16 protected groups.

The quantization policy is unchanged. Every INT8 tile is rounded to BF16 before
the dot product; GEMM accumulation can still differ from the native library.
Preparation belongs outside timed inference and does not cache decoded weights.
"""
import math
import torch
import triton
import triton.language as tl


def prepare(module):
    """Build the tiny group-to-exception lookup once, without forward-time sync."""
    groups = math.ceil(module.in_features / module.group_size)
    ids = module.protected_groups.detach().cpu().long()
    if len(ids) != len(torch.unique(ids)) or bool(((ids < 0) | (ids >= groups)).any()):
        raise ValueError('Invalid protected groups')
    mapping = torch.full((groups,), -1, dtype=torch.int32)
    mapping[ids] = torch.arange(len(ids), dtype=torch.int32)
    module.register_buffer('_w8_group_map', mapping.to(module.qweight.device), persistent=False)
    module._w8_identity_inverse = bool((module.inverse_channel_scales.detach().cpu() == 1).all())


@triton.jit
def _decode(Q, S, I, P, V, O, K: tl.constexpr, N: tl.constexpr,
            G: tl.constexpr, NG: tl.constexpr, NP: tl.constexpr,
            IDENTITY: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    row, column = i // K, i % K
    valid = i < N
    q = tl.load(Q+i, valid, other=0).to(tl.float32)
    s = tl.load(S+row*NG+column//G, valid, other=0)
    w = q*s
    if not IDENTITY:
        inv = tl.load(I+column, valid, other=0)
        w = w*inv
    w = w.to(tl.bfloat16)
    if NP:
        slot = tl.load(P+column//G, valid, other=-1)
        original = tl.load(V+(row*NP+slot)*G+column%G,
                           valid & (slot >= 0), other=0)
        w = tl.where(slot >= 0, original, w)
    tl.store(O+i, w, valid)


def decode(module):
    out = torch.empty_like(module.qweight, dtype=torch.bfloat16)
    _decode[(triton.cdiv(out.numel(), 1024),)](
        module.qweight, module.scales, module.inverse_channel_scales,
        module._w8_group_map, module.protected_values, out,
        module.in_features, out.numel(), module.group_size, module.scales.shape[1],
        module.protected_groups.numel(), module._w8_identity_inverse, 1024,
        num_warps=4, enable_fp_fusion=False)
    return out


@triton.jit
def _decode_groups(Q, S, I, P, V, O, K: tl.constexpr, N: tl.constexpr,
                   G: tl.constexpr, NG: tl.constexpr, NP: tl.constexpr,
                   IDENTITY: tl.constexpr, BR: tl.constexpr, BG: tl.constexpr):
    group = tl.program_id(0)
    rows = tl.program_id(1)*BR+tl.arange(0, BR)
    offsets = tl.arange(0, BG)
    cols = group*G+offsets
    mask = (rows[:, None] < N) & (cols[None, :] < K) & (offsets[None, :] < G)
    slot = tl.load(P+group)
    if slot >= 0:
        w = tl.load(V+(rows[:, None]*NP+slot)*G+offsets[None, :], mask, other=0)
    else:
        q = tl.load(Q+rows[:, None]*K+cols[None, :], mask, other=0).to(tl.float32)
        s = tl.load(S+rows*NG+group, rows < N, other=0)
        w = q*s[:, None]
        if not IDENTITY:
            w = w*tl.load(I+cols, (cols < K) & (offsets < G), other=0)[None, :]
        w = w.to(tl.bfloat16)
    tl.store(O+rows[:, None]*K+cols[None, :], w, mask)


def decode_groups(module, block_rows=32):
    out = torch.empty_like(module.qweight, dtype=torch.bfloat16)
    _decode_groups[(module.scales.shape[1], triton.cdiv(module.out_features, block_rows))](
        module.qweight, module.scales, module.inverse_channel_scales,
        module._w8_group_map, module.protected_values, out, module.in_features,
        module.out_features, module.group_size, module.scales.shape[1],
        module.protected_groups.numel(), module._w8_identity_inverse, block_rows,
        triton.next_power_of_2(module.group_size), num_warps=4, enable_fp_fusion=False)
    return out


@triton.jit
def _decode_rows(Q, S, I, P, V, O, K: tl.constexpr, N: tl.constexpr,
                 G: tl.constexpr, NG: tl.constexpr, NP: tl.constexpr,
                 IDENTITY: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(1)
    col = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
    mask = col < K
    q = tl.load(Q+row*K+col, mask, other=0).to(tl.float32)
    scale = tl.load(S+row*NG+col//G, mask, other=0)
    w = q*scale
    if not IDENTITY:
        w = w*tl.load(I+col, mask, other=0)
    w = w.to(tl.bfloat16)
    if NP:
        slot = tl.load(P+col//G, mask, other=-1)
        original = tl.load(V+(row*NP+slot)*G+col%G, mask & (slot >= 0), other=0)
        w = tl.where(slot >= 0, original, w)
    tl.store(O+row*K+col, w, mask)


def decode_rows(module, block=1024):
    out = torch.empty_like(module.qweight, dtype=torch.bfloat16)
    _decode_rows[(triton.cdiv(module.in_features, block), module.out_features)](
        module.qweight, module.scales, module.inverse_channel_scales,
        module._w8_group_map, module.protected_values, out, module.in_features,
        module.out_features, module.group_size, module.scales.shape[1],
        module.protected_groups.numel(), module._w8_identity_inverse, block,
        num_warps=4, enable_fp_fusion=False)
    return out


@triton.jit
def _gemm(A, Q, S, I, P, V, BIAS, O, M: tl.constexpr, N: tl.constexpr,
          K: tl.constexpr, G: tl.constexpr, NG: tl.constexpr, NP: tl.constexpr,
          IDENTITY: tl.constexpr, HAS_BIAS: tl.constexpr,
          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0)*BM + tl.arange(0, BM)
    cols = tl.program_id(1)*BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for block in range(tl.cdiv(K, BK)):
        k = block*BK+ks
        a = tl.load(A+rows[:, None]*K+k[None, :],
                    (rows[:, None] < M) & (k[None, :] < K), other=0)
        mask = (cols[None, :] < N) & (k[:, None] < K)
        q = tl.load(Q+cols[None, :]*K+k[:, None], mask, other=0).to(tl.float32)
        s = tl.load(S+cols[None, :]*NG+k[:, None]//G, mask, other=0)
        w = q*s
        if not IDENTITY:
            inv = tl.load(I+k, k < K, other=0)
            w = w*inv[:, None]
        w = w.to(tl.bfloat16)
        if NP:
            slot = tl.load(P+k//G, k < K, other=-1)
            original = tl.load(V+(cols[None, :]*NP+slot[:, None])*G+k[:, None]%G,
                               mask & (slot[:, None] >= 0), other=0)
            w = tl.where(slot[:, None] >= 0, original, w)
        acc = tl.dot(a, w, acc)
    if HAS_BIAS:
        acc += tl.load(BIAS+cols, cols < N, other=0)[None, :].to(tl.float32)
    tl.store(O+rows[:, None]*N+cols[None, :], acc,
             (rows[:, None] < M) & (cols[None, :] < N))


@triton.jit
def _gemv(A, Q, S, I, P, V, TMP, M: tl.constexpr, N: tl.constexpr,
          K: tl.constexpr, G: tl.constexpr, NG: tl.constexpr, NP: tl.constexpr,
          IDENTITY: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SPLIT: tl.constexpr):
    cols = tl.program_id(0)*BN+tl.arange(0, BN)
    part = tl.program_id(1)
    row = tl.program_id(2)
    k = part*BK+tl.arange(0, BK)
    mask = (cols[:, None] < N) & (k[None, :] < K)
    q = tl.load(Q+cols[:, None]*K+k[None, :], mask, other=0).to(tl.float32)
    s = tl.load(S+cols[:, None]*NG+k[None, :]//G, mask, other=0)
    w = q*s
    if not IDENTITY:
        w = w*tl.load(I+k, k < K, other=0)[None, :]
    w = w.to(tl.bfloat16)
    if NP:
        slot = tl.load(P+k//G, k < K, other=-1)
        original = tl.load(V+(cols[:, None]*NP+slot[None, :])*G+k[None, :]%G,
                           mask & (slot[None, :] >= 0), other=0)
        w = tl.where(slot[None, :] >= 0, original, w)
    a = tl.load(A+row*K+k, k < K, other=0).to(tl.float32)
    partial = tl.sum(w.to(tl.float32)*a[None, :], 1)
    tl.store(TMP+(row*SPLIT+part)*N+cols, partial, cols < N)


@triton.jit
def _reduce(TMP, BIAS, O, N: tl.constexpr, M: tl.constexpr, SPLIT: tl.constexpr,
            HAS_BIAS: tl.constexpr, BS: tl.constexpr, BN: tl.constexpr):
    cols = tl.program_id(0)*BN+tl.arange(0, BN)
    row = tl.program_id(1)
    part = tl.arange(0, BS)
    value = tl.load(TMP+(row*SPLIT+part[:, None])*N+cols[None, :],
                    (part[:, None] < SPLIT) & (cols[None, :] < N), other=0)
    total = tl.sum(value, 0)
    if HAS_BIAS:
        total += tl.load(BIAS+cols, cols < N, other=0).to(tl.float32)
    tl.store(O+row*N+cols, total, cols < N)


def linear(module, value, *, kind='gemm', config=None):
    if value.requires_grad:
        raise RuntimeError('W8A16 operators are inference-only')
    if value.dtype != torch.bfloat16 or value.device != module.qweight.device:
        raise ValueError('Expected colocated BF16 input')
    if value.shape[-1] != module.in_features:
        raise ValueError('Input feature mismatch')
    if not hasattr(module, '_w8_group_map'):
        raise RuntimeError('Call prepare() before inference')
    if module.bias is not None and kind in ('gemm', 'gemv'):
        # Preserve the native library's bias rounding/order, which differs from
        # an FP32 fused epilogue on this runtime. Llama MLPs have no bias.
        return torch.nn.functional.linear(value, decode_rows(module), module.bias)
    if kind == 'decode':
        return torch.nn.functional.linear(value, decode(module), module.bias)
    if kind == 'decode_groups':
        return torch.nn.functional.linear(value, decode_groups(module, (config or (32,))[0]), module.bias)
    if kind == 'decode_rows':
        return torch.nn.functional.linear(value, decode_rows(module, (config or (1024,))[0]), module.bias)
    x = value.reshape(-1, module.in_features).contiguous()
    m, k = x.shape; n = module.out_features
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    if m == 0:
        return out.reshape(*value.shape[:-1], n)
    common = (x, module.qweight, module.scales, module.inverse_channel_scales,
              module._w8_group_map, module.protected_values)
    dims = (m, n, k, module.group_size, module.scales.shape[1],
            module.protected_groups.numel(), module._w8_identity_inverse)
    if kind == 'gemv':
        bn, bk = config or (4, 1024)
        split = triton.cdiv(k, bk)
        temp = torch.empty((m, split, n), device=x.device, dtype=torch.float32)
        _gemv[(triton.cdiv(n, bn), split, m)](*common, temp, *dims, bn, bk, split,
                                           num_warps=4, enable_fp_fusion=False)
        _reduce[(triton.cdiv(n, 128), m)](temp, module.bias, out, n, m, split,
                module.bias is not None, triton.next_power_of_2(split), 128, num_warps=4)
    elif kind == 'gemm':
        bm, bn, bk = config or (16, 64, 64)
        _gemm[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
            *common, module.bias, out, *dims, module.bias is not None, bm, bn, bk,
            num_warps=4, enable_fp_fusion=False)
    else:
        raise ValueError('Unknown kernel kind')
    return out.reshape(*value.shape[:-1], n)
