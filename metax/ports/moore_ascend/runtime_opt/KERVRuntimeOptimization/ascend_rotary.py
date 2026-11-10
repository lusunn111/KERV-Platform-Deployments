"""Ascend RoPE: rotate all heads of a token in one vector tile.

The Moore kernel loops over heads, causing many small transfers on Ascend.
This tile follows the physical token/head/dimension layout of QKV outputs.
Sequence length and tensor strides remain runtime arguments to reuse binaries.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _rotary(q, k, cos, sin, positions, oq, ok,
            qh: tl.constexpr, qt: tl.constexpr, qd: tl.constexpr,
            kh: tl.constexpr, kt: tl.constexpr, kd: tl.constexpr,
            oh: tl.constexpr, ot: tl.constexpr, od: tl.constexpr,
            ph: tl.constexpr, pt: tl.constexpr, pd: tl.constexpr,
            ct: tl.constexpr, cd: tl.constexpr, st: tl.constexpr, sd: tl.constexpr,
            HEADS_Q: tl.constexpr, HEADS_K: tl.constexpr, D: tl.constexpr,
            DIRECT: tl.constexpr, BLOCK: tl.constexpr):
    token = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    head = offsets // D
    dim = offsets % D
    partner = (dim + D // 2) % D
    sign = tl.where(dim < D // 2, -1.0, 1.0)
    if DIRECT:
        pos = token
    else:
        pos = tl.load(positions + token).to(tl.int32)
    small = tl.arange(0, D)
    c = tl.gather(tl.load(cos + pos * ct + small * cd).to(tl.float32), dim, 0)
    s = tl.gather(tl.load(sin + pos * st + small * sd).to(tl.float32), dim, 0)
    x = tl.load(q + head * qh + token * qt + dim * qd, head < HEADS_Q, 0).to(tl.float32)
    r = tl.gather(x, head * D + partner, 0) * sign
    y = (x * c).to(q.dtype.element_ty).to(tl.float32) + (r * s).to(q.dtype.element_ty).to(tl.float32)
    tl.store(oq + head * oh + token * ot + dim * od, y, head < HEADS_Q)
    xk = tl.load(k + head * kh + token * kt + dim * kd, head < HEADS_K, 0).to(tl.float32)
    rk = tl.gather(xk, head * D + partner, 0) * sign
    yk = (xk * c).to(k.dtype.element_ty).to(tl.float32) + (rk * s).to(k.dtype.element_ty).to(tl.float32)
    tl.store(ok + head * ph + token * pt + dim * pd, yk, head < HEADS_K)


def fused_rotary(query, key, cos, sin, positions=None):
    if query.shape[0] != 1 or key.shape[0] != 1:
        raise ValueError('Ascend RoPE requires batch=1')
    direct = positions is None
    d = query.shape[-1]
    cos = cos.reshape(-1, d)
    sin = sin.reshape(-1, d)
    if positions is not None:
        positions = positions.contiguous()
    oq = torch.empty((1, query.shape[2], query.shape[1], d), device=query.device, dtype=query.dtype).transpose(1, 2)
    ok = torch.empty((1, key.shape[2], key.shape[1], d), device=key.device, dtype=key.dtype).transpose(1, 2)
    _rotary[(query.shape[2],)](query, key, cos, sin, cos if direct else positions,
        oq, ok, *query.stride()[1:], *key.stride()[1:],
        *oq.stride()[1:], *ok.stride()[1:],
        *cos.stride(), *sin.stride(),
        HEADS_Q=query.shape[1], HEADS_K=key.shape[1], D=d,
        DIRECT=direct, BLOCK=triton.next_power_of_2(max(query.shape[1],key.shape[1])*d),
        num_warps=4, enable_fp_fusion=False)
    return oq, ok
