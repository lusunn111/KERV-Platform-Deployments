"""Single-allocation W8 dequantization; native MUSA/CUDA GEMM follows it."""
import torch
import triton
import triton.language as tl


@triton.jit
def _dequantize(Q, S, O, K: tl.constexpr, TOTAL: tl.constexpr,
                GROUP: tl.constexpr, GROUPS: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < TOTAL
    scale_index = (index // K) * GROUPS + (index % K) // GROUP
    q = tl.load(Q + index, mask=valid, other=0).to(tl.float32)
    scale = tl.load(S + scale_index, mask=valid, other=0)
    tl.store(O + index, q * scale, mask=valid)


def dequantize(qweight, scales, group_size, dtype):
    output = torch.empty(qweight.shape, device=qweight.device, dtype=dtype)
    _dequantize[(triton.cdiv(qweight.numel(), 1024),)](
        qweight, scales, output, qweight.shape[1], qweight.numel(),
        group_size, scales.shape[1], 1024, num_warps=4)
    return output
