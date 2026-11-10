"""Opt-in Ascend Triton SwiGLU probe with an explicit BF16 rounding boundary.

No production dispatch is changed by importing this module. The intermediate
SiLU is rounded to BF16 before multiplication, matching the unfused graph's
rounding locations; exp implementations still require numerical validation.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _rounded_swiglu(combined, output, count, width: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = index // width
    col = index % width
    gate = tl.load(combined + row * (2 * width) + col, index < count, other=0).to(tl.float32)
    up = tl.load(combined + row * (2 * width) + width + col, index < count, other=0).to(tl.float32)
    silu = (gate / (1.0 + tl.exp(-gate))).to(tl.bfloat16).to(tl.float32)
    tl.store(output + index, silu * up, index < count)


@triton.jit
def _rounded_swiglu_rows(combined, output, width: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    gate = tl.load(combined + row * (2 * width) + col, col < width, other=0).to(tl.float32)
    up = tl.load(combined + row * (2 * width) + width + col, col < width, other=0).to(tl.float32)
    silu = (gate / (1.0 + tl.exp(-gate))).to(tl.bfloat16).to(tl.float32)
    tl.store(output + row * width + col, silu * up, col < width)


def rounded_swiglu(combined, block=4096):
    if combined.device.type != 'npu' or combined.dtype != torch.bfloat16 or not combined.is_contiguous():
        raise ValueError('Expected contiguous NPU BF16 gate/up input')
    if combined.ndim < 2 or combined.shape[-1] % 2:
        raise ValueError('Expected paired gate/up features')
    width = combined.shape[-1] // 2
    output = torch.empty((*combined.shape[:-1], width), dtype=combined.dtype, device=combined.device)
    _rounded_swiglu_rows[(output.numel()//width, triton.cdiv(width,block))](
        combined, output, width, BLOCK=block)
    return output
