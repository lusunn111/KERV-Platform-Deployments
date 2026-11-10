"""Ascend adapters using FlagGems and its active FlagTree/Triton backend."""

import torch
import triton
import triton.language as tl
from flag_gems.utils import pointwise_dynamic


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def _silu_mul_with_rounding(x, y):
    values = x.to(tl.float32)
    silu = tl.fdiv(values, 1.0 + tl.exp(-values))
    # Match LLaMA's separate BF16 SiLU output before multiplying Up.
    return silu.to(x.dtype).to(tl.float32) * y.to(tl.float32)


def silu_mul_out(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    _silu_mul_with_rounding(gate, up, out0=gate)
    return gate


def rms_norm(hidden: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    import flag_gems
    return flag_gems.rms_norm(hidden, [hidden.shape[-1]], weight, eps)
