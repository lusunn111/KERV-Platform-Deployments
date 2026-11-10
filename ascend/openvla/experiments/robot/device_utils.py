"""Device helpers shared by the KERV inference entry points.

The upstream release assumes CUDA.  Ascend uses the same PyTorch tensor API
through ``torch_npu`` but exposes a separate device namespace and synchronizer.
This module keeps that distinction at the application boundary so model code
can continue to use tensors' own devices internally.
"""

from __future__ import annotations

import os
from typing import Optional

import torch


def _npu_available() -> bool:
    try:
        import torch_npu  # noqa: F401

        return bool(torch.npu.is_available() and torch.npu.device_count() > 0)
    except Exception:
        return False


def accelerator_type(preferred: Optional[str] = None) -> str:
    """Return the selected backend: ``cuda``, ``npu`` or ``cpu``."""

    value = str(preferred or os.environ.get("KERV_DEVICE_TYPE", "auto")).lower()
    if value in {"ascend", "npu"}:
        return "npu" if _npu_available() else "cpu"
    if value == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value not in {"auto", "cpu"}:
        raise ValueError(f"unsupported KERV device type: {value}")
    if value == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if _npu_available():
        return "npu"
    return "cpu"


def get_device(preferred: Optional[str] = None, index: int = 0) -> torch.device:
    backend = accelerator_type(preferred)
    if backend == "cpu":
        return torch.device("cpu")
    return torch.device(f"{backend}:{int(index)}")


def set_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.set_device(device)
    elif device.type == "npu":
        if not _npu_available():
            raise RuntimeError("torch_npu is required for an NPU device")
        torch.npu.set_device(device)


def synchronize(device: Optional[torch.device] = None) -> None:
    backend = device.type if device is not None else accelerator_type()
    if backend == "cuda":
        torch.cuda.synchronize(device)
    elif backend == "npu":
        torch.npu.synchronize(device)


def supports_async_host_transfer(device: torch.device) -> bool:
    # Pinning and non-blocking transfers are CUDA-specific in this release.
    return device.type == "cuda"
