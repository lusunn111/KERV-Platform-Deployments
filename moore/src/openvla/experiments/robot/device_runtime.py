"""Device selection for CUDA, Moore Threads MUSA, and CPU inference."""

import importlib.util
import os

import torch


def get_device():
    backend = os.environ.get("KERV_DEVICE_BACKEND", "auto").lower()
    if backend not in {"auto", "cuda", "musa", "cpu"}:
        raise ValueError(f"Unknown KERV_DEVICE_BACKEND: {backend}")
    if backend in {"auto", "musa"} and importlib.util.find_spec("torch_musa"):
        import torch_musa  # noqa: F401 -- registers the MUSA backend

        if torch.musa.is_available():
            return torch.device("musa:0")
    if backend == "musa":
        raise RuntimeError("MUSA requested, but torch_musa or a visible MUSA device is unavailable")
    if backend in {"auto", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda:0")
    if backend == "cuda":
        raise RuntimeError("CUDA requested, but no CUDA device is available")
    return torch.device("cpu")


def synchronize(device=None):
    device = get_device() if device is None else torch.device(device)
    if device.type in {"cuda", "musa"}:
        getattr(torch, device.type).synchronize(device)


def seed_device(seed):
    device = get_device()
    if device.type in {"cuda", "musa"}:
        getattr(torch, device.type).manual_seed_all(seed)
    if device.type == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def is_accelerator(value):
    return isinstance(value, torch.Tensor) and value.device.type in {"cuda", "musa"}


def get_accelerator(device=None):
    device = get_device() if device is None else torch.device(device)
    return torch.musa if device.type == "musa" else torch.cuda


def new_graph():
    backend = get_accelerator()
    return backend.MUSAGraph() if get_device().type == "musa" else backend.CUDAGraph()
