"""Exact vectorized MUSA decode; compilation is explicit and outside inference."""
import ctypes
import hashlib
import os
from pathlib import Path
import subprocess

import torch
from . import calibrated_w8a16_ops as reference

_library = None


def load_library(build_directory):
    global _library
    source = Path(__file__).parent / 'csrc' / 'calibrated_decode.mu'
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    directory = Path(build_directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f'calibrated_decode_{digest[:16]}.so'
    if not target.exists():
        temporary = target.with_suffix(f'.{os.getpid()}.so')
        subprocess.run(['/usr/local/musa/bin/mcc', '-mtgpu', '-x', 'cuda',
                        '--cuda-gpu-arch=mp_31', '-O3', '-ffp-contract=off',
                        '-shared', '-fPIC', str(source), '-o', str(temporary),
                        '-L/usr/local/musa/lib', '-lmusart'], check=True)
        temporary.replace(target)
    library = ctypes.CDLL(str(target))
    library.kerv_decode8.argtypes = [ctypes.c_void_p]*6 + [ctypes.c_int]*6 + [ctypes.c_void_p]
    library.kerv_decode8.restype = ctypes.c_int
    _library = library
    return {'source_sha256':digest, 'library':str(target),
            'library_sha256':hashlib.sha256(target.read_bytes()).hexdigest()}


def decode_into(module, out, threads=128):
    if module.qweight.device.type != 'musa':
        raise ValueError('Native decoder requires MUSA weights')
    if not hasattr(module, '_w8_group_map'):
        raise RuntimeError('Call calibrated_w8a16_ops.prepare() before inference')
    if not all(t.is_contiguous() for t in (module.qweight, module.scales,
            module.inverse_channel_scales, module.protected_values)):
        raise ValueError('Contiguous quantized buffers required')
    if out.shape != module.qweight.shape or out.dtype != torch.bfloat16 or not out.is_contiguous():
        raise ValueError('Expected contiguous BF16 output matching qweight')
    if out.device != module.qweight.device:
        raise ValueError('Output must be on the weight device')
    if module.in_features % 8 or module.group_size % 8:
        # The scalar Triton implementation handles arbitrary group/tail shapes.
        out.copy_(reference.decode_rows(module))
        return out
    if _library is None:
        raise RuntimeError('Call load_library() before inference')
    if threads not in (64, 128, 256):
        raise ValueError('threads must be 64, 128 or 256')
    with torch.musa.device(module.qweight.device):
        stream = torch.musa.current_stream(module.qweight.device)._as_parameter_
        code = _library.kerv_decode8(
            module.qweight.data_ptr(), module.scales.data_ptr(),
            module.inverse_channel_scales.data_ptr(), module._w8_group_map.data_ptr(),
            module.protected_values.data_ptr(), out.data_ptr(), module.out_features,
            module.in_features, module.group_size, module.protected_groups.numel(),
            int(module._w8_identity_inverse), threads, stream)
    if code:
        raise RuntimeError(f'MUSA vector decode launch failed: {code}')
    return out


def decode(module, threads=128):
    return decode_into(module, torch.empty_like(module.qweight, dtype=torch.bfloat16), threads)
