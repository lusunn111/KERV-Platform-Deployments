"""Explicit opt-in BF16 lossless storage backend for the existing FlagScale path.

The legacy w8a16 enable field is used only as a CPU-before-device load hook and
fusion exclusion. This backend performs NO INT8 quantization or reduced-precision
GEMM; the manifest and bit-exact accelerator audit identify its actual behavior.
"""
import atexit
import json
import os
from pathlib import Path


def register():
    from experiments.robot import flagos_ops
    from KERVRuntimeOptimization.lossless_bf16 import compress_model, LosslessBF16Linear, raw_hash
    if getattr(flagos_ops, '_lossless_registered', False): return
    original = flagos_ops.enable_w8a16_quantization

    def backend(model, enabled, groups=None, min_speedup=1.05, strict=False,
                manifest_path=None, backend='auto', group_size=128, policy_path=None):
        if not enabled:
            return original(model, False, groups=groups, manifest_path=manifest_path, backend=backend)
        selected = set(groups.split(',')) if isinstance(groups, str) else set(groups or [])
        if selected != {'verify_mlp', 'verify_attention'} or group_size != 128 or policy_path:
            raise ValueError('Lossless backend requires verify_mlp,verify_attention; group128; no INT8 policy')
        manifest = getattr(model, '_lossless_storage_manifest', None)
        if manifest is None:
            manifest = compress_model(model, group_size)
            manifest['legacy_enable_field'] = 'w8a16; dispatch hook only, not INT8'
            model._lossless_storage_manifest = manifest
            model._flagos_w8a16_manifest = manifest
            model._flagos_w8a16_enabled = True
            print('[lossless BF16 storage]', manifest['module_count'], 'projections; saved bytes',
                  manifest['estimated_weight_storage_saving_bytes'], flush=True)
        if manifest_path:
            path = Path(manifest_path);path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(manifest, indent=2) + '\n')
        return manifest

    flagos_ops.enable_w8a16_quantization = backend
    flagos_ops._lossless_registered = True
    from experiments.robot.libero import run_kerv_libero as evaluation
    evaluation.enable_w8a16_quantization = backend
    original_model = evaluation.get_model
    def audited_model(cfg):
        model = original_model(cfg)
        manifest = model._lossless_storage_manifest
        target = os.environ.get('KERV_ACCEPTANCE_CONTEXT')
        root = Path(target).parent if target else Path(os.environ.get('KERV_REPLAY_OUTPUT') or cfg.flagos_w8a16_manifest_path).parent
        if os.environ.get('KERV_REPLAY_OUTPUT'): root = Path(os.environ['KERV_REPLAY_OUTPUT'])
        root.mkdir(parents=True, exist_ok=True)
        modules = dict(model.named_modules()); proof = []
        for item in manifest['modules']:
            module = modules[item['name']]
            if module.packed.device.type != 'musa': raise RuntimeError('Lossless storage requires real MUSA buffers')
            digest = raw_hash(module.dequantize())
            if digest != item['original_sha256']: raise RuntimeError('Accelerator bit-exact check failed: ' + item['name'])
            proof.append({**item, 'gpu_decoded_sha256': digest, 'bit_exact': True,
                          'buffers': {n: {'dtype': str(t.dtype), 'device': str(t.device),
                                          'bytes': t.numel()*t.element_size()} for n,t in module.named_buffers()}})
        def emit(row):
            with (root/'compression.jsonl').open('a') as stream: stream.write(json.dumps(row)+'\n')
        emit({'phase':'loaded', 'implementation':manifest['implementation'], 'lossy_quantization':False,
              'gpu_bit_exact_verified_modules':len(proof), 'modules':proof})
        def final():
            emit({'phase':'final', 'module_count':len(proof), 'modules':[
                {'name':n,'forward_calls':m.forward_calls} for n,m in model.named_modules() if isinstance(m,LosslessBF16Linear)]})
        atexit.register(final)
        return model
    evaluation.get_model = audited_model


if __name__ == '__main__':
    register()
    from experiments.robot.libero.run_kerv_libero import eval_libero
    eval_libero()
