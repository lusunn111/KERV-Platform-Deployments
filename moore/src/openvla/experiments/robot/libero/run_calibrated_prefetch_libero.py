"""Explicit two-slot decode prefetch; unchanged lossy policy/native GEMMs."""
import atexit
import json
import os
from pathlib import Path

def register():
    from experiments.robot.libero.run_calibrated_libero import register as policy_register,digest_tensor
    policy_register()
    from experiments.robot.libero import run_kerv_libero as evaluation
    if getattr(evaluation,'_calibrated_prefetch_registered',False):return
    original_model=evaluation.get_model
    def audited(cfg):
        import torch
        from KERVRuntimeOptimization.calibrated_prefetch import install
        model=original_model(cfg);pipeline=install(model)
        proof=pipeline.audit(digest_tensor)
        expected={x['name']:x['decoded_weight_sha256'] for x in model._calibrated_manifest['modules']}
        if {x['name']:x['decoded_sha256'] for x in proof}!=expected:raise RuntimeError('Prefetched weights changed')
        context=os.environ.get('KERV_ACCEPTANCE_CONTEXT');replay=os.environ.get('KERV_REPLAY_OUTPUT')
        root=Path(context).parent if context else Path(replay) if replay else Path(cfg.flagos_w8a16_manifest_path).parent.parent
        root.mkdir(parents=True,exist_ok=True)
        model._calibrated_manifest.update(implementation='musa_calibrated_two_slot_prefetch_native_bf16_gemm',
            operator_backend='prefetch2',workspace_bytes=pipeline.workspace_bytes,lookup_storage_bytes=sum(m._w8_group_map.numel()*4 for _,m in pipeline.named_modules))
        def emit(d):
            with (root/'calibrated_prefetch.jsonl').open('a') as f:f.write(json.dumps(d)+'\n')
        emit({'phase':'loaded','policy_sha256':model._calibrated_manifest['policy_sha256'],'modules':proof,
              'workspace_bytes':pipeline.workspace_bytes,'pool_slots':2,'quantization_policy_changed':False})
        def final():
            torch.musa.synchronize()
            emit({'phase':'final','stats':pipeline.stats,'fallback_calls':pipeline.fallback_calls,
                  'modules':[{'name':n,'prefetch_forward_calls':pipeline.module_calls[i],
                              'forward_calls':m.forward_calls} for i,(n,m) in enumerate(pipeline.named_modules)]})
        atexit.register(final)
        return model
    evaluation.get_model=audited;evaluation._calibrated_prefetch_registered=True

if __name__=='__main__':
    register()
    from experiments.robot.libero.run_kerv_libero import eval_libero
    eval_libero()
