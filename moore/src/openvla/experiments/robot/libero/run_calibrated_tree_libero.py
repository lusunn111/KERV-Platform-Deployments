"""Calibrated row decoder plus host-batched, unpruned Kalman tree buffers.

Kalman predictions still use the original scalar implementation. Attention,
GEMM, RoPE, draft-tree selection and tree sizes retain the native settings.
"""
import atexit
import json
import os
from pathlib import Path

def register():
    from experiments.robot.libero.run_calibrated_kernel_libero import register as rows_register
    rows_register()
    from experiments.robot.libero import run_kerv_libero as evaluation
    if getattr(evaluation,'_calibrated_tree_registered',False):return
    original=evaluation.get_model
    def audited(cfg):
        if not cfg.use_flagos_tree_builder or cfg.flagos_tree_static_runtime or cfg.flagos_kalman_batch:
            raise ValueError('Host tree backend requires enabled tree builder, static_runtime=false, kalman_batch=false')
        model=original(cfg)
        model._calibrated_manifest['runtime_control_backend']='host_batched_buffers_scalar_kalman'
        context=os.environ.get('KERV_ACCEPTANCE_CONTEXT');replay=os.environ.get('KERV_REPLAY_OUTPUT')
        root=Path(context).parent if context else Path(replay) if replay else Path(cfg.flagos_w8a16_manifest_path).parent.parent
        def final():
            data={'phase':'final','tree_builder_enabled':model._flagos_tree_builder_enabled,
                  'tree_builder_hits':model._flagos_tree_builder_hit_count,
                  'static_tree_runtime':model._flagos_static_tree_runtime_enabled,
                  'kalman_batch_enabled':model._flagos_kalman_batch_enabled,
                  'policy_sha256':model._calibrated_manifest['policy_sha256'],
                  'scope':'host tree traffic optimization; no pruning, no Kalman numerical batching'}
            with (root/'calibrated_tree_runtime.jsonl').open('a') as f:f.write(json.dumps(data)+'\n')
        atexit.register(final)
        return model
    evaluation.get_model=audited;evaluation._calibrated_tree_registered=True

if __name__=='__main__':
    register()
    from experiments.robot.libero.run_kerv_libero import eval_libero
    eval_libero()
