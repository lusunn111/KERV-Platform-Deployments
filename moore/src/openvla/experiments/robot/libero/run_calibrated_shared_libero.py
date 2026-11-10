"""Frozen mixed weights and guarded attention with shared Kalman estimates."""
import atexit
import hashlib
import json
import os
from pathlib import Path
from types import MethodType


def register():
    from experiments.robot.libero.run_calibrated_safe_libero import register as safe_register
    safe_register()
    from experiments.robot.libero import run_kerv_libero as evaluation
    if getattr(evaluation,'_calibrated_shared_registered',False):return
    original=evaluation.get_model
    def audited(cfg):
        if not cfg.use_flagos_tree_builder or cfg.flagos_tree_static_runtime or not cfg.flagos_kalman_batch:
            raise ValueError('Shared history requires host tree builder, static_runtime=false, kalman_batch=true')
        model=original(cfg)
        model._calibrated_shared_calls=0;model._calibrated_shared_chains=0
        method=model.predict_kf_chains_batched
        def counted(self,prefixes,lengths):
            self._calibrated_shared_calls+=1;self._calibrated_shared_chains+=len(prefixes)
            return method(prefixes,lengths)
        model.predict_kf_chains_batched=MethodType(counted,model)
        model._calibrated_manifest['runtime_control_backend']='host_tree_shared_scalar_kalman_history'
        context=os.environ.get('KERV_ACCEPTANCE_CONTEXT');replay=os.environ.get('KERV_REPLAY_OUTPUT')
        root=Path(context).parent if context else Path(replay) if replay else Path(cfg.flagos_w8a16_manifest_path).parent.parent
        root.mkdir(parents=True,exist_ok=True)
        implementation=Path(__import__(type(model).__module__,fromlist=['__file__']).__file__)
        def emit(phase):
            data={'phase':phase,'backend':'host_tree_shared_scalar_kalman_history',
                  'calls':model._calibrated_shared_calls,'chains':model._calibrated_shared_chains,
                  'configured_kalman_batch':cfg.flagos_kalman_batch,
                  'kalman_batch_enabled':getattr(model,'_flagos_kalman_batch_enabled',None),
                  'tree_builder_enabled':getattr(model,'_flagos_tree_builder_enabled',None),
                  'static_tree_runtime':getattr(model,'_flagos_static_tree_runtime_enabled',None),
                  'policy_sha256':model._calibrated_manifest['policy_sha256'],
                  'implementation_sha256':hashlib.sha256(implementation.read_bytes()).hexdigest(),
                  'quantization_policy_changed':False,'persistent_decoded_weight_bytes':0}
            with (root/'calibrated_shared_history.jsonl').open('a') as f:f.write(json.dumps(data)+'\n')
        emit('loaded');atexit.register(emit,'final')
        return model
    evaluation.get_model=audited;evaluation._calibrated_shared_registered=True


if __name__=='__main__':
    register()
    from experiments.robot.libero.run_kerv_libero import eval_libero
    eval_libero()
