"""Opt-in calibrated row/tree runtime with the MUSA eager Softmax fix."""
import atexit
import json
import os
import hashlib
from pathlib import Path


def register():
    from experiments.robot.libero.run_calibrated_kernel_libero import register as rows_register
    rows_register()
    from experiments.robot.libero import run_kerv_libero as evaluation
    if getattr(evaluation,'_calibrated_safe_registered',False):return
    original=evaluation.get_model
    def audited(cfg):
        from KERVRuntimeOptimization.musa_safe_softmax import patch_llama_attention_softmax
        model=original(cfg)
        modules=patch_llama_attention_softmax(model)
        context=os.environ.get('KERV_ACCEPTANCE_CONTEXT');replay=os.environ.get('KERV_REPLAY_OUTPUT')
        root=Path(context).parent if context else Path(replay) if replay else Path(cfg.flagos_w8a16_manifest_path).parent.parent
        root.mkdir(parents=True,exist_ok=True)
        from KERVRuntimeOptimization import musa_safe_softmax
        implementation_sha=hashlib.sha256(Path(musa_safe_softmax.__file__).read_bytes()).hexdigest()
        def emit(phase):
            data={'phase':phase,'implementation':'native_softmax_with_triton_row_validation_and_selective_repair','modules':modules,
                  'implementation_file_sha256':implementation_sha,
                  'calls':{name:module._musa_safe_softmax_calls for name,module in model.named_modules()
                           if getattr(module,'_musa_safe_softmax_enabled',False)},
                  'repaired_rows':{name:int(module._musa_softmax_repaired_rows.cpu()) for name,module in model.named_modules()
                           if getattr(module,'_musa_safe_softmax_enabled',False)},
                  'policy_sha256':model._calibrated_manifest['policy_sha256']}
            with (root/'safe_softmax.jsonl').open('a') as f:f.write(json.dumps(data)+'\n')
        emit('loaded');atexit.register(emit,'final')
        return model
    evaluation.get_model=audited;evaluation._calibrated_safe_registered=True


if __name__=='__main__':
    register()
    from experiments.robot.libero.run_kerv_libero import eval_libero
    eval_libero()
