"""Original BF16 weights with the same guarded/shared-history runtime."""
import atexit
import json
import os
from pathlib import Path


def register():
    from experiments.robot.libero import run_kerv_libero as evaluation
    if getattr(evaluation,'_precision_reference_registered',False):return
    original=evaluation.get_model
    def audited(cfg):
        from KERVRuntimeOptimization.musa_safe_softmax import patch_llama_attention_softmax
        model=original(cfg);names=patch_llama_attention_softmax(model)
        context=os.environ.get('KERV_ACCEPTANCE_CONTEXT');replay=os.environ.get('KERV_REPLAY_OUTPUT')
        root=Path(context).parent if context else Path(replay) if replay else Path(cfg.local_log_dir).parent
        root.mkdir(parents=True,exist_ok=True)
        def emit(phase):
            data={'phase':phase,'compressed_modules':0,'guarded_attention_modules':names,
                  'kalman_batch_hits':getattr(model,'_flagos_kalman_batch_hits',0),
                  'softmax_guard_calls':{n:m._musa_safe_softmax_calls for n,m in model.named_modules() if getattr(m,'_musa_safe_softmax_enabled',False)}}
            with (root/'precision_reference.jsonl').open('a') as f:f.write(json.dumps(data)+'\n')
        emit('loaded');atexit.register(emit,'final');return model
    evaluation.get_model=audited;evaluation._precision_reference_registered=True


if __name__=='__main__':
    register()
    from experiments.robot.libero.run_kerv_libero import eval_libero
    eval_libero()
