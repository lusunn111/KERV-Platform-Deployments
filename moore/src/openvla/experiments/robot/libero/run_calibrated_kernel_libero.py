"""Opt-in single-pass row decode, retaining the native BF16 GEMM."""
import atexit
import json
import os
from pathlib import Path
from types import MethodType


def register():
    from experiments.robot.libero.run_calibrated_libero import register as register_policy, digest_tensor
    register_policy()
    from experiments.robot.libero import run_kerv_libero as evaluation
    from KERVRuntimeOptimization.calibrated_int8 import CalibratedInt8Linear
    from KERVRuntimeOptimization import calibrated_w8a16_ops as ops
    if getattr(evaluation,'_calibrated_kernel_registered',False): return
    original_model = evaluation.get_model

    def audited(cfg):
        model = original_model(cfg)
        context = os.environ.get('KERV_ACCEPTANCE_CONTEXT')
        replay = os.environ.get('KERV_REPLAY_OUTPUT')
        root = Path(context).parent if context else Path(replay) if replay else Path(cfg.flagos_w8a16_manifest_path).parent.parent
        root.mkdir(parents=True,exist_ok=True)
        proof=[]
        expected={e['name']:e['decoded_weight_sha256'] for e in model._calibrated_manifest['modules']}
        for name,module in model.named_modules():
            if not isinstance(module,CalibratedInt8Linear): continue
            if module.qweight.device.type!='musa': raise RuntimeError('MUSA operator backend required')
            tensors=[module.qweight,module.scales,module.inverse_channel_scales,module.protected_values]
            if not all(t.is_contiguous() for t in tensors): raise ValueError('Contiguous quantized buffers required')
            ops.prepare(module)
            module._w8_decode_calls=0
            def fast_decode(self):
                self._w8_decode_calls+=1
                return ops.decode_rows(self,1024)
            module.dequantize=MethodType(fast_decode,module)
            actual=digest_tensor(module.dequantize())
            if actual!=expected[name]: raise RuntimeError('Row decoder changed weight bits: '+name)
            proof.append({'name':name,'decoded_sha256':actual,'bit_exact':True,
                          'lookup_bytes':module._w8_group_map.numel()*4})
        if len(proof)!=model._calibrated_manifest['module_count']: raise RuntimeError('Missing calibrated modules')
        model._calibrated_manifest['implementation']='musa_calibrated_row_decode_1024_native_bf16_gemm'
        model._calibrated_manifest['operator_backend']='row_decode_1024'
        model._calibrated_manifest['lookup_storage_bytes']=sum(x['lookup_bytes'] for x in proof)
        def emit(data):
            with (root/'calibrated_kernel.jsonl').open('a') as f:f.write(json.dumps(data)+'\n')
        emit({'phase':'loaded','backend':'row_decode_1024','modules':proof,
              'policy_sha256':model._calibrated_manifest['policy_sha256'],
              'persistent_decoded_weight_bytes':0,'quantization_policy_changed':False})
        def final():
            emit({'phase':'final','modules':[{'name':n,'forward_calls':m.forward_calls,
                   'decode_calls':m._w8_decode_calls} for n,m in model.named_modules() if isinstance(m,CalibratedInt8Linear)]})
        atexit.register(final)
        print('[calibrated kernel] row_decode_1024; bit-exact weights',len(proof),flush=True)
        return model
    evaluation.get_model=audited
    evaluation._calibrated_kernel_registered=True


if __name__=='__main__':
    register()
    from experiments.robot.libero.run_kerv_libero import eval_libero
    eval_libero()
