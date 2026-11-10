"""Explicit opt-in calibrated INT8/BF16 backend, CPU preparation before MUSA."""
import atexit
import hashlib
import json
import os
from pathlib import Path


def digest_tensor(value):
    import torch
    return hashlib.sha256(value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def register():
    import torch
    from experiments.robot import flagos_ops
    from KERVRuntimeOptimization.calibrated_int8 import CalibratedInt8Linear
    if getattr(flagos_ops,'_calibrated_registered',False):return
    original=flagos_ops.enable_w8a16_quantization
    def backend(model,enabled,groups=None,min_speedup=0.,strict=True,manifest_path=None,backend='auto',group_size=32,policy_path=None):
        if not enabled:return original(model,False,groups=groups,manifest_path=manifest_path,backend=backend)
        if set(str(groups).split(','))!={'verify_mlp'} or group_size!=32 or not policy_path:
            raise ValueError('Calibrated backend requires verify_mlp, group32 and an explicit policy')
        path=Path(policy_path);raw=path.read_bytes();policy_sha=hashlib.sha256(raw).hexdigest()
        if getattr(model,'_calibrated_manifest',None):
            if model._calibrated_manifest['policy_sha256']!=policy_sha:raise ValueError('Reload before changing policy')
            manifest=model._calibrated_manifest
        else:
            policy=json.loads(raw)
            if policy.get('schema')!='kerv_calibrated_int8_bf16_v2' or not policy.get('complete') or not policy.get('modules'):
                raise ValueError('Incomplete or incompatible calibrated policy')
            names=[x['name'] for x in policy['modules']]
            if len(names)!=len(set(names)) or policy['module_count']!=len(names):raise ValueError('Invalid module list')
            prepared=[]
            for entry in policy['modules']:
                linear=model.get_submodule(entry['name'])
                if linear.weight.device.type!='cpu' or list(linear.weight.shape)!=entry['shape']:
                    raise ValueError('CPU weight preparation/shape mismatch')
                if digest_tensor(linear.weight)!=entry['original_weight_sha256']:raise ValueError('Source weight hash mismatch')
                checkpoint=path.parent/entry['file']
                if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=entry['sha256']:raise ValueError('Quantized payload hash mismatch')
                quant=CalibratedInt8Linear.from_payload(torch.load(checkpoint,map_location='cpu',weights_only=True))
                if digest_tensor(quant.dequantize())!=entry['decoded_weight_sha256']:raise ValueError('CPU decoded hash mismatch')
                prepared.append((entry,quant))
            for entry,quant in prepared:
                parent,_,child=entry['name'].rpartition('.')
                setattr(model.get_submodule(parent),child,quant)
            manifest={'enabled':True,'fallback':False,'implementation':'musa_calibrated_gptq_int8_bf16_groups_native_bf16_gemm',
                'lossy_quantization':True,'requested_groups':['verify_mlp'],'selected_groups':['verify_mlp'],
                'group_size':32,'module_count':len(prepared),'modules':policy['modules'],
                'policy_sha256':policy_sha,'policy_path':str(path),
                'estimated_weight_storage_saving_bytes':policy['estimated_static_saving_bytes'],
                'alpha':policy['alpha'],'protected_fraction':policy['protected_fraction'],
                'accuracy':'must be measured; no precision guarantee from calibration alone'}
            model._calibrated_manifest=manifest;model._flagos_w8a16_manifest=manifest;model._flagos_w8a16_enabled=True
            print('[calibrated INT8/BF16]',len(prepared),'modules; storage saving bytes',policy['estimated_static_saving_bytes'],flush=True)
        if manifest_path:
            p=Path(manifest_path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(manifest,indent=2)+'\n')
        return manifest
    flagos_ops.enable_w8a16_quantization=backend;flagos_ops._calibrated_registered=True
    from experiments.robot.libero import run_kerv_libero as evaluation
    evaluation.enable_w8a16_quantization=backend
    original_model=evaluation.get_model
    def audited(cfg):
        model=original_model(cfg);manifest=model._calibrated_manifest
        context=os.environ.get('KERV_ACCEPTANCE_CONTEXT');replay=os.environ.get('KERV_REPLAY_OUTPUT')
        root=Path(context).parent if context else Path(replay) if replay else Path(cfg.flagos_w8a16_manifest_path).parent.parent
        root.mkdir(parents=True,exist_ok=True);proof=[]
        for entry in manifest['modules']:
            module=model.get_submodule(entry['name'])
            if module.qweight.device.type!='musa' or module.qweight.dtype!=torch.int8:raise RuntimeError('Expected real INT8 MUSA weights')
            actual=digest_tensor(module.dequantize())
            if actual!=entry['decoded_weight_sha256']:raise RuntimeError('Accelerator decoded weight differs from CPU payload')
            proof.append({'name':entry['name'],'decoded_sha256':actual,'cpu_gpu_decode_bit_exact':True,
                'protected_group_count':module.protected_groups.numel(),'buffers':{
                    n:{'device':str(b.device),'dtype':str(b.dtype),'bytes':b.numel()*b.element_size()} for n,b in module.named_buffers()}})
        def emit(data):
            with (root/'calibrated_quantization.jsonl').open('a') as f:f.write(json.dumps(data)+'\n')
        emit({'phase':'loaded','policy_sha256':manifest['policy_sha256'],'module_count':len(proof),'modules':proof})
        def final():
            emit({'phase':'final','modules':[{'name':n,'forward_calls':m.forward_calls} for n,m in model.named_modules() if isinstance(m,CalibratedInt8Linear)]})
        atexit.register(final)
        return model
    evaluation.get_model=audited


if __name__=='__main__':
    register()
    from experiments.robot.libero.run_kerv_libero import eval_libero
    eval_libero()
