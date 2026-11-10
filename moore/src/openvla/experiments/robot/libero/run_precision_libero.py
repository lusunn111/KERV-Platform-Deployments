"""Precision-first storage: mixed packed weights, optional exact BF16 storage, native GEMM."""
import atexit
import hashlib
import json
import os
from pathlib import Path
from types import MethodType


def digest(value):
    import torch
    return hashlib.sha256(value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def register():
    import torch
    from experiments.robot import flagos_ops
    from KERVRuntimeOptimization.calibrated_int8 import CalibratedInt8Linear
    from KERVRuntimeOptimization.packed_intn import PackedIntLinear
    from KERVRuntimeOptimization.packed_pow2 import PackedPow2Linear
    from KERVRuntimeOptimization.lossless_bf16 import compress_model,LosslessBF16Linear
    from KERVRuntimeOptimization.nearzero_bf16 import compress_nearzero,NearZeroBF16Linear
    from KERVRuntimeOptimization.inline_exact_bf16 import compress_inline_exact
    if getattr(flagos_ops,'_precision_storage_registered',False):return
    original=flagos_ops.enable_w8a16_quantization
    def backend(model,enabled,groups=None,min_speedup=0.,strict=True,manifest_path=None,
                backend='auto',group_size=32,policy_path=None):
        if not enabled:return original(model,False,groups=groups,manifest_path=manifest_path,backend=backend)
        if set(str(groups).split(','))!={'verify_mlp','verify_attention'} or not policy_path:
            raise ValueError('Precision storage requires both verifier groups and an explicit policy')
        path=Path(policy_path);raw=path.read_bytes();policy=json.loads(raw);sha=hashlib.sha256(raw).hexdigest()
        if policy.get('schema')!='kerv_precision_first_storage_v1' or not policy.get('complete'):
            raise ValueError('Incomplete precision-first policy')
        manifest=getattr(model,'_precision_storage_manifest',None)
        if manifest:
            if manifest['policy_sha256']!=sha:raise ValueError('Reload before changing precision policy')
        else:
            entries=policy.get('modules',[]);names=[x['name'] for x in entries]
            if len(names)!=len(set(names)) or policy.get('module_count')!=len(names):raise ValueError('Invalid module list')
            exact_groups=set(policy.get('exact_bf16_groups',['verify_mlp','verify_attention']))
            nearzero_groups=set(policy.get('nearzero_bf16_groups',[]))
            inline_groups=set(policy.get('inline_exact_bf16_groups',[]))
            if not exact_groups<={'verify_mlp','verify_attention'} or not nearzero_groups<={'verify_mlp'} or not inline_groups<={'verify_mlp'}:
                raise ValueError('Unsupported storage group')
            if inline_groups and (exact_groups or nearzero_groups or entries):raise ValueError('Inline exact requires an exclusive policy')
            if nearzero_groups and (exact_groups or entries):raise ValueError('Near-zero pilot requires an exclusive policy')
            if not exact_groups and not entries and not nearzero_groups and not inline_groups:raise ValueError('Policy must select quantization or exact storage')
            prepared=[]
            for entry in entries:
                linear=model.get_submodule(entry['name'])
                if linear.weight.device.type!='cpu' or digest(linear.weight)!=entry['original_weight_sha256']:
                    raise ValueError('Original CPU BF16 weight hash mismatch')
                payload=path.parent/entry['file']
                if hashlib.sha256(payload.read_bytes()).hexdigest()!=entry['sha256']:raise ValueError('Quantized payload hash mismatch')
                content=torch.load(payload,weights_only=True,map_location='cpu')
                encoding=entry.get('encoding','calibrated_int8')
                loaders={'calibrated_int8':CalibratedInt8Linear,'packed_intn':PackedIntLinear,'packed_pow2':PackedPow2Linear}
                if encoding not in loaders:raise ValueError('Unknown quantized weight encoding')
                loader=loaders[encoding]
                q=loader.from_payload(content)
                if digest(q.dequantize())!=entry['decoded_weight_sha256']:raise ValueError('Quantized decoded hash mismatch')
                prepared.append((entry,q))
            for entry,q in prepared:
                parent,_,child=entry['name'].rpartition('.');setattr(model.get_submodule(parent),child,q)
            exact=compress_model(model,128,groups=exact_groups) if exact_groups else {'modules':[]}
            nearzero=compress_nearzero(model,128,groups=nearzero_groups) if nearzero_groups else {'modules':[]}
            inline=compress_inline_exact(model,128,groups=inline_groups) if inline_groups else {'modules':[]}
            records=[{'name':e['name'],'shape':e['shape'],'original_sha256':e['original_weight_sha256'],
                      'decoded_sha256':e['decoded_weight_sha256'],'before_bytes':e['before_bytes'],
                      'storage_bytes':q.storage_bytes,'storage':f'packed_pow2_w{q.bits}a16' if isinstance(q,PackedPow2Linear) else f'packed_w{q.bits}a16' if isinstance(q,PackedIntLinear) else 'int8_bf16_groups'} for e,q in prepared]
            records += [{'name':e['name'],'shape':e['shape'],'original_sha256':e['original_sha256'],
                         'decoded_sha256':e['original_sha256'],'before_bytes':e['weight_bytes_before'],
                         'storage_bytes':e['weight_bytes_after'],'storage':'exact_bf16_exp_delta12'} for e in exact['modules']]
            records += nearzero['modules']+inline['modules']
            manifest={'enabled':True,'fallback':False,'implementation':'precision_first_mixed_packed_exact_bf16_storage_native_gemm',
                      'module_count':len(records),'int8_module_count':sum(isinstance(q,CalibratedInt8Linear) for _,q in prepared),
                      'packed_intn_module_count':sum(isinstance(q,PackedIntLinear) for _,q in prepared),
                      'nearzero_module_count':len(nearzero['modules']),'inline_exact_module_count':len(inline['modules']),
                      'lossy_module_count':len(prepared)+len(nearzero['modules']),'exact_module_count':len(exact['modules'])+len(inline['modules']),
                      'lossy_quantization':bool(prepared or nearzero['modules']),'selected_groups':sorted(exact_groups | nearzero_groups | inline_groups | ({'verify_mlp'} if prepared else set())),
                      'requested_groups':['verify_mlp','verify_attention'],'modules':records,'policy_sha256':sha,
                      'policy_path':str(path),'group_size':128,'int8_group_size':32,
                      'estimated_weight_storage_saving_bytes':sum(x['before_bytes']-x['storage_bytes'] for x in records),
                      'legacy_enable_field':'w8a16 dispatch hook; actual storage is identified per module',
                      'accuracy':'exact BF16 storage has zero weight error; each lossy subset requires closed-loop accuracy validation'}
            model._precision_storage_manifest=manifest;model._flagos_w8a16_manifest=manifest;model._flagos_w8a16_enabled=True
            print('[precision storage]',manifest['int8_module_count'],'INT8;',manifest['packed_intn_module_count'],'packed intN;',manifest['exact_module_count'],'exact BF16;',manifest['nearzero_module_count'],
                  'near-zero; saved',manifest['estimated_weight_storage_saving_bytes'],flush=True)
        if manifest_path:
            target=Path(manifest_path);target.parent.mkdir(parents=True,exist_ok=True)
            target.write_text(json.dumps(manifest,indent=2)+'\n')
        return manifest
    flagos_ops.enable_w8a16_quantization=backend;flagos_ops._precision_storage_registered=True
    from experiments.robot.libero import run_kerv_libero as evaluation
    evaluation.enable_w8a16_quantization=backend;original_model=evaluation.get_model
    def audited(cfg):
        from KERVRuntimeOptimization import calibrated_w8a16_ops as ops
        from KERVRuntimeOptimization.musa_safe_softmax import patch_llama_attention_softmax
        if not cfg.use_flagos_tree_builder or cfg.flagos_tree_static_runtime or not cfg.flagos_kalman_batch:
            raise ValueError('Expected host tree and shared Kalman history')
        model=original_model(cfg);manifest=model._precision_storage_manifest;proof=[]
        for entry in manifest['modules']:
            module=model.get_submodule(entry['name'])
            if isinstance(module,CalibratedInt8Linear):
                ops.prepare(module)
                def decode(self):return ops.decode_rows(self,1024)
                module.dequantize=MethodType(decode,module)
            actual=digest(module.dequantize())
            if actual!=entry['decoded_sha256']:raise RuntimeError('MUSA decoded hash mismatch: '+entry['name'])
            proof.append({**entry,'gpu_decoded_sha256':actual,'expected_decode_exact':True})
        patch_llama_attention_softmax(model)
        context=os.environ.get('KERV_ACCEPTANCE_CONTEXT');replay=os.environ.get('KERV_REPLAY_OUTPUT')
        root=Path(context).parent if context else Path(replay) if replay else Path(cfg.flagos_w8a16_manifest_path).parent.parent
        root.mkdir(parents=True,exist_ok=True)
        def emit(data):
            with (root/'precision_storage.jsonl').open('a') as f:f.write(json.dumps(data)+'\n')
        emit({'phase':'loaded','policy_sha256':manifest['policy_sha256'],'modules':proof,
              'lossy_quantization':manifest['lossy_quantization'],'manifest':manifest})
        def final():
            emit({'phase':'final','policy_sha256':manifest['policy_sha256'],
                  'modules':[{'name':n,'forward_calls':m.forward_calls} for n,m in model.named_modules()
                             if isinstance(m,(CalibratedInt8Linear,PackedIntLinear,LosslessBF16Linear,NearZeroBF16Linear))],
                  'kalman_batch_hits':getattr(model,'_flagos_kalman_batch_hits',0),
                  'softmax_guard_calls':{n:m._musa_safe_softmax_calls for n,m in model.named_modules()
                                         if getattr(m,'_musa_safe_softmax_enabled',False)},
                  'softmax_repaired_rows':{n:int(m._musa_softmax_repaired_rows.cpu()) for n,m in model.named_modules()
                                          if getattr(m,'_musa_safe_softmax_enabled',False)}})
        atexit.register(final);return model
    evaluation.get_model=audited


if __name__=='__main__':
    register()
    from experiments.robot.libero.run_kerv_libero import eval_libero
    eval_libero()
