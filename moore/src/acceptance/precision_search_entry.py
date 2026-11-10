"""Resident accuracy screening against original BF16 on identical replay inputs.

Co-resident copies are intentional for selection. Their memory is not a
deployment-memory measurement. The capture is previously seen development data.
"""
import hashlib
import os
from pathlib import Path
from types import MethodType
from acceptance.common import dump,load


class Finished(BaseException):pass


def run(cfg,model,processor):
    import torch
    from torch import nn
    from KERVRuntimeOptimization.lossless_bf16 import LosslessBF16Linear,raw_hash
    from KERVRuntimeOptimization.calibrated_int8 import CalibratedInt8Linear
    from KERVRuntimeOptimization import calibrated_w8a16_ops as ops
    from KERVRuntimeOptimization.weight_only_int8 import group_for
    from KERVRuntimeOptimization.musa_safe_softmax import patch_llama_attention_softmax
    from acceptance.replay_entry import worker
    root=Path(os.environ['KERV_PRECISION_ROOT']);experiment=root/'screening'
    old_path=Path('/workspace/outputs/calibrated_quantization/20261026T045030Z_v2/model_refined/policy.json')
    old=load(old_path);original={};compressed={};quantized={};proof=[]
    device=next(model.parameters()).device
    patch_llama_attention_softmax(model)
    for name,module in model.named_modules():
        if isinstance(module,nn.Linear) and group_for(name) in {'verify_mlp','verify_attention'} and not name.endswith('.q_proj'):
            original[name]=module
    with torch.inference_mode():
        for i,(name,module) in enumerate(original.items()):
            cpu=nn.Linear(module.in_features,module.out_features,bias=False,device='meta',dtype=torch.bfloat16)
            cpu.weight=nn.Parameter(module.weight.detach().cpu(),requires_grad=False)
            if module.bias is not None:cpu.bias=nn.Parameter(module.bias.detach().cpu(),requires_grad=False)
            coded=LosslessBF16Linear(cpu,128).to(device)
            actual=raw_hash(coded.dequantize())
            assert actual==coded.original_sha256,name
            compressed[name]=coded
            proof.append({'name':name,'original_sha256':coded.original_sha256,'gpu_decoded_sha256':actual,
                          'before_bytes':module.weight.numel()*2,'storage_bytes':coded.storage_bytes,
                          'shape':list(module.weight.shape),'lossless':True})
            if (i+1)%16==0:print('EXACT_CODEC',i+1,len(original),flush=True)
        expected={x['name']:x for x in proof}
        for entry in old['modules']:
            path=old_path.parent/entry['file']
            assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']
            assert expected[entry['name']]['original_sha256']==entry['original_weight_sha256']
            q=CalibratedInt8Linear.from_payload(torch.load(path,weights_only=True)).to(device)
            ops.prepare(q)
            def decode(self):return ops.decode_rows(self,1024)
            q.dequantize=MethodType(decode,q)
            assert raw_hash(q.dequantize())==entry['decoded_weight_sha256']
            quantized[entry['name']]=q
        dump(experiment/'weight_proof.json',{'original_weights_and_exact_codec':proof,
             'quantized_modules':old['modules'],'scope':'CPU codec roundtrip and actual MUSA decode checked; all copies co-resident'})
        modes=[('bf16',set(),False),('old_int8',set(quantized),False)]
        for label in ['conservative_up8','conservative_up4']:
            policy=load(root/label/'policy.json');modes.append((label,{x['name'] for x in policy['modules']},True))
        modes.append(('exact_bf16',set(),True))
        for label,quant_names,use_codec in modes:
            out=experiment/label;out.mkdir(exist_ok=False)
            for name,linear in original.items():
                replacement=quantized[name] if name in quant_names else compressed[name] if use_codec else linear
                parent,_,child=name.rpartition('.');setattr(model.get_submodule(parent),child,replacement)
            records=[{'name':name,'storage':'int8_bf16_groups' if name in quant_names else 'exact_bf16_exp_delta12' if use_codec else 'native_bf16',
                      'decoded_sha256':next(x['decoded_weight_sha256'] for x in old['modules'] if x['name']==name) if name in quant_names else expected[name]['original_sha256'],
                      'before_bytes':expected[name]['before_bytes'],
                      'storage_bytes':quantized[name].storage_bytes if name in quant_names else compressed[name].storage_bytes if use_codec else expected[name]['before_bytes']}
                     for name in original]
            manifest={'mode':label,'module_count':len(records),'int8_modules':len(quant_names),
                      'exact_codec_modules':len(original)-len(quant_names) if use_codec else 0,
                      'estimated_weight_storage_saving_bytes':sum(x['before_bytes']-x['storage_bytes'] for x in records),
                      'lossy_quantization':bool(quant_names),'modules':records,
                      'quality':'accuracy screening against BF16; not independent SR acceptance'}
            dump(out/'precision_manifest.json',manifest);model._flagos_w8a16_manifest=manifest
            os.environ['KERV_REPLAY_OUTPUT']=str(out);os.environ['KERV_REPLAY_PROFILE']=label
            print('MODE',label,'int8',len(quant_names),'saved',manifest['estimated_weight_storage_saving_bytes'],flush=True)
            worker(cfg,model,processor,label)
            dump(out/'execution_proof.json',{'mode':label,'quantized_forward_calls':{n:quantized[n].forward_calls for n in quant_names},
                'exact_forward_calls':{n:compressed[n].forward_calls for n in original if use_codec and n not in quant_names},
                'softmax_guard_calls':{n:m._musa_safe_softmax_calls for n,m in model.named_modules() if getattr(m,'_musa_safe_softmax_enabled',False)}})
    raise Finished()


def main():
    from experiments.robot.libero import run_kerv_libero as evaluation
    root=Path(os.environ['KERV_PRECISION_ROOT']);(root/'screening').mkdir(exist_ok=False)
    dump(root/'screening/worker_start.json',{'pid':os.getpid()})
    original_model=evaluation.get_model;original_processor=evaluation.get_processor;holder={}
    def model_hook(cfg):holder['model']=original_model(cfg);return holder['model']
    def processor_hook(cfg):
        processor=original_processor(cfg);run(cfg,holder['model'],processor)
    evaluation.get_model=model_hook;evaluation.get_processor=processor_hook
    try:evaluation.eval_libero()
    except Finished:pass


if __name__=='__main__':main()
