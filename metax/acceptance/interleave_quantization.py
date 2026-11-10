"""Interleave two resident models to reduce shared-host and test-order drift.

Diagnostic dual residency is not the deployment memory measurement. All reported
latencies exclude loading, warmup, and the separately collected device traces.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import torch
import torch_npu

from diagnose_quantization import configure, VerifyCounter, write
from experiments.robot.openvla_utils import get_vla, get_processor, get_vla_action
from experiments.robot.tool_utils import compute_dynamic_threshold
from experiments.robot.flagos_ops import enable_linear_fusion
from experiments.robot.ascend_qkv_ops import verifier_qkv_scope,verifier_qkv_layer_ids


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--samples',type=int,default=64)
    p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--trace-model',action='store_true')
    p.add_argument('--quant-group-size',type=int,choices=[32,64,128],default=128)
    p.add_argument('--quant-policy',type=Path)
    p.add_argument('--quant-profile',choices=['mlp_int8','mlp_dynamic_int8'],default='mlp_int8')
    p.add_argument('--reference-quant-policy',type=Path)
    p.add_argument('--candidate-qkv-fusion',action='store_true')
    p.add_argument('--candidate-qkv-layers',default='all')
    p.add_argument('--candidate-qkv-backend',choices=['concat','bmm'],default='concat')
    p.add_argument('--candidate-qkv-rows',default=','.join(map(str,range(1,513))))
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1]
    files=sorted((root/'outputs/npu_ar_tensorflow/observations').glob('*.npz'))[:a.samples]
    assert len(files)==a.samples
    observations=[]
    for f in files:
        with np.load(f,allow_pickle=False) as d:
            observations.append(({'full_image':d['image'].copy()},str(d['task'])))
    plan={'physical_device':os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),'samples':a.samples,
        'measured_repeats':a.repeats,'warmup_sequences':1,'order':'AB/BA alternating per observation and repeat',
        'quant_layers':'7:25','group_size':a.quant_group_size,
        'quant_profile':a.quant_profile,
        'policy_sha256':hashlib.sha256(a.quant_policy.read_bytes()).hexdigest() if a.quant_policy else None,
        'reference_policy_sha256':hashlib.sha256(a.reference_quant_policy.read_bytes()).hexdigest() if a.reference_quant_policy else None,
        'reference_kind':'quantized_policy' if a.reference_quant_policy else 'bf16',
        'display_names':{'bf16':'previous_quantized_candidate' if a.reference_quant_policy else 'bf16',
                         'int8':'candidate_with_qkv_fusion' if a.candidate_qkv_fusion else 'candidate'},
        'candidate_qkv_fusion':a.candidate_qkv_fusion,
        'candidate_qkv_layers':a.candidate_qkv_layers,
        'candidate_qkv_backend':a.candidate_qkv_backend,
        'candidate_qkv_rows':a.candidate_qkv_rows,
        'both_models_resident':True,'memory_is_diagnostic_only':True,
        'input_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in files}}
    write(a.output/'plan.json',plan)
    models={};configs={};processors={};counters={};shapes={};hooks=[]
    for label in ['bf16','int8']:
        policy=a.reference_quant_policy if label=='bf16' else a.quant_policy
        os.environ['KERV_NPU_QUANT']='none' if label=='bf16' and policy is None else a.quant_profile
        os.environ['KERV_NPU_QUANT_LAYERS']='7:25';os.environ['KERV_NPU_QUANT_GROUP_SIZE']=str(a.quant_group_size)
        os.environ['KERV_NPU_QUANT_POLICY']=str(policy) if policy else ''
        torch.manual_seed(7);np.random.seed(7)
        cfg=SimpleNamespace(pretrained_checkpoint=str(root/'checkpoints/openvla-libero-goal'),
            spec_checkpoint=str(root/'checkpoints/kerv-drafter'),use_spec=True,parallel_draft=False,
            accept_threshold=9,accept_threshold_start=15,load_in_8bit=False,load_in_4bit=False)
        if label=='int8' and a.candidate_qkv_fusion:
            cfg.use_flagos_linear_fusion=True
            cfg.flagos_qkv_fusion_layers=a.candidate_qkv_layers
            cfg.flagos_qkv_fusion_backend=a.candidate_qkv_backend
            cfg.flagos_qkv_fusion_rows=a.candidate_qkv_rows
        model=get_vla(cfg).eval();configure(model,a.output,label)
        if label=='int8' and a.candidate_qkv_fusion:
            manifest=enable_linear_fusion(model=verifier_qkv_scope(model,a.candidate_qkv_layers),enabled=True,
                qkv_rows=a.candidate_qkv_rows,qkv_input_sizes='4096',
                gate_up_rows='',gate_up_input_sizes='',swiglu_rows='',swiglu_input_sizes='',
                swiglu_backend='native_inplace',add_rms_norm_rows='',add_rms_norm_input_sizes='',
                add_rms_norm_backend='native_inplace',record=True,record_once=True,
                log_path=str(a.output/'candidate_qkv_hits.jsonl'),
                manifest_path=str(a.output/'candidate_qkv_manifest.json'))
            expected=len(verifier_qkv_layer_ids(model,a.candidate_qkv_layers))
            assert manifest['qkv_group_count']==expected and manifest['gate_up_group_count']==0
            assert manifest['cpu_prepacked_qkv_group_count']==expected
            assert manifest.get('bmm_qkv_group_count',0)==(expected if a.candidate_qkv_backend.startswith('bmm') else 0)
        processor=get_processor(cfg);processor._flagos_prompt_token_cache_enabled=False
        models[label]=model;configs[label]=cfg;processors[label]=processor
        counters[label]=VerifyCounter();model._acceptance_recorder=counters[label];shapes[label]=[]
        mlp=model.base_model.language_model.model.layers[7].mlp
        layer=mlp._dynamic_gate_up if getattr(mlp,'_dynamic_direct_mlp',False) else mlp.gate_proj
        hooks.append(layer.register_forward_pre_hook(lambda m,x,key=label:shapes[key].append(list(x[0].shape))))

    def rollout_reset():
        for model in models.values():
            model.start_rollout(max_steps=220,kalman_cfg={'enabled':True,'tree_enabled':True},store_history=True)

    def action(label,index,history):
        obs,task=observations[index];counters[label].events=[];shapes[label].clear()
        torch.npu.synchronize();begin=time.perf_counter()
        result,(end,start)=get_vla_action(models[label],processors[label],configs[label].pretrained_checkpoint,
            obs,task,'libero_goal',return_time=True,center_crop=True,generate_mode='speculative',history=history,
            step_idx=index,max_steps=220,dynamic_threshold=compute_dynamic_threshold(index,220,15,3),
            use_kalman=True,kalman_tree_enabled=True)
        torch.npu.synchronize()
        assert result.shape==(7,) and np.isfinite(result).all()
        return dict(index=index,model_ms=1000*(end-start),pipeline_ms=1000*(time.perf_counter()-begin),
            action=result.tolist(),verify=list(counters[label].events),verifier_shapes=list(shapes[label]))

    rows={'bf16':[],'int8':[]}
    for repeat in range(a.repeats+1):
        rollout_reset();histories={'bf16':[],'int8':[]}
        for index in range(a.samples):
            order=['bf16','int8'] if (index+repeat)%2==0 else ['int8','bf16']
            for position,label in enumerate(order):
                row=action(label,index,histories[label]);row.update(repeat=repeat,position=position)
                histories[label].append(np.asarray(row['action']))
                if repeat:rows[label].append(row)
            if (index+1)%16==0:print('interleaved sequence',repeat,'action',index+1,flush=True)
        if repeat:write(a.output/'partial.json',rows)
    summary={}
    for label,data in rows.items():
        seqs=[[r['action'] for r in data if r['repeat']==i] for i in range(1,a.repeats+1)]
        assert all(np.array_equal(seqs[0],s) for s in seqs)
        events=[e for r in data for e in r['verify']]
        summary[label]={'mean_model_ms':float(np.mean([r['model_ms'] for r in data])),
            'median_model_ms':float(np.median([r['model_ms'] for r in data])),
            'mean_pipeline_ms':float(np.mean([r['pipeline_ms'] for r in data])),
            'verify_cycles':len(events),'mean_raw_acceptance':float(np.mean([e['accepted'] for e in events])),
            'mean_verifier_tokens_per_action':sum(s[1] for r in data for s in r['verifier_shapes'])/len(data),
            'within_profile_actions_exact':True,'steps':len(data)}
    write(a.output/'results.json',dict(plan=plan,summary=summary,rows=rows))
    print('SUMMARY',summary,flush=True)
    if a.trace_model:
        for hook in hooks:hook.remove()
        for label in ['bf16','int8']:
            rollout_reset()
            with torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(a.output/'traces'/label)),
                record_shapes=True,with_stack=False,profile_memory=False):
                with torch.profiler.record_function('QUANT_DIAG_MODEL::'+label):
                    result=action(label,0,[])
            write(a.output/f'{label}_traced_action.json',result)


if __name__=='__main__':main()
