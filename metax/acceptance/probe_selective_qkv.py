"""Screen fixed middle-layer QKV fusion on actual verifier inputs from init0 replay.

Capture and microbenchmarks are diagnostic and excluded from deployment timing.
Selection 7:25 is frozen before the screen; this does not fit layers to task labels.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import statistics
import time
import numpy as np
import torch
import torch_npu
from diagnose_quantization import configure,write
from experiments.robot.openvla_utils import get_vla,get_processor,get_vla_action
from experiments.robot.tool_utils import compute_dynamic_threshold


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1]
    torch.manual_seed(7);np.random.seed(7)
    cfg=SimpleNamespace(pretrained_checkpoint=str(root/'checkpoints/openvla-libero-goal'),
        spec_checkpoint=str(root/'checkpoints/kerv-drafter'),use_spec=True,parallel_draft=False,
        accept_threshold=9,accept_threshold_start=15,load_in_8bit=False,load_in_4bit=False)
    model=get_vla(cfg).eval();configure(model,a.output,'screen')
    layers=model.base_model.language_model.model.layers
    captures={};hooks=[]
    for index,layer in enumerate(layers):
        def capture(module,args,index=index):
            x=args[0];m=x.numel()//x.shape[-1]
            if m>8 and (index,m) not in captures and sum(k[0]==index for k in captures)<3:
                captures[index,m]=x.detach().cpu().clone()
        hooks.append(layer.self_attn.q_proj.register_forward_pre_hook(capture))
    processor=get_processor(cfg);history=[]
    model.start_rollout(max_steps=220,kalman_cfg={'enabled':True,'tree_enabled':True},store_history=True)
    files=sorted((root/'outputs/npu_ar_tensorflow/observations').glob('*.npz'))[:8]
    with torch.inference_mode():
        for index,f in enumerate(files):
            with np.load(f,allow_pickle=False) as d:obs={'full_image':d['image'].copy()};task=str(d['task'])
            action,_=get_vla_action(model,processor,cfg.pretrained_checkpoint,obs,task,'libero_goal',return_time=True,
                center_crop=True,generate_mode='speculative',history=history,step_idx=index,max_steps=220,
                dynamic_threshold=compute_dynamic_threshold(index,220,15,3),use_kalman=True,kalman_tree_enabled=True)
            history.append(action)
        for hook in hooks:hook.remove()
        torch.save(captures,a.output/'actual_inputs.pt')
        rows=[]
        for (index,m),host_x in sorted(captures.items()):
            x=host_x.npu();attn=layers[index].self_attn
            weights=[getattr(attn,n).weight for n in ['q_proj','k_proj','v_proj']]
            # Diagnostic temporary concatenation is not the deployed CPU prepack path.
            packed=torch.cat(weights,dim=0)
            sizes=[w.shape[0] for w in weights]
            def separate():return tuple(torch.nn.functional.linear(x,w) for w in weights)
            def fused():return torch.nn.functional.linear(x,packed).split(sizes,dim=-1)
            ref=separate();actual=fused()
            result=dict(layer=index,rows=m,selected=7<=index<25,
                mismatches=sum(int((u!=v).sum()) for u,v in zip(ref,actual)),
                relative_l2=max(float((u.float()-v.float()).norm()/u.float().norm()) for u,v in zip(ref,actual)),
                max_abs_error=max(float((u.float()-v.float()).abs().max()) for u,v in zip(ref,actual)),
                finite=all(bool(y.isfinite().all()) for y in actual),samples_ms={'separate':[],'fused':[]})
            for repeat in range(4):
                for name,fn in ([('separate',separate),('fused',fused)] if repeat%2==0 else [('fused',fused),('separate',separate)]):
                    for _ in range(10):fn()
                    torch.npu.synchronize();start=time.perf_counter()
                    for _ in range(50):fn()
                    torch.npu.synchronize();result['samples_ms'][name].append(1000*(time.perf_counter()-start)/50)
            result['median_ms']={k:statistics.median(v) for k,v in result['samples_ms'].items()}
            result['change_percent']=100*(result['median_ms']['fused']/result['median_ms']['separate']-1)
            rows.append(result);print(json.dumps(result),flush=True)
            del packed,x,ref,actual
    selected=[r for r in rows if r['selected']]
    report=dict(fixed_selection='7:25',physical_device=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
        input_scope='Actual q_proj inputs from unchanged balanced quantized verifier; first 8 init0 observations',
        input_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
        captures_sha256=hashlib.sha256((a.output/'actual_inputs.pt').read_bytes()).hexdigest(),
        gate=dict(all_layers_captured=len(set(r['layer'] for r in rows))==32,
                  finite=all(r['finite'] for r in selected),local_l2_under_1e_3=all(r['relative_l2']<1e-3 for r in selected),
                  summed_medians_lower=sum(r['median_ms']['fused'] for r in selected)<sum(r['median_ms']['separate'] for r in selected)),
        rows=rows,scope='Local error and microbench only; full closed-loop quality and same-card model timing required')
    write(a.output/'results.json',report)
    print('GATES',report['gate'],flush=True)
    assert all(report['gate'].values())


if __name__=='__main__':main()
