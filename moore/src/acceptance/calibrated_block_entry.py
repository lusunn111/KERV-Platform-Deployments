"""Pilot: real multi-task activations, calibrated MLP block and fixed histories.

All calibration/search memory and latency are excluded from deployment claims.
This stage never touches the proposed final task/state test set.
"""
import hashlib
import json
import os
from pathlib import Path
import time
from acceptance.common import dump, rows, sha256


class Finished(BaseException): pass


def pilot(cfg, model, processor):
    import numpy as np
    import torch
    from KERVRuntimeOptimization.calibrated_int8 import calibrate_linear
    from experiments.robot.libero.libero_utils import get_libero_image, quat2axisangle
    from experiments.robot.robot_utils import get_action, set_seed_everywhere
    from experiments.robot.tool_utils import compute_dynamic_threshold
    from libero.libero import benchmark
    root=Path(os.environ['KERV_CALIBRATION_ROOT']);output=root/os.environ.get('KERV_BLOCK_PILOT_DIR','block_pilot')
    output.mkdir(parents=True,exist_ok=False)
    model._flagos_suppress_runtime_stats=True
    suite=benchmark.get_benchmark_dict()['libero_goal']()
    processor._flagos_prompt_token_cache_enabled=False
    processor._flagos_pinned_image_buffer_enabled=False
    processor._flagos_prompt_cache_model=str(cfg.pretrained_checkpoint)
    mlp=model.base_model.language_model.model.layers[15].mlp
    original={name:getattr(mlp,name) for name in ['gate_proj','up_proj','down_proj']}
    requests=[]
    for task in range(3):
        folder=root/'runs'/f'capture_C_task{task}_state3_s7'
        requests.append((folder,task,list(range(0,48,2)),'train'))
    requests.append((root/'inputs/diagnostic_capture',0,list(range(0,141,4)),'validation'))
    current={};samples={'train':{'prompt':[],'decode':[]},'validation':{'prompt':[],'decode':[]}}
    records=[]
    def hook(module,args):
        x=args[0].detach().reshape(-1,args[0].shape[-1])
        kind='prompt' if current['call']==0 else 'decode'
        current['call']+=1
        count=min(len(x),32 if kind=='prompt' else 16)
        index=torch.linspace(0,len(x)-1,count,device=x.device).long()
        samples[current['split']][kind].append(x[index].cpu())
    handle=mlp.gate_proj.register_forward_pre_hook(hook)
    def replay(folder,task,indices,split,save=True):
        frames=np.load(folder/'observations.npz');captured=rows(folder/'steps.jsonl');runtime=rows(folder/'runtime_stats.jsonl')
        set_seed_everywhere(7)
        model.start_rollout(max_steps=220,kalman_cfg={'enabled':True,'tree_enabled':True,'process_var':1.,'measurement_var':.001},store_history=True)
        result=[]
        for i in indices:
            current.update(split=split,frame=i,call=0)
            obs={key:np.array(frames[key][i],copy=True) for key in frames.files}
            fingerprint=hashlib.sha256()
            for key in ('agentview_image','robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos'):
                value=obs[key];fingerprint.update(key.encode());fingerprint.update(str(value.shape).encode());fingerprint.update(str(value.dtype).encode());fingerprint.update(value.tobytes())
            assert fingerprint.hexdigest()==captured[i]['observation_hash']
            model._token_history=[list(x['kalman_history_token_ids']) for x in runtime[:i]]
            image=get_libero_image(obs,224)
            observation={'full_image':image,'state':np.concatenate((obs['robot0_eef_pos'],quat2axisangle(obs['robot0_eef_quat']),obs['robot0_gripper_qpos']))}
            action,_=get_action(cfg,model,observation,suite.get_task(task).language,processor=processor,return_time=True,
                generate_mode='speculative',history=[np.array(x['raw_action']) for x in captured[:i]],step_idx=i,max_steps=220,
                dynamic_threshold=compute_dynamic_threshold(i,220,15,3),use_kalman=True,kalman_process_var=1.,
                kalman_measurement_var=.001,kalman_history_window=None,kalman_tree_enabled=True)
            result.append({'task':task,'frame':i,'capture':str(folder),'observation_hash':fingerprint.hexdigest(),
                           'action_tokens':list(model._last_action_token_ids_cpu),'raw_action':np.asarray(action).tolist()})
        return result
    with torch.inference_mode():
        for folder,task,indices,split in requests:
            print('COLLECT',split,task,len(indices),flush=True)
            data=replay(folder,task,indices,split);records.extend(data)
        handle.remove()
        generator=torch.Generator(device='cpu').manual_seed(2050)
        assembled={};counts={}
        for split in samples:
            pieces=[];counts[split]={}
            for kind in samples[split]:
                if not samples[split][kind]:raise RuntimeError('Missing calibration category: '+split+'/'+kind)
                values=torch.cat(samples[split][kind]);cap=(2048 if kind=='prompt' else 6144) if split=='train' else (1024 if kind=='prompt' else 3072)
                index=torch.randperm(len(values),generator=generator)[:cap]
                pieces.append(values[index]);counts[split][kind]={'available':len(values),'selected':len(index)}
            assembled[split]=torch.cat(pieces)
        del samples
        torch.save({'inputs':assembled,'sampling_counts':counts,'frames':records},output/'activations.pt')
        dump(output/'sampling.json',{'counts':counts,'frames':len(records),'train_frames':72,'validation_frames':36,
             'coverage':'Actual native KERV prompt and tree/decode activations; no separate teacher-forcing rows in this pilot',
             'limitation':'Small pilot subset, not full2048-frame calibration or final test'})
        train=assembled['train'].to(next(mlp.parameters()).device);valid=assembled['validation'].to(train.device)
        target=mlp(valid).detach();variants=[('rtn32',0.,0.,'rtn'),('gptq32',0.,0.,'gptq'),
            ('gptq32_p5',0.,.05,'gptq'),('awq_gptq32_p5',.5,.05,'gptq')]
        summary=[];best=None
        for name,alpha,protect,method in variants:
            print('VARIANT',name,flush=True);begin=time.perf_counter();calibrated={};metadata={}
            for projection in ['gate_proj','up_proj']:
                print('QUANTIZE',name,projection,flush=True)
                quant=calibrate_linear(original[projection],train,32,protect,alpha,method,execution_device=train.device)
                metadata[projection]=quant.calibration;calibrated[projection]=quant.to(train.device)
                setattr(mlp,projection,calibrated[projection])
            down_inputs=mlp.act_fn(mlp.gate_proj(train))*mlp.up_proj(train)
            print('QUANTIZE',name,'down_proj',flush=True)
            quant=calibrate_linear(original['down_proj'],down_inputs,32,protect,alpha,method,execution_device=train.device)
            metadata['down_proj']=quant.calibration;calibrated['down_proj']=quant.to(train.device);mlp.down_proj=calibrated['down_proj']
            actual=mlp(valid);difference=actual.float()-target.float()
            relative=float(difference.square().mean().sqrt()/target.float().square().mean().sqrt())
            payload={projection:module.payload() for projection,module in calibrated.items()}
            torch.save(payload,output/(name+'.pt'))
            info={'variant':name,'relative_output_rmse':relative,'mean_absolute_output_error':float(difference.abs().mean()),
                'maximum_output_error':float(difference.abs().max()),'seconds':time.perf_counter()-begin,
                'storage_bytes':sum(q.storage_bytes for q in calibrated.values()),'metadata':metadata}
            summary.append(info);dump(output/'reconstruction.json',summary)
            print('RESULT',name,'relative_RMSE',relative,flush=True)
            for projection,module in original.items():setattr(mlp,projection,module)
        rtn=summary[0]['relative_output_rmse']
        chosen=min(summary[1:],key=lambda x:x['relative_output_rmse'])
        decision={'rtn32_relative_output_rmse':rtn,'chosen':chosen['variant'],
            'chosen_relative_output_rmse':chosen['relative_output_rmse'],
            'relative_rmse_ratio_to_rtn':chosen['relative_output_rmse']/rtn,
            'reconstruction_gate':'PASS' if chosen['relative_output_rmse']<rtn else 'FAIL',
            'precision_scope':'one MLP block on diagnostic activations; not full-model accuracy or SR'}
        dump(output/'decision.json',decision)
        if decision['reconstruction_gate']=='PASS':
            payload=torch.load(output/(chosen['variant']+'.pt'),map_location='cpu',weights_only=False)
            from KERVRuntimeOptimization.calibrated_int8 import CalibratedInt8Linear
            for projection,data in payload.items():setattr(mlp,projection,CalibratedInt8Linear.from_payload(data).to(train.device))
            print('MODEL_REPLAY',chosen['variant'],flush=True)
            candidate=replay(*requests[-1][:3],'validation',False)
            baseline=records[-36:]
            errors=[abs(x-y) for a,b in zip(baseline,candidate) for x,y in zip(a['raw_action'],b['raw_action'])]
            dump(output/'action_replay.json',{'frames':len(candidate),'exact_action_tokens':sum(a['action_tokens']==b['action_tokens'] for a,b in zip(baseline,candidate)),
                'raw_action_mae':sum(errors)/len(errors),'maximum_raw_action_error':max(errors),
                'gripper_matches':sum(a['raw_action'][-1]==b['raw_action'][-1] for a,b in zip(baseline,candidate)),
                'baseline':baseline,'candidate':candidate,'scope':'one calibrated MLP block; same captured observations/history; not SR'})
            for projection,module in original.items():setattr(mlp,projection,module)
        print('BLOCK_PILOT_COMPLETE',decision,flush=True)


def main():
    from experiments.robot.libero import run_kerv_libero as evaluation
    original_model=evaluation.get_model;original_processor=evaluation.get_processor;holder={}
    def model_hook(cfg):holder['model']=original_model(cfg);return holder['model']
    def processor_hook(cfg):
        processor=original_processor(cfg);pilot(cfg,holder['model'],processor);raise Finished()
    evaluation.get_model=model_hook;evaluation.get_processor=processor_hook
    try:evaluation.eval_libero()
    except Finished:pass
    finally:
        if 'model' in holder and hasattr(holder['model'],'close_flagos_runtime_stats'):holder['model'].close_flagos_runtime_stats()


if __name__=='__main__':main()
