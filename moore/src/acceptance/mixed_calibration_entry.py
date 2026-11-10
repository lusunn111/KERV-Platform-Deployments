"""Select actual MLP projections for INT8 using fixed-action calibration gates.

Both original BF16 and candidate INT8 weights remain resident during search;
search memory is explicitly excluded from deployment memory measurements.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import time
from acceptance.common import dump, load, rows, sha256, distribution


class Finished(BaseException):
    pass


def calibrate(cfg, model, processor):
    import numpy as np
    import torch
    from torch import nn
    from KERVRuntimeOptimization.weight_only_int8 import Int8WeightOnlyLinear, group_for
    from experiments.robot.libero.libero_utils import get_libero_image, quat2axisangle
    from experiments.robot.robot_utils import get_action, set_seed_everywhere
    from experiments.robot.tool_utils import compute_dynamic_threshold
    output = Path(os.environ['KERV_MIXED_CALIBRATION_OUTPUT'])
    output.mkdir(parents=True, exist_ok=False)
    capture = Path(os.environ['KERV_MIXED_CAPTURE'])
    data, runtime = rows(capture/'steps.jsonl'), rows(capture/'runtime_stats.jsonl')
    frames = np.load(capture/'observations.npz')
    selected_frames = json.loads(os.environ.get('KERV_MIXED_CALIBRATION_FRAMES','[0,20,40,60,80,100,120,140]'))
    group_size = int(os.environ.get('KERV_MIXED_GROUP_SIZE','128'))
    assert selected_frames == sorted(set(selected_frames)) and all(0<=i<141 for i in selected_frames)
    assert len(data) == 141 and len(runtime) == 141
    inputs, histories, token_histories = [], [], []
    for i, expected in enumerate(data):
        obs = {key: np.array(frames[key][i], copy=True) for key in frames.files}
        fingerprint = hashlib.sha256()
        for key in ('agentview_image','robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos'):
            value = obs[key]
            fingerprint.update(key.encode()); fingerprint.update(str(value.shape).encode())
            fingerprint.update(str(value.dtype).encode()); fingerprint.update(value.tobytes())
        assert fingerprint.hexdigest() == expected['observation_hash']
        inputs.append({'full_image': get_libero_image(obs,224), 'state': np.concatenate((
            obs['robot0_eef_pos'],quat2axisangle(obs['robot0_eef_quat']),obs['robot0_gripper_qpos']))})
        histories.append([np.array(x['raw_action']) for x in data[:i]])
        token_histories.append([list(x['kalman_history_token_ids']) for x in runtime[:i]])
    processor._flagos_prompt_token_cache_enabled = bool(cfg.flagos_prompt_token_cache)
    processor._flagos_pinned_image_buffer_enabled = False
    processor._flagos_prompt_cache_model = str(cfg.pretrained_checkpoint)
    model._flagos_suppress_runtime_stats = True
    candidates = [(name, module) for name, module in model.named_modules()
                  if isinstance(module, nn.Linear) and group_for(name) == 'verify_mlp']
    assert len(candidates) == 96
    originals = dict(candidates); quantized = {}; savings = {}; shapes = {}
    print('PREPARE 96 INT8 candidates; BF16 backups retained for search only', flush=True)
    for name, linear in candidates:
        cpu = nn.Linear(linear.in_features, linear.out_features, bias=False, device='meta', dtype=linear.weight.dtype)
        cpu.weight = nn.Parameter(linear.weight.detach().cpu(), requires_grad=False)
        if linear.bias is not None: cpu.bias = nn.Parameter(linear.bias.detach().cpu(), requires_grad=False)
        quant = Int8WeightOnlyLinear.from_linear(cpu,group_size).to(linear.weight.device)
        quantized[name] = quant
        savings[name] = linear.weight.numel()*linear.weight.element_size() - quant.qweight.numel() - quant.scales.numel()*4
        shapes[name] = list(linear.weight.shape)
    dump(output/'protocol.json', {'frames':141,'calibration_frames':selected_frames,'heldout_frames':141-len(selected_frames),
        'capture_sha256':sha256(capture/'observations.npz'),'group_size':group_size,'target_static_saving_bytes':3_000_000_000,
        'gate':'Every calibration action token vector must match an observed BF16 reference vector for that frame',
        'order':'gate/up before down; center layers before boundary layers',
        'memory':'BF16 backups and INT8 candidates co-resident; search memory is not deployment memory'})

    def run(indices, label, save=False):
        set_seed_everywhere(17)
        model.start_rollout(max_steps=220,kalman_cfg={'enabled':True,'tree_enabled':True,'process_var':1.,'measurement_var':.001},store_history=True)
        result=[]
        for i in indices:
            model._token_history = [list(x) for x in token_histories[i]]
            begin=time.perf_counter()
            action, times=get_action(cfg,model,inputs[i],'open the middle drawer of the cabinet',processor=processor,
                return_time=True,generate_mode='speculative',history=histories[i],step_idx=i,max_steps=220,
                dynamic_threshold=compute_dynamic_threshold(i,220,15,3),use_kalman=True,
                kalman_process_var=1.,kalman_measurement_var=.001,kalman_history_window=None,kalman_tree_enabled=True)
            tokens=list(model._last_action_token_ids_cpu)
            assert len(tokens)==len(action)==7 and np.isfinite(action).all()
            result.append({'frame':i,'observation_hash':data[i]['observation_hash'],'action_tokens':tokens,
                           'raw_action':np.asarray(action).tolist(),'model_seconds':float(times[0]-times[1]),
                           'pipeline_seconds':time.perf_counter()-begin})
        if save:
            with (output/(label+'.jsonl')).open('w') as stream:
                for item in result:stream.write(json.dumps(item,allow_nan=False)+'\n')
        return result

    print('BF16 WARM full input set',flush=True)
    run(range(141),'bf16_warmup',True)
    reference=run(range(141),'bf16_reference',True)
    observed={i:{tuple(reference[i]['action_tokens'])} for i in selected_frames}
    for repeat in range(2):
        for item in run(selected_frames,'bf16_repeat_'+str(repeat),True):
            observed[item['frame']].add(tuple(item['action_tokens']))
    unstable=[i for i in selected_frames if len(observed[i])>1]
    dump(output/'bf16_calibration_references.json', {'tokens':{str(i):[list(x) for x in sorted(v)] for i,v in observed.items()},'unstable_frames':unstable})
    def mismatches(items):return [x for x in items if tuple(x['action_tokens']) not in observed[x['frame']]]
    def set_module(name,module):
        parent,_,child=name.rpartition('.')
        setattr(model.get_submodule(parent),child,module)
    def order(name):
        layer=int(re.search(r'\.layers\.(\d+)\.',name).group(1))
        return (name.endswith('down_proj'),abs(layer-15.5),layer,name)
    chosen=[]; total=0; trials=[]
    preferred = json.loads(os.environ.get('KERV_MIXED_PREFERRED_MODULES','[]'))
    rank = {name:i for i,name in enumerate(preferred)}
    ordered = sorted(originals,key=lambda name:(0,rank[name]) if name in rank else (1,order(name)))
    for index,name in enumerate(ordered):
        set_module(name,quantized[name])
        actual=run(selected_frames,'trial')
        bad=mismatches(actual)
        if not bad:chosen.append(name);total+=savings[name]
        else:set_module(name,originals[name])
        trial={'trial':index,'module':name,'accepted':not bad,'mismatch_frames':[x['frame'] for x in bad],
               'int8_module_count':len(chosen),'static_saving_bytes':total,'actions':actual}
        trials.append(trial)
        with (output/'trials.jsonl').open('a') as stream:stream.write(json.dumps(trial,allow_nan=False)+'\n')
        print('TRIAL',index,name,'ACCEPT' if not bad else 'KEEP_BF16','chosen',len(chosen),'saving',total,flush=True)
        if total>=3_000_000_000:break
    # Check the final cumulative policy twice, backtracking if repetitions vary.
    backtracks=[]
    while True:
        failures=[x for repeat in range(2) for x in mismatches(run(selected_frames,'final_calibration_'+str(repeat),True))]
        if not failures:break
        if not chosen:raise RuntimeError('BF16 calibration itself changed beyond observed references')
        name=chosen.pop();set_module(name,originals[name]);total-=savings[name];backtracks.append(name)
        print('BACKTRACK',name,flush=True)
    if not chosen:raise RuntimeError('No INT8 projection passed calibration')
    policy={'schema':'kerv_mixed_w8_bf16_v1','group_size':group_size,'int8_modules':sorted(chosen),
        'weight_shapes':{name:shapes[name] for name in sorted(chosen)},
        'bf16_modules':sorted(set(originals)-set(chosen)),'calibration_frames':selected_frames,
        'capture_sha256':sha256(capture/'observations.npz'),'base_revision':'fa5ae1e7509348889295bba8e08621d8b55e9baf',
        'drafter_sha256':'d3f62ae500f27b356083f997b24fa7a67e5dda86e5083fa15985f45250009aae',
        'estimated_weight_storage_saving_bytes':total,'calibration_observed_reference_gate':'PASS',
        'bf16_unstable_calibration_frames':unstable,'backtracked_modules':backtracks,
        'quality_scope':'Calibration only; full held-out replay and independent closed-loop tests required'}
    dump(output/'policy.json',policy)
    run(range(141),'mixed_warmup',True)
    actual=run(range(141),'mixed_validation',True)
    heldout=[i for i in range(141) if i not in selected_frames]
    errors=[abs(a-b) for left,right in zip(reference,actual) for a,b in zip(left['raw_action'],right['raw_action'])]
    summary={'int8_module_count':len(chosen),'bf16_mlp_module_count':96-len(chosen),'static_saving_bytes':total,
        'calibration_frames':len(selected_frames),'calibration_gate':'PASS',
        'all_token_matches':sum(a['action_tokens']==b['action_tokens'] for a,b in zip(reference,actual)),
        'all_frames':141,'heldout_token_matches':sum(reference[i]['action_tokens']==actual[i]['action_tokens'] for i in heldout),
        'heldout_frames':len(heldout),'raw_action_absolute_error':distribution(errors),
        'maximum_raw_action_absolute_error':max(errors),'bf16_model_seconds':distribution(x['model_seconds'] for x in reference),
        'mixed_model_seconds':distribution(x['model_seconds'] for x in actual),'memory_eligible':False,
        'scope':'Search-resident model; no environment or SR; production reload and memory checks remain required'}
    dump(output/'summary.json',summary)
    print('CALIBRATION COMPLETE',json.dumps(summary),flush=True)


def main():
    from experiments.robot.libero import run_kerv_libero as evaluation
    get_model,get_processor=evaluation.get_model,evaluation.get_processor
    holder={}
    def model_hook(cfg):
        model=get_model(cfg);holder['model']=model;return model
    def processor_hook(cfg):
        processor=get_processor(cfg)
        calibrate(cfg,holder['model'],processor)
        raise Finished()
    evaluation.get_model=model_hook;evaluation.get_processor=processor_hook
    try:evaluation.eval_libero()
    except Finished:pass
    finally:
        if 'model' in holder and hasattr(holder['model'],'close_flagos_runtime_stats'):holder['model'].close_flagos_runtime_stats()


if __name__=='__main__':main()
