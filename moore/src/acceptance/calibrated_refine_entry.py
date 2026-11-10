"""Output-aware BF16 block restoration under a frozen static storage budget.

All selection uses existing diagnostic V frames, never the final test T.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
from acceptance.common import dump,rows

class Finished(BaseException):pass
class Captured(BaseException):pass


def refine(cfg,model,processor):
    import numpy as np
    import torch
    from torch import nn
    from safetensors import safe_open
    from experiments.robot.libero.libero_utils import get_libero_image,quat2axisangle
    from experiments.robot.robot_utils import get_action
    from libero.libero import benchmark
    root=Path(os.environ['KERV_CALIBRATION_ROOT']);out=root/'model_refined';out.mkdir(exist_ok=False)
    policy=json.loads((root/'model_quant/policy.json').read_text());lm=model.base_model.language_model
    device=next(model.parameters()).device;index=json.loads((Path(cfg.pretrained_checkpoint)/'model.safetensors.index.json').read_text())['weight_map']
    current={};quantized={};original={};examples=[]
    for entry in policy['modules']:
        name=entry['name'];quantized[name]=model.get_submodule(name)
        key=name.removeprefix('base_model.')+'.weight'
        with safe_open(str(Path(cfg.pretrained_checkpoint)/index[key]),framework='pt',device='cpu') as file:weight=file.get_tensor(key)
        module=nn.Linear(weight.shape[1],weight.shape[0],bias=False,device='meta',dtype=torch.bfloat16)
        module.weight=nn.Parameter(weight.to(device),requires_grad=False);module.eval();original[name]=module
    def install(restored):
        for name,module in quantized.items():
            layer=int(name.split('.layers.')[1].split('.')[0]);parent,_,child=name.rpartition('.')
            setattr(model.get_submodule(parent),child,original[name] if layer in restored else module)
    def hook(module,args,kwargs):
        current['hidden']=args[0].detach().clone()
        current['mask']=kwargs['attention_mask'].detach().clone()
        raise Captured()
    handle=lm.model.layers[0].register_forward_pre_hook(hook,with_kwargs=True)
    op=model.predict_action
    def intercept(input_ids=None,**kwargs):
        if int(input_ids[0,-1])!=29871:input_ids=torch.cat((input_ids,torch.full((1,1),29871,device=device,dtype=torch.long)),1)
        ids=torch.cat((input_ids,torch.tensor(current['tokens'],device=device).reshape(1,7)),1)
        lm.tree_mask=None;lm._flagos_external_causal_mask=None
        model.base_model(input_ids=ids,attention_mask=torch.ones_like(ids),pixel_values=kwargs['pixel_values'],use_cache=False)
    model.predict_action=intercept
    folder=root/'inputs/diagnostic_capture';frames=np.load(folder/'observations.npz');data=rows(folder/'steps.jsonl')
    task=benchmark.get_benchmark_dict()['libero_goal']().get_task(0).language
    with torch.inference_mode():
        try:
            for i in range(0,141,4):
                current['tokens']=data[i]['action_tokens']
                obs={'full_image':get_libero_image({k:frames[k][i] for k in frames.files},224),
                     'state':np.concatenate((frames['robot0_eef_pos'][i],quat2axisangle(frames['robot0_eef_quat'][i]),frames['robot0_gripper_qpos'][i]))}
                try:get_action(cfg,model,obs,task,processor=processor,generate_mode='speculative')
                except Captured:pass
                examples.append({'hidden':current['hidden'],'mask':current['mask'],'frame':i})
        finally:handle.remove();model.predict_action=op
        def logits():
            result=[]
            for e in examples:
                lm.tree_mask=None;lm._flagos_external_causal_mask=e['mask']
                output=lm.model_forward(inputs_embeds=e['hidden'],use_cache=False,return_dict=True,output_attentions=False,output_hidden_states=False)
                result.append(lm.lm_head(output.last_hidden_state[:,-8:-1,:]).float().cpu())
            return result
        layers=sorted({int(x.split('.layers.')[1].split('.')[0]) for x in quantized})
        install(set(layers));teacher=logits();repeat=logits()
        baseline_variation=sum(int((a.argmax(-1)!=b.argmax(-1)).sum()) for a,b in zip(teacher,repeat))
        dump(out/'bf16_repeat.json',{'different_top1_positions':baseline_variation,'positions':len(teacher)*7,
            'maximum_logit_change':max(float((a-b).abs().max()) for a,b in zip(teacher,repeat))})
        norm_stats=model.get_action_stats('libero_goal')
        gripper_mask=norm_stats.get('mask',[True]*7)[-1]
        def grip(tokens):
            indices=(model.vocab_size-tokens.numpy()-1).clip(0,len(model.bin_centers)-1)
            value=model.bin_centers[indices]
            if gripper_mask:value=.5*(value+1)*(norm_stats['q99'][-1]-norm_stats['q01'][-1])+norm_stats['q01'][-1]
            return value>0
        def evaluate(restored):
            install(set(restored));actual=logits();wrong=0;kl=0.;maximum=0.;gripper_changes=0;pose_error=0;maximum_pose_error=0
            for a,b in zip(teacher,actual):
                ta,tb=a.argmax(-1),b.argmax(-1);wrong+=int((ta!=tb).sum());gripper_changes+=int((grip(ta[:,-1])!=grip(tb[:,-1])).sum())
                bins_a=(model.vocab_size-ta-1).clamp(0,len(model.bin_centers)-1)
                bins_b=(model.vocab_size-tb-1).clamp(0,len(model.bin_centers)-1)
                pose=(bins_a[:,:6]-bins_b[:,:6]).abs();pose_error+=int(pose.sum());maximum_pose_error=max(maximum_pose_error,int(pose.max()))
                la,lb=a.log_softmax(-1),b.log_softmax(-1);kl+=float((la.exp()*(la-lb)).sum())
                maximum=max(maximum,float((a-b).abs().max()))
            saving=sum(x['before_bytes']-x['storage_bytes'] for x in policy['modules'] if int(x['name'].split('.layers.')[1].split('.')[0]) not in restored)
            return {'restored_layers':sorted(restored),'wrong_top1_positions':wrong,'gripper_execution_differences':gripper_changes,'pose_bin_absolute_error_sum':pose_error,'maximum_pose_bin_error':maximum_pose_error,
                    'mean_KL':kl/(len(teacher)*7),'maximum_logit_error':maximum,'static_saving_bytes':saving}
        trials=[];base=evaluate([]);trials.append(base);dump(out/'trials.json',trials);print('REFINE_BASE',base,flush=True)
        leave=[]
        for layer in layers:
            score=evaluate([layer]);leave.append(score);trials.append(score);dump(out/'trials.json',trials)
            print('REFINE_RESTORE',layer,score['wrong_top1_positions'],score['mean_KL'],flush=True)
        key=lambda x:(x['gripper_execution_differences'],x['pose_bin_absolute_error_sum'],x['maximum_pose_bin_error'],x['wrong_top1_positions'],x['mean_KL'])
        ranked=[x['restored_layers'][0] for x in sorted(leave,key=key)]
        combinations=[base]
        for count in [1,2,4,6]:
            restored=ranked[:count]
            score=next((x for x in leave if x['restored_layers']==sorted(restored)),None) or evaluate(restored)
            trials.append(score);combinations.append(score);dump(out/'trials.json',trials)
            print('REFINE_COMBINATION',score,flush=True)
        eligible=[x for x in combinations if x['static_saving_bytes']>=2_000_000_000]
        chosen=min(eligible,key=key)
        selected=copy.deepcopy(policy)
        selected['modules']=[x for x in selected['modules'] if int(x['name'].split('.layers.')[1].split('.')[0]) not in chosen['restored_layers']]
        for entry in selected['modules']:entry['file']='../model_quant/'+entry['file']
        selected['module_count']=len(selected['modules']);selected['estimated_static_saving_bytes']=chosen['static_saving_bytes']
        selected['restored_output_sensitive_layers']=chosen['restored_layers']
        selected['quality']='NOT_PROMOTED; output-aware selection on seen diagnostic V only'
        dump(out/'policy.json',selected)
        dump(out/'decision.json',{'base':base,'selected':chosen,'ranked_layers':ranked,'bf16_repeat_different_top1_positions':baseline_variation,
            'criterion':'executed gripper changes, pose bin absolute/max error, all top1 changes, then KL; static savings >=2e9; combinations0/1/2/4/6',
            'scope':'Seen diagnostic V only; no final test; actual KERV replay and closed-loop required'})
        print('REFINE_COMPLETE',chosen,flush=True)


def main():
    from experiments.robot.libero.run_calibrated_libero import register
    register()
    from experiments.robot.libero import run_kerv_libero as evaluation
    om,op=evaluation.get_model,evaluation.get_processor;holder={}
    def get_model(cfg):holder['model']=om(cfg);return holder['model']
    def get_processor(cfg):
        processor=op(cfg);refine(cfg,holder['model'],processor);raise Finished()
    evaluation.get_model=get_model;evaluation.get_processor=get_processor
    try:evaluation.eval_libero()
    except Finished:pass

if __name__=='__main__':main()
