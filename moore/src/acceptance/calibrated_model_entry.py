"""Sequential block calibration on multimodal teacher-forced contexts.

Later blocks see actual quantized upstream activations. Original BF16 teacher
states are advanced separately. This pilot uses a small C/V subset, not T.
"""
import json
import os
from pathlib import Path
import time
from acceptance.common import dump, rows, sha256


class Finished(BaseException): pass
class Captured(BaseException): pass


def calibrate_model(cfg,model,processor):
    import numpy as np
    import torch
    from KERVRuntimeOptimization.calibrated_int8 import calibrate_linear
    from experiments.robot.libero.libero_utils import get_libero_image,quat2axisangle
    from experiments.robot.robot_utils import get_action
    from libero.libero import benchmark
    root=Path(os.environ['KERV_CALIBRATION_ROOT'])
    source=Path(os.environ.get('KERV_CALIBRATION_SOURCE_ROOT',str(root)))
    bits=int(os.environ.get('KERV_CALIBRATION_BITS','8'));scale_kind=os.environ.get('KERV_CALIBRATION_SCALE_KIND','fp32')
    out=root/os.environ.get('KERV_MODEL_OUTPUT_DIR','model_quant');out.mkdir(exist_ok=False)
    selected_layers={int(x) for x in os.environ.get('KERV_CALIBRATION_LAYER_LIST',','.join(str(i) for i in range(2,28))).split(',')}
    pilot=json.loads((source/'block_pilot_v2/decision.json').read_text())
    assert pilot['reconstruction_gate']=='PASS'
    candidates=json.loads((source/'block_pilot_v2/reconstruction.json').read_text())
    chosen=next(x for x in candidates if x['variant']==pilot['chosen'])
    parameters=chosen['metadata']['gate_proj'];alpha=parameters['alpha'];protect=parameters['protected_fraction_requested']
    device=next(model.parameters()).device;lm=model.base_model.language_model
    suite=benchmark.get_benchmark_dict()['libero_goal']();examples=[];holder={}
    original_predict=model.predict_action
    def first_layer(module,args,kwargs):
        holder['hidden']=args[0].detach().cpu()
        holder['kwargs']={k:(v.detach().cpu() if isinstance(v,torch.Tensor) else v) for k,v in kwargs.items()}
        raise Captured()
    handle=lm.model.layers[0].register_forward_pre_hook(first_layer,with_kwargs=True)
    def intercept(input_ids=None,**kwargs):
        if int(input_ids[0,-1])!=29871:input_ids=torch.cat((input_ids,torch.full((1,1),29871,device=device,dtype=torch.long)),1)
        action=torch.tensor(holder['tokens'],device=device,dtype=torch.long).reshape(1,7)
        ids=torch.cat((input_ids,action),1)
        lm.tree_mask=None;lm._flagos_external_causal_mask=None
        model.base_model(input_ids=ids,attention_mask=torch.ones_like(ids),pixel_values=kwargs['pixel_values'],
            use_cache=False,output_hidden_states=False,return_dict=True)
        raise RuntimeError('First decoder layer was not captured')
    model.predict_action=intercept
    requests=[(source/'runs'/f'capture_C_task{task}_state3_s7',task,list(range(48)),'C') for task in range(3)]
    requests.append((source/'inputs/diagnostic_capture',0,list(range(0,141,4)),'V'))
    with torch.inference_mode():
        try:
            for folder,task,indices,split in requests:
                frames=np.load(folder/'observations.npz');data=rows(folder/'steps.jsonl')
                for i in indices:
                    holder['tokens']=data[i]['action_tokens'];holder.pop('hidden',None)
                    observation={'full_image':get_libero_image({key:frames[key][i] for key in frames.files},224),
                        'state':np.concatenate((frames['robot0_eef_pos'][i],quat2axisangle(frames['robot0_eef_quat'][i]),frames['robot0_gripper_qpos'][i]))}
                    try:get_action(cfg,model,observation,suite.get_task(task).language,processor=processor,generate_mode='speculative')
                    except Captured:pass
                    assert 'hidden' in holder
                    examples.append({'hidden':holder['hidden'],'kwargs':holder['kwargs'],'split':split,'task':task,'frame':i,'observation_hash':data[i]['observation_hash']})
                print('EMBEDDINGS',split,task,len(indices),flush=True)
        finally:handle.remove();model.predict_action=original_predict
        dump(out/'protocol.json',{'chosen_single_block_variant':pilot['chosen'],'alpha':alpha,'protected_fraction':protect,
            'precision_bits':bits,'scale_kind':scale_kind,'selected_layers':sorted(selected_layers),'calibration_source':str(source),
            'calibration_examples':144,'validation_examples':36,'method':'GPTQ with actual BF16 decode error feedback and frozen BF16 groups',
            'sequential':'independent original teacher and actual quantized candidate states, layer by layer',
            'rows_per_frame':'7 prompt + 3 copies of7 action prediction positions (25/75 mass)',
            'coverage':'multimodal teacher forcing on 3 C tasks; no actual tree-candidate rows in this full-model pilot',
            'limitations':'small C/V pilot; V is prior-seen diagnostic task0 trajectory; no final test T executed',
            'rollback_gate':'additional layer output relative RMSE >1.5% or cumulative >3% on V restores that MLP to BF16'})
        teacher=[x['hidden'] for x in examples];candidate=[x.clone() for x in teacher]
        modules=[];layer_stats=[]
        def call(layer,hidden,index):
            kw={k:(v.to(device) if isinstance(v,torch.Tensor) else v) for k,v in examples[index]['kwargs'].items()}
            return layer(hidden.to(device),**kw)[0]
        for layer_id,layer in enumerate(lm.model.layers):
            begin=time.perf_counter();print('LAYER',layer_id,'start',flush=True)
            reference=[call(layer,x,i).cpu() for i,x in enumerate(teacher)]
            if layer_id not in selected_layers:
                candidate=[call(layer,x,i).cpu() for i,x in enumerate(candidate)];teacher=reference;continue
            mlp=layer.mlp;original={p:getattr(mlp,p) for p in ['gate_proj','up_proj','down_proj']};collected=[];active={}
            def collect(module,args):
                if active['split']!='C':return
                x=args[0].detach().reshape(-1,args[0].shape[-1]);length=len(x)
                action=x[length-8:length-1];prompt_index=torch.linspace(0,length-9,7,device=device).long()
                collected.append(torch.cat((x[prompt_index],action,action,action)).cpu())
            hook=mlp.gate_proj.register_forward_pre_hook(collect)
            before=[]
            for i,x in enumerate(candidate):
                active['split']=examples[i]['split'];before.append(call(layer,x,i).cpu())
            hook.remove();inputs=torch.cat(collected).to(device)
            quantized={};info={}
            for projection in ['gate_proj','up_proj']:
                print('LAYER',layer_id,'QUANTIZE',projection,flush=True)
                q=calibrate_linear(original[projection],inputs,32,protect,alpha,execution_device=device,
                                   precision_bits=bits,scale_kind=scale_kind)
                info[projection]=q.calibration;quantized[projection]=q.to(device);setattr(mlp,projection,quantized[projection])
            down=mlp.act_fn(mlp.gate_proj(inputs))*mlp.up_proj(inputs)
            print('LAYER',layer_id,'QUANTIZE down_proj',flush=True)
            q=calibrate_linear(original['down_proj'],down,32,protect,alpha,execution_device=device,
                               precision_bits=bits,scale_kind=scale_kind)
            info['down_proj']=q.calibration;quantized['down_proj']=q.to(device);mlp.down_proj=quantized['down_proj']
            after=[call(layer,x,i).cpu() for i,x in enumerate(candidate)]
            def relative(left,right):
                numerator=denominator=0.
                for i,(a,b) in enumerate(zip(left,right)):
                    if examples[i]['split']!='V':continue
                    numerator+=float((a.float()-b.float()).square().sum());denominator+=float(b.float().square().sum())
                return (numerator/denominator)**.5
            added=relative(after,before);cumulative=relative(after,reference)
            keep=added<=.015 and cumulative<=.03
            if keep:
                for projection,q in quantized.items():
                    name=f'base_model.language_model.model.layers.{layer_id}.mlp.{projection}'
                    if bits!=8:
                        from KERVRuntimeOptimization.calibrated_packed import pack_calibrated
                        q=pack_calibrated(q.cpu())
                    filename=f'layer{layer_id:02d}_{projection}.pt';torch.save(q.payload(),out/filename)
                    original_hash=__import__('hashlib').sha256(original[projection].weight.detach().contiguous().view(torch.int16).cpu().numpy().tobytes()).hexdigest()
                    decoded=q.dequantize().contiguous().view(torch.int16).cpu().numpy().tobytes()
                    modules.append({'name':name,'file':filename,'sha256':sha256(out/filename),'shape':[q.out_features,q.in_features],
                        'encoding':'calibrated_int8' if bits==8 else 'packed_pow2' if scale_kind=='pow2' else 'packed_intn',
                        'bits':bits,
                        'original_weight_sha256':original_hash,'decoded_weight_sha256':__import__('hashlib').sha256(decoded).hexdigest(),
                        'storage_bytes':q.storage_bytes,'before_bytes':original[projection].weight.numel()*2,'calibration':info[projection]})
                candidate=after
            else:
                for projection,value in original.items():setattr(mlp,projection,value)
                candidate=before
            teacher=reference
            stat={'layer':layer_id,'quantized':keep,'added_relative_rmse_V':added,'cumulative_relative_rmse_V':cumulative,
                  'seconds':time.perf_counter()-begin,'calibration':info};layer_stats.append(stat)
            dump(out/'layers.json',layer_stats)
            print('LAYER',layer_id,('KEEP_INT8' if bits==8 else f'KEEP_W{bits}') if keep else 'RESTORE_BF16','local',added,'cumulative',cumulative,flush=True)
            del inputs,down,original,quantized,before,after,collected
        # Teacher forcing: compare actual full-vocabulary logits on the7 action prediction positions.
        exact=total=0;kl_sum=0.;maximum=0.;tokens=[]
        for i,example in enumerate(examples):
            if example['split']!='V':continue
            a=teacher[i][:,-8:-1,:].to(device);b=candidate[i][:,-8:-1,:].to(device)
            za=lm.lm_head(lm.model.norm(a)).float();zb=lm.lm_head(lm.model.norm(b)).float()
            la=za.log_softmax(-1);lb=zb.log_softmax(-1)
            same=int((za.argmax(-1)==zb.argmax(-1)).sum());exact+=same;total+=7
            kl_sum+=float((la.exp()*(la-lb)).sum(-1).sum());maximum=max(maximum,float((za-zb).abs().max()))
            tokens.append({'frame':example['frame'],'same_top1':same,'teacher_top1':za.argmax(-1).cpu().tolist(),'candidate_top1':zb.argmax(-1).cpu().tolist()})
        dump(out/'teacher_forcing.json',{'prediction_positions':total,'same_top1':exact,'mean_KL':kl_sum/total,
             'maximum_logit_error':maximum,'records':tokens,'scope':'fixed captured action prefix; not KERV action replay or SR'})
        if not modules:raise RuntimeError('No block passed reconstruction gates')
        policy={'schema':'kerv_calibrated_int8_bf16_v2' if bits==8 else 'kerv_precision_first_storage_v1',
            'complete':True,'group_size':32,'module_count':len(modules),'modules':modules,
            'exact_bf16_groups':[],
            'base_revision':'fa5ae1e7509348889295bba8e08621d8b55e9baf','calibration_runtime':'native','alpha':alpha,'protected_fraction':protect,
            'estimated_static_saving_bytes':sum(x['before_bytes']-x['storage_bytes'] for x in modules),
            'quality':'NOT_PROMOTED; teacher-forcing diagnostic only; independent replay and closed-loop required'}
        dump(out/'policy.json',policy)
        print('MODEL_QUANT_COMPLETE',len(modules),'saving',policy['estimated_static_saving_bytes'],flush=True)


def main():
    from experiments.robot.libero import run_kerv_libero as evaluation
    om,op=evaluation.get_model,evaluation.get_processor;holder={}
    def get_model(cfg):
        holder['model']=om(cfg)
        if int(os.environ.get('KERV_CALIBRATION_BITS','8'))!=8:
            from KERVRuntimeOptimization.musa_safe_softmax import patch_llama_attention_softmax
            patch_llama_attention_softmax(holder['model'])
        return holder['model']
    def get_processor(cfg):
        result=op(cfg);calibrate_model(cfg,holder['model'],result);raise Finished()
    evaluation.get_model=get_model;evaluation.get_processor=get_processor
    try:evaluation.eval_libero()
    except Finished:pass
    finally:
        if 'model' in holder and hasattr(holder['model'],'close_flagos_runtime_stats'):holder['model'].close_flagos_runtime_stats()

if __name__=='__main__':main()
