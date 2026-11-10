"""GPTQ on actual BF16 KERV prompt/tree activations; closed-loop validation follows.

Each MLP is calibrated independently on native upstream activations. Downstream
projection inputs include its calibrated gate/up error. This is diagnostic C/V
reconstruction, not task-success evidence or sequential full-model correction.
"""
import hashlib
import os
from pathlib import Path
import time
from acceptance.common import dump, rows, sha256


class Finished(BaseException):
    pass


def calibrate_model(cfg, model, processor):
    import numpy as np
    import torch
    from KERVRuntimeOptimization.calibrated_int8 import calibrate_linear
    from KERVRuntimeOptimization.calibrated_packed import pack_calibrated
    from experiments.robot.libero.libero_utils import get_libero_image, quat2axisangle
    from experiments.robot.robot_utils import get_action, set_seed_everywhere
    from experiments.robot.tool_utils import compute_dynamic_threshold
    from libero.libero import benchmark

    torch.set_num_threads(8)
    root = Path(os.environ['KERV_CALIBRATION_ROOT'])
    source = Path(os.environ['KERV_CALIBRATION_SOURCE_ROOT'])
    bits = int(os.environ.get('KERV_CALIBRATION_BITS', '10'))
    scale_kind = os.environ.get('KERV_CALIBRATION_SCALE_KIND', 'pow2')
    selected = sorted(int(x) for x in os.environ['KERV_CALIBRATION_LAYER_LIST'].split(','))
    out = root / os.environ.get('KERV_MODEL_OUTPUT_DIR', 'tree_gptq_pow2_w10')
    out.mkdir(exist_ok=False)
    device = next(model.parameters()).device
    layers = model.base_model.language_model.model.layers
    suite = benchmark.get_benchmark_dict()['libero_goal']()
    model._flagos_suppress_runtime_stats = True
    processor._flagos_prompt_token_cache_enabled = False
    processor._flagos_pinned_image_buffer_enabled = False
    processor._flagos_prompt_cache_model = str(cfg.pretrained_checkpoint)
    requests = [(source/'runs'/f'capture_C_task{t}_state3_s7', t, list(range(0,48,2)), 'C') for t in range(3)]
    requests.append((source/'inputs/diagnostic_capture', 0, list(range(0,141,4)), 'V'))
    protocol = {'precision_bits':bits, 'scale_kind':scale_kind, 'selected_layers':selected,
                'group_size':32, 'protected_fraction':.05, 'alpha':0.,
                'calibration_frames':72, 'diagnostic_frames':36,
                'coverage':'Actual BF16 KERV prompt and speculative verification tree activations',
                'sampling':'Per invocation: up to32 prompt or16 tree rows; prompt/tree sampled separately with CPU seed449',
                'independence':'Each MLP sees original BF16 upstream inputs; down sees quantized gate/up outputs',
                'limitations':'V is a previously seen diagnostic trajectory; reconstruction does not establish task success',
                'local_gate':'MLP output relative RMSE <=1%; failing blocks stay native BF16',
                'sources':[{'path':str(p), 'split':s, 'task':t, 'indices':ix,
                            'steps_sha256':sha256(p/'steps.jsonl'), 'observations_sha256':sha256(p/'observations.npz'),
                            'runtime_sha256':sha256(p/'runtime_stats.jsonl')} for p,t,ix,s in requests]}
    dump(out/'protocol.json', protocol)
    samples = {i:{s:{k:[] for k in ('prompt','tree')} for s in ('C','V')} for i in selected}
    current = {}; handles = []; frames_record = []
    def make_hook(layer_id):
        def hook(module, args):
            x = args[0].detach().reshape(-1,args[0].shape[-1])
            call = current['calls'].get(layer_id,0)
            current['calls'][layer_id] = call+1
            kind = 'prompt' if call==0 else 'tree'
            count = min(len(x),32 if kind=='prompt' else 16)
            index = torch.linspace(0,len(x)-1,count,device=x.device).long()
            samples[layer_id][current['split']][kind].append(x[index].cpu())
        return hook
    for i in selected:
        handles.append(layers[i].mlp.gate_proj.register_forward_pre_hook(make_hook(i)))
    with torch.inference_mode():
        try:
            for folder,task,indices,split in requests:
                captured = rows(folder/'steps.jsonl'); runtime = rows(folder/'runtime_stats.jsonl')
                set_seed_everywhere(7)
                model.start_rollout(max_steps=220, kalman_cfg={'enabled':True,'tree_enabled':True,'process_var':1.,'measurement_var':.001}, store_history=True)
                with np.load(folder/'observations.npz') as frames:
                    for index in indices:
                        current.update(split=split,calls={})
                        obs = {key:np.array(frames[key][index],copy=True) for key in frames.files}
                        fingerprint = hashlib.sha256()
                        for key in ('agentview_image','robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos'):
                            value = obs[key]
                            fingerprint.update(key.encode()); fingerprint.update(str(value.shape).encode())
                            fingerprint.update(str(value.dtype).encode()); fingerprint.update(value.tobytes())
                        if fingerprint.hexdigest()!=captured[index]['observation_hash']:
                            raise RuntimeError('Capture observation hash mismatch')
                        model._token_history = [list(x['kalman_history_token_ids']) for x in runtime[:index]]
                        observation = {'full_image':get_libero_image(obs,224), 'state':np.concatenate((obs['robot0_eef_pos'],quat2axisangle(obs['robot0_eef_quat']),obs['robot0_gripper_qpos']))}
                        get_action(cfg,model,observation,suite.get_task(task).language,processor=processor,return_time=True,
                                   generate_mode='speculative',history=[np.array(x['raw_action']) for x in captured[:index]],
                                   step_idx=index,max_steps=220,dynamic_threshold=compute_dynamic_threshold(index,220,15,3),
                                   use_kalman=True,kalman_process_var=1.,kalman_measurement_var=.001,
                                   kalman_history_window=None,kalman_tree_enabled=True)
                        frames_record.append({'task':task,'frame':index,'split':split,'observation_hash':fingerprint.hexdigest(),
                                              'calls_per_layer':dict(current['calls'])})
                print('COLLECTED',split,task,len(indices),flush=True)
        finally:
            for handle in handles: handle.remove()
        assembled = {}; counts = {}; generator = torch.Generator(device='cpu').manual_seed(449)
        for i in selected:
            assembled[i] = {}; counts[i] = {}
            for split in ('C','V'):
                pieces = []; counts[i][split] = {}
                for kind in ('prompt','tree'):
                    values_list = samples[i][split][kind]
                    if not values_list: raise RuntimeError(f'Missing actual {split}/{kind} activations at layer{i}')
                    values = torch.cat(values_list)
                    cap = (2048 if kind=='prompt' else 6144) if split=='C' else (1024 if kind=='prompt' else 3072)
                    index = torch.randperm(len(values),generator=generator)[:cap]
                    pieces.append(values[index]); counts[i][split][kind] = {'available':len(values),'selected':len(index)}
                assembled[i][split] = torch.cat(pieces)
        del samples
        torch.save({'inputs':assembled,'sampling_counts':counts,'frames':frames_record},out/'activations.pt')
        dump(out/'sampling.json',{'counts':counts,'frames':frames_record})
        modules = []; stats = []
        def weight_hash(tensor):
            return hashlib.sha256(tensor.detach().contiguous().view(torch.int16).cpu().numpy().tobytes()).hexdigest()
        for layer_id in selected:
            started = time.perf_counter(); mlp = layers[layer_id].mlp
            original = {p:getattr(mlp,p) for p in ('gate_proj','up_proj','down_proj')}
            train = assembled[layer_id]['C'].to(device); valid = assembled[layer_id]['V'].to(device)
            target = mlp(valid).detach(); quantized = {}
            try:
                for projection in ('gate_proj','up_proj'):
                    print('QUANTIZE',layer_id,projection,flush=True)
                    q = calibrate_linear(original[projection],train,32,.05,0.,'gptq',execution_device=device,
                                         precision_bits=bits,scale_kind=scale_kind).to(device)
                    quantized[projection] = q; setattr(mlp,projection,q)
                down = mlp.act_fn(mlp.gate_proj(train))*mlp.up_proj(train)
                print('QUANTIZE',layer_id,'down_proj',flush=True)
                q = calibrate_linear(original['down_proj'],down,32,.05,0.,'gptq',execution_device=device,
                                     precision_bits=bits,scale_kind=scale_kind).to(device)
                quantized['down_proj'] = q; mlp.down_proj = q
                actual = mlp(valid); diff = actual.float()-target.float()
                relative = float(diff.square().mean().sqrt()/target.float().square().mean().sqrt())
                keep = relative<=.01
                if keep:
                    for projection,q in quantized.items():
                        decoded_hash = weight_hash(q.dequantize())
                        packed = pack_calibrated(q.cpu())
                        if weight_hash(packed.dequantize())!=decoded_hash:
                            raise RuntimeError('CPU export differs from actual MUSA BF16 decode')
                        filename = f'layer{layer_id:02d}_{projection}.pt'
                        torch.save(packed.payload(),out/filename)
                        modules.append({'name':f'base_model.language_model.model.layers.{layer_id}.mlp.{projection}',
                                        'file':filename,'sha256':sha256(out/filename),'shape':[packed.out_features,packed.in_features],
                                        'encoding':'packed_pow2' if scale_kind=='pow2' else 'packed_intn','bits':bits,
                                        'original_weight_sha256':weight_hash(original[projection].weight),
                                        'decoded_weight_sha256':decoded_hash,'storage_bytes':packed.storage_bytes,
                                        'before_bytes':original[projection].weight.numel()*2,'calibration':q.calibration})
                stats.append({'layer':layer_id,'quantized':keep,'relative_mlp_rmse_V':relative,
                              'maximum_mlp_error_V':float(diff.abs().max()),'seconds':time.perf_counter()-started})
                dump(out/'layers.json',stats)
                print('LAYER',layer_id,'KEEP' if keep else 'RESTORE_BF16','relative',relative,flush=True)
            finally:
                for projection,module in original.items(): setattr(mlp,projection,module)
            del train,valid,target,actual,diff,down,quantized,q
        if not modules: raise RuntimeError('No block passed local reconstruction gate')
        dump(out/'policy.json',{'schema':'kerv_precision_first_storage_v1','complete':True,'module_count':len(modules),
                              'modules':modules,'exact_bf16_groups':[], 'group_size':32,
                              'base_revision':'fa5ae1e7509348889295bba8e08621d8b55e9baf',
                              'estimated_static_saving_bytes':sum(x['before_bytes']-x['storage_bytes'] for x in modules),
                              'quality':'NOT_PROMOTED; actual tree activation reconstruction only; closed-loop success required'})
        print('TREE_GPTQ_COMPLETE',len(modules),flush=True)


def main():
    from experiments.robot.libero import run_kerv_libero as evaluation
    om,op = evaluation.get_model,evaluation.get_processor
    holder = {}
    def get_model(cfg):
        from KERVRuntimeOptimization.musa_safe_softmax import patch_llama_attention_softmax
        holder['model'] = om(cfg)
        patch_llama_attention_softmax(holder['model'])
        return holder['model']
    def get_processor(cfg):
        processor = op(cfg); calibrate_model(cfg,holder['model'],processor); raise Finished()
    evaluation.get_model = get_model; evaluation.get_processor = get_processor
    try: evaluation.eval_libero()
    except Finished: pass
    finally:
        if 'model' in holder and hasattr(holder['model'],'close_flagos_runtime_stats'):
            holder['model'].close_flagos_runtime_stats()


if __name__=='__main__': main()
