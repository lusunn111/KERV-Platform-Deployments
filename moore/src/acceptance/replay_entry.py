"""Resident fixed-observation replay worker, with explicit per-round requests.

KERV workers use the existing FlagScale model/operator setup. The separate B0
process imports only the isolated SpecVLA implementation. No simulator runs.
"""
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
from acceptance.common import dump, load, rows, distribution


class ReplayFinished(BaseException):
    pass


def worker(cfg, model, processor, profile):
    import numpy as np
    from experiments.robot.libero.libero_utils import get_libero_image, quat2axisangle
    from experiments.robot.robot_utils import get_action, normalize_gripper_action, invert_gripper_action, set_seed_everywhere
    from experiments.robot.openvla_utils import get_vla_action
    output = Path(os.environ["KERV_REPLAY_OUTPUT"])
    capture = Path(os.environ["KERV_REPLAY_CAPTURE"])
    captured = rows(capture / "steps.jsonl")
    captured_runtime = rows(capture / "runtime_stats.jsonl")
    frames = np.load(capture / "observations.npz")
    assert len(captured) == len(captured_runtime) == len(frames["agentview_image"])
    task = "open the middle drawer of the cabinet"
    processor._flagos_prompt_token_cache_enabled = bool(getattr(cfg, "flagos_prompt_token_cache", False))
    processor._flagos_pinned_image_buffer_enabled = False
    processor._flagos_prompt_cache_model = str(cfg.pretrained_checkpoint)
    model._flagos_suppress_runtime_stats = True
    module = __import__(type(model).__module__, fromlist=["tree_decoding"])
    real_tree_decoding = module.tree_decoding
    shapes = []
    # Hash actual loaded logical parameters, outside all timed passes.
    parameter_hashes={}
    for name,parameter in model.named_parameters():
        raw=parameter.detach().contiguous().reshape(-1).view(__import__('torch').uint8).cpu().numpy()
        parameter_hashes[name]={"shape":list(parameter.shape),"dtype":str(parameter.dtype),"sha256":hashlib.sha256(raw.tobytes()).hexdigest()}
    dump(output/'parameter_hashes.json',parameter_hashes)
    def traced_tree(*args, **kwargs):
        shapes.append({"draft_token_shape":list(args[2].shape), "prompt_embedding_shape":list(args[1].shape)})
        result=real_tree_decoding(*args, **kwargs)
        if getattr(model,'_acceptance_replay_frame',-1)==77:
            import torch
            index=sum(1 for x in output.glob('warmup_step077_cycle*.pt'))
            torch.save({"draft_tokens":args[2].detach().cpu(),"retrieve_indices":args[6].detach().cpu(),
                        "prompt_embeddings":args[1].detach().cpu(),"logits":result[0].detach().cpu()},
                       output/('warmup_step077_cycle%d.pt'%index))
        return result
    module.tree_decoding = traced_tree
    real_predict=model.predict_action
    def traced_predict(*args, **kwargs):
        if getattr(model,'_acceptance_replay_frame',-1)==77:
            import torch
            torch.save({key:value.detach().cpu() for key,value in kwargs.items() if isinstance(value,torch.Tensor)},
                       output/'warmup_step077_model_inputs.pt')
        return real_predict(*args,**kwargs)
    model.predict_action=traced_predict

    def one_pass(label):
        set_seed_everywhere(int(cfg.seed))
        if hasattr(model, "start_rollout"):
            model.start_rollout(max_steps=220, kalman_cfg={"enabled":True,"tree_enabled":True,"process_var":1.,"measurement_var":.001}, store_history=True)
        result=[]
        for i, expected in enumerate(captured):
            model._acceptance_replay_frame=i
            obs = {key:np.array(frames[key][i],copy=True) for key in frames.files}
            fingerprint=hashlib.sha256()
            for key in ("agentview_image", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"):
                value=obs[key]
                fingerprint.update(key.encode());fingerprint.update(str(value.shape).encode());fingerprint.update(str(value.dtype).encode());fingerprint.update(value.tobytes())
            assert fingerprint.hexdigest()==expected["observation_hash"],(i,"capture hash mismatch")
            history=[np.array(x["raw_action"]) for x in captured[:i]]
            if hasattr(model,"_token_history"):
                model._token_history=[list(x["kalman_history_token_ids"]) for x in captured_runtime[:i]]
            start=time.perf_counter()
            image=get_libero_image(obs,224)
            observation={"full_image":image,"state":np.concatenate((obs['robot0_eef_pos'],quat2axisangle(obs['robot0_eef_quat']),obs['robot0_gripper_qpos']))}
            if profile == "specvla":
                action,ts=get_vla_action(model,processor,cfg.pretrained_checkpoint,observation,task,"libero_goal",return_time=True,center_crop=True,generate_mode='speculative')
            else:
                from experiments.robot.tool_utils import compute_dynamic_threshold
                action,ts=get_action(cfg,model,observation,task,processor=processor,return_time=True,generate_mode='speculative',
                    history=history,step_idx=i,max_steps=220,dynamic_threshold=compute_dynamic_threshold(i,220,15,3),
                    use_kalman=True,kalman_process_var=1.,kalman_measurement_var=.001,kalman_history_window=None,kalman_tree_enabled=True)
            raw=np.array(action,copy=True)
            executed=invert_gripper_action(normalize_gripper_action(np.array(raw,copy=True),binarize=True))
            pipeline=time.perf_counter()-start
            tokens=list(model._last_action_token_ids_cpu)
            assert len(tokens)==len(raw)==7 and np.isfinite(raw).all()
            result.append({"frame":i,"profile":profile,"pass":label,"observation_hash":expected["observation_hash"],
                           "model_seconds":float(ts[0]-ts[1]),"pipeline_seconds":pipeline,"action_tokens":tokens,
                           "raw_action":raw.tolist(),"executed_action":executed.tolist(),
                           "tokens_match_capture":tokens==expected["action_tokens"]})
        with (output/(label+'.jsonl')).open('w') as f:
            for row in result:f.write(json.dumps(row,allow_nan=False)+'\n')
        report={"profile":profile,"pass":label,"frames":len(result),"model_seconds":distribution(x['model_seconds'] for x in result),
                "pipeline_seconds":distribution(x['pipeline_seconds'] for x in result),
                "tokens_matching_capture":sum(x['tokens_match_capture'] for x in result)}
        dump(output/(label+'.json'),report)
        return report

    try:
        warm=one_pass('warmup')
        module.tree_decoding=real_tree_decoding
        model.predict_action=real_predict
        dump(output/'warmup_shapes.json',shapes)
        dump(output/'ready.json',{"profile":profile,"pid":os.getpid(),"frames":len(captured),"warmup":warm,
            "quantization_manifest":getattr(model,'_flagos_w8a16_manifest',None),
            "int8_weight_buffer_count":sum(1 for name,buf in model.named_buffers() if name.endswith('qweight') and str(buf.dtype)=='torch.int8'),
            "parameter_devices":sorted({str(p.device) for p in model.parameters()}),"seed":cfg.seed,
            "model_source":module.__file__,"history_protocol":"same captured raw-action and Kalman-token prefix before every frame",
            "runtime_stats_during_timing":False,"profiler_during_timing":False})
        for round_id in range(3):
            request=output/('request_%d.json'%round_id)
            while not request.exists():
                if (output/'stop').exists():return
                time.sleep(.05)
            report=one_pass('measure_%d'%round_id)
            dump(output/('done_%d.json'%round_id),report)
        while not (output/'stop').exists():time.sleep(.05)
    finally:
        module.tree_decoding=real_tree_decoding
        model.predict_action=real_predict
        if hasattr(model,'close_flagos_runtime_stats'):model.close_flagos_runtime_stats()


def main():
    dump(Path(os.environ['KERV_REPLAY_OUTPUT'])/'worker_start.json',{'pid':os.getpid(),'output':os.environ['KERV_REPLAY_OUTPUT']})
    profile=os.environ['KERV_REPLAY_PROFILE']
    if profile=='specvla':
        from experiments.robot.openvla_utils import get_vla, get_processor
        from experiments.robot.robot_utils import set_seed_everywhere
        cfg=SimpleNamespace(pretrained_checkpoint='/workspace/checkpoints/openvla-libero-goal',spec_checkpoint='/workspace/checkpoints/kerv-drafter',
                            use_spec=True,load_in_8bit=False,load_in_4bit=False,parallel_draft=False,accept_threshold=9,
                            model_family='openvla',center_crop=True,task_suite_name='libero_goal',seed=17)
        set_seed_everywhere(17)
        worker(cfg,get_vla(cfg),get_processor(cfg),profile)
        return
    from experiments.robot.libero import run_kerv_libero as evaluation
    original_model=evaluation.get_model;original_processor=evaluation.get_processor;holder={}
    def model_hook(cfg):
        model=original_model(cfg);holder['model']=model;return model
    def processor_hook(cfg):
        processor=original_processor(cfg)
        worker(cfg,holder['model'],processor,profile)
        raise ReplayFinished()
    evaluation.get_model=model_hook;evaluation.get_processor=processor_hook
    try:evaluation.eval_libero()
    except ReplayFinished:pass


if __name__=='__main__':
    main()
