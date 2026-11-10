"""Real upstream SpecVLA case. Must run with the isolated port first in PYTHONPATH."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import json
import os
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-steps", type=int, default=8)
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--init-state-id", type=int, default=0)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    import numpy as np
    import torch
    from libero.libero import benchmark
    from experiments.robot import openvla_utils
    from experiments.robot import robot_utils
    from experiments.robot.libero.libero_utils import get_libero_env, get_libero_image, get_libero_dummy_action, save_rollout_video, quat2axisangle
    from acceptance.collect_metrics import Collector
    from acceptance.common import dump
    if "/ports/specvla_musa/" not in openvla_utils.__file__:
        raise RuntimeError("Refusing a mislabeled baseline; wrong module: " + openvla_utils.__file__)
    args.output.mkdir(parents=True, exist_ok=True)
    os.chdir(args.output)
    cfg = SimpleNamespace(pretrained_checkpoint="/workspace/checkpoints/openvla-libero-goal",
                          spec_checkpoint="/workspace/checkpoints/kerv-drafter", use_spec=True,
                          load_in_8bit=False, load_in_4bit=False, parallel_draft=False,
                          accept_threshold=9, model_family="openvla", center_crop=True,
                          task_suite_name="libero_goal", seed=args.seed)
    robot_utils.set_seed_everywhere(args.seed)
    torch.musa.manual_seed_all(args.seed)
    audit = Collector.from_environment(cfg)
    model = openvla_utils.get_vla(cfg)
    if audit:
        audit.loaded(model)
    processor = openvla_utils.get_processor(cfg)
    dump(args.output/"baseline_identity.json", {"implementation":"upstream SpecVLA MUSA port", "generation_module":__import__(type(model).__module__,fromlist=['x']).__file__,
        "drafter_module":__import__(type(model.ea_layer).__module__,fromlist=['x']).__file__,
        "action_utils_module":openvla_utils.__file__, "fixed_accept_threshold":9,
        "draft_total_tokens":model.ea_layer.total_tokens, "draft_depth":model.ea_layer.depth,
        "draft_top_k":model.ea_layer.top_k,"kalman":False,"dynamic_threshold":False,
        "framework_rng_seed":args.seed,"environment_seed":0})
    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    task = suite.get_task(args.task_id)
    env, description = get_libero_env(task, "openvla", resolution=256)
    frames = []; done=False
    try:
        env.reset()
        obs = env.set_init_state(suite.get_task_init_states(args.task_id)[args.init_state_id])
        for _ in range(10):
            obs,_,_,_=env.step(get_libero_dummy_action("openvla"))
        for i in range(args.max_steps):
            if audit:
                audit.begin_step(args.task_id,1,i,obs)
            image=get_libero_image(obs,224); frames.append(image)
            observation={"full_image":image,"state":np.concatenate((obs['robot0_eef_pos'],quat2axisangle(obs['robot0_eef_quat']),obs['robot0_gripper_qpos']))}
            with audit.transfer_context() if audit else __import__('contextlib').nullcontext():
                raw, times=openvla_utils.get_vla_action(model,processor,cfg.pretrained_checkpoint,observation,description,"libero_goal",return_time=True,center_crop=True,generate_mode='speculative')
            raw=np.array(raw,copy=True)
            action=robot_utils.invert_gripper_action(robot_utils.normalize_gripper_action(np.array(raw,copy=True),binarize=True))
            if audit:
                audit.policy_finished()
            obs,_,done,_=env.step(action.tolist())
            tokens=model._last_action_token_ids_cpu
            if audit:
                events=list(audit.pending.get('verify',[]))
                audit.finish_step(raw,action,done,times[0]-times[1],tokens)
            else:
                events=[]
            with (args.output/'actions.jsonl').open('a') as f:
                f.write(json.dumps({'task_id':args.task_id,'episode':1,'step':i,'raw_action':raw.tolist(),'executed_action':action.tolist(),'done':bool(done),'threshold':9,'latency_seconds':times[0]-times[1]})+'\n')
            with (args.output/'runtime_stats.jsonl').open('a') as f:
                f.write(json.dumps({'step_index':i,'action_token_ids':tokens,'accept_lengths':[x['accepted_prefix_length'] for x in events], 'cycle_count':len(events),'forced_kalman':False,'generated_token_count':len(tokens)})+'\n')
            print('SpecVLA step',i,'seconds',times[0]-times[1], 'accept',[x['accepted_prefix_length'] for x in events],flush=True)
            if done:
                break
        if audit:
            audit.finish_episode(done,args.max_steps)
        save_rollout_video(frames,1,success=done,task_description=description)
    except BaseException as error:
        if audit:
            audit.error(error)
        raise
    finally:
        env.close()
        if audit:
            audit.close()


if __name__ == '__main__':
    main()
