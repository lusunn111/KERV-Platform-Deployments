"""Independent real upstream SpecVLA LIBERO/FlagScale entry; shared IO only."""
import json
import os
from pathlib import Path
import time
import draccus
import numpy as np
import torch
from libero.libero import benchmark
from experiments.robot.libero.run_kerv_libero import GenerateConfig
from experiments.robot.libero.libero_utils import (get_libero_dummy_action, get_libero_env,
    get_libero_image, save_rollout_video)
from experiments.robot.robot_utils import (set_seed_everywhere, get_image_resize_size,
    normalize_gripper_action, invert_gripper_action)
from experiments.robot.openvla_utils import get_processor, get_vla_action
from acceptance.collect_metrics import Recorder
from ports.metax.specvla.adapter import get_specvla

@draccus.wrap()
def main(cfg: GenerateConfig):
    if cfg.task_suite_name != 'libero_goal' or cfg.max_planning_steps != 220:
        raise ValueError('This paired evaluation fixes Goal / 220 planning horizon')
    if cfg.use_kalman_tree or cfg.use_kalman_fallback:
        raise ValueError('True upstream SpecVLA has no Kalman completion/tree')
    set_seed_everywhere(cfg.seed)
    variant = os.environ['KERV_METAX_SPECVLA']
    cfg.accept_threshold = int(os.environ['KERV_METAX_SPECVLA_THRESHOLD'])
    recorder = Recorder(os.environ['KERV_ACCEPTANCE_DIR'], torch)
    cfg.unnorm_key = cfg.task_suite_name
    model = get_specvla(cfg, variant)
    model._acceptance_recorder = recorder
    recorder.loaded(model)
    processor = get_processor(cfg)
    suite = benchmark.get_benchmark_dict()[cfg.task_suite_name]()
    offset = int(os.environ.get('KERV_LIBERO_INITIAL_STATE_OFFSET', '0'))
    successes = 0; count = 0
    log_path = Path(cfg.local_log_dir); log_path.mkdir(parents=True, exist_ok=True)
    try:
        for task_id in range(min(suite.n_tasks, cfg.max_tasks)):
            env, text = get_libero_env(suite.get_task(task_id), cfg.model_family, resolution=256)
            try:
                states = suite.get_task_init_states(task_id)
                for trial in range(cfg.num_trials_per_task):
                    initial = offset + trial
                    env.reset(); obs = env.set_init_state(states[initial])
                    # Upstream eagenerate resets drafter KV before every action.
                    # No KERV rollout/history path is attached to this model.
                    model.ea_layer.reset_kv(); model.tree_mask = None
                    model.base_model.language_model.tree_mask = None
                    for _ in range(cfg.num_steps_wait):
                        obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
                    images = []; done = False
                    try:
                        for step in range(cfg.max_episode_steps):
                            recorder.begin_step(task_id, initial, step, obs['agentview_image'],
                                np.concatenate((obs['robot0_eef_pos'], obs['robot0_eef_quat'], obs['robot0_gripper_qpos'])))
                            pipeline_start = time.perf_counter()
                            image = get_libero_image(obs, get_image_resize_size(cfg)); images.append(image)
                            with recorder.transfer_context():
                                raw, timing = get_vla_action(model, processor, cfg.pretrained_checkpoint,
                                    {'full_image': image}, text, cfg.unnorm_key,
                                    return_time=True, center_crop=cfg.center_crop, generate_mode='speculative')
                            raw = np.array(raw, copy=True)
                            if raw.shape != (7,) or not np.isfinite(raw).all():
                                raise RuntimeError(f'Upstream returned invalid action {raw}')
                            action = invert_gripper_action(normalize_gripper_action(raw, binarize=True))
                            pipeline_s = time.perf_counter()-pipeline_start
                            env_start = time.perf_counter()
                            obs, _, done, _ = env.step(action.tolist())
                            recorder.end_step(raw, action, done, timing[0]-timing[1], pipeline_s,
                                time.perf_counter()-env_start, model.accept_threshold, model._last_action_token_ids_cpu)
                            if cfg.action_log_path:
                                with Path(cfg.action_log_path).open('a') as f:
                                    f.write(json.dumps(dict(task_id=task_id, initial_state_id=initial,
                                        step=step, raw_action=raw.tolist(), executed_action=action.tolist(),
                                        action_tokens=model._last_action_token_ids_cpu,
                                        generated_tokens=model._last_specvla_generated,
                                        done=bool(done), latency_seconds=timing[0]-timing[1]))+'\n')
                            if done:
                                break
                    except Exception as e:
                        recorder.end_episode(False, cfg.max_episode_steps, cfg.max_planning_steps)
                        recorder.buffers['episodes'][-1].update(termination_reason='exception', error=repr(e))
                        recorder.flush()
                        raise
                    recorder.end_episode(done, cfg.max_episode_steps, cfg.max_planning_steps)
                    successes += int(done); count += 1
                    print(f'SpecVLA {variant}: task{task_id}/init{initial}: success={done}, steps={step+1}; total={successes}/{count}', flush=True)
                    save_rollout_video(images, count, done, text)
            finally:
                env.close()
    finally:
        recorder.close()

if __name__ == '__main__':
    main()
