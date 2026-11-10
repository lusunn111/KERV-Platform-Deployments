#!/usr/bin/env python3
"""Compose an auditable MetaX config, then use the actual FlagScale worker."""
import argparse
import hashlib
import tarfile
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import sys
from omegaconf import OmegaConf


def main():
    root = Path(__file__).resolve().parents[2]
    port = root / 'ports/moore_ascend'
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', choices=['native', 'optimized', 'w8a16', 'bf16_packed', 'optimized_v2', 'bf16_packed_v2', 'numerical_native_rms', 'numerical_fixed', 'numerical_fixed_packed', 'upstream_original', 'boundary_fixed'], default='native')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--max-tasks', type=int, default=1)
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--max-episode-steps', type=int, default=220)
    parser.add_argument('--planning-horizon', type=int, default=220)
    parser.add_argument('--v2-features',default=None)
    parser.add_argument('--specvla-threshold',type=int,default=9, help='Pinned upstream relaxed entry=9; strict Spec entry=0; explicit baseline identity')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--initial-state', type=int, default=0)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--prepare-only', action='store_true', help='CPU config/source snapshot only; no Torch import, device query, worker or measurement')
    parser.add_argument('--capture-inputs', action='store_true', help='Diagnostic only: retain real pre-crop images')
    args = parser.parse_args()
    if not args.prepare_only:
        import torch
        if not getattr(torch.version, 'maca', None):
            raise RuntimeError('MetaX vendor PyTorch required; refusing ordinary CUDA/CPU torch')
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('Expose exactly one available MetaX GPU')
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    names = subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard'],cwd=root,text=True).splitlines()
    source_files = sorted({name for name in names if (Path(name).suffix in {'.py','.cpp','.h','.yaml','.yml','.sh','.toml','.txt','.json'}
                           or Path(name).name.startswith(('LICENSE','NOTICE'))) and (root/name).is_file()})
    source_hashes = {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in source_files}
    with tarfile.open(output/'source.tar.gz','w:gz') as archive:
        for name in source_files:archive.add(root/name,arcname=name,recursive=False)
    (output/'source-state.json').write_text(json.dumps(dict(
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        files=source_hashes),indent=2)+'\n')

    env = os.environ.copy()
    env.update(KERV_METAX_V2='', KERV_METAX_QUANT='none', KERV_METAX_SPECVLA='')
    env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + env['PATH']
    env.update(KERV_DEVICE_BACKEND='metax', KERV_ACTION_PROTOCOL='seven',
               KERV_RELEASE_ROOT=str(port), KERV_RUNTIME_OPT_ROOT=str(port / 'runtime_opt'),
               FLAGSCALE_ROOT=str(root / 'third_party/FlagScale'), KERV_PYTHON=sys.executable,
               KERV_BASE_CHECKPOINT=str(root / 'checkpoints/openvla-libero-goal'),
               KERV_DRAFT_CHECKPOINT=str(root / 'checkpoints/kerv-drafter'),
               KERV_OUTPUT_ROOT=str(output), LIBERO_CONFIG_PATH=str(root / '.libero'),
               KERV_LIBERO_ROOT=str(root / 'third_party/LIBERO'),
               KERV_ACCEPTANCE_DIR=str(output / 'raw'), KERV_ACCEPTANCE_RUN_ID=output.name,
               KERV_LIBERO_INITIAL_STATE_OFFSET=str(args.initial_state),
               PYOPENGL_PLATFORM='osmesa', MUJOCO_GL='osmesa', HF_HUB_OFFLINE='1',
               OMP_NUM_THREADS='16', TF_NUM_INTRAOP_THREADS='1', TF_NUM_INTEROP_THREADS='1')
    env['PYTHONPATH'] = os.pathsep.join(map(str, [port / 'runtime_opt', port, port / 'openvla',
                                               root / 'third_party/FlagScale', root / 'third_party/LIBERO', root]))
    os.environ.update(env)
    cfg = OmegaConf.merge(OmegaConf.load(port / 'runtime_opt/FlagScale/configs/kerv_libero_goal.yaml'),
                          OmegaConf.load(root / 'ports/metax/native.yaml'))
    if args.profile in {'optimized', 'w8a16', 'bf16_packed', 'optimized_v2', 'bf16_packed_v2', 'numerical_native_rms', 'numerical_fixed', 'numerical_fixed_packed'}:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(root / 'ports/metax/optimized.yaml'))
    if args.profile in {'optimized_v2','bf16_packed_v2'}:
        cfg=OmegaConf.merge(cfg,OmegaConf.load(root/'ports/metax/optimized_v2.yaml'))
        if args.v2_features is not None:cfg.experiment.envs.KERV_METAX_V2=args.v2_features
    if args.profile == 'numerical_fixed_packed':
        cfg=OmegaConf.merge(cfg,OmegaConf.load(root/'ports/metax/numerical_fixed.yaml'),
                            OmegaConf.load(root/'ports/metax/numerical_fixed_packed.yaml'))
    if args.profile in {'numerical_native_rms','numerical_fixed'}:
        cfg=OmegaConf.merge(cfg,OmegaConf.load(root/('ports/metax/'+args.profile+'.yaml')))
    if args.profile=='bf16_packed_v2':
        cfg=OmegaConf.merge(cfg,OmegaConf.load(root/'ports/metax/bf16_packed_v2.yaml'))
    if args.profile == 'w8a16':
        cfg.experiment.envs.KERV_METAX_QUANT = 'w8a16_group32'
        cfg.inference.runtime.run_id_note = 'metax-w8a16-group32-experimental'
    if args.profile=='bf16_packed':
        cfg.experiment.envs.KERV_METAX_QUANT='bf16_lossless'
        cfg.inference.runtime.run_id_note='metax-bf16-lossless-dictionary4'
    if args.profile in {'upstream_original','boundary_fixed'}:
        cfg=OmegaConf.merge(cfg,OmegaConf.load(root/'ports/metax/specvla.yaml'))
        cfg.experiment.envs.KERV_METAX_SPECVLA=args.profile
        cfg.experiment.envs.KERV_METAX_SPECVLA_THRESHOLD=str(args.specvla_threshold)
        cfg.inference.runtime.source_root=str(root)
        cfg.inference.runtime.entrypoint='ports/metax/specvla/eval.py'
        cfg.inference.runtime.run_id_note='specvla-'+args.profile+'-threshold'+str(args.specvla_threshold)
    if args.capture_inputs:
        cfg.experiment.envs.KERV_METAX_CAPTURE_INPUTS = str(output / 'observations')
    cfg.inference.runtime.max_tasks = args.max_tasks
    cfg.inference.runtime.num_trials_per_task = args.trials
    cfg.inference.runtime.max_episode_steps = args.max_episode_steps
    cfg.inference.runtime.max_planning_steps = args.planning_horizon
    cfg.inference.runtime.seed = args.seed
    cfg.action = 'dryrun' if args.dry_run else 'test'
    OmegaConf.save(cfg, output / 'metax.yaml', resolve=True)
    if args.prepare_only:
        (output/'prepared.json').write_text(json.dumps(dict(status='prepared_only',profile=args.profile,
            worker_launched=False,gpu_queried=False,model_loaded=False,arguments=vars(args)),indent=2)+'\n')
        return
    command = [sys.executable, str(root / 'third_party/FlagScale/run.py'),
               '--config-path', str(output), '--config-name', 'metax']
    (output / 'launch.json').write_text(json.dumps(dict(command=command, profile=args.profile,
        torch=torch.__version__, maca=torch.version.maca, device=torch.cuda.get_device_name(),device_properties=str(torch.cuda.get_device_properties(0)),
        physical_gpu=env.get('MACA_VISIBLE_DEVICES'), logical_device='cuda:0',
        planning_horizon=args.planning_horizon, step_limit=args.max_episode_steps,seed=args.seed,initial_state=args.initial_state), indent=2)+'\n')
    stop = threading.Event()
    def sample_vendor():
        with (output / 'vendor-memory.log').open('w') as stream:
            while not stop.is_set():
                stream.write(f'UTC_UNIX_SECONDS {time.time()}\n'); stream.flush()
                subprocess.run(['mx-smi', '-i', env.get('MACA_VISIBLE_DEVICES', '0'),
                                '--show-memory', '--show-process'], stdout=stream, stderr=stream)
                stream.flush()
                stop.wait(1.0)
    sampler = threading.Thread(target=sample_vendor, daemon=True)
    sampler.start()
    try:
        result = subprocess.run(command, cwd=root / 'third_party/FlagScale', env=env)
    finally:
        stop.set(); sampler.join(timeout=5)

    (output / 'exit_code.txt').write_text(str(result.returncode)+'\n')
    raise SystemExit(result.returncode)

if __name__ == '__main__':
    main()
