"""One warmed, paired BF16/W8 replay on identical captured inputs and histories."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import time
from acceptance.common import dump, load, rows, sha256, distribution
from acceptance.run_matrix import make_command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--quantized-profile', choices=('quantized', 'quantized_g32', 'mixed', 'lossless_bf16_12', 'calibrated'), default='quantized')
    parser.add_argument('--baseline-profile', choices=('native', 'optimized', 'optimized_unfused'), default='optimized')
    parser.add_argument('--policy-path', type=Path)
    parser.add_argument('--experiment-name', default='quantized_fixed_input_v1')
    args = parser.parse_args()
    root = Path('/root/KERV')
    campaign, capture = args.campaign.resolve(), args.capture.resolve()
    if Path(args.experiment_name).name != args.experiment_name: raise ValueError('Use a single experiment directory name')
    experiment = campaign / 'replays' / args.experiment_name
    experiment.mkdir(parents=True, exist_ok=False)
    device = subprocess.check_output(['mthreads-gmi', '-i', '0'], text=True)
    (experiment / 'device_before.log').write_text(device)
    if 'No running processes found' not in device:
        raise RuntimeError('An idle GPU0 is required; no other jobs will be stopped')
    frames = rows(capture / 'steps.jsonl')
    protocol = {'capture': str(capture.relative_to(root)), 'capture_sha256': sha256(capture / 'observations.npz'),
                'frames': len(frames), 'profiles': [args.baseline_profile, args.quantized_profile], 'seed': 17,
                'warmup_full_passes': 1, 'measured_full_passes': 1,
                'history': 'identical captured raw actions and Kalman token prefixes before each frame',
                'scope': 'fixed inputs, no environment or SR; co-resident memory excluded; codec is lossless, INT8 is lossy',
                'timing': 'all measured frames including first; excludes load, hashing, warmup, profiler and runtime log flushing'}
    dump(experiment / 'protocol.json', protocol)
    processes, handles = {}, {}
    def wait_for(path, profile):
        start = time.monotonic()
        while not path.exists():
            if processes[profile].poll() is not None:
                raise RuntimeError(profile + ' exited before ' + path.name)
            if time.monotonic() - start > 600: raise TimeoutError(str(path))
            time.sleep(.2)
    try:
        for profile in protocol['profiles']:
            output = experiment / profile
            output.mkdir()
            case = {'profile': profile, 'kind': 'replay', 'timeout_seconds': '1800',
                    'max_steps': '220', 'planning_steps': '220', 'task_id': '0', 'init_state_id': '0', 'seed': '17'}
            if profile in ('mixed', 'calibrated'):
                if args.policy_path is None: raise ValueError('Mixed replay requires --policy-path')
                case['policy_path'] = str(args.policy_path)
            command = make_command(root, case, output / 'context.json', 'kerv-musa-runtime')
            command = [('KERV_ACCEPTANCE_CONTEXT=' if x.startswith('KERV_ACCEPTANCE_CONTEXT=') else x) for x in command]
            env = ['-e', 'KERV_REPLAY_PROFILE=' + profile,
                   '-e', 'KERV_REPLAY_CAPTURE=/workspace/' + str(capture.relative_to(root)),
                   '-e', 'KERV_REPLAY_OUTPUT=/workspace/' + str(output.relative_to(root))]
            command = command[:2] + env + command[2:]
            entry = 'calibrated_replay_entry.py' if profile == 'calibrated' else 'lossless_replay_entry.py' if profile == 'lossless_bf16_12' else 'replay_entry.py'
            command += ['--override', 'inference.runtime.entrypoint=/workspace/src/acceptance/' + entry]
            dump(output / 'command.json', {'argv': command, 'start_timestamp': datetime.now(timezone.utc).isoformat()})
            handles[profile] = (output / 'launcher.log').open('w')
            processes[profile] = subprocess.Popen(command, cwd=root, stdout=handles[profile], stderr=subprocess.STDOUT)
            print('WARM', profile, flush=True)
            wait_for(output / 'ready.json', profile)
            print('READY', profile, flush=True)
        for profile in protocol['profiles']:
            output = experiment / profile
            dump(output / 'request_0.json', {'round': 0})
            print('MEASURE', profile, flush=True)
            wait_for(output / 'done_0.json', profile)
        left, right = [rows(experiment / profile / 'measure_0.jsonl') for profile in protocol['profiles']]
        assert len(left) == len(right) == len(frames)
        for i, (a, b, captured) in enumerate(zip(left, right, frames)):
            assert a['frame'] == b['frame'] == i
            assert a['observation_hash'] == b['observation_hash'] == captured['observation_hash']
        errors = [[abs(x - y) for x, y in zip(a['raw_action'], b['raw_action'])] for a, b in zip(left, right)]
        import math
        stats = {profile: {'model_seconds': distribution(x['model_seconds'] for x in data),
                           'pipeline_seconds': distribution(x['pipeline_seconds'] for x in data)}
                 for profile, data in zip(protocol['profiles'], (left, right))}
        result = {'protocol': protocol, 'profiles': stats, 'same_observation_hashes': len(frames),
                  'exact_action_token_matches': sum(a['action_tokens'] == b['action_tokens'] for a, b in zip(left, right)),
                  'exact_raw_action_matches': sum(a['raw_action'] == b['raw_action'] for a, b in zip(left, right)),
                  'raw_action_absolute_error': distribution(x for error in errors for x in error),
                  'per_frame_raw_action_l2_error': distribution(math.sqrt(sum(x*x for x in error)) for error in errors),
                  'maximum_raw_action_absolute_error': max(x for error in errors for x in error),
                  'per_dimension_raw_action_mae': [sum(error[i] for error in errors) / len(errors) for i in range(7)],
                  'executed_gripper_matches': sum(a['executed_action'][-1] == b['executed_action'][-1] for a, b in zip(left, right)),
                  'model_mean_speedup': stats[args.baseline_profile]['model_seconds']['mean'] / stats[args.quantized_profile]['model_seconds']['mean'],
                  'pipeline_mean_speedup': stats[args.baseline_profile]['pipeline_seconds']['mean'] / stats[args.quantized_profile]['pipeline_seconds']['mean'],
                  'all_finite': all(math.isfinite(x) for error in errors for x in error),
                  'quality_status': 'fixed-input differences reported; closed-loop SR and weight bit-exact audits are separate'}
        assert result['all_finite']
        dump(experiment / 'results.json', result)
        print('REPLAY COMPLETE', experiment, flush=True)
    finally:
        for profile in processes: (experiment / profile / 'stop').touch()
        for profile, process in processes.items():
            try: code = process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                pid = int(load(experiment / profile / 'worker_start.json')['pid'])
                subprocess.run(['docker', 'exec', 'kerv-musa-runtime', 'kill', '-TERM', str(pid)])
                code = process.wait(timeout=30)
            handles[profile].close()
            record = load(experiment / profile / 'command.json')
            record.update(exit_code=code, end_timestamp=datetime.now(timezone.utc).isoformat())
            dump(experiment / profile / 'command.json', record)


if __name__ == '__main__': main()
