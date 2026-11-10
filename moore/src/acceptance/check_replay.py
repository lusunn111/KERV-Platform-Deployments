"""Recompute fixed replay completeness and action equality from every raw frame."""
import argparse
import math
from pathlib import Path
from acceptance.common import dump, load, rows, sha256, distribution


def check(campaign):
    experiment = campaign / 'replays/fixed_input_v1'
    protocol = load(experiment / 'protocol.json')
    results = load(experiment / 'results.json')
    capture = Path('/root/KERV') / protocol['capture']
    n = protocol['frames']
    errors = []
    if sha256(capture / 'observations.npz') != protocol['capture_sha256']:
        errors.append('Captured observations hash changed')
    captured = rows(capture / 'steps.jsonl')
    streams = {}
    rounds = []
    for profile in ('native', 'optimized', 'specvla'):
        streams[profile] = []
        if load(experiment / profile / 'command.json').get('exit_code') != 0:
            errors.append(profile + ': worker did not exit successfully')
        for pass_id in range(protocol['measured_full_passes_per_profile']):
            label = 'measure_%d' % pass_id
            data = rows(experiment / profile / (label + '.jsonl'))
            if len(data) != n or [x['frame'] for x in data] != list(range(n)):
                errors.append(profile + '/' + label + ': missing/duplicate/reordered frames')
            for item in data:
                frame = item['frame']
                if item['profile'] != profile or item['pass'] != label:
                    errors.append(profile + '/' + label + ': identity mismatch')
                if item['observation_hash'] != captured[frame]['observation_hash']:
                    errors.append(profile + '/' + label + ': different observation')
                for key in ('raw_action', 'executed_action', 'action_tokens'):
                    if len(item[key]) != 7 or not all(math.isfinite(v) for v in item[key]):
                        errors.append(profile + '/' + label + ': invalid action')
                if not all(math.isfinite(item[k]) and item[k] > 0 for k in ('model_seconds', 'pipeline_seconds')):
                    errors.append(profile + '/' + label + ': invalid duration')
            rounds.append({'profile': profile, 'round': pass_id + 1, 'actions': len(data),
                           'model_seconds': distribution(x['model_seconds'] for x in data),
                           'pipeline_seconds': distribution(x['pipeline_seconds'] for x in data)})
            streams[profile].extend(data)
    native, optimized = streams['native'], streams['optimized']
    pairs = list(zip(native, optimized))
    equal = {key: sum(a[key] == b[key] for a, b in pairs)
             for key in ('action_tokens', 'raw_action', 'executed_action', 'observation_hash')}
    total = n * protocol['measured_full_passes_per_profile']
    profile_statistics = {}
    for profile, data in streams.items():
        stats = {'actions': len(data), 'model_seconds': distribution(x['model_seconds'] for x in data),
                 'pipeline_seconds': distribution(x['pipeline_seconds'] for x in data)}
        original = data[:n]
        stats['within_profile_token_matches'] = sum(x['action_tokens'] == original[x['frame']]['action_tokens'] for x in data)
        if stats['actions'] != results['profiles'][profile]['actions'] or stats['model_seconds'] != results['profiles'][profile]['model_seconds']:
            errors.append(profile + ': published results differ from raw recomputation')
        profile_statistics[profile] = stats
    correctness = all(value == total for value in equal.values()) and len(native) == len(optimized) == total
    report = {'evidence_integrity': 'PASS' if not errors else 'FAIL', 'errors': errors,
              'native_optimized_action_preservation': 'PASS' if correctness else 'FAIL',
              'total_actions_per_profile': total, 'equal_records': equal, 'profiles': profile_statistics,
              'model_mean_speedup': profile_statistics['native']['model_seconds']['mean'] / profile_statistics['optimized']['model_seconds']['mean'],
              'pipeline_mean_speedup': profile_statistics['native']['pipeline_seconds']['mean'] / profile_statistics['optimized']['pipeline_seconds']['mean'],
              'rounds': rounds, 'scope': 'Fixed captured observations and histories only; not closed-loop quality or single-model memory'}
    dump(campaign / 'reports/replay_evidence_checks.json', report)
    print({key: report[key] for key in ('evidence_integrity', 'native_optimized_action_preservation', 'total_actions_per_profile', 'equal_records', 'model_mean_speedup', 'pipeline_mean_speedup')})
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    check(parser.parse_args().campaign)
