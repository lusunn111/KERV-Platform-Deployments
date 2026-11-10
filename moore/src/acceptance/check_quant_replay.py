"""Independently audit the paired quantization replay without running a model."""
import argparse
import math
from pathlib import Path
from acceptance.common import load, rows, dump, sha256, distribution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--experiment-name', default='quantized_fixed_input_v1')
    args = parser.parse_args()
    c = args.campaign
    experiment = c / 'replays' / args.experiment_name
    published = load(experiment / 'results.json')
    protocol = published['protocol']
    capture = Path('/root/KERV') / protocol['capture']
    assert sha256(capture / 'observations.npz') == protocol['capture_sha256']
    reference = rows(capture / 'steps.jsonl')
    streams = []
    for profile in protocol['profiles']:
        assert load(experiment / profile / 'command.json')['exit_code'] == 0
        data = rows(experiment / profile / 'measure_0.jsonl')
        assert len(data) == protocol['frames'] == len(reference)
        assert [x['frame'] for x in data] == list(range(len(reference)))
        for item, expected in zip(data, reference):
            assert item['profile'] == profile and item['pass'] == 'measure_0'
            assert item['observation_hash'] == expected['observation_hash']
            for field in ('action_tokens', 'raw_action', 'executed_action'):
                assert len(item[field]) == 7 and all(math.isfinite(x) for x in item[field])
            for field in ('model_seconds', 'pipeline_seconds'):
                assert math.isfinite(item[field]) and item[field] > 0
        for field in ('model_seconds', 'pipeline_seconds'):
            assert distribution(x[field] for x in data) == published['profiles'][profile][field]
        streams.append(data)
    left, right = streams
    errors = [abs(x-y) for a, b in zip(left, right) for x, y in zip(a['raw_action'], b['raw_action'])]
    assert distribution(errors) == published['raw_action_absolute_error']
    assert max(errors) == published['maximum_raw_action_absolute_error']
    tokens = sum(a['action_tokens'] == b['action_tokens'] for a, b in zip(left, right))
    gripper = sum(a['executed_action'][-1] == b['executed_action'][-1] for a, b in zip(left, right))
    assert tokens == published['exact_action_token_matches']
    assert gripper == published['executed_gripper_matches']
    result = {'evidence_integrity': 'PASS', 'frames_per_profile': len(left), 'same_input_hashes': len(left),
              'exact_action_token_matches': tokens, 'gripper_matches': gripper,
              'all_actions_and_durations_finite': True, 'worker_exit_codes': [0, 0],
              'raw_action_absolute_error': distribution(errors),
              'scope': 'Data integrity; not closed-loop SR or lossless quality approval'}
    dump(c / 'reports' / (args.experiment_name + '_evidence_checks.json' if args.experiment_name != 'quantized_fixed_input_v1' else 'replay_evidence_checks.json'), result)
    print(result)


if __name__ == '__main__': main()
