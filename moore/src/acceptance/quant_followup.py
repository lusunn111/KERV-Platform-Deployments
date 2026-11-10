"""Finish registered group32 checks and a fixed-input error comparison serially."""
import argparse
import subprocess
import sys
import time
from pathlib import Path
from acceptance.common import load, dump


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    args = parser.parse_args()
    c = args.campaign
    waiting = c / 'operations/full_quality_matrix/command.json'
    start = time.monotonic()
    while 'exit_code' not in load(waiting):
        if time.monotonic() - start > 3600: raise TimeoutError('Original registered queue did not finish')
        time.sleep(2)
    if load(waiting)['exit_code'] != 0:
        raise RuntimeError('Original matrix infrastructure failed; do not hide it by continuing')
    def recorded(name, command):
        subprocess.run([sys.executable, '-m', 'acceptance.record_command', '--campaign', str(c),
                        '--id', name, '--', *command], check=True)
    matrix = [sys.executable, '-m', 'acceptance.run_matrix', '--campaign', str(c),
              '--cases', str(c / 'inventory/cases_addendum_01.csv')]
    recorded('group32_state0', [*matrix, '--case-id', 'full_quantized_g32_s7_state0'])
    case = load(c / 'runs/full_quantized_g32_s7_state0/run.json')
    if case['status'] != 'PASS': raise RuntimeError('Group32 execution failed')
    if case['task_success']:
        recorded('group32_additional_states', [*matrix, '--case-id', 'full_quantized_g32_s7_state1',
                                              '--case-id', 'full_quantized_g32_s7_state2'])
    profile = 'quantized_g32' if case['task_success'] else 'quantized'
    dump(c / 'inventory/replay_selection.json', {'selected_for_error_measurement': profile,
         'reason': 'Pre-registered policy: group32 if state0 completes, otherwise retain the original group128 candidate',
         'not_default_promotion': True, 'group32_state0_success': case['task_success']})
    for attempt in range(16):
        device = subprocess.check_output(['mthreads-gmi', '-i', '0'], text=True)
        if 'No running processes found' in device: break
        if attempt == 15: raise RuntimeError('GPU0 occupied; no external job will be stopped')
        time.sleep(2)
    recorded('fixed_input_quantization_error', [sys.executable, '-m', 'acceptance.quant_replay',
             '--campaign', str(c), '--capture', str(c / 'inputs/fixed_input_capture'),
             '--quantized-profile', profile])


if __name__ == '__main__': main()
