"""Run the unchanged standalone reproducer and record its actual mapped libraries."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--library-dir', type=Path, required=True)
    p.add_argument('--runtime-dir', type=Path)
    p.add_argument('--executable', type=Path, required=True)
    p.add_argument('arguments', nargs=argparse.REMAINDER)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    args = a.arguments[1:] if a.arguments[:1] == ['--'] else a.arguments
    env = os.environ.copy()
    prefix = [str(a.library_dir.resolve())]
    if a.runtime_dir:
        prefix.append(str(a.runtime_dir.resolve()))
    prefix.append('/usr/local/musa/lib')
    env['LD_LIBRARY_PATH'] = ':'.join(prefix)
    env['MUSA_VISIBLE_DEVICES'] = '0'
    command = [str(a.executable), *args]
    record = {'argv':command, 'LD_LIBRARY_PATH':env['LD_LIBRARY_PATH'],
              'MUSA_VISIBLE_DEVICES':'0'}
    log_path = a.output/'stdout_stderr.log'
    with log_path.open('w') as log:
        child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        record['pid'] = child.pid
        latest_maps = []
        while child.poll() is None:
            try:
                paths = sorted({line.split()[-1] for line in
                    Path(f'/proc/{child.pid}/maps').read_text().splitlines()
                    if any(name in line for name in ['libmudnn', 'libmusart', 'libmusa.so'])})
                if paths:
                    latest_maps = paths
                    (a.output/'mapped_libraries.json').write_text(json.dumps(paths, indent=2)+'\n')
            except FileNotFoundError:
                pass
            time.sleep(0.25)
        record['exit_code'] = child.returncode
    text = log_path.read_text()
    match = re.search(r'RESULT iterations=(\d+) failures=(\d+)', text)
    record['result'] = {'iterations':int(match[1]), 'failures':int(match[2])} if match else None
    mudnn = [s for s in latest_maps if '/libmudnn' in s]
    record['all_mudnn_from_candidate'] = bool(mudnn) and all(
        Path(s).resolve().parent == a.library_dir.resolve() for s in mudnn)
    record['mapped_libraries'] = latest_maps
    record['library_selection_verified'] = record['all_mudnn_from_candidate'] and any(
        Path(s).name.startswith('libmudnn_ops.so') for s in mudnn)
    if not record['library_selection_verified']:
        record['classification'] = 'SETUP_OR_LIBRARY_SELECTION_FAILED'
    elif record['result'] is None:
        record['classification'] = 'NO_NUMERICAL_RESULT'
    elif record['result']['failures']:
        record['classification'] = 'NUMERICAL_ERROR_REPRODUCED'
    else:
        record['classification'] = 'NO_ERROR_OBSERVED_IN_THIS_RUN'
    (a.output/'results.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2), flush=True)
    if not record['library_selection_verified']:
        raise SystemExit(2)
    raise SystemExit(child.returncode)


if __name__ == '__main__':
    main()
