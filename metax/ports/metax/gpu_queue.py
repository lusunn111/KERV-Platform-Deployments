"""Host-only, fail-closed GPU0 queue. No Torch import or GPU context while waiting."""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
LOCK = Path('/tmp/kerv-metax-card0.lock')
MARKER = '<!-- metax-model-performance-live -->'

def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def atomic(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)

def parse_device(text):
    # -i 0 --show-memory --show-process. Missing/unrecognized data is never idle.
    memory = re.search(r'^\s+vram used\s+:\s+(\d+) KB', text, re.M)
    process_header = '| Process:' in text and 'Usage(MiB)' in text
    pids = [int(x) for x in re.findall(r'^\|\s*0\s+(\d+)\s+\S+\s+\d+\s*\|', text, re.M)]
    # If a process row is present but not parseable, fail closed as well.
    rows = re.findall(r'^\|\s*0\s+[^|]+\|', text, re.M)
    valid = memory is not None and process_header and len(rows) == len(pids) and 'GPU#0' in text
    return dict(valid=valid, pids=pids, memory_kib=int(memory[1]) if memory else None)

def device():
    result = subprocess.run(['mx-smi', '-i', '0', '--show-memory', '--show-process'],
                            capture_output=True, text=True, timeout=20)
    value = parse_device(result.stdout)
    value['valid'] &= result.returncode == 0
    value['observed_utc'] = utc()
    return value

def idle(value, max_kib):
    return value['valid'] and not value['pids'] and value['memory_kib'] <= max_kib

def source_hashes():
    files = subprocess.check_output(['git', 'ls-files'], cwd=ROOT, text=True).splitlines()
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files
            if name not in {'docs/METAX_PROGRESS.md', 'docs/METAX_MODEL_PERFORMANCE.md'} and (ROOT / name).is_file()}

def verify_freeze(plan):
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if head != plan['git_head'] or source_hashes() != plan['source_hashes']:
        raise RuntimeError('Frozen Git/config/source differs; regenerate and explicitly re-arm the queue')
    dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).splitlines()
    if any(line[3:] not in {'docs/METAX_PROGRESS.md','docs/METAX_MODEL_PERFORMANCE.md'} for line in dirty):
        raise RuntimeError('Worktree changed outside live reports; refusing GPU launch')
    if shutil.disk_usage(ROOT).free < plan.get('min_free_disk_bytes',8_000_000_000):
        raise RuntimeError('Less than reserved 8 GB free disk; refusing an incomplete capture')
    matrix=Path(plan['matrix_path'])
    if hashlib.sha256(matrix.read_bytes()).hexdigest()!=plan['evaluation_matrix_sha256']:
        raise RuntimeError('Frozen development/holdout matrix changed')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--interval', type=float, default=60)
    p.add_argument('--max-idle-kib', type=int, default=950 * 1024,
                   help='historical base 858-872 MiB; do not raise to accommodate a model')
    a = p.parse_args()
    if a.interval < 60:
        p.error('Real polling interval must be at least 60 seconds')
    out = a.directory.resolve(); out.mkdir(parents=True, exist_ok=True)
    owner = (out / 'supervisor.lock').open('a+')
    fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    completed = {}
    if (out / 'queue-state.json').exists():
        completed = json.loads((out / 'queue-state.json').read_text()).get('jobs', {})
        if any(x['status'] == 'running' for x in completed.values()):
            raise RuntimeError('Interrupted running job needs audit before recovery; no implicit rerun')
    def record(status, **values):
        state = dict(status=status, updated_utc=utc(), supervisor_pid=os.getpid(), jobs=completed, **values)
        atomic(out / 'queue-state.json', state)
        with (out / 'queue-events.jsonl').open('a') as f:
            f.write(json.dumps(state, ensure_ascii=False) + '\n')
        path = ROOT / 'docs/METAX_PROGRESS.md'
        text = path.read_text()
        section = (f'{MARKER}\n## 模型性能任务实时状态（{state["updated_utc"]}）\n\n'
                   f'- 状态：`{status}`；等待不计实验运行。物理卡0 → 逻辑cuda:0。\n'
                   f'- 当前信息：`{json.dumps(values, ensure_ascii=False)}`\n'
                   f'- 队列状态/日志：`{out.relative_to(ROOT)}/queue-state.json`、`queue-events.jsonl`；'
                   '每60秒检查，连续两次无进程且显存≤950MiB后锁内复查。\n'
                   f'{MARKER}\n')
        if text.count(MARKER) == 2:
            before, _, after = text.split(MARKER)
            path.write_text(before + section + after)
        else:
            path.write_text(text.split('\n', 1)[0] + '\n\n' + section + '\n' + text.split('\n', 1)[1])
    streak = 0
    while True:
        observation = device()
        streak = streak + 1 if idle(observation, a.max_idle_kib) else 0
        plan_path = out / 'queue.json'
        if not plan_path.exists():
            record('waiting_gpu', device=observation, idle_streak=streak, queue='CPU preparation; not armed')
        else:
            plan = json.loads(plan_path.read_text())
            pending = [j for j in plan['jobs'] if j['id'] not in completed]
            if not pending:
                record('completed' if all(j['status'] == 'completed' for j in completed.values()) else 'failed',
                       device=observation, note='Prepared stage finished; subsequent selection/holdout requires audit')
                return
            job = pending[0]
            failed = [key for key in job.get('requires', [])
                      if completed.get(key, {}).get('status') != 'completed']
            if failed:
                completed[job['id']]=dict(status='failed', reason='Dependency failed or absent; not executed', dependencies=failed, attempted=False)
                record('failed', job=job['id'], dependencies=failed)
                continue
            if streak >= 2:
                with LOCK.open('a+') as lock:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        record('waiting_gpu', device=observation, idle_streak=streak, reason='KERV lock busy')
                    else:
                        check = device()
                        if idle(check, a.max_idle_kib):
                            try:
                                verify_freeze(plan)
                                log = out / (job['id'] + '.log')
                                if log.exists():
                                    raise RuntimeError('Existing log without terminal job state; audit required')
                                completed[job['id']] = dict(status='running', started_utc=utc(), command=job['command'])
                                record('running', job=job['id'], device=check, log=str(log))
                                start = time.monotonic()
                                with log.open('x') as stream:
                                    result = subprocess.run(job['command'], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                                completed[job['id']].update(status='completed' if result.returncode == 0 else 'failed',
                                    exit_code=result.returncode, finished_utc=utc(), runtime_seconds=time.monotonic()-start)
                                record(completed[job['id']]['status'], job=job['id'], exit_code=result.returncode)
                                with (out/'summary.log').open('a') as summary_log:
                                    subprocess.run([sys.executable,'-m','ports.metax.model_performance_summary','--directory',str(out)],
                                        cwd=ROOT,stdout=summary_log,stderr=subprocess.STDOUT)
                            except Exception as e:
                                record('failed', job=job['id'], reason=str(e))
                                return
                        else:
                            record('waiting_gpu', device=check, idle_streak=0, reason='Lock-time recheck busy')
                        streak = 0
            else:
                record('waiting_gpu', device=observation, idle_streak=streak, next_job=job['id'])
        time.sleep(a.interval)

if __name__ == '__main__':
    main()
