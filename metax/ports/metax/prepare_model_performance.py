"""CPU audit of development/holdout separation and resumable serial job manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from ports.metax.gpu_queue import ROOT, source_hashes, atomic

BASE = ROOT / 'outputs/metax_model_performance_20261105'
PROFILES = ['upstream_original', 'boundary_fixed', 'native', 'numerical_fixed', 'numerical_fixed_packed']

def prepare(out):
    out.mkdir(parents=True, exist_ok=True)
    # Read only trial/init metadata, not held-out images or outcomes. Prior
    # development is registered even if a run was truncated or unsuccessful.
    uses = {}; scanned = 0
    for path in sorted((ROOT/'outputs').glob('**/raw/episodes.jsonl')):
        if out in path.parents:continue
        for line in path.read_text().splitlines():
            row=json.loads(line);initial=row.get('init_state_id')
            if initial is None:continue
            uses.setdefault(int(initial),set()).add(str(path.relative_to(ROOT)))
            scanned += 1
    available=[i for i in range(2,50) if i not in uses]
    holdout=available[:5]
    if len(holdout)!=5:raise RuntimeError('Five untouched initial states unavailable')
    matrix=dict(seed=7, suite='libero_goal', horizon=220, episode_limit=220,
        stable_reference='cf41388', stable_profile='numerical_fixed',
        development=[dict(task=t,init=i) for t in range(3) for i in [0,1]],
        holdout=[dict(task=t,init=i) for t in range(10) for i in holdout],
        holdout_initial_states=holdout, historical_init_usage={str(i):sorted(v) for i,v in uses.items()},
        prior_episode_metadata_rows=scanned,
        holdout_status='prepared_not_executed; freeze candidate selection after development audit, then arm separately',
        quality_caveat='50 paired cases are a finite stage evaluation; not proof of population SR loss <5%',
        baseline_identity='Pinned upstream SpecVLA relaxed fixed-threshold9. Strict upstream Spec threshold0 is available separately, not silently substituted. Both upstream_original and boundary_fixed retained.')
    atomic(out/'evaluation-matrix.json',matrix)
    jobs=[]
    def job(name, profile, tasks, trials, steps, requires=()):
        target=f'/workspace/outputs/{out.name}/{name}'
        cmd=['docker','exec','-w','/workspace','-e','PYTHONUNBUFFERED=1','kerv-metax',
             '/workspace/.venv-metax/bin/python','ports/metax/run.py','--profile',profile,
             '--max-tasks',str(tasks),'--trials',str(trials),'--max-episode-steps',str(steps),
             '--planning-horizon','220','--seed','7','--initial-state','0','--output-dir',target]
        jobs.append(dict(id=name,profile=profile,command=cmd,requires=list(requires),
            output_dir=str(out/name),expected_episodes=tasks*trials,diagnostic=False,
            role='smoke_not_SR' if steps<220 else 'development',
            parameters=dict(tasks=tasks,trials=trials,episode_limit=steps,initial_state=0,seed=7,horizon=220)))
    # All algorithms have a fresh process. Do not let a broken original baseline
    # block independent corrected/KERV runs; each full run depends on its own smoke.
    for profile in PROFILES:
        job(profile+'_smoke8',profile,1,1,8)
    for profile in PROFILES:
        job(profile+'_dev6',profile,3,2,220,[profile+'_smoke8'])
    atomic(out/'jobs-prepared.json',dict(stage='A/B/D development only; C selection and 50-case holdout await measured audit',jobs=jobs))
    return matrix,jobs

def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,default=BASE);p.add_argument('--arm',action='store_true');a=p.parse_args()
    matrix,jobs=prepare(a.directory.resolve())
    if a.arm:
        evidence=json.loads((a.directory/'cpu-checks.json').read_text())
        assert evidence['status']=='passed' and not evidence['gpu_context_initialized']
        assert not (a.directory/'queue.json').exists(),'Do not overwrite an armed queue'
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        dirty=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).splitlines()
        assert all(line[3:] in {'docs/METAX_PROGRESS.md','docs/METAX_MODEL_PERFORMANCE.md'} for line in dirty),dirty
        frozen=dict(git_head=head,source_hashes=source_hashes(),jobs=jobs,
            matrix_path=str(a.directory.resolve()/'evaluation-matrix.json'),
            evaluation_matrix_sha256=hashlib.sha256((a.directory/'evaluation-matrix.json').read_bytes()).hexdigest(),
            min_free_disk_bytes=8_000_000_000)
        atomic(a.directory/'queue.json',frozen)
    print(json.dumps(dict(jobs=len(jobs),holdout_initial_states=matrix['holdout_initial_states'],armed=a.arm),indent=2))

if __name__=='__main__':main()
