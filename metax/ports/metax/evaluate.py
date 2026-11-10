"""Paired task/state quality, acceptance, time and memory from raw MetaX runs."""
import argparse
import json
from pathlib import Path
from summarize import rows,summarize

p=argparse.ArgumentParser();p.add_argument('--native',type=Path,required=True);p.add_argument('--optimized',type=Path,required=True);p.add_argument('--quantized',type=Path);p.add_argument('--lossless',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
roots={'B1':a.native,'B2':a.optimized}
if a.quantized:roots['Q1']=a.quantized
if a.lossless:roots['Q2']=a.lossless
runs={name:summarize(root) for name,root in roots.items()}
episodes={name:rows(root/'raw/episodes.jsonl') for name,root in roots.items()}
keys=lambda data:{(e['task_id'],e['init_state_id']) for e in data}
reference_keys=keys(episodes['B1'])
assert reference_keys and len(reference_keys)==len(episodes['B1'])
for name,run in runs.items():
    assert run['exit_code']=='0' and run['smoke_episodes']==0
    assert keys(episodes[name])==reference_keys and len(episodes[name])==len(reference_keys)
    assert all(e['planning_horizon']==e['step_limit']==220 for e in episodes[name])
comparisons={}
for name in list(runs)[1:]:
    baseline=runs['B1' if name=='B2' else 'B2'];test=runs[name]
    base_sr=baseline['task_success_rate'];sr=test['task_success_rate']
    comparisons[name]=dict(reference='B1' if name=='B2' else 'B2',
        success_count=test['successful_episodes'],denominator=test['complete_episodes'],
        success_percentage_point_delta=100*(sr-base_sr),
        relative_success_delta_percent=100*(sr-base_sr)/base_sr if base_sr else None,
        actual_accepted_prefix_delta=test['mean_accepted_prefix']-baseline['mean_accepted_prefix'],
        allocated_saving_bytes=baseline['allocated_peak']['bytes']-test['allocated_peak']['bytes'],
        reserved_saving_bytes=baseline['reserved_peak']['bytes']-test['reserved_peak']['bytes'],
        model_time_ratio=baseline['model']['sum_seconds']/test['model']['sum_seconds'],
        pipeline_time_ratio=baseline['policy_pipeline']['sum_seconds']/test['policy_pipeline']['sum_seconds'],
        closed_loop_time_ratio=baseline['closed_loop_seconds']/test['closed_loop_seconds'],
        same_policy_work=baseline['policy_steps']==test['policy_steps'],
        note='Task-set wall/model sums retain cold actions. Changed episode lengths and quality are shown explicitly; not a pure kernel benchmark.')
result=dict(runs=runs,episodes=episodes,comparisons=comparisons,
    B0=None,B0_status='pending user decision on upstream stopping boundary; no fabricated SpecVLA baseline',
    formal_acceptance_gain_gt_0_5=None,formal_success_loss_lt_5_percent=None,
    scope='LIBERO Goal pilot, paired task/state/seed7; no A100/other-hardware historical data',
    limitations=['Small pilot, not a population quality guarantee','Cold first action retained in closed loop','Device observed peak is 1s sampling, separate from framework peaks'])
a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(comparisons=comparisons),indent=2))
