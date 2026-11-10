"""Require a complete same-condition pair before emitting speed ratios."""
import argparse
import json
from pathlib import Path
from summarize import rows,summarize

parser=argparse.ArgumentParser();parser.add_argument('native',type=Path);parser.add_argument('optimized',type=Path);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--reference-label',default='B1');parser.add_argument('--test-label',default='B2')
a=parser.parse_args();left=rows(a.native/'raw/steps.jsonl');right=rows(a.optimized/'raw/steps.jsonl')
assert len(left)==len(right)>0
fields=['task_id','init_state_id','step_id','observation_hash','action_tokens','raw_action','executed_action','done','threshold']
checks={key:all(x[key]==y[key] for x,y in zip(left,right)) for key in fields}
A=summarize(a.native);B=summarize(a.optimized)
assert A['complete_episodes']==B['complete_episodes']>0
assert A['exit_code']==B['exit_code']=='0'
ratios={}
for name,ta,tb in [('model',A['model']['sum_seconds'],B['model']['sum_seconds']),
                   ('policy_pipeline',A['policy_pipeline']['sum_seconds'],B['policy_pipeline']['sum_seconds']),
                   ('closed_loop',A['closed_loop_seconds'],B['closed_loop_seconds'])]:
    ratios[name]=dict(native_seconds=ta,optimized_seconds=tb,reference_over_test=ta/tb,
        speed_increase_percent=100*(ta/tb-1),time_reduction_percent=100*(1-tb/ta))
report=dict(report_schema='paired_comparison_v2',reference_label=a.reference_label,test_label=a.test_label,native=A,optimized=B,exact=checks,ratios=ratios,
    B0_status='pending independent upstream boundary decision/port; no B0 ratios',
    acceptance_delta=B['mean_accepted_prefix']-A['mean_accepted_prefix'],
    success_percentage_point_delta=100*(B['task_success_rate']-A['task_success_rate']),
    relative_success_delta=(B['task_success_rate']-A['task_success_rate'])/A['task_success_rate'] if A['task_success_rate'] else None,
    limitation='Cold first action retained; complete loop includes simulation. A finite paired pilot cannot establish population quality.')
a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(exact=checks,ratios=ratios),indent=2))
assert all(checks.values()),'Paired observations/actions differ; inspect quality separately'
