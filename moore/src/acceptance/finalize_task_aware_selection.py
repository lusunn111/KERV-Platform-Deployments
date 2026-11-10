"""Select only a verified task-preserving policy; retain rejected experiments."""
import argparse
from pathlib import Path
import shutil
from acceptance.common import load,rows,dump,sha256
from acceptance.check_task_aware_pair import check


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True)
    a=p.parse_args();c=a.campaign
    reference=c/'runs/control_fresh_bf16_task0_state0_s7'
    candidate=c/'runs/comparison_old_exact_mlp_task0_state0_s7'
    result=check(reference,candidate)
    assert result['both_succeeded']
    diagnostic=result['trajectory_diagnostic'];steps=result['reference']['steps']
    assert diagnostic['same_length'] and all(diagnostic[k]['equal_matching_step_records']==steps for k in ('observation_hash','action_tokens','raw_action','executed_action'))
    assert result['original_exact_hashes_checked']==96
    dump(c/'reports/recommended_exact_recheck.json',result)
    source=Path(load(candidate/'run.json')['policy_path']);dest=c/'recommended_exact_mlp/deployment_policy.json'
    dest.parent.mkdir(exist_ok=True)
    if dest.exists():assert sha256(dest)==sha256(source)
    else:shutil.copyfile(source,dest)
    assert sha256(source)==sha256(dest)
    families={}
    for file in sorted((c/'runs').glob('pilot_*/run.json')):
        run=load(file)
        if run['status']!='PASS':continue
        family=run['run_id'].split('_task')[0]
        families.setdefault(family,[]).append({'run_id':run['run_id'],'success':run['task_success'],
                                              'steps':rows(file.parent/'episodes.jsonl')[0]['steps']})
    assert len(families)==7 and all(any(not x['success'] for x in values) for values in families.values())
    before=result['reference']['model_seconds_excluding_first']['mean']
    after=result['candidate']['model_seconds_excluding_first']['mean']
    selection={'status':'RETAIN_VERIFIED_EXACT_BF16','selected_policy':str(dest.relative_to(c)),
               'selected_policy_sha256':sha256(dest),'same_policy_as_rechecked_source':str(source),
               'lossy_quantization':False,'module_count':96,
               'reason':'All7 lossy families failed task0 pilot; new exact-decoder prototypes did not demonstrate stable Linear gains. Retain existing exact MLP codec.',
               'current_recheck':{'reference':reference.name,'candidate':candidate.name,'both_succeeded':True,
                                 'identical_actions_observations':steps,'original_weight_hashes_checked':96,
                                 'static_weight_saving_bytes':result['static_weight_saving_bytes'],
                                 'allocated_peak_saving_bytes':result['allocated_peak_saving_bytes'],
                                 'reserved_peak_saving_bytes':result['reserved_peak_saving_bytes'],
                                 'bf16_steady_model_ms':before*1000,'exact_steady_model_ms':after*1000,
                                 'latency_increase_percent':(after/before-1)*100},
               'lossy_families':families,
               'reserved_fresh_validation':'Not executed: no new candidate passed pilot/performance gates; original lists preserved',
               'historical_scope':'Prior sealed campaign had2 successful BF16/exact pairs,262 identical actions; current task0 is a repeat, not a new third task/state',
               'scope':'One Moore node; zero additional weight error, local task preservation; no new lossy quality pass, no new speed gain or full-suite SR claim'}
    dump(c/'inventory/selection.json',selection)
    print(selection)


if __name__=='__main__':main()
