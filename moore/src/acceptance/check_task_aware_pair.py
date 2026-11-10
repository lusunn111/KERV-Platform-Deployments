"""Check same-runtime closed-loop pairs, allowing action and trajectory differences."""
import argparse
from pathlib import Path
from acceptance.common import load, rows, dump, sha256
from acceptance.task_aware_quantization_report import summarize_run


def check(reference, candidate):
    left,right = [summarize_run(p) for p in (reference,candidate)]
    metas = [load(p/'run.json') for p in (reference,candidate)]
    for item,meta,folder in zip((left,right),metas,(reference,candidate)):
        assert item['execution']=='PASS' and item['exit_code']==0,folder
        episodes=rows(folder/'episodes.jsonl'); steps=rows(folder/'steps.jsonl')
        assert len(episodes)==1 and episodes[0]['steps']==len(steps)>0
        assert item['success']==meta['task_success'] and not rows(folder/'errors.jsonl')
        assert not load(folder/'collection.json')['incomplete_episode']
    for key in ('task_id','init_state_id','seed','max_steps','planning_steps','action_protocol'):
        assert metas[0][key]==metas[1][key],key
    assert rows(reference/'steps.jsonl')[0]['observation_hash']==rows(candidate/'steps.jsonl')[0]['observation_hash']
    configs=[load(p/'collection.json')['resolved_generate_config'] for p in (reference,candidate)]
    for key in ('center_crop','use_spec','max_planning_steps','use_kalman_tree','kalman_process_var',
                'kalman_measurement_var','kalman_history_window','flagos_tree_static_runtime','flagos_kalman_batch',
                'use_flagos_tree_builder','use_flagos_linear_fusion'):
        assert configs[0][key]==configs[1][key],key
    assert configs[0]['use_flagos_tree_builder'] and configs[0]['flagos_kalman_batch']
    assert not configs[0]['flagos_tree_static_runtime'] and not configs[0]['use_flagos_linear_fusion']
    proof=rows(candidate/'precision_storage.jsonl')
    loaded=next(x for x in proof if x['phase']=='loaded'); final=next(x for x in proof if x['phase']=='final')
    assert loaded['policy_sha256']==sha256(metas[1]['policy_path'])
    assert len(loaded['modules'])==loaded['manifest']['module_count']==len(final['modules'])
    assert all(x['decoded_sha256']==x['gpu_decoded_sha256'] for x in loaded['modules'])
    exact=[x for x in loaded['modules'] if x['storage'].startswith('exact_bf16')]
    assert all(x['original_sha256']==x['decoded_sha256'] for x in exact)
    assert all(x['forward_calls']>0 for x in final['modules'])
    ref=next(x for x in rows(reference/'precision_reference.jsonl') if x['phase']=='final')
    for audit in (ref,final):
        assert audit['kalman_batch_hits']>0
        assert len(audit['softmax_guard_calls'])==32 and all(x>0 for x in audit['softmax_guard_calls'].values())
    a,b=[rows(p/'steps.jsonl') for p in (reference,candidate)]
    trajectory={'same_length':len(a)==len(b)}
    for key in ('observation_hash','action_tokens','raw_action','executed_action'):
        trajectory[key]={'equal_matching_step_records':sum(u[key]==v[key] for u,v in zip(a,b)),
                         'first_difference_index':next((i for i,(u,v) in enumerate(zip(a,b)) if u[key]!=v[key]),None)}
    return {'reference':left,'candidate':right,'execution_integrity':'PASS','same_initial_observation':True,
            'task_success_regression':left['success'] and not right['success'],
            'both_succeeded':left['success'] and right['success'],'trajectory_diagnostic':trajectory,
            'gpu_decoded_hashes_checked':len(loaded['modules']), 'original_exact_hashes_checked':len(exact),
            'static_weight_saving_bytes':loaded['manifest']['estimated_weight_storage_saving_bytes'],
            'allocated_peak_saving_bytes':left['peak_allocated_bytes']-right['peak_allocated_bytes'],
            'reserved_peak_saving_bytes':left['peak_reserved_bytes']-right['peak_reserved_bytes'],
            'scope':'One local task/state/seed pair. Action differences allowed. No suite-level success-rate bound or cross-trajectory action MAE.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=check(a.reference,a.candidate);dump(a.output,result)
    print({k:v for k,v in result.items() if k not in ('reference','candidate')})
    if result['task_success_regression']:raise SystemExit(2)


if __name__=='__main__':main()
