"""Compare new BF16/INT8 runs without requiring quantized actions to match bitwise."""
import argparse, csv, json, math
from pathlib import Path
from acceptance.aggregate import rows, stats


def summarize(path):
    meta=json.loads((path/'run.json').read_text());metrics=path/'metrics'
    steps=rows(metrics/'steps.jsonl');episodes=rows(metrics/'episodes.jsonl');memory=rows(metrics/'memory.jsonl')
    runtime=rows(path/'model/runtime_stats.jsonl')
    ids=[(s['episode_id'],s['step_id']) for s in steps];assert len(ids)==len(set(ids))
    for s in steps:
        assert len(s['raw_action'])==len(s['executed_action'])==len(s['action_tokens'])==7
        assert all(math.isfinite(x) for x in s['raw_action']+s['executed_action'])
    for ep in episodes:
        assert ep['steps']==sum(s['episode_id']==ep['episode_id'] for s in steps)
    for s in runtime:
        assert s['generated_token_count']==7 and s['action_token_ids']==s['kalman_history_token_ids']
    q=json.loads((metrics/'quantization.json').read_text()) if (metrics/'quantization.json').exists() else None
    if meta.get('quantization') in {'mlp_int8','mlp_dynamic_int8'} and meta['status']=='PASS':
        assert q is not None
        expected_modules=len(q['modules']) if q.get('modules') else 3*len(q.get('layer_indices',list(range(32))))
        assert q['module_count']==expected_modules and q['forward_calls']>0 and q['fallback'] is False
        assert q['converted_on']=='cpu before NPU transfer'
    full=[ep for ep in episodes if ep['termination_reason']!='truncated_smoke']
    return {'run_id':path.name,'profile':meta.get('profile'),'execution_status':meta['status'],
        'physical_device_id':meta.get('physical_device_id'),'steps':len(steps),'full_episodes':len(full),
        'successes':sum(ep['success'] for ep in full),'success_rate':sum(ep['success'] for ep in full)/len(full) if full else None,
        'episode_results':[{k:ep[k] for k in ['task_id','init_state_id','success','steps','termination_reason','closed_loop_wall_seconds']} for ep in episodes],
        'model_seconds':stats([s['model_seconds'] for s in steps]),
        'pipeline_seconds':stats([s['pipeline_seconds'] for s in steps]),
        'allocated_peak_bytes':max([m.get('allocated_peak_bytes',0) for m in memory],default=0),
        'reserved_peak_bytes':max([m.get('reserved_peak_bytes',0) for m in memory],default=0),
        'quantization':({**{k:q[k] for k in ['profile','module_count','forward_calls','unique_storage_saving_bytes','weight_dtype','activation_dtype','group_size','converted_on']},
            'layer_indices':q.get('layer_indices',list(range(32))),
            'implementation':q.get('implementation'),
            'fused_gate_up':q.get('fused_gate_up',False),'direct_mlp':q.get('direct_mlp',False),
            'fused_swiglu_quant':q.get('fused_swiglu_quant',False),
            'replaced_projection_count':q.get('replaced_projection_count',q['module_count']),
            'activation_rounding':q.get('activation_rounding'),
            'preserved_bf16_modules':q.get('preserved_bf16_modules',[]),
            'precision_counts':q.get('precision_counts'),
            'native_swiglu_quant_projection_count':q.get('native_swiglu_quant_projection_count'),
            'silu_backend':q.get('silu_backend','native'),
            'group_counts':q.get('group_counts'), 'policy_sha256':q.get('policy_sha256')}) if q else None}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--campaign-dir',required=True,type=Path);a=parser.parse_args()
    reports=a.campaign_dir/'reports';reports.mkdir(exist_ok=True)
    runs=[summarize(p.parent) for p in sorted((a.campaign_dir/'runs').glob('*/run.json')) if json.loads(p.read_text()).get('status')!='RUNNING']
    by={x['run_id']:x for x in runs};comparisons={}
    for label,bf,quant in [('same_chip_task0','bf16_reference','int8_candidate'),
        ('middle18_task0','bf16_reference','int8_middle18'),
        ('goal10_pilot','bf16_goal10','int8_goal10'),('middle18_goal10_pilot','bf16_goal10','int8_middle18_goal10')]:
        if bf not in by or quant not in by:continue
        b,q=by[bf],by[quant]
        if not b['full_episodes'] or not q['full_episodes']:continue
        assert [(ep['task_id'],ep['init_state_id']) for ep in b['episode_results']]==[(ep['task_id'],ep['init_state_id']) for ep in q['episode_results']]
        saving=b['allocated_peak_bytes']-q['allocated_peak_bytes']
        comparisons[label]={'baseline':bf,'candidate':quant,'allocated_peak_saving_bytes':saving,
            'allocated_peak_saving_GB':saving/1e9,'allocated_peak_saving_GiB':saving/2**30,
            'saving_at_least_2GB':saving>=2_000_000_000,'saving_at_least_2GiB':saving>=2*2**30,
            'reserved_peak_saving_bytes':b['reserved_peak_bytes']-q['reserved_peak_bytes'],
            'native_model_mean_s':b['model_seconds']['mean'],'int8_model_mean_s':q['model_seconds']['mean'],
            'bf16_over_int8_model_speedup':b['model_seconds']['mean']/q['model_seconds']['mean'],
            'bf16_success':b['successes'],'int8_success':q['successes'],'episodes_per_profile':b['full_episodes'],
            'success_rate_change_percentage_points':100*(q['success_rate']-b['success_rate']),
            'relative_success_rate_loss':(b['success_rate']-q['success_rate'])/b['success_rate'] if b['success_rate'] else None,
            'same_chip':b['physical_device_id']==q['physical_device_id'],
            'timing_scope':'observed closed-loop step latencies, includes first action; trajectories may differ, not same-input speed attribution'}
    report={'campaign_id':a.campaign_dir.name,'authorization':'project permits quantization per user confirmation2026-10-26',
        'runs':runs,'comparisons':comparisons,'quality_gate':'finite seven-token actions and task success pilot; exact BF16 action parity not required',
        'formal_scope':'OpenVLA Goal only; not full300episode protocol or true Spec-VLA B0 comparison'}
    (reports/'quantization_results.json').write_text(json.dumps(report,indent=2)+'\n')
    with (reports/'quantization_results.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['run_id','profile','steps','full_episodes','successes','allocated_peak_bytes','reserved_peak_bytes','model_mean_s']);w.writeheader()
        for r in runs:w.writerow({**{k:r[k] for k in w.fieldnames if k!='model_mean_s'},'model_mean_s':r['model_seconds']['mean']})
    print(json.dumps({'comparisons':comparisons},ensure_ascii=False))

if __name__=='__main__':main()
