"""Validate verifier-only QKV hits and report current paired measurements."""
import argparse
import json
from pathlib import Path
import statistics
from acceptance.aggregate_quantization import summarize


def validate_scope(manifest, indices, require_cpu=False, scoped=False):
    assert manifest['qkv_group_count']==len(indices) and manifest['gate_up_group_count']==0
    if require_cpu:assert manifest.get('cpu_prepacked_qkv_group_count')==len(indices)
    expected = [f'layer_{i}' for i in indices] if scoped else [f'model.layers.{i}.self_attn' for i in indices]
    assert manifest['qkv_modules']==expected, 'Unexpected QKV scope or layer order'


def timing(path,require_cpu=False,indices=None):
    if indices is None:indices=list(range(32))
    data=json.loads((path/'results.json').read_text())
    assert data['plan']['candidate_qkv_fusion']
    manifest=json.loads((path/'candidate_qkv_manifest.json').read_text())
    validate_scope(manifest,indices,require_cpu,data['plan'].get('candidate_qkv_layers','all')!='all')
    backend=data['plan'].get('candidate_qkv_backend','concat')
    assert manifest.get('bmm_qkv_group_count',0)==(len(indices) if backend=='bmm' else 0)
    rows_requested=data['plan'].get('candidate_qkv_rows',','.join(map(str,range(1,513))))
    assert manifest['qkv_fused_rows']==list(map(int,rows_requested.split(',')))
    result=dict(qkv_backend=backend,reference_kind=data['plan']['reference_kind'],physical_device=data['plan']['physical_device'],
                samples=data['plan']['samples'],repeats=data['plan']['measured_repeats'],per_repeat=[],
                candidate_policy_sha256=data['plan']['policy_sha256'],
                reference_policy_sha256=data['plan'].get('reference_policy_sha256'))
    for label,rows in data['rows'].items():
        values=sorted(r['model_ms'] for r in rows)
        index=.95*(len(values)-1);lo=int(index);hi=min(lo+1,len(values)-1)
        result[label]=dict(mean_ms=statistics.mean(values),median_ms=statistics.median(values),
                           p95_ms=values[lo]+(values[hi]-values[lo])*(index-lo),
                           verify_cycles=data['summary'][label]['verify_cycles'],
                           pipeline_mean_ms=data['summary'][label]['mean_pipeline_ms'],steps=len(rows))
        assert len(rows)==data['plan']['samples']*data['plan']['measured_repeats']
    result['change_percent']=100*(result['int8']['mean_ms']/result['bf16']['mean_ms']-1)
    for repeat in range(1,data['plan']['measured_repeats']+1):
        means={k:statistics.mean(r['model_ms'] for r in rows if r['repeat']==repeat) for k,rows in data['rows'].items()}
        result['per_repeat'].append(dict(repeat=repeat,reference_ms=means['bf16'],candidate_ms=means['int8'],
                                        change_percent=100*(means['int8']/means['bf16']-1)))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('campaign',type=Path);p.add_argument('--previous',type=Path,required=True)
    p.add_argument('--cachefix-reference',type=Path)
    p.add_argument('--require-cpu-prepack',action='store_true')
    p.add_argument('--require-cached-views',action='store_true')
    p.add_argument('--qkv-backend',choices=['concat','bmm'],default='concat')
    p.add_argument('--require-previous-behavior-parity',action='store_true')
    p.add_argument('--timing-campaign',type=Path,help='Separate timing campaign if only launcher serialization changed')
    p.add_argument('--qkv-layer-range',help='Explicit half-open verifier range; default all 32')
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    indices=list(range(32))
    if a.qkv_layer_range:
        start,stop=map(int,a.qkv_layer_range.split(':'))
        assert 0<=start<stop<=32
        indices=list(range(start,stop))
    results={}
    for init in [1,2]:
        candidate_dir=a.campaign/'runs'/f'qkv_init{init}'
        candidate=summarize(candidate_dir)
        previous=summarize(a.previous/'runs'/f'hybrid_init{init}')
        bf16=summarize(root/'outputs/acceptance'/('quant_accuracy_20261026T045119Z' if init==1 else 'quant_noslow_20261026T054934Z')/'runs'/f'bf16_init{init}')
        for run in [candidate,previous,bf16]:assert run['execution_status']=='PASS' and run['full_episodes']==10
        manifest=json.loads((candidate_dir/'model/manifests/linear_fusion.json').read_text())
        assert manifest['enabled']
        validate_scope(manifest,indices,a.require_cpu_prepack,bool(a.qkv_layer_range))
        assert manifest.get('bmm_qkv_group_count',0)==(len(indices) if a.qkv_backend=='bmm' else 0)
        if a.qkv_backend=='bmm':assert manifest['qkv_fused_rows']==list(range(257,321))
        if a.require_cached_views:assert manifest.get('cached_view_qkv_group_count')==len(indices)
        assert manifest.get('scripted_bmm_qkv_group_count',0)==0
        assert all(manifest[k]==0 for k in ['gate_up_group_count','swiglu_module_count','add_rms_norm_module_count'])
        assert all('ea_layer' not in name for name in manifest['qkv_modules'])
        hits=[json.loads(s) for s in (candidate_dir/'model/manifests/linear_fusion_hits.jsonl').read_text().splitlines()]
        assert hits and all(h['operator']=='qkv_linear' and h['device']=='npu' and h['shape'][-1]==4096 for h in hits)
        for reference in [previous,bf16]:
            assert [(x['task_id'],x['init_state_id']) for x in reference['episode_results']]==[(x['task_id'],x['init_state_id']) for x in candidate['episode_results']]
        assert candidate['quantization']['policy_sha256']==previous['quantization']['policy_sha256']
        def driver_peak(directory,device):
            samples=[json.loads(s) for s in (directory/'vendor_memory.jsonl').read_text().splitlines()]
            values=[r['devices'][str(device)]['used_mb'] for r in samples if str(device) in r['devices']]
            return max(values) if values else None
        bf16_dir=root/'outputs/acceptance'/('quant_accuracy_20261026T045119Z' if init==1 else 'quant_noslow_20261026T054934Z')/'runs'/f'bf16_init{init}'
        old_success={e['task_id']:e['success'] for e in previous['episode_results']}
        new_success={e['task_id']:e['success'] for e in candidate['episode_results']}
        results[str(init)]=dict(candidate=candidate,previous=previous,bf16=bf16,
            gained_tasks=[k for k,v in new_success.items() if v and not old_success[k]],
            lost_tasks=[k for k,v in new_success.items() if not v and old_success[k]],
            allocated_saving_vs_bf16_bytes=bf16['allocated_peak_bytes']-candidate['allocated_peak_bytes'],
            reserved_saving_vs_bf16_bytes=bf16['reserved_peak_bytes']-candidate['reserved_peak_bytes'],
            success_not_lower_than_previous=candidate['successes']>=previous['successes'],
            cpu_prepacked_qkv_groups=manifest.get('cpu_prepacked_qkv_group_count',0),
            cached_view_qkv_groups=manifest.get('cached_view_qkv_group_count',0),
            driver_hbm_sampled_peak=dict(candidate_mb=driver_peak(candidate_dir,candidate['physical_device_id']),
                bf16_mb=driver_peak(bf16_dir,bf16['physical_device_id']),sample_interval_seconds=10,
                units='MB as labeled by npu-smi; sampled,not continuous peak',
                same_physical_device=candidate['physical_device_id']==bf16['physical_device_id']),
            qkv_groups=manifest['qkv_group_count'],recorded_qkv_hit_shapes=len(hits))
    timing_campaign=a.timing_campaign or a.campaign
    timings={name:timing(timing_campaign/name,a.require_cpu_prepack,indices) for name in ['timing_vs_previous','timing_vs_bf16']}
    assert timings['timing_vs_previous']['reference_kind']=='quantized_policy'
    assert timings['timing_vs_bf16']['reference_kind']=='bf16'
    for name in timings:
        plan=json.loads((timing_campaign/name/'plan.json').read_text())
        assert plan.get('candidate_qkv_layers','all')==(a.qkv_layer_range or 'all')
        assert plan.get('candidate_qkv_backend','concat')==a.qkv_backend
        if a.require_cached_views:
            manifest=json.loads((timing_campaign/name/'candidate_qkv_manifest.json').read_text())
            assert manifest.get('cached_view_qkv_group_count')==len(indices)
            assert manifest.get('scripted_bmm_qkv_group_count',0)==0
    expected=results['1']['candidate']['quantization']['policy_sha256']
    assert results['2']['candidate']['quantization']['policy_sha256']==expected
    assert all(t['candidate_policy_sha256']==expected for t in timings.values())
    assert timings['timing_vs_previous']['reference_policy_sha256']==expected
    checks=dict(memory_target_met=all(r['allocated_saving_vs_bf16_bytes']>=2_000_000_000 for r in results.values()),
                quality_counts_not_lower_than_previous=all(r['success_not_lower_than_previous'] for r in results.values()),
                measured_mean_latency_not_higher=all(t['change_percent']<=0 for t in timings.values()))
    behavior_parity={}
    if a.require_previous_behavior_parity:
        for init in [1,2]:
            before=[json.loads(s) for s in (a.previous/'runs'/f'hybrid_init{init}'/'metrics/steps.jsonl').read_text().splitlines()]
            after=[json.loads(s) for s in (a.campaign/'runs'/f'qkv_init{init}'/'metrics/steps.jsonl').read_text().splitlines()]
            fields=['episode_id','step_id','observation_hash','image_hash','action_tokens','raw_action','executed_action','done','threshold']
            counts={k:sum(b[k]==q[k] for b,q in zip(before,after)) for k in fields}
            exact=len(before)==len(after) and all(n==len(after) for n in counts.values())
            behavior_parity[f'init{init}']=dict(previous_steps=len(before),candidate_steps=len(after),matched=counts,exact=exact)
        data=json.loads((timing_campaign/'timing_vs_previous/results.json').read_text())
        before,after=data['rows']['bf16'],data['rows']['int8']
        fields=['repeat','index','action','verify','verifier_shapes']
        counts={k:sum(b[k]==q[k] for b,q in zip(before,after)) for k in fields}
        exact=len(before)==len(after) and all(n==len(after) for n in counts.values())
        behavior_parity['timing_vs_previous']=dict(previous_steps=len(before),candidate_steps=len(after),matched=counts,exact=exact)
        checks['previous_behavior_exact_on_measured_inputs']=all(r['exact'] for r in behavior_parity.values())
    out=dict(qkv_backend=a.qkv_backend,previous_behavior_parity=behavior_parity,qkv_layer_indices=indices,timing_campaign=str(timing_campaign),quality_and_memory=results,timings=timings,observed_checks=checks,
             recommend_new_candidate=all(checks.values()),population_quality_loss_bound_proven=False,
             legacy_timing_keys=dict(bf16='reference (previous quantized policy or BF16,see reference_kind)',int8='new QKV candidate'),
             scope='20 new exploratory Goal10 episodes; reused init1/init2; each timing comparison has its own same-chip reference')
    out['observed_reserved_memory_target_met']=all(r['reserved_saving_vs_bf16_bytes']>=2_000_000_000 for r in results.values())
    if a.cachefix_reference:
        parity={}
        for init in [1,2]:
            old_dir=a.cachefix_reference/'runs'/f'qkv_init{init}'
            old=summarize(old_dir);new=results[str(init)]['candidate']
            before=[json.loads(s) for s in (old_dir/'metrics/steps.jsonl').read_text().splitlines()]
            after=[json.loads(s) for s in (a.campaign/'runs'/f'qkv_init{init}'/'metrics/steps.jsonl').read_text().splitlines()]
            assert len(before)==len(after)
            fields=['observation_hash','image_hash','action_tokens','raw_action','executed_action','done','threshold']
            counts={k:sum(b[k]==q[k] for b,q in zip(before,after)) for k in fields}
            assert all(n==len(after) for n in counts.values()),'Allocator-only change has behavior differences; investigate'
            parity[str(init)]=dict(steps=len(after),matched=counts,
                reserved_peak_before=old['reserved_peak_bytes'],reserved_peak_after=new['reserved_peak_bytes'])
        for name in timings:
            before=json.loads((a.cachefix_reference/name/'results.json').read_text())['rows']['int8']
            after=json.loads((a.campaign/name/'results.json').read_text())['rows']['int8']
            fields=['action','verify','verifier_shapes']
            counts={k:sum(b[k]==q[k] for b,q in zip(before,after)) for k in fields}
            assert len(before)==len(after) and all(n==len(after) for n in counts.values())
            parity[name]=dict(steps=len(after),matched=counts)
        out['cachefix_parity']=parity
    reports=a.campaign/'reports';reports.mkdir(exist_ok=True)
    (reports/'qkv_results.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(checks=checks,recommend_new_candidate=out['recommend_new_candidate'],
        successes={k:{n:v[n]['successes'] for n in ['bf16','previous','candidate']} for k,v in results.items()},
        mean_changes={k:v['change_percent'] for k,v in timings.items()}),indent=2))


if __name__=='__main__':main()
