"""Read immutable Ascend evidence and produce a consolidated local evaluation.

Does not run models or write into a source campaign. P95 is nearest-rank,
independent of the interpolated percentile used by some historical reports.
"""
import argparse
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
from acceptance.aggregate import stats

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/acceptance'
POLICY='363e245a3b8265d4165cf18e78b1491fecffa1e8e6019fe2a2506661db3c62d7'
RUNS={
 'bf16': [('quant_accuracy_20261026T045119Z','bf16_init1'),('quant_noslow_20261026T054934Z','bf16_init2')],
 'mixed': [('weight_only_hybrid4_20261027',f'hybrid_init{i}') for i in [1,2]],
 'concat32': [('qkv_cpu_pack_20261027',f'qkv_init{i}') for i in [1,2]],
 'concat18': [('qkv_middle18_fixed_20261027',f'qkv_init{i}') for i in [1,2]],
 'bmm': [('qkv_bmm_guarded_20261027',f'qkv_init{i}') for i in [1,2]],
 'bmm_cached': [('qkv_bmm_cached_20261027',f'qkv_init{i}') for i in [1,2]],
}
TIMINGS={
 'mixed_vs_bf16':BASE/'weight_only_hybrid4_20261027/timing_full157',
 'cached_vs_mixed':BASE/'qkv_bmm_cached_20261027/timing_vs_previous',
 'cached_vs_bf16':BASE/'qkv_bmm_cached_20261027/timing_vs_bf16',
 'concat32_vs_mixed':BASE/'qkv_cpu_pack_20261027/timing_vs_previous',
 'concat18_vs_mixed':BASE/'qkv_middle18_20261027/timing_vs_previous',
 'bmm_vs_mixed':BASE/'qkv_bmm_guarded_20261027/timing_vs_previous',
}


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def wilson(success,total):
    z=1.959963984540054;p=success/total;den=1+z*z/total
    center=(p+z*z/(2*total))/den
    half=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    return [center-half,center+half]


class Evaluation:
    def __init__(self):self.evidence={};self.steps={}
    def track(self,path,role):
        path=path.resolve();name=str(path.relative_to(ROOT))
        if name not in self.evidence:self.evidence[name]=dict(path=name,bytes=path.stat().st_size,sha256=sha(path),roles=[])
        if role not in self.evidence[name]['roles']:self.evidence[name]['roles'].append(role)
        return path
    def json(self,path,role):return json.loads(self.track(path,role).read_text())
    def rows(self,path,role):return [json.loads(s) for s in self.track(path,role).read_text().splitlines() if s.strip()]
    def run(self,campaign,run_id,label):
        path=BASE/campaign/'runs'/run_id;role='quality_memory_'+label
        meta=self.json(path/'run.json',role);assert meta['status']=='PASS' and meta['exit_code']==0,path
        steps=self.rows(path/'metrics/steps.jsonl',role);eps=self.rows(path/'metrics/episodes.jsonl',role)
        verify=self.rows(path/'metrics/verify.jsonl',role);memory=self.rows(path/'metrics/memory.jsonl',role)
        runtime=self.rows(path/'model/runtime_stats.jsonl',role)
        assert len(eps)==10 and {e['task_id'] for e in eps}==set(range(10))
        assert len({(s['episode_id'],s['step_id']) for s in steps})==len(steps)
        assert len({(v['episode_id'],v['step_id'],v['verify_id']) for v in verify})==len(verify)
        for e in eps:
            subset=[s for s in steps if s['episode_id']==e['episode_id']]
            assert e['steps']==len(subset) and [s['step_id'] for s in subset]==list(range(e['steps']))
            assert e['termination_reason'] in {'success','step_limit'}
            assert e['success'] or e['steps']==220
        for s in steps:
            assert len(s['raw_action'])==len(s['executed_action'])==len(s['action_tokens'])==7
            assert all(math.isfinite(x) for x in s['raw_action']+s['executed_action'])
        assert runtime and all(s['generated_token_count']==7 and s['action_token_ids']==s['kalman_history_token_ids'] for s in runtime)
        q=None
        if label!='bf16':
            q=self.json(path/'metrics/quantization.json',role)
            assert q['policy_sha256']==POLICY and q['forward_calls']>0 and q['fallback'] is False
            assert q['converted_on']=='cpu before NPU transfer'
        accepted=[v['accepted_prefix_length'] for v in verify]
        assert accepted and all(isinstance(x,int) and x>=0 for x in accepted)
        vendor=self.rows(path/'vendor_memory.jsonl',role);device=str(meta['physical_device_id'])
        hbm=max(r['devices'][device]['used_mb'] for r in vendor if device in r['devices'])
        hits=self.rows(path/'model/manifests/embodied_ops_hits.jsonl','operator_execution_'+label)
        self.track(path/'source_manifest.sha256','runtime_source_'+label)
        source_manifest=BASE/campaign/'inventory/source_after_manifest.sha256'
        if source_manifest.exists():self.track(source_manifest,'runtime_source_'+label)
        self.steps[label,eps[0]['init_state_id']]=steps
        return dict(label=label,campaign=campaign,run_id=run_id,raw_path=str(path.relative_to(ROOT)),
            seed=meta['seed'],physical_device=meta['physical_device_id'],initial_state=eps[0]['init_state_id'],
            start_timestamp=meta['start_timestamp'],end_timestamp=meta['end_timestamp'],
            success=sum(e['success'] for e in eps),total=len(eps),steps=len(steps),episodes=eps,
            model_ms=stats([1000*s['model_seconds'] for s in steps]),
            pipeline_ms=stats([1000*s['pipeline_seconds'] for s in steps]),
            closed_loop_seconds=stats([e['closed_loop_wall_seconds'] for e in eps]),
            allocated_peak_bytes=max(m.get('allocated_peak_bytes',0) for m in memory),
            reserved_peak_bytes=max(m.get('reserved_peak_bytes',0) for m in memory),
            driver_hbm_sampled_peak_mb=hbm,driver_sample_interval_seconds=10,
            raw_acceptance=stats(accepted),zero_acceptance_events=accepted.count(0),
            quantization=None if q is None else {k:q.get(k) for k in ['policy_sha256','precision_counts','implementation','forward_calls','activation_rounding','silu_backend']},
            observed_operators=sorted({h['operator'] for h in hits}),
            observed_hit_devices=sorted({d for h in hits for d in h.get('devices',[])}))
    def timing(self,path,name):
        data=self.json(path/'results.json','timing_'+name);plan=data['plan'];rows=data['rows']
        assert plan['samples']==157 and plan['measured_repeats']==3 and plan['policy_sha256']==POLICY
        assert plan['both_models_resident'] and plan['memory_is_diagnostic_only']
        result=dict(path=str(path.relative_to(ROOT)),physical_device=plan['physical_device'],
            reference_kind=plan.get('reference_kind','bf16'),input_sha256=plan['input_sha256'],profiles={},per_repeat=[])
        for label,values in rows.items():
            assert len(values)==471 and len({(r['repeat'],r['index']) for r in values})==471
            for repeat in [1,2,3]:assert [r['index'] for r in values if r['repeat']==repeat]==list(range(157))
            actions=[[r['action'] for r in values if r['repeat']==rep] for rep in [1,2,3]]
            assert actions[0]==actions[1]==actions[2]
            result['profiles'][label]=dict(model_ms=stats([r['model_ms'] for r in values]),
                pipeline_ms=stats([r['pipeline_ms'] for r in values]),
                verify_cycles=sum(len(r['verify']) for r in values))
        for repeat in [1,2,3]:
            means={k:stats([r['model_ms'] for r in values if r['repeat']==repeat])['mean'] for k,values in rows.items()}
            result['per_repeat'].append(dict(repeat=repeat,change_percent=100*(means['int8']/means['bf16']-1)))
        a,b=result['profiles']['bf16'],result['profiles']['int8']
        result['change_percent']=100*(b['model_ms']['mean']/a['model_ms']['mean']-1)
        result['speedup']=a['model_ms']['mean']/b['model_ms']['mean']
        result['actions_and_workload_exact']=all(all(a[k]==b[k] for k in ['action','verify','verifier_shapes']) for a,b in zip(rows['bf16'],rows['int8']))
        if 'candidate_qkv_manifest.json' in [p.name for p in path.iterdir()]:self.json(path/'candidate_qkv_manifest.json','operator_'+name)
        return result
    def legacy(self):
        campaign=BASE/'ascend_20261019T0730Z';profiles={'native':{'model':[],'pipeline':[]},'runtime':{'model':[],'pipeline':[]}}
        for name in ['round1_AB','round2_BA','round3_AB']:
            data=self.json(campaign/'replays'/name/'benchmark.json','historical_runtime_replay')
            assert data['actions_exact'] and data['samples']==157
            assert data['profiles']['native']['actions']==data['profiles']['runtime']['actions']
            for profile in profiles:
                profiles[profile]['model']+=data['profiles'][profile]['raw_model_ms']
                profiles[profile]['pipeline']+=data['profiles'][profile]['raw_pipeline_ms']
        profiles={p:{k:stats(v) for k,v in a.items()} for p,a in profiles.items()}
        for p in profiles:assert profiles[p]['model']['count']==471
        pair={}
        for label in ['paired_B1','paired_B2']:
            path=campaign/'runs'/label;meta=self.json(path/'run.json','historical_closed_loop')
            assert meta['status']=='PASS'
            pair[label]=dict(steps=self.rows(path/'metrics/steps.jsonl','historical_closed_loop'),
                episodes=self.rows(path/'metrics/episodes.jsonl','historical_closed_loop'))
        a,b=pair['paired_B1']['steps'],pair['paired_B2']['steps'];fields=['episode_id','step_id','observation_hash','raw_action','executed_action','action_tokens','done']
        assert len(a)==len(b)==115 and all(x[k]==y[k] for x,y in zip(a,b) for k in fields)
        closed={k:dict(model_ms=stats([s['model_seconds']*1000 for s in r['steps']]),
            pipeline_ms=stats([s['pipeline_seconds']*1000 for s in r['steps']]),
            wall_seconds=sum(e['closed_loop_wall_seconds'] for e in r['episodes'])) for k,r in pair.items()}
        for name in ['T03_placement_and_feedback.md','T13_integration.md','placement_trace_summary.json','T12_resource_limitations.md','B0_source_audit.md']:
            self.track(campaign/'reports'/name,'historical_coverage_limits')
        return dict(date='2026-10-19',profiles=profiles,
            model_speedup=profiles['native']['model']['mean']/profiles['runtime']['model']['mean'],
            pipeline_speedup=profiles['native']['pipeline']['mean']/profiles['runtime']['pipeline']['mean'],
            closed_loop=closed,paired_actions_exact=True,
            comparison='KERV native B1 vs optimized B2, not SpecVLA B0',
            limitation='One task/init0/seed7; shared-host CPU interference recorded in original campaign')


def write(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);inv=a.output/'inventory';inv.mkdir()
    e=Evaluation();runs={label:[e.run(camp,name,label) for camp,name in cases] for label,cases in RUNS.items()}
    timings={name:e.timing(path,name) for name,path in TIMINGS.items()}
    assert all(t['input_sha256']==timings['mixed_vs_bf16']['input_sha256'] for t in timings.values())
    quality={};pair_counts={}
    for label,values in runs.items():
        total=sum(r['total'] for r in values);success=sum(r['success'] for r in values)
        quality[label]=dict(success=success,total=total,rate=success/total,wilson95_descriptive=wilson(success,total))
        if label=='bf16':continue
        old={(r['initial_state'],ep['task_id']):ep['success'] for r in runs['bf16'] for ep in r['episodes']}
        new={(r['initial_state'],ep['task_id']):ep['success'] for r in values for ep in r['episodes']}
        assert old.keys()==new.keys()
        pair_counts[label]={name:sum((old[k],new[k])==outcome for k in old) for name,outcome in [
            ('both_success',(True,True)),('lost',(True,False)),('gained',(False,True)),('both_failure',(False,False))]}
    parity={}
    for i in [1,2]:
        before=e.steps['mixed',i];after=e.steps['bmm_cached',i]
        fields=['episode_id','step_id','observation_hash','image_hash','action_tokens','raw_action','executed_action','done','threshold']
        counts={k:sum(x[k]==y[k] for x,y in zip(before,after)) for k in fields}
        assert len(before)==len(after) and all(v==len(after) for v in counts.values())
        parity[str(i)]=dict(steps=len(after),exact=True,matched=counts)
    assert timings['cached_vs_mixed']['actions_and_workload_exact']
    legacy=e.legacy()
    for label in runs:
        for r,b in zip(runs[label],runs['bf16']):
            assert r['initial_state']==b['initial_state'] and r['seed']==b['seed']==7
            r['allocated_saving_vs_bf16_bytes']=b['allocated_peak_bytes']-r['allocated_peak_bytes']
            r['reserved_saving_vs_bf16_bytes']=b['reserved_peak_bytes']-r['reserved_peak_bytes']
            r['driver_same_chip_as_bf16']=b['physical_device']==r['physical_device']
    policy=e.track(BASE/'weight_only_hybrid4_20261027/policy.pt','frozen_quantization_policy')
    assert sha(policy)==POLICY
    for name in ['weight_only_hybrid4_20261027','qkv_bmm_cached_20261027']:
        e.track(BASE/name/'inventory/weight_manifest.json','checkpoint_hash_manifest')
        e.track(BASE/name/'inventory/source_after_manifest.sha256','frozen_runtime_source')
    unit=e.track(BASE/'qkv_bmm_cached_20261027/inventory/unit.log','npu_unit_tests9')
    assert 'Ran 9 tests' in unit.read_text() and '\nOK' in unit.read_text()
    e.track(BASE/'ascend_20261019T0730Z/inventory/mechanism_tests.log','historical_mechanism_tests4')
    frozen_matches=0
    for campaign in sorted({c for values in RUNS.values() for c,_ in values} | {'ascend_20261019T0730Z'}):
        directory=BASE/campaign
        manifest=next((directory/n for n in ['evidence.sha256','checksums.sha256'] if (directory/n).exists()),None)
        if manifest is None:continue
        recorded={}
        for line in manifest.read_text().splitlines():
            digest,sep,name=line.partition('  ')
            if sep:recorded[name.lstrip('*')]=digest
        for name,record in list(e.evidence.items()):
            path=ROOT/name
            if directory in path.parents:
                relative=str(path.relative_to(directory))
                if relative in recorded:
                    assert record['sha256']==recorded[relative],f'Frozen evidence mismatch: {name}'
                    frozen_matches+=1
        e.track(manifest,'original_evidence_seal')
    report=dict(recomputed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),new_model_runs_this_delivery=0,
        scope='Current Ascend only; immutable historical experiments recomputed, not third-party or whole-project acceptance',
        p95_method='nearest-rank, ordered[ceil(0.95*N)-1]; historical interpolated P95 unchanged',
        runs=runs,quality=quality,paired_quality=pair_counts,timings=timings,behavior_parity=parity,legacy_runtime=legacy,
        original_frozen_hash_matches=frozen_matches,policy_sha256=POLICY,recommended_quantized_profile='mixed',default_precision='BF16',
        formal_acceptance_status='BLOCKED',population_quality_loss_bound_proven=False,
        recommendations=dict(mixed_observed_memory_pass=all(r["allocated_saving_vs_bf16_bytes"]>=2_000_000_000 for r in runs["mixed"]),
            mixed_observed_mean_latency_not_higher=timings['mixed_vs_bf16']['change_percent']<=0,
            cached_observed_memory_pass=all(r['allocated_saving_vs_bf16_bytes']>=2_000_000_000 for r in runs['bmm_cached']),
            cached_observed_mean_latency_not_higher=all(timings[n]['change_percent']<=0 for n in ['cached_vs_mixed','cached_vs_bf16'])))
    write(a.output/'evaluation.json',report)
    with (a.output/'quality.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['profile','initial_state','success','total','steps','physical_device','allocated_peak_bytes','reserved_peak_bytes','allocated_saving_vs_bf16_bytes','driver_hbm_sampled_peak_mb','driver_same_chip_as_bf16']);w.writeheader()
        for label,values in runs.items():
            for r in values:w.writerow(dict(profile=label,**{k:r[k] for k in w.fieldnames if k!='profile'}))
    with (a.output/'timing.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['comparison','role','samples','mean_ms','median_ms','p95_nearest_rank_ms','physical_device','reference_kind','change_percent']);w.writeheader()
        for name,t in timings.items():
            for role,profile in t['profiles'].items():
                s=profile['model_ms'];w.writerow(dict(comparison=name,role=role,samples=s['count'],mean_ms=s['mean'],median_ms=s['median'],p95_nearest_rank_ms=s['p95_nearest_rank'],physical_device=t['physical_device'],reference_kind=t['reference_kind'],change_percent=t['change_percent']))
    with (a.output/'episodes.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['profile','campaign','run_id','seed','task_id','init_state_id','success','steps','termination_reason','closed_loop_wall_seconds']);w.writeheader()
        for label,values in runs.items():
            for r in values:
                for ep in r['episodes']:w.writerow({**{k:ep[k] for k in w.fieldnames if k in ep},'profile':label,'campaign':r['campaign'],'run_id':r['run_id'],'seed':r['seed']})
    e.track(ROOT/'TestOutline.md','test_protocol');e.track(ROOT/'docs/acceptance/decisions/D09_quantization_20261026.md','quantization_authorization')
    e.track(ROOT/'third_party/FlagScale/examples/kerv/README.md','local_flagscale_source')
    e.track(ROOT/'scripts/run_kerv_moore_ascend.sh','current_launcher');e.track(ROOT/'scripts/run_kerv_quant_qkv_ascend.sh','current_launcher')
    e.track(ROOT/'docs/acceptance/sources/2025-1215-课题任务书.pdf','primary_requirement_source')
    with (a.output/'evidence_index.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['path','bytes','sha256','roles']);w.writeheader()
        for name,v in sorted(e.evidence.items()):w.writerow({**v,'roles':';'.join(v['roles'])})
    write(inv/'input_manifest.json',e.evidence)
    shutil.copy2(__file__,inv/'evaluate_ascend_delivery.py')
    try:
        result=subprocess.run(['npu-smi','info'],capture_output=True,text=True)
        hardware_text=result.stdout+result.stderr
    except FileNotFoundError:
        hardware_text='npu-smi unavailable on this statistics-only host; no new hardware measurement\n'
    (inv/'npu_state.txt').write_text(hardware_text)
    write(a.output/'validation.json',dict(status='PASS',scope='Raw evidence validation and recomputation only',
        source_files=len(e.evidence),quality_runs=sum(map(len,runs.values())),quality_episodes=sum(q['total'] for q in quality.values()),
        unique_goal_task_initial_state_seed_cases=20,exact_cached_steps=sum(v['steps'] for v in parity.values()),
        timing_comparisons=len(timings),p95_method='nearest-rank',source_campaigns_modified=False))
    print(json.dumps(dict(output=str(a.output),status='PASS',quality=quality,recommendations=report['recommendations']),ensure_ascii=False))

if __name__=='__main__':main()
