"""Rebuild current-stage paired metrics; pending/failed cases are never removed."""
import argparse
from collections import defaultdict, Counter
import datetime as dt
import json
import math
from pathlib import Path
import random
import statistics
from ports.metax.summarize import summarize, rows

ROOT=Path(__file__).resolve().parents[2]
PROFILES=['upstream_original','boundary_fixed','native','numerical_fixed','numerical_fixed_packed']

def stats(values):
    return dict(n=len(values),sum_seconds=sum(values),mean_ms=1000*statistics.mean(values),
        median_ms=1000*statistics.median(values),p95_ms=1000*sorted(values)[math.ceil(.95*len(values))-1]) if values else None

def wilson(success,n):
    if not n:return None
    z=1.959963984540054;p=success/n;d=1+z*z/n
    mid=(p+z*z/(2*n))/d;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [mid-half,mid+half]

def summarize_run(path,expected):
    if not path.exists():return dict(status='pending',expected_episodes=expected)
    result=summarize(path)
    episodes=rows(path/'raw/episodes.jsonl');steps=rows(path/'raw/steps.jsonl');events=rows(path/'raw/verify.jsonl')
    allkeys={(e['task_id'],e['init_state_id']) for e in episodes}
    assert len(allkeys)==len(episodes),'Duplicate episode records'
    groups=defaultdict(list)
    for row in steps:groups[row['episode_id']].append(row)
    per=[];warm=[]
    for key,seq in groups.items():
        seq.sort(key=lambda r:r['step_id']);assert seq[0]['step_id']==0
        assert [r['step_id'] for r in seq]==list(range(len(seq)))
        warm.extend(seq[1:])
        per.append(dict(episode=key,first_model_ms=1000*seq[0]['model_seconds'],
            including_first={k:stats([r[k] for r in seq]) for k in ['model_seconds','pipeline_seconds']},
            excluding_first={k:stats([r[k] for r in seq[1:]]) for k in ['model_seconds','pipeline_seconds']}))
    conservation=[];by_step=defaultdict(list)
    for event in events:by_step[event['episode_id'],event['step_id']].append(event)
    for key,seq in by_step.items():
        tokens=sum(e.get('initial_action_root_length',0)+e.get('committed_verifier_bonus_length',0)+
                   e.get('committed_prefix_length',0)+e.get('compensation_length',0)+e.get('padding_length',0) for e in seq)
        conservation.append(tokens==7)
    result.update(status='completed' if result['exit_code']=='0' and len(episodes)==expected else ('failed' if result['exit_code'] not in {None,'0'} else 'incomplete'),
        expected_episodes=expected,attempted_episodes=len(episodes),missing_episodes=expected-len(episodes),
        episode_results=[dict(task=e['task_id'],init=e['init_state_id'],success=e['success'],
            termination=e['termination_reason'],steps=e['steps']) for e in episodes],
        per_episode=per,including_first={k:stats([r[k] for r in steps]) for k in ['model_seconds','pipeline_seconds']},
        excluding_each_first_weighted={k:stats([r[k] for r in warm]) for k in ['model_seconds','pipeline_seconds']},
        sr_wilson95=wilson(sum(e['success'] for e in episodes),len(episodes)),
        # Failed attempted episodes count in this denominator. Not-yet-run cases
        # remain missing, never called successful nor quietly excluded as complete.
        sr_completed_attempts=sum(e['success'] for e in episodes)/len(episodes) if episodes else None,
        sr_expected_matrix_bounds=[sum(e['success'] for e in episodes)/expected,
            (sum(e['success'] for e in episodes)+expected-len(episodes))/expected],
        seven_token_conservation=all(conservation) if conservation else None,
        events_nonprefix_retained=sum(not e.get('retained_candidate_is_prefix',True) for e in events),
        discarded_roots=sum(e.get('discarded_root_length',0) for e in events),
        discarded_candidates=sum(e.get('discarded_candidate_length',0) for e in events),
        observed_rejection_positions=dict(Counter(str(e.get('action_tokens_before',0)+1+e['accepted_prefix_length']) for e in events
            if e['accepted_prefix_length']<e['proposed_length'])),
        thresholds=dict(Counter(str(e['threshold']) for e in events)))
    return result

def paired(a,b):
    A={(r['task'],r['init']):int(r['success']) for r in a.get('episode_results',[])}
    B={(r['task'],r['init']):int(r['success']) for r in b.get('episode_results',[])}
    if a['status']!='completed' or b['status']!='completed' or A.keys()!=B.keys():
        return dict(status='pending_complete_pair')
    keys=sorted(A);delta=[B[k]-A[k] for k in keys]
    rng=random.Random(7);samples=sorted(statistics.mean(rng.choices(delta,k=len(delta))) for _ in range(10000))
    return dict(status='complete',n=len(keys),success_reference=sum(A.values()),success_test=sum(B.values()),
        sr_delta_pp=100*statistics.mean(delta),
        sr_relative_delta=(sum(B.values())-sum(A.values()))/sum(A.values()) if sum(A.values()) else None,
        paired_bootstrap95_delta_pp=[100*samples[249],100*samples[9749]],
        discordant=dict(test_only=sum(d==1 for d in delta),reference_only=sum(d==-1 for d in delta)),
        mean_committed_acceptance_delta=b['mean_committed_prefix']-a['mean_committed_prefix'],
        model_reference_over_test=a['model']['sum_seconds']/b['model']['sum_seconds'],
        pipeline_reference_over_test=a['policy_pipeline']['sum_seconds']/b['policy_pipeline']['sum_seconds'],
        loop_reference_over_test=a['closed_loop_seconds']/b['closed_loop_seconds'],
        allocated_saving_bytes=a['allocated_peak']['bytes']-b['allocated_peak']['bytes'],
        reserved_saving_bytes=a['reserved_peak']['bytes']-b['reserved_peak']['bytes'],
        caveat='Same paired task/init/seed, own trajectories; different step counts may affect totals. Bootstrap describes this finite batch, not population proof; matching outcomes alone can give degenerate intervals.')

def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,default=ROOT/'outputs/metax_model_performance_20261105');a=p.parse_args()
    out=a.directory.resolve();reports={profile:summarize_run(out/(profile+'_dev6'),6) for profile in PROFILES}
    comparisons={name:paired(reports[ref],reports[test]) for name,ref,test in [
        ('corrected_spec_to_native','boundary_fixed','native'),('corrected_spec_to_fixed','boundary_fixed','numerical_fixed'),
        ('native_to_fixed','native','numerical_fixed'),('same_fixed_to_packed','numerical_fixed','numerical_fixed_packed'),
        ('spec_to_packed','boundary_fixed','numerical_fixed_packed'),('native_to_packed','native','numerical_fixed_packed')]}
    result=dict(updated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),runs=reports,comparisons=comparisons,
        baseline_identity='Real pinned SpecVLA Relaxed threshold9, original and corrected separately; strict0 available, final taskbook B0 identity must be stated explicitly',
        primary_acceptance='per verification actually submitted consecutive candidate prefix including zero; excludes root/bonus/KF/padding; original tail discards recorded separately',
        thresholds_not_equivalent='SpecVLA fixed9 vs KERV dynamic15..3 and KF, part of declared algorithms',
        target_acceptance_comparator='boundary_fixed SpecVLA, not KERV native/optimized delta',
        old_results_imported=False)
    (out/'model-performance-results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    if (ROOT/'docs/METAX_MODEL_PERFORMANCE.md').exists():
        path=ROOT/'docs/METAX_MODEL_PERFORMANCE.md';text=path.read_text();marker='<!-- measured-results -->'
        block=marker+'\n\n本阶段开发集原始结果（不拼接历轮数值）：\n\n| 配置 | 状态 | SR成功/已执行/计划 | 模型/策略/闭环 s | allocated / reserved B | 主接受长度 |\n|---|---|---|---|---|---|\n'
        for profile,r in reports.items():
            if r['status']=='pending':block+=f'| {profile} | pending，等待GPU | — / — / 6 | — | — | — |\n';continue
            block+=f"| {profile} | {r['status']} | {r['successful_episodes']}/{r['attempted_episodes']}/6 | {r['model']['sum_seconds']:.3f}/{r['policy_pipeline']['sum_seconds']:.3f}/{r['closed_loop_seconds']:.3f} | {r['allocated_peak']['bytes']} / {r['reserved_peak']['bytes']} | {r['mean_committed_prefix']} |\n"
        block+='\n原版仅单列，不用其边界错误放大倍率。配对差/区间、每回合去首步均值/中位/P95、含首步及设备采样见 `model-performance-results.json`。未完整的配置不计算正式配对倍率。\n\n'+marker
        if text.count(marker)==2:
            pre,_,post=text.split(marker);path.write_text(pre+block+post)
    print(json.dumps(dict(statuses={p:r['status'] for p,r in reports.items()},comparisons=comparisons),indent=2))
if __name__=='__main__':main()
