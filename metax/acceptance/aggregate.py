"""Recompute descriptive local results from immutable raw records."""
import argparse, csv, json, math, statistics
from fractions import Fraction
from pathlib import Path


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()] if path.exists() else []


def stats(values):
    if not values:return {'count':0,'mean':None,'median':None,'p95_nearest_rank':None}
    if not all(isinstance(x,(int,float)) and math.isfinite(x) for x in values):raise ValueError('Non-finite or nonnumeric sample')
    ordered=sorted(values)
    return {'count':len(values),'mean':statistics.mean(values),'median':statistics.median(values),
        'p95_nearest_rank':ordered[math.ceil(.95*len(values))-1]}


def acceptance_improvement_pass(delta): return Fraction(str(delta))>Fraction(1,2)

def success_loss_pass(base_success,base_total,new_success,new_total):
    return Fraction(base_success,base_total)-Fraction(new_success,new_total)<Fraction(1,20)


def summarize(directory):
    meta=json.loads((directory/'run.json').read_text());metrics=directory/'metrics'
    steps=rows(metrics/'steps.jsonl');verify=rows(metrics/'verify.jsonl');episodes=rows(metrics/'episodes.jsonl')
    memory=rows(metrics/'memory.jsonl');transfers=rows(metrics/'transfers.jsonl')
    ids=[(x['episode_id'],x['step_id']) for x in steps]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate step ID: '+directory.name)
    vids=[(x['episode_id'],x['step_id'],x['verify_id']) for x in verify]
    if len(vids)!=len(set(vids)):raise ValueError('Duplicate verify ID: '+directory.name)
    for s in steps:
        for name in ['raw_action','executed_action']:
            assert len(s[name])==7 and all(math.isfinite(x) for x in s[name])
        assert len(s['action_tokens'])==7
    for ep in episodes:
        assert ep['steps']==sum(x['episode_id']==ep['episode_id'] for x in steps)
    samples={name:stats([x[name] for x in steps]) for name in ['model_seconds','pipeline_seconds','env_seconds','wall_seconds','cpu_seconds']}
    accepted=[x['accepted_prefix_length'] for x in verify]
    for x in accepted:assert isinstance(x,int) and x>=0
    peaks={name:max([x.get(name,0) for x in memory if isinstance(x.get(name),(int,float))],default=None)
        for name in ['allocated_peak_bytes','reserved_peak_bytes','cpu_rss_bytes']}
    full=[x for x in episodes if x['termination_reason']!='truncated_smoke']
    transfer_totals={d:sum(x['bytes'] for x in transfers if x['direction']==d) for d in ['H2D','D2H']}
    diagnostic = meta.get('transfer_diagnostic') or any('profile_case.py' in x for x in meta.get('extra_overrides',[]))
    return {'run_id':directory.name,'profile':meta.get('profile'),'execution_status':meta['status'],
        'mode':'diagnostic' if diagnostic else meta.get('mode'),'physical_device_id':meta.get('physical_device_id'),'seed':meta.get('seed'),
        'episode_count':len(episodes),'full_episode_count':len(full),'successes':sum(x['success'] for x in full),
        'success_rate':sum(x['success'] for x in full)/len(full) if full else None,
        'steps':len(steps),'verify_events':len(verify),'verify_zero_acceptance_events':accepted.count(0),
        'raw_posterior_accept_length':stats(accepted),'formal_acceptance_status':'BLOCKED: padding/compensation provenance requires audit',
        'timing':samples,'closed_loop_seconds':[x['closed_loop_wall_seconds'] for x in episodes],
        'memory_bytes':peaks,'diagnostic_transfer_bytes':transfer_totals if transfers else None,
        'transfer_status':'PARTIAL_ATEN_DIAGNOSTIC' if transfers else 'NOT_MEASURED',
        'actual_dma_copy_seconds':None,'formal_bandwidth_status':'BLOCKED: D05 and device trace missing',
        'formal_acceptance_result':'BLOCKED: B0/A100/M2/P2/P3/SKU/decisions/third-party prerequisites missing',
        'confidence_interval':'not estimable from one episode per configuration'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign-dir',type=Path,required=True);a=p.parse_args()
    out=a.campaign_dir/'reports';out.mkdir(exist_ok=True)
    results=[summarize(x.parent) for x in sorted((a.campaign_dir/'runs').glob('*/run.json'))]
    report={'scope':'Ascend local staged self-test, not formal acceptance','runs':results,'pair':None}
    dirs=a.campaign_dir/'runs'
    if (dirs/'paired_B1/metrics/steps.jsonl').exists() and (dirs/'paired_B2/metrics/steps.jsonl').exists():
        x=rows(dirs/'paired_B1/metrics/steps.jsonl');y=rows(dirs/'paired_B2/metrics/steps.jsonl')
        equal=len(x)==len(y) and all((u['episode_id'],u['step_id'],u['observation_hash'],u['raw_action'],u['executed_action'],u['action_tokens'],u['done'])==
            (v['episode_id'],v['step_id'],v['observation_hash'],v['raw_action'],v['executed_action'],v['action_tokens'],v['done']) for u,v in zip(x,y))
        r={z['run_id']:z for z in results};b1=r['paired_B1'];b2=r['paired_B2']
        report['pair']={'actions_tokens_observations_exact':equal,'native_steps':len(x),'optimized_steps':len(y),
            'model_speedup_including_first':b1['timing']['model_seconds']['mean']/b2['timing']['model_seconds']['mean'],
            'pipeline_speedup_including_first':b1['timing']['pipeline_seconds']['mean']/b2['timing']['pipeline_seconds']['mean'],
            'closed_loop_wall_speedup':sum(b1['closed_loop_seconds'])/sum(b2['closed_loop_seconds']),
            'allocated_peak_saving_bytes':b1['memory_bytes']['allocated_peak_bytes']-b2['memory_bytes']['allocated_peak_bytes'],
            'reserved_peak_saving_bytes':b1['memory_bytes']['reserved_peak_bytes']-b2['memory_bytes']['reserved_peak_bytes'],
            'baseline':'KERV B1, not Spec-VLA B0'}
    (out/'results.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    with (out/'T11_T12_selftest.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['run_id','profile','execution_status','steps','verify_events','successes','full_episode_count','model_mean_s','pipeline_mean_s','allocated_peak_bytes','reserved_peak_bytes'])
        writer.writeheader()
        for r in results:
            writer.writerow({**{k:r[k] for k in ['run_id','profile','execution_status','steps','verify_events','successes','full_episode_count']},
                'model_mean_s':r['timing']['model_seconds']['mean'],'pipeline_mean_s':r['timing']['pipeline_seconds']['mean'],**{k:r['memory_bytes'][k] for k in ['allocated_peak_bytes','reserved_peak_bytes']}})
    text=['# 本轮昇腾自测结果','',report['scope'],'','| Run | 模式 | 状态 | 步数 | 成功/完整回合 | 模型均值ms | P95 ms |','|---|---|---|---:|---:|---:|---:|']
    for r in results:
        t=r['timing']['model_seconds'];text.append(f"| {r['run_id']} | {r['mode']} | {r['execution_status']} | {r['steps']} | {r['successes']}/{r['full_episode_count']} | {(t['mean'] or 0)*1000:.3f} | {(t['p95_nearest_rank'] or 0)*1000:.3f} |")
    text+=['','diagnostic各组含重型采集开销，不用于正式计时。smoke是截断采集，成功率分母0；策略失败和超步数保留。首步和全部逐步数据均保留。',
        '原始posterior接受长度保留0事件，但padding/补偿来源未审计，不作任务书ΔL>0.5判据。',
        'ATen传输仅诊断，实际DMA时间/隐式后端拷贝未覆盖，不作100MB/s通过证据。',
        '显存为含加载/初始化的框架全程峰值；厂商设备用量单独10秒采样，不等于进程100ms峰值。',
        '该表不证明A100、Spec-VLA基线、两模型三平台、三芯片或第三方验收已完成。']
    (out/'selftest_report.md').write_text('\n'.join(text)+'\n')
    print(json.dumps({'runs':len(results),'pair':report['pair']},ensure_ascii=False))

if __name__=='__main__':main()
