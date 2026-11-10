"""Validate paired task identities and report the accuracy-aware PTQ pilot."""
import argparse
import json
from pathlib import Path

from acceptance.aggregate_quantization import summarize


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('campaign',type=Path);a=p.parse_args();base=a.campaign
    names=['bf16_calibration_init0','int8_g128_init0','int8_g32_init0','bf16_init1','int8_g128_init1','int8_g32_init1','int8_calibrated_init1']
    reports={}
    for name in names:
        path=base/'runs'/name
        meta=json.loads((path/'run.json').read_text());assert meta['status']=='PASS',name
        r=summarize(path);assert r['full_episodes']==10,name
        state=0 if name.endswith('init0') else 1
        assert [(e['task_id'],e['init_state_id']) for e in r['episode_results']]==[(i,state) for i in range(10)]
        r['timing_scope']='diagnostic capture; not a speed comparison' if meta.get('calibration') else 'descriptive closed-loop latency; separate card and trajectories'
        reports[name]=r
    comparisons={}
    for name in ['int8_g128_init1','int8_g32_init1','int8_calibrated_init1']:
        b=reports['bf16_init1'];q=reports[name]
        pairs=list(zip(b['episode_results'],q['episode_results']))
        saving=b['allocated_peak_bytes']-q['allocated_peak_bytes']
        comparisons[name]={'bf16_successes':b['successes'],'candidate_successes':q['successes'],'episodes_per_profile':10,
            'success_rate_change_percentage_points':100*(q['success_rate']-b['success_rate']),
            'relative_success_rate_loss':(b['success_rate']-q['success_rate'])/b['success_rate'] if b['success_rate'] else None,
            'bf16_only_success_tasks':[x['task_id'] for x,y in pairs if x['success'] and not y['success']],
            'candidate_only_success_tasks':[y['task_id'] for x,y in pairs if y['success'] and not x['success']],
            'allocated_peak_saving_bytes':saving,'allocated_peak_saving_GB':saving/1e9,'allocated_peak_saving_GiB':saving/2**30,
            'saving_at_least_2GiB':saving>=2*2**30,'formal_2_percentage_point_loss_bound_proven':False}
    policy=json.loads((base/'calibrated/policy.json').read_text());calibration=json.loads((base/'calibrated/summary.json').read_text())
    q=reports['int8_calibrated_init1']['quantization']
    assert q['policy_sha256']==calibration['policy_sha256'] and q['module_count']==54
    assert q['group_counts']=={'32':policy['group_counts']['32'],'64':0,'128':policy['group_counts']['128']}
    timing={}
    for name in ['timing_g32','timing_calibrated']:
        r=json.loads((base/name/'results.json').read_text());b=r['summary']['bf16'];q=r['summary']['int8']
        assert r['plan']['samples']==64 and b['steps']==q['steps']==128
        timing[name]={'bf16_mean_model_ms':b['mean_model_ms'],'quant_mean_model_ms':q['mean_model_ms'],
            'quant_latency_change_percent':100*(q['mean_model_ms']/b['mean_model_ms']-1),
            'bf16_over_quant_latency_ratio':b['mean_model_ms']/q['mean_model_ms'],
            'verify_cycles':{'bf16':b['verify_cycles'],'quant':q['verify_cycles']},
            'scope':'same-chip interleaved fixed64 observations, warm64 and measured128 each; independent generated histories; no loading/capture/profiler'}
    report={'campaign':base.name,'scope':'current Ascend only; paired init1 is held out from calibration, ten episodes per profile',
        'method':'activation-calibrated group and global-projection clipping search, not GPTQ or task-aware KL optimization',
        'runs':reports,'validation_comparisons':comparisons,'calibration':calibration,'timing':timing,
        'limitations':['only ten held-out episodes per profile; no formal2pp or5% guarantee',
            'tasks are shared with calibration; held out by initial state/trajectory, not held-out task identities',
            'full-episode memory comparisons across chips of this machine; timing measured separately on one chip',
            'GPTQ dependencies were absent; no claim that a complete GPTQ or full sensitivity-selected BF16/INT8 pipeline was evaluated',
            'no new KERV/native total speedup measured']}
    (base/'reports').mkdir(exist_ok=True)
    (base/'reports/results.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    text=['# 昇腾精度优先量化试验','',
        '方法：真实BF16激活校准，按投影搜索clip比例，在group128基础上最多24个投影升级group32。不是完整GPTQ。',
        '', '## 校准/开发集：init0','', '| 配置 | 成功/10 | 总动作 | allocated峰值GB |','|---|---:|---:|---:|']
    for name in names[:3]:
        r=reports[name];text.append(f"| {name} | {r['successes']}/10 | {r['steps']} | {r['allocated_peak_bytes']/1e9:.3f} |")
    text+=['','## 独立初始状态验证：init1','','配置冻结后评估，未根据该组结果更改policy。','',
        '| Task | BF16结果/步数 | group128结果/步数 | group32结果/步数 | 校准混合结果/步数 |','|---|---|---|---|---|']
    for task in range(10):
        values=[]
        for name in names[3:]:
            ep=reports[name]['episode_results'][task];values.append(('成功' if ep['success'] else '失败')+'/'+str(ep['steps']))
        text.append('| '+str(task)+' | '+' | '.join(values)+' |')
    text+=['','| 验证配置 | 成功/10 | allocated峰值GB | 对BF16节省GB/GiB |','|---|---:|---:|---:|']
    for name in names[3:]:
        r=reports[name];c=comparisons.get(name)
        saving=f"{c['allocated_peak_saving_GB']:.3f}/{c['allocated_peak_saving_GiB']:.3f}" if c else '参考'
        text.append(f"| {name} | {r['successes']}/10 | {r['allocated_peak_bytes']/1e9:.3f} | {saving} |")
    text+=['','## 同卡逐帧交替时延','','| 配置 | BF16 ms | INT8 ms | INT8时延变化 |','|---|---:|---:|---:|']
    for name,r in timing.items():text.append(f"| {name} | {r['bf16_mean_model_ms']:.3f} | {r['quant_mean_model_ms']:.3f} | {r['quant_latency_change_percent']:+.2f}% |")
    text+=['','## 范围','',
        '全部失败保留；小样本不证明成功率损失≤2个百分点。校准误差下降不等于闭环质量一定提升。',
        '各单模型全程显存包含加载和首次执行；同卡时延测试临时双模型驻留，其内存不用于显存结论。',
        'BF16/init0激活采集有额外操作，其时延只作诊断，不进入加速比较。',
        '原始记录、输入哈希、校准policy、模块权重哈希和失败测试均保留于本目录。']
    (base/'summary.md').write_text('\n'.join(text)+'\n')
    print(json.dumps({'successes':{n:r['successes'] for n,r in reports.items()},'timing':timing,'validation_comparisons':comparisons},indent=2))


if __name__=='__main__':main()
