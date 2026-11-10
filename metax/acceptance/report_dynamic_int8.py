"""Report deployment memory, paired task outcomes, and separate same-chip timing."""
import argparse
import json
from pathlib import Path
import statistics

from acceptance.aggregate_quantization import summarize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('campaign', type=Path)
    p.add_argument('--baseline', default='bf16_init2')
    p.add_argument('--baseline-dir', type=Path)
    p.add_argument('--candidate', default='dynamic_int8_init2')
    p.add_argument('--timing', nargs='+', default=['timing_dynamic'])
    p.add_argument('--additional-baseline', type=Path)
    p.add_argument('--additional-candidate')
    p.add_argument('--other-candidates', nargs='*', default=[])
    p.add_argument('--other-timing', nargs='*', default=[])
    p.add_argument('--report-stem', default='dynamic_results')
    a = p.parse_args()
    runs = {}
    for name in [a.baseline, a.candidate]:
        run = summarize(a.baseline_dir if name==a.baseline and a.baseline_dir else a.campaign / 'runs' / name)
        assert run['execution_status'] == 'PASS' and run['full_episodes'] == 10
        runs[name] = run
    b, q = runs[a.baseline], runs[a.candidate]
    assert [(e['task_id'], e['init_state_id']) for e in b['episode_results']] == [
        (e['task_id'], e['init_state_id']) for e in q['episode_results']]
    timing = []
    for name in a.timing:
        data = json.loads((a.campaign / name / 'results.json').read_text())
        assert data['plan'].get('reference_kind','bf16')=='bf16', 'Use a report with explicit quantized-reference labels'
        sb, sq = data['summary']['bf16'], data['summary']['int8']
        assert data['plan']['policy_sha256'] == q['quantization']['policy_sha256']
        assert sb['steps'] == sq['steps'] == data['plan']['samples'] * data['plan']['measured_repeats']
        per_repeat = []
        for repeat in sorted({row['repeat'] for row in data['rows']['bf16']}):
            mb = statistics.mean(row['model_ms'] for row in data['rows']['bf16'] if row['repeat'] == repeat)
            mq = statistics.mean(row['model_ms'] for row in data['rows']['int8'] if row['repeat'] == repeat)
            per_repeat.append(dict(repeat=repeat, bf16_ms=mb, candidate_ms=mq, change_percent=100*(mq/mb-1)))
        percentiles = {}
        for label, rows in data['rows'].items():
            values = sorted(row['model_ms'] for row in rows)
            def percentile(fraction):
                index = (len(values)-1)*fraction
                lo = int(index)
                hi = min(lo+1,len(values)-1)
                return values[lo]+(values[hi]-values[lo])*(index-lo)
            percentiles[label] = dict(p50_ms=percentile(.5), p95_ms=percentile(.95),
                                      mean_pipeline_ms=data['summary'][label]['mean_pipeline_ms'])
        timing.append(dict(name=name, bf16_ms=sb['mean_model_ms'], candidate_ms=sq['mean_model_ms'],
                           change_percent=100*(sq['mean_model_ms']/sb['mean_model_ms']-1),
                           steps_per_profile=sb['steps'], per_repeat=per_repeat,
                           verification_cycles=dict(bf16=sb['verify_cycles'], candidate=sq['verify_cycles']),
                           policy_sha256=data['plan']['policy_sha256'], percentiles=percentiles))
    saving = b['allocated_peak_bytes'] - q['allocated_peak_bytes']
    result = dict(runs=runs, timing=timing, allocated_peak_saving_bytes=saving,
                  allocated_peak_saving_GB=saving/1e9, allocated_peak_saving_GiB=saving/2**30,
                  bf16_successes=b['successes'], candidate_successes=q['successes'],
                  observed_memory_target_met=saving>=2_000_000_000,
                  observed_timing_target_met=all(t['change_percent']<=0 for t in timing),
                  observed_success_count_not_lower=q['successes']>=b['successes'],
                  population_quality_loss_bound_proven=False,
                  scope='Current Ascend; ten paired Goal tasks, one initial state; same-chip timing measured separately from deployment memory')
    additional = None
    if a.additional_baseline or a.additional_candidate:
        assert a.additional_baseline and a.additional_candidate
        extra_b = summarize(a.additional_baseline)
        extra_q = summarize(a.campaign / 'runs' / a.additional_candidate)
        assert extra_b['execution_status'] == extra_q['execution_status'] == 'PASS'
        assert extra_b['full_episodes'] == extra_q['full_episodes'] == 10
        assert [(e['task_id'], e['init_state_id']) for e in extra_b['episode_results']] == [
            (e['task_id'], e['init_state_id']) for e in extra_q['episode_results']]
        assert extra_q['quantization']['policy_sha256'] == q['quantization']['policy_sha256']
        additional = dict(baseline_source=str(a.additional_baseline), baseline=extra_b, candidate=extra_q)
        result['additional_quality_pair'] = additional
        result['observed_success_count_not_lower'] &= extra_q['successes'] >= extra_b['successes']
        result['total_bf16_successes'] = b['successes'] + extra_b['successes']
        result['total_candidate_successes'] = q['successes'] + extra_q['successes']
        result['episodes_per_profile_across_two_initial_states'] = 20
        result['scope'] = ('Current Ascend; two initial states of the same ten Goal tasks, 20 trajectories per profile; '
                           'separate same-chip timing and full deployment memory')
    result['other_candidates'] = {name: summarize(a.campaign / 'runs' / name) for name in a.other_candidates}
    assert all(run['execution_status'] == 'PASS' and run['full_episodes'] == 10 for run in result['other_candidates'].values())
    result['other_timing'] = {name: json.loads((a.campaign / name / 'results.json').read_text())['summary']
                              for name in a.other_timing}
    (a.campaign / 'reports').mkdir(exist_ok=True)
    (a.campaign / 'reports' / (a.report_stem+'.json')).write_text(json.dumps(result, indent=2) + '\n')
    lines = ['# 昇腾混合量化省内存与时延验证', '',
             '| 配置 | 成功/10 | allocated峰值GB |', '|---|---:|---:|',
             f"| BF16 | {b['successes']}/10 | {b['allocated_peak_bytes']/1e9:.3f} |",
             f"| 量化候选 | {q['successes']}/10 | {q['allocated_peak_bytes']/1e9:.3f} |", '',
             f'峰值节省 {saving/1e9:.3f}GB / {saving/2**30:.3f}GiB，包含加载和首次执行。', '',
             '| 同卡交替对照 | BF16 ms | 候选 ms | 时延变化 |', '|---|---:|---:|---:|']
    for t in timing:
        lines.append(f"| {t['name']} | {t['bf16_ms']:.3f} | {t['candidate_ms']:.3f} | {t['change_percent']:+.2f}% |")
    lines += ['', '## 逐任务', '', '| Task | BF16结果/动作数 | 候选结果/动作数 |', '|---|---|---|']
    for eb, eq in zip(b['episode_results'], q['episode_results']):
        lines.append(f"| {eb['task_id']} | {eb['success']}/{eb['steps']} | {eq['success']}/{eq['steps']} |")
    if additional:
        eb, eq = additional['baseline'], additional['candidate']
        lines += ['', '## 另一初态的质量对照', '',
                  f"BF16 {eb['successes']}/10，候选 {eq['successes']}/10。BF16数据来自 {a.additional_baseline}。",
                  '该组历史BF16记录只用于任务质量对照，不与新时延拼接计算加速比。']
    if result['other_candidates']:
        lines += ['', '## 其他候选（相同init2）', '', '| 配置 | 成功/10 | allocated峰值GB |', '|---|---:|---:|']
        for name, run in result['other_candidates'].items():
            lines.append(f"| {name} | {run['successes']}/10 | {run['allocated_peak_bytes']/1e9:.3f} |")
    lines += ['', '配置使用init0激活校准；不同精度允许不同动作和验证树。',
              '时延不包含加载、校准、预热、profiler；临时双模型驻留内存不用于省显存结论。',
              '每配置每初态十回合；两初态共享任务身份。该样本不能证明总体成功率损失边界，也不能代表所有任务或硬件负载。']
    (a.campaign / 'summary.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k not in {'runs','additional_quality_pair','other_candidates','other_timing'}}, indent=2))


if __name__ == '__main__':
    main()
