"""Recompute Moore W8 storage, closed-loop quality and memory comparisons."""
import argparse
import csv
from pathlib import Path
from acceptance.aggregate import aggregate
from acceptance.common import dump, load, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    args = parser.parse_args()
    c = args.campaign
    result = aggregate(c)
    full = [x for x in result['runs'] if x['kind'] == 'full']
    for item in full:
        item['model_load_seconds'] = next((x['elapsed_since_collector_start_seconds'] for x in
            rows(c / 'runs' / item['run_id'] / 'phases.jsonl') if x['phase'] == 'model_loaded'), None)
    dump(c / 'reports/local_results.json', result)
    proof = []
    for item in result['runs']:
        if not item['profile'].startswith('quantized'): continue
        records = rows(c / 'runs' / item['run_id'] / 'quantization.jsonl')
        loaded = next((x for x in records if x['phase'] == 'after_model_load'), None)
        ended = next((x for x in records if x['phase'] == 'process_exit'), None)
        buffers = loaded['execution_buffers'] if loaded else []
        calls = ended['module_forward_calls'] if ended else {}
        manifest = loaded['manifest'] if loaded else {}
        expected_group_size = 32 if item['profile'] == 'quantized_g32' else 128
        ok = (item['status'] == 'PASS' and len(buffers) == manifest.get('module_count') == 96
              and manifest.get('group_size') == expected_group_size
              and not manifest.get('fallback', True) and len(calls) == 96
              and all(value > 0 for value in calls.values())
              and all(x['weight_device'] == 'musa:0' and x['weight_dtype'] == 'torch.int8'
                      and x['scale_dtype'] == 'torch.float32' for x in buffers))
        proof.append({'run_id': item['run_id'], 'execution_storage_and_use': 'PASS' if ok else 'INCOMPLETE_OR_FAIL',
                      'int8_module_count': len(buffers), 'modules_used': sum(value > 0 for value in calls.values()),
                      'minimum_forward_calls': min(calls.values()) if calls else None,
                      'manifest': manifest})
        if item['status'] == 'PASS' and not ok:
            raise ValueError('Completed quantized run lacks storage/execution proof: ' + item['run_id'])
    dump(c / 'reports/quantization_evidence.json', proof)
    pairs = []
    for quantized in [x for x in full if x['profile'].startswith('quantized')]:
        for baseline in [x for x in full if x['profile'] in ('specvla', 'optimized', 'optimized_unfused') and
                         all(x[k] == quantized[k] for k in ('task_id', 'init_state_id', 'seed', 'physical_device_id'))]:
            pair = {'baseline': baseline['run_id'], 'quantized': quantized['run_id'],
                    'state': quantized['init_state_id'], 'execution_complete': baseline['status'] == quantized['status'] == 'PASS',
                    'baseline_task_success': baseline['task_success'], 'quantized_task_success': quantized['task_success']}
            if pair['execution_complete']:
                for kind in ('allocated', 'reserved'):
                    key = 'maximum_peak_' + kind + '_bytes'
                    saved = baseline[key] - quantized[key]
                    pair[kind + '_peak_saving_bytes'] = saved
                    pair[kind + '_saving_ge2GB'] = saved >= 2_000_000_000
                    pair[kind + '_saving_ge2GiB'] = saved >= 2 * 1024**3
                for field in ('model_seconds_excluding_first', 'pipeline_seconds'):
                    pair[field + '_mean_speedup'] = baseline[field]['mean'] / quantized[field]['mean']
                left, right = [rows(c / 'runs' / x['run_id'] / 'steps.jsonl') for x in (baseline, quantized)]
                pair['same_initial_observation_hash'] = left[0]['observation_hash'] == right[0]['observation_hash']
                pair['same_steps'] = len(left) == len(right)
                pair['token_equal_records_in_overlap'] = sum(a['action_tokens'] == b['action_tokens'] for a, b in zip(left, right))
                pair['latency_scope'] = 'descriptive: independently evolving trajectories, not lossless speedup'
            pairs.append(pair)
    dump(c / 'reports/memory_and_quality_comparisons.json', pairs)
    columns = ['run_id', 'profile', 'init_state_id', 'seed', 'status', 'task_success', 'steps',
               'model_load_seconds', 'first_model_seconds', 'maximum_peak_allocated_bytes', 'maximum_peak_reserved_bytes',
               'vendor_sampled_device_peak_bytes', 'steady_peak_allocated_bytes', 'steady_peak_reserved_bytes']
    with (c / 'reports/results.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader(); writer.writerows(full)
    f = lambda value: '未完成' if value is None else ('是' if value is True else '否' if value is False else str(value))
    text = ['# 摩尔 W8A16 量化实测结果', '', 'Campaign：`' + c.name + '`。根据用户本轮确认，将量化纳入允许的优化范围；本轮使用独立配置，不改变默认BF16入口。', '',
            '## 配置与执行事实', '',
            'W8：INT8分组对称权重，主候选group_size=128，追加候选group_size=32，FP32 scale；A16：BF16激活/GEMM。仅verifier的96个MLP投影量化，attention、vision、drafter、embedding、lm_head、KV保持原精度。CPU上转换后才搬到MUSA；每层前向临时反量化，不保存全模型BF16副本。这是显存优化路径，不宣称原生INT8 GEMM加速。', '',
            'group128选中权重存储从8,657,043,456降为4,463,788,032 bytes，静态节省4,193,255,424 bytes；group32为4,869,586,944 bytes，节省3,787,456,512 bytes（均含FP32 scale，不含未改变的bias）。实际全模型显存以实测峰值为准。group32在观察到group128任务失败后追加，失败与新增用例分别保留，不作为原结果的覆盖重试。', '',
            '量化开启时BF16投影融合不叠加，树/Kalman/阈值设置保留。通过专用`quantized` profile启动；错误严格退出，不允许将BF16回退写成量化成功。', '',
            '## 全部完整闭环', '', '| Run | 官方state | 执行 | 成功 | 动作数 | 排除首步模型均值(s) | pipeline均值(s) | allocated峰值(GB) | reserved峰值(GB) |',
            '| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for x in full:
        text.append('| ' + ' | '.join([x['run_id'], x['init_state_id'], x['status'], f(x['task_success']), f(x.get('steps')),
            f(round(x['model_seconds_excluding_first']['mean'], 6)) if x.get('steps') else '未完成',
            f(round(x['pipeline_seconds']['mean'], 6)) if x.get('steps') else '未完成',
            f(round(x['maximum_peak_allocated_bytes']/1e9, 6)) if x.get('maximum_peak_allocated_bytes') else '未完成',
            f(round(x['maximum_peak_reserved_bytes']/1e9, 6)) if x.get('maximum_peak_reserved_bytes') else '未完成']) + ' |')
    text += ['', 'task0，官方state0/1/2，框架seed7；env.seed仍固定0，但三个官方初始状态分别注入。最多220策略步，稳定化10步另计。每种状态的BF16/量化顺序为AB、BA、AB；B0另做state0。冒烟8步不算任务失败，也不进入SR。执行PASS仅指运行与采集完成，任务成功另列。', '',
             '## 峰值显存：基线减量化，正数表示节省', '',
             '| 基线 → 量化 | allocated节省(GB) | reserved节省(GB) | allocated≥2GB/2GiB | reserved≥2GB/2GiB | 初始观测相同 |',
             '| --- | ---: | ---: | --- | --- | --- |']
    for p in pairs:
        if p['execution_complete']:
            text.append('| '+p['baseline']+' → '+p['quantized']+' | '+f(round(p['allocated_peak_saving_bytes']/1e9, 6))+' | '+f(round(p['reserved_peak_saving_bytes']/1e9, 6))+' | '+f(p['allocated_saving_ge2GB'])+'/'+f(p['allocated_saving_ge2GiB'])+' | '+f(p['reserved_saving_ge2GB'])+'/'+f(p['reserved_saving_ge2GiB'])+' | '+f(p['same_initial_observation_hash'])+' |')
    text += ['', '峰值从模型加载前开始采集，首动作后另建稳态窗口，取两窗口最大值；没有排除BF16整模型上卡峰值来人为降低量化峰值，量化是在CPU准备后才上卡。allocated与reserved独立比较；整卡vendor采样峰值、实际采样间隔及冷/稳态分布见JSON。不同轨迹/动作数造成的工作量差异也需保留，不能将此表解释成相同动作工作量下的全部收益。', '',
             '## 任务质量及部署状态', '']
    for profile in ('optimized', 'optimized_unfused', 'quantized', 'quantized_g32'):
        group = next(x for x in result['groups'] if x['profile'] == profile)
        text.append('- '+profile+'：成功 '+str(group['task_successes'])+'/'+str(group['executed_cases_including_errors'])+' 个已执行完整case；执行故障 '+str(group['execution_failures'])+'；未执行 '+str(group['not_started_or_blocked'])+'。')
    text += ['', '本地三个初始状态只能说明该子集的任务完成情况，不能证明完整LIBERO SR或损失<5%。量化是有损变换，不要求/宣称与BF16逐token相同；同输入的动作误差另以固定回放记录。原有BF16跨seed/跨进程不一致问题没有在本轮被修复，不把量化作为其修复。BF16 optimized与量化版还存在Linear融合配置差异，追加optimized_unfused控制回合以免将所有退化归因于INT8；各控制仍为小样本，不能单独确立一般因果。', '',
             '量化实际执行证明见`reports/quantization_evidence.json`：逐run检查96个INT8 MUSA buffer、FP32 scale、非fallback manifest及所有量化模块的真实forward次数。若尚未完成，保留INCOMPLETE_OR_FAIL。', '']
    text += ['模型加载时间（含CPU转换）、首步模型耗时及全部冷/稳态数值另列在`reports/results.csv`；CPU量化耗时不混入逐动作计时，也不从初始化显存采集中排除。', '']
    replay_file = c / 'replays/quantized_fixed_input_v1/results.json'
    if replay_file.exists():
        replay = load(replay_file)
        text += ['## 同输入固定回放', '',
                 '每组'+str(replay['protocol']['frames'])+'帧，先完整预热一次，再计时一次。输入与历史逐帧固定；观测hash相同'+str(replay['same_observation_hashes'])+'，token完全相同'+str(replay['exact_action_token_matches'])+'；执行夹爪相同'+str(replay['executed_gripper_matches'])+'。', '',
                 '7维raw action绝对误差均值 '+str(replay['raw_action_absolute_error']['mean'])+'，P95 '+str(replay['raw_action_absolute_error']['p95'])+'，最大 '+str(replay['maximum_raw_action_absolute_error'])+'；逐维MAE、每帧L2和全部记录保留。该误差为checkpoint动作空间数值，不额外假定物理单位；回放不产生SR。', '',
                 'BF16/量化模型均值比 '+str(replay['model_mean_speedup'])+'×，pipeline均值比 '+str(replay['pipeline_mean_speedup'])+'×；不同算法输出、量化误差及样本范围均披露，不称为无损加速。两个worker同卡驻留、串行计算，回放显存不参与单模型显存表。', '']
        text += ['换算为耗时变化：量化版模型均值增加 '+format((1/replay['model_mean_speedup']-1)*100,'.2f')+'%，pipeline均值增加 '+format((1/replay['pipeline_mean_speedup']-1)*100,'.2f')+'%。本轮不能声称量化带来时延收益。', '']
    text += ['## 直接运行及复算', '', '以下是实验入口，不代表量化候选已通过任务质量门禁；默认入口仍为BF16 optimized。', '', '```bash', 'cd /root/KERV',
             'KERV_MUSA_PROFILE=quantized src/scripts/run_musa.sh --max-tasks 1 --trials 1 --max-episode-steps 220', '',
             '# group32追加候选：',
             'KERV_MUSA_PROFILE=quantized src/scripts/run_musa.sh --max-tasks 1 --trials 1 --max-episode-steps 220 \\',
             '  --override inference.runtime.w8a16.group_size=32', '',
             'PYTHONPATH=src python3 -m acceptance.quantization_report --campaign '+str(c), '```', '',
             '不要使用`--load_in_8bit`/`--load_in_4bit`代替此入口：它们是另一个checkpoint加载接口，当前MUSA仍明确拒绝。专用profile为INT8权重存储+BF16逐层计算，当前只验证verifier MLP组合。', '',
             '## 证据与正式范围', '',
             '配置、命令、退出码、视频、逐步动作/接受/时延/内存及量化manifest均在`runs/`；预登记与环境/权重哈希在`inventory/`；固定回放在`replays/`。历史2026-10-19报告和原始证据保持原样，本轮作为独立版本与独立测试记录。', '',
             '正式验收仍需既定模型/平台/芯片覆盖、SR统计和相关口径材料；本机量化允许范围已经获得用户确认，不再重复征求许可。本轮结果不代替A100实验、第三方验证或项目签认。']
    (c / 'summary.md').write_text('\n'.join(text)+'\n')
    print(c / 'summary.md')


if __name__ == '__main__': main()
