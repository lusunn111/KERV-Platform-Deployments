"""Render Moore-only self-test reports from raw recomputation, without formal PASS claims."""
import argparse
import csv
import json
from pathlib import Path
from acceptance.aggregate import aggregate
from acceptance.common import dump, load, rows, committed_candidate_prefix, distribution


def fmt(x, digits=6):
    return "未测" if x is None else (str(x) if isinstance(x, (str, int, bool)) else format(x, ".%df" % digits))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True)
    args=p.parse_args()
    c=args.campaign; r=aggregate(c); reports=c/'reports'
    dump(reports/'local_results.json',r)
    full=[x for x in r['runs'] if x['kind']=='full']
    flat=[]
    for x in full:
        flat.append({**{k:x.get(k) for k in ('run_id','profile','model_id','platform_id','chip_id','task_id','init_state_id','seed','status','task_success','steps','closed_loop_wall_seconds','maximum_peak_allocated_bytes','maximum_peak_reserved_bytes','vendor_sampled_device_peak_bytes')},
                     'model_mean_seconds':x.get('model_seconds',{}).get('mean'),
                     'model_mean_excluding_first_seconds':x.get('model_seconds_excluding_first',{}).get('mean'),
                     'model_p95_excluding_first_seconds':x.get('model_seconds_excluding_first',{}).get('p95'),
                     'pipeline_mean_seconds':x.get('pipeline_seconds',{}).get('mean'),
                     'posterior_acceptance_mean':x.get('accepted_prefix_length',{}).get('mean'),
                     'committed_candidate_prefix_mean':x.get('committed_candidate_prefix_length',{}).get('mean'),
                     'formal_acceptance_status':'BLOCKED'})
    if flat:
        for filename in ('T11_results.csv','T12_results.csv'):
            with (reports/filename).open('w') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    text=['# 摩尔节点新一轮验收自测结果','', 'Campaign：`'+c.name+'`。本报告由新运行的原始记录重算，范围限定为本机MTT S5000；不是课题整体或第三方验收结论。','',
          '## 完整闭环','', '| Run | 配置 | seed | 执行 | 任务成功 | 动作数 | 首步(s) | 排除首步均值(s) | 策略均值(s) | 闭环墙钟(s) | allocated峰值(GiB) | reserved峰值(GiB) |',
          '| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for x in full:
        text.append('| '+' | '.join([x['run_id'],x['profile'],x['seed'],x['status'],str(x.get('task_success')),str(x.get('steps','—')),fmt(x.get('first_model_seconds')),fmt(x.get('model_seconds_excluding_first',{}).get('mean')),fmt(x.get('pipeline_seconds',{}).get('mean')),fmt(x.get('closed_loop_wall_seconds')),fmt(x.get('maximum_peak_allocated_bytes')/1024**3 if x.get('maximum_peak_allocated_bytes') is not None else None),fmt(x.get('maximum_peak_reserved_bytes')/1024**3 if x.get('maximum_peak_reserved_bytes') is not None else None)])+' |')
    text += ['', '所有完整回合为task0、官方state0、最多220步；框架seed为7/17/27，现用LIBERO环境seed固定0。相同初始状态的重复不代表跨任务独立样本。冷启动、全部动作均值、中位数/P95、验证事件及显存原始bytes见 `reports/local_results.json`。','', '## KERV原生/优化逐步一致性','', '| seed | 对照 | 状态 | 全部动作/token一致 | 观测hash一致数 | 排除首步模型均值比 | 策略均值比 | 闭环墙钟比 |','| --- | --- | --- | --- | ---: | ---: | ---: | ---: |']
    for pair in r['pairs']:
        seed=next(x['seed'] for x in full if x['run_id']==pair['native'])
        speed_columns=([fmt(pair.get(key)) for key in ('model_seconds_excluding_first_mean_speedup','pipeline_seconds_mean_speedup','closed_loop_wall_speedup')]
                       if pair['status']=='PASS' else ['轨迹不同，不作无损加速判据']*3)
        text.append('| '+' | '.join([seed,pair['native']+' / '+pair['optimized'],pair['status'],str(pair.get('all_actions_and_tokens_identical')),str(pair.get('observation_hash_equal_records','—')),*speed_columns])+' |')
    successes=[x for x in r['pairs'] if x['status']=='PASS']
    if successes:
        speeds=[x['model_seconds_excluding_first_mean_speedup'] for x in successes]
        text += ['', '仅一致性通过配对的排除首步模型均值比观测范围：'+fmt(min(speeds))+'–'+fmt(max(speeds))+' 倍。它是本次观测范围，不是置信区间或跨平台SLA。']
    text += ['', '三组配对仅seed 7全部动作一致；seed 17在第78个动作、seed 27在第107个动作首次token分歧，分歧发生时观测hash相同。native seed 17达到220步仍未成功，optimized成功。不能宣称三组无损；原始JSON中不同轨迹的比值仅为描述性统计。新进程capture与此前optimized seed 17也有分歧，原因未定位，不单独归因于某个融合算子。']
    text += ['', '## 真实SpecVLA对照与接受长度','', '| Run | 验证次数 | posterior候选接受均值 | 动作边界内实际提交候选前缀均值 | 零接受事件 | 补偿事件 |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for x in full:
        if x.get('verification_events'):
            text.append('| '+' | '.join([x['run_id'],str(x['verification_events']),fmt(x['accepted_prefix_length']['mean']),fmt(x['committed_candidate_prefix_length']['mean']),str(x['zero_acceptance_events']),str(x['compensation_events'])])+' |')
    comparisons=[]
    base=next((x for x in full if x['profile']=='specvla' and x['status']=='PASS'),None)
    if base:
        for target in full:
            if target['status']!='PASS' or target['profile']=='specvla' or any(target[k]!=base[k] for k in ('task_id','init_state_id','seed','physical_device_id','model_id','platform_id')):
                continue
            item={'baseline':base['run_id'],'tested':target['run_id'],'same_registered_task_state_seed':True,
                  'posterior_mean_delta':target['accepted_prefix_length']['mean']-base['accepted_prefix_length']['mean'],
                  'committed_prefix_mean_delta':target['committed_candidate_prefix_length']['mean']-base['committed_candidate_prefix_length']['mean'],
                  'whole_episode_policy_seconds_speedup':sum(s['pipeline_seconds'] for s in rows(c/'runs'/base['run_id']/'steps.jsonl'))/sum(s['pipeline_seconds'] for s in rows(c/'runs'/target['run_id']/'steps.jsonl')),
                  'closed_loop_wall_speedup':base['closed_loop_wall_seconds']/target['closed_loop_wall_seconds'],
                  'allocated_peak_saving_bytes':base['maximum_peak_allocated_bytes']-target['maximum_peak_allocated_bytes'],
                  'reserved_peak_saving_bytes':base['maximum_peak_reserved_bytes']-target['maximum_peak_reserved_bytes'],
                  'formal_status':'BLOCKED: Moore self-test; A100/model-platform coverage/decisions not satisfied'}
            comparisons.append(item)
            text += ['', 'B0 `'+base['run_id']+'` → `'+target['run_id']+'`：整回合策略区间之和比 '+fmt(item['whole_episode_policy_seconds_speedup'])+'，闭环墙钟比 '+fmt(item['closed_loop_wall_speedup'])+'；posterior均值差 '+fmt(item['posterior_mean_delta'])+'，提交候选前缀均值差 '+fmt(item['committed_prefix_mean_delta'])+'。']
    dump(reports/'B0_comparisons.json',comparisons)
    text += ['', '接受分母为每次实际验证，保留全拒绝0。posterior候选认可与最后七维动作边界内提交不是同一值；两种定义同时保留，D04未签认，不选择更有利的数值作正式通过结论。B0/B1/B2算法之间无需动作相同；B1/B2的系统优化保持要求另行检查。', '', '## 资源及传输','', '各项峰值分别报告，正节省表示测试版更少，负数表示测试版占用更多。不得将reserved与allocated混算，也不把设备总显存冒充进程或模型显存。','', '| B0对照 | allocated峰值节省(bytes) | reserved峰值节省(bytes) | 正式≥2GB判断 |','| --- | ---: | ---: | --- |']
    for x in comparisons:
        text.append('| '+x['tested']+' | '+str(x['allocated_peak_saving_bytes'])+' | '+str(x['reserved_peak_saving_bytes'])+' | 两种实测口径均未达到节省2GB |')
    text += ['', '诊断轮logical copy payload仅用于定位传输，物理DMA、自定义算子内部拷贝及完整性未认证；因此100M/s正式指标保持BLOCKED。诊断的逐次同步和profiler会影响时间，不进入上方完整回合性能表。', '', '## 失败与修复记录','', '- `smoke_optimized_s7`：8步模型执行完毕，但退出时inf配置默认值导致collection.json写出失败；原结果保留并更正为FAIL，唯一重试ID另列。','- 首次新增机制测试有语法错误，修正后保留原日志并重新回归。统计测试随后因加入SpecVLA组而暴露按数组位置选native的夹具错误，改为按profile查找后通过。','- 第一轮完整队列在上一进程结束后的设备空闲检查处停止，尚未创建KERV用例；保存调度日志，增加最多30秒只读释放等待后继续，未终止其他作业。', '', '## 正式指标边界','', '本轮仅为摩尔H1、M1/P1已登记子集的自测。缺A100参考、M2、P2/P3认可及完整矩阵、D01–D08签认、完整传输认证和独立第三方测试；不将本地PASS扩展为课题1.1/1.2/1.3通过。FlagScale目前核实的是本地桥接执行，公开发布和上游接收另需证据。', '', '## 复算与证据','', '```bash','cd /root/KERV','PYTHONPATH=src python3 -m acceptance.aggregate --campaign '+str(c),'PYTHONPATH=src python3 -m acceptance.report --campaign '+str(c),'PYTHONPATH=src python3 -m acceptance.check_evidence --campaign '+str(c)+' --check-source-hashes --check-weight-hashes','```','', '原始命令/退出码见operations与runs/*/run.json；动作、接受事件、内存、采样、配置、录像均在对应run内。evidence_index.csv与checksums.sha256提供交付索引。']
    replay_file=c/'reports/replay_evidence_checks.json'
    if replay_file.exists():
        replay=load(replay_file)
        text += ['', '## 固定输入回放（与闭环分开）', '',
                 '141帧，每配置完整预热一次后计时三轮，共'+str(replay['total_actions_per_profile'])+'个动作。每帧固定观测和采集时历史；不执行仿真，不计算SR，不用于单模型显存对比。', '',
                 '原生/优化动作保持：'+replay['native_optimized_action_preservation']+'；逐项相同记录数：`'+str(replay['equal_records'])+'`。', '',
                 '模型均值比 '+fmt(replay['model_mean_speedup'])+'×，pipeline均值比 '+fmt(replay['pipeline_mean_speedup'])+'×。这是本输入集的预热回放统计，不能消除上文新进程闭环差异。全部423条计时记录保留；没有排除每轮第一条。', '',
                 '逐帧原始记录见replays/fixed_input_v1，独立复算见reports/replay_evidence_checks.json。']
    (c/'summary.md').write_text('\n'.join(text)+'\n')
    (reports/'T11_report.md').write_text('# 技术指标1.1本地自测\n\n正式状态：BLOCKED。真实B0、B1、B2本机结果、两种接受长度及配对限制见 ../summary.md 与 local_results.json。未复现A100原始65秒条件，不以摩尔单任务结果判正式通过。\n')
    (reports/'T12_report.md').write_text('# 技术指标1.2本地自测\n\n正式状态：BLOCKED。实际CPU/MUSA分工见 docs/acceptance/placement.md；双向logical payload与诊断copy时间保留，不等于完整physical DMA。allocated/reserved及设备采样均报告；B0同case显存差见B0_comparisons.json。不在口径未签认或结果未达阈值时声称节省2GB。\n')
    (reports/'T13_integration.md').write_text('# 技术指标1.3摩尔本机集成核查\n\n本轮H1两项核心执行及B1/B2从run_kerv.py经FlagScale/inference_kerv.py启动的记录保存在runs内；实际自定义KERV代码执行，不是vLLM引擎。third_party/FlagScale本地修改和补丁见inventory；核心模型代码仍在src/openvla及runtime_opt，不能由桥接成功推断两个功能整体已进入公开FlagScale仓库。\n\n公开发布：未由本轮执行。上游接收：未证明。其他两款芯片、智源VLA及独立第三方：非本机已验收事项。1.3整体保持BLOCKED。\n')
    semantic=[]
    for x in r['runs']:
        for event in rows(c/'runs'/x['run_id']/'verify.jsonl'):
            normalized=committed_candidate_prefix(event)
            raw=event.get('committed_accepted_prefix_length')
            if raw is not None and raw!=normalized:
                semantic.append({'run_id':x['run_id'],'step_id':event['step_id'],'verify_id':event['verify_id'],
                                 'raw_telemetry_value':raw,'normalized_candidate_prefix':normalized,
                                 'reason':'Initial secondary committed-prefix telemetry included terminal EOS in ordinary update length; recomputed from actual action-budget clamp. Posterior raw acceptance and actions unchanged.'})
    dump(reports/'verification_semantics_audit.json',{'corrected_in_derived_statistics_only':True,'affected_secondary_records':semantic,
         'formula':'ordinary min(posterior_accept, max(0,6-tokens_before)); direct completion min(posterior_accept,7)',
         'raw_files_preserved':True})
    print(c/'summary.md')


if __name__=='__main__':
    main()
