"""Generate the precision-first report from retained screening and cold runs."""
import argparse
from pathlib import Path
from acceptance.common import load,rows,dump


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True)
    a=p.parse_args();c=a.campaign;screen=load(c/'reports/screening_checks.json')
    assert screen['status']=='PASS' and load(c/'reports/dependencies_checked.json')['status']=='PASS'
    closed=[load(c/'reports'/f'closedloop_task{t}_state{s}.json') for t,s in [(0,0),(1,10)]]
    assert all(x['precision_parity']=='MATCH' for x in closed)
    assert load(c/'operations/precision_storage_tests_v2/command.json')['exit_code']==0
    names=[list(x['runs']) for x in closed]
    latency={}
    for i,label in enumerate(['bf16','exact_mlp']):
        samples=[row['model_seconds'] for item in closed for row in rows(c/'runs'/list(item['runs'])[i]/'steps.jsonl')[1:]]
        latency[label]={'frames':len(samples),'mean_ms':sum(samples)/len(samples)*1000}
    total=sum(x['common_steps'] for x in closed)
    summary={'campaign':c.name,'selected':'exact_mlp/deployment_policy.json',
             'mechanism':'Lossless BF16 storage for96 MLP projections, original BF16 computation; zero INT8 modules',
             'weight_error':'zero additional BF16 weight error; all96 decoded SHA match original',
             'screening':screen,'closed_loop_pairs':closed,'closed_loop_exact_actions':total,
             'closed_loop_exact_observations':total,'closed_loop_raw_max_abs':max(x['max_abs'] for x in closed),
             'local_task_success':{'bf16':sum(x['task_success'][0] for x in closed),'exact_mlp':sum(x['task_success'][1] for x in closed),'cases_per_profile':len(closed)},
             'steady_model_latency':latency,'steady_latency_change_percent':(latency['exact_mlp']['mean_ms']/latency['bf16']['mean_ms']-1)*100,
             'tests_passed':5,'scope':'Two local task/state/seed pairs, not full LIBERO SR or cross-chip acceptance',
             'installed_stack_changed':False,'default_entry_changed':False}
    dump(c/'reports/local_results.json',summary)
    lines=['# 摩尔 S5000 精度优先存储配置实测报告','',f'日期：2026-10-27。Campaign：`{c.name}`。',
           '', '## 结论',
           '', '本轮将推荐候选改为 **96 个 verifier MLP 投影的无损 BF16 存储压缩**，attention 等其余投影保持原生 BF16。实际推理使用原生 BF16 GEMM。主体12-bit编码加指数基值和BF16异常值可以精确还原原始16-bit权重，不属于有损W12量化，也不使用INT8计算。',
           '', f'- 两对独立冷启动 LIBERO 闭环中，BF16 和候选各成功 **{summary["local_task_success"]["bf16"]}/2、{summary["local_task_success"]["exact_mlp"]}/2**；共 **{total}/{total}** 条动作/token、raw数值及观测全部一致，raw最大误差为0。',
           '- 推荐候选每次加载均核验96个压缩投影的原始/实际MUSA解码SHA一致；不存在整模型BF16展开缓存。',
           '- 推荐候选静态权重节省 **2,122,448,562 bytes**。冷进程allocated/reserved实际峰值差见下表，不能用静态预算替代。',
           f'- 代价：两对相同轨迹、各排除首步后的模型平均耗时 **{latency["bf16"]["mean_ms"]:.3f}→{latency["exact_mlp"]["mean_ms"]:.3f} ms**，增加 **{summary["steady_latency_change_percent"]:.2f}%**。本轮解决精度偏差，未取得相对BF16的速度收益。',
           '- 新增针对性测试5项通过，checkpoint依赖11个文件重新SHA校验通过。两例成功不代表完整LIBERO套件SR。',
           '', '## 同一BF16参考下的候选筛选',
           '', '先在同一resident模型中保持原始BF16权重，并准备既有INT8模块与无损BF16压缩模块；切换候选后，各自完整预热141帧，再测141帧。每帧恢复同一捕获观测、raw action历史和Kalman token历史。所有模式使用相同guarded Softmax、host tree builder、共享原标量Kalman历史和原生BF16 GEMM。',
           '', '| 候选 | INT8/无损压缩投影数 | 动作一致 | raw MAE | 最大误差 | 静态节省 GB | 模型均值 ms |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in screen['results']:
        lines.append(f'| {r["mode"]} | {r["int8_modules"]}/{r["exact_modules"]} | {r["exact_actions"]}/141 | {r["raw_MAE"]:.9g} | {r["raw_max_abs"]:.9g} | {r["static_saving_bytes"]/1e9:.6f} | {r["model_mean_ms"]:.3f} |')
    lines += ['', '原BF16参考的预热/测量动作一致141/141。上述141帧已多次用于研发，不能称作未见测试；按候选固定顺序测量，没有用这一轮筛选对照宣称稳定性能倍率。多份权重同卡驻留，筛选显存不能作为部署峰值。',
              '', '8/4投影候选仅保留历史敏感性排序较低层中的up_proj，其他可压缩投影使用无损BF16。它们复用已有顺序校准payload，没有针对新稀疏布局重新校准，因此只能评价这两个具体候选，不能据此断言所有少量INT8方案都无效。两者仍只有117/141动作一致，未推荐。',
              '', '全192投影无损候选达到141/141一致，但约327ms，BF16参考约234ms。为减小解码开销，随后追加MLP-only配置：只压缩96个大投影，attention保持原生。它的96个解码权重全部与全无损候选及BF16参考一致，并以独立冷启动的完整闭环验证。不要把全192投影的141/141直接标成MLP-only单独完成的回放。',
              '', '## 冷启动完整闭环与显存',
              '', '每个run独立进程，GPU0串行执行；加载前开始采集显存，保留首步和整个回合峰值。task0/state0/seed7是历史已见场景；通过后才扩展task1/state10/seed7，该组合未参与本轮141帧筛选。所有case最多220个策略动作，初始化稳定化仿真步数另计。',
              '', '| task/state | BF16/候选成功 | 两端动作数 | 动作/观测一致 | allocated节省 GB | reserved节省 GB |',
              '| --- | --- | ---: | ---: | ---: | ---: |']
    for r in closed:
        lines.append(f'| {r["task_id"]}/{r["state_id"]} | {r["task_success"][0]}/{r["task_success"][1]} | {r["steps"][0]}/{r["steps"][1]} | {r["exact_actions"]}/{r["exact_observations"]} | {r["allocated_peak_saving_bytes"]/1e9:.6f} | {r["reserved_peak_saving_bytes"]/1e9:.6f} |')
    lines += ['', '| task/state | 实现 | allocated峰值 bytes | reserved峰值 bytes | 排除首步模型均值 ms |',
              '| --- | --- | ---: | ---: | ---: |']
    for r in closed:
        for name,d in r['runs'].items():
            lines.append(f'| {r["task_id"]}/{r["state_id"]} | {name} | {d["peak_allocated_bytes"]} | {d["peak_reserved_bytes"]} | {d["model_seconds_excluding_first"]["mean"]*1000:.3f} |')
    lines += ['', 'GB=10⁹ bytes；GiB=2³⁰ bytes。所有原始首步耗时保留在steps.jsonl，上表稳态均值按每个case排除第一个策略动作计算，合并均值按其余帧数加权。动作、观测完全一致使两端工作量可比较；没有将之前量化内部1.1669×与本轮结果相乘。',
              '', '## 实现、验证及失败保留',
              '', '新增run_precision_libero.py与run_musa_precision.sh，支持可选INT8子集加指定组的无损BF16压缩；推荐policy没有INT8条目。模型先在CPU完成编码再迁移MUSA，避免先把完整BF16 verifier上卡再压缩。每个投影使用时临时解码，原生GEMM消费BF16张量；attention、vision、drafter、head及KV精度不变。',
              '', 'lossless_bf16.compress_model新增显式groups选择，默认仍处理原有MLP/attention范围；编码类和GPU解码kernel与筛选阶段完全相同，AST与文件字节比较见inventory/codec_identity.json。推荐入口loaded审计96个权重SHA，final审计真实forward、Kalman与32个注意力Softmax调用。',
              '', '精度参考是本轮实际加载的原始BF16权重。筛选检查器将192个CPU/设备解码SHA与BF16模型的实际参数SHA匹配；冷进程再核验96个MLP解码SHA与该参考。INT8候选只核验其冻结payload解码，不将其称作原BF16一致。独立CPU检查器从逐帧JSONL重新计算动作/观测一致数、raw误差和内存差。',
              '', '首次MLP-only冷启动因Hydra逗号字符串缺少内层引号，在模型执行前失败；原full_exact_mlp_task0_state0_s7保留为FAIL，唯一重试_r1成功。该次没有启动episode，作为部署失败单列，不伪装为任务成功。inventory/cases.csv为最初全压缩队列，addendum_01切换候选到MLP-only；重复列出的基线ID引用同一case，不重复计数，初始未执行全压缩队列不计SR。实际执行次数按runs中唯一run ID核对。',
              '', '5项测试覆盖exact-only、INT8/exact混合、MLP-only选择、损坏payload和不完整policy。precision_storage.jsonl、precision_reference.jsonl和collection.json保留加载/退出与真实调用证据；每个成功完整run包含动作、runtime、内存与视频。原始checkpoint只读，11个依赖文件重新SHA检查通过。',
              '', '## 环境与适用边界',
              '', '单张物理GPU0 MTT S5000 80GiB，kerv-musa-runtime容器，驱动3.3.5-server，torch/torch_musa2.9.1、Triton MUSA3.2、Transformers4.40.1。沿用FlagScale启动及本地FlagOS/KERV适配，使用现有OpenVLA LIBERO-Goal verifier和SpecVLA drafter。没有升级SDK、muDNN、框架，没有修改原始checkpoint或重新训练。',
              '', '本轮推荐的是零新增权重误差的BF16存储方案，不能称为INT8质量验收通过，也不能称为提高了模型相对原BF16的任务能力。旧75投影有损策略仍未通过质量门禁。两个本地task/state/seed对照验证不覆盖完整任务成功率矩阵、真实机器人或跨芯片验收；没有证明套件SR下降≤1个百分点。默认BF16入口保留，推荐方案需通过显式入口选择。',
              '', '## 操作与证据',
              '', '- [部署、policy字段及复核命令](../../src/docs/MUSA_PRECISION_FIRST.md)。',
              '- 选择结果：inventory/selection.json；推荐policy：exact_mlp/deployment_policy.json。',
              '- 同输入筛选：screening/及reports/screening_checks.json；真实闭环：runs/及reports/closedloop_*.json。',
              '- 原始命令与失败日志：operations/；源码版本：各run/source_manifest.sha256与source_screening/。',
              '- 最终源码：inventory/source_final.tar.gz；依赖SHA：inventory/weight_manifest.json；交付校验：checksums.sha256。',
              '', '完整性校验通过表示该批记录可以复算，与正式任务书全部验收通过分开。']
    text='\n'.join(lines)+'\n';target=Path('/root/KERV/docs/acceptance/MOORE_PRECISION_REFINEMENT_20261027.md');target.write_text(text)
    (c/'summary.md').write_text(text.replace('../../src/','../../../src/'))
    print({'report':str(target),'exact_actions':total,'local_task_success':summary['local_task_success'],'latency':latency})


if __name__=='__main__':main()
