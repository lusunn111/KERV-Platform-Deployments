"""Create the follow-up calibrated runtime report from retained evidence."""
import argparse
from pathlib import Path
import re
from acceptance.common import load,rows,dump

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--root',type=Path,default=Path('/root/KERV'));a=p.parse_args();c=a.campaign;r=a.root
    experiment=c/'replays/same_policy_tree_v1';check=load(experiment/'evidence_checks.json');assert check['status']=='PASS'
    replay=load(experiment/'results.json');profile=load(c/'diagnostics/profile_row_backend/profile.json')
    tree=load(c/'diagnostics/tree_same_input_v3/tree_shadow_checks.json')
    assert tree['all_buffers_bit_exact'] and tree['calls']
    prefetch=load(c/'prefetch_rotation75/results.json')
    smoke={}
    for name in ['rows','tree_rows']:
        out=c/'runs'/('smoke_'+name+'_s7');run=load(out/'run.json');assert run['status']=='PASS'
        mem=rows(out/'memory.jsonl');steps=rows(out/'steps.jsonl');assert len(steps)==8
        smoke[name]={'execution':run['status'],'steps':len(steps),
            'peak_allocated_bytes':max(x['peak_allocated_bytes'] or 0 for x in mem),
            'peak_reserved_bytes':max(x['peak_reserved_bytes'] or 0 for x in mem),
            'tokens':[x['action_tokens'] for x in steps],'observations':[x['observation_hash'] for x in steps]}
    smoke['exact_tokens']=sum(x==y for x,y in zip(smoke['rows']['tokens'],smoke['tree_rows']['tokens']))
    smoke['exact_observations']=sum(x==y for x,y in zip(smoke['rows']['observations'],smoke['tree_rows']['observations']))
    audit=rows(c/'runs/smoke_tree_rows_s7/calibrated_tree_runtime.jsonl')[0]
    assert audit['tree_builder_hits']>0 and not audit['kalman_batch_enabled'] and not audit['static_tree_runtime']
    tests=(c/'operations/regression_suite/stdout_stderr.log').read_text();assert load(c/'operations/regression_suite/command.json')['exit_code']==0
    total=int(re.search(r'Ran (\d+) tests',tests)[1]);m=re.search(r'OK \(skipped=(\d+)\)',tests);skipped=int(m[1]) if m else 0
    speed=check['pooled_speedup'];model_speed=speed['model_seconds'];pipe_speed=speed['pipeline_seconds']
    stable=all(x['model_speedup']>1 for x in replay['rounds'])
    repeatability={}
    for label in ['quant_reference','quant_kernel']:
        first=rows(experiment/label/'measure_0.jsonl')
        repeatability[label]={str(i):[x['frame'] for x,y in zip(first,rows(experiment/label/('measure_%d.jsonl'%i)))
                                     if x['action_tokens']!=y['action_tokens']] for i in [1,2]}
    cross={str(i):[x['frame'] for x,y in zip(rows(experiment/'quant_reference'/('measure_%d.jsonl'%i)),
                                             rows(experiment/'quant_kernel'/('measure_%d.jsonl'%i)))
                   if x['action_tokens']!=y['action_tokens']] for i in [0,1,2]}
    dump(c/'reports/repeatability_detail.json',{'within_profile_vs_round0':repeatability,'cross_profile_frames':cross,
         'scope':'actual measured differences retained; does not locate numerical root cause'})
    result={'campaign':c.name,'same_policy_replay':check,'cold_smoke':smoke,'profile_scope':profile['scope'],
        'profile_decode_stream_elapsed_ms':profile['decode_sum_ms'],'profile_linear_stream_elapsed_ms':profile['linear_sum_ms'],
        'tree_same_input_calls':len(tree['calls']),'tree_buffers_bit_exact':True,
        'prefetch_rotation':prefetch,'tests':{'total':total,'passed':total-skipped,'skipped':skipped},
        'repeatability_detail':{'within_profile_vs_round0':repeatability,'cross_profile_frames':cross},
        'performance_status':'BENEFIT_IN_THREE_ROUND_REPLAY' if stable else 'NO_STABLE_BENEFIT',
        'policy_quality_status':'NOT_PASSED; policy unchanged, no new SR claim','default_bf16_changed':False}
    dump(c/'reports/local_results.json',result)
    lines=['# KERV 摩尔 S5000 校准量化推理链路继续优化报告','',f'日期：2026-10-27。Campaign：`{c.name}`。',
        '', '## 实测结果',
        '', '本轮先剖析实际动作，再将已有FlagOS/KERV批量树缓冲路径接入校准量化入口。量化权重及保护组不变，继续使用上一轮row解码算子和原生BF16 GEMM。树不剪枝、不重排，Kalman预测继续逐分支执行原标量函数。',
        '',f'- 相对上一轮row量化后端，三轮141帧合并模型倍率 **{model_speed:.4f}×**，平均耗时变化 **{(1/model_speed-1)*100:+.3f}%**；pipeline倍率 **{pipe_speed:.4f}×**，耗时变化 **{(1/pipe_speed-1)*100:+.3f}%**。这里的参考不是BF16 native，也不是SpecVLA B0。',
        f'- 三轮共 **{check["total_frames"]}** 条动作，token完全一致 **{check["exact_actions"]}/{check["total_frames"]}**，raw动作完全一致 **{check["exact_raw_actions"]}/{check["total_frames"]}**，执行夹爪一致 **{check["gripper_matches"]}/{check["total_frames"]}**。独立worker差异及各自重复性均保留。',
        f'- 实际同输入诊断 **{len(tree["calls"])} 次树扩展**，tokens、候选路径、mask及position ids四类tensor全部逐位一致；75个量化投影解码SHA全部匹配同一冻结policy。',
        f'- 树优化实际命中 **{check["tree_builder_hits"]} 次**（包含预热）；完整回归 **{total}项：{total-skipped}通过、{skipped}跳过**。',
        '- 双槽提前解码已实现，正确性测试通过，但本机75投影轮换基准未提速，未作为主性能方案。',
        '- 原量化策略对BF16的精度验收仍为 **NOT_PASSED**。本轮没有改量化机制，也没有重跑完整任务成功率矩阵。默认BF16配置保留。',
        '', '## 环境、模型和依据',
        '', '物理GPU0 MTT S5000 80GiB，MUSA驱动3.3.5-server；torch/torch_musa2.9.1、Triton MUSA3.2、Transformers4.40.1。沿用kerv-musa-runtime容器及FlagScale启动桥接，未升级依赖或驱动。使用现有OpenVLA LIBERO-Goal verifier和SpecVLA drafter。实际源码含本地适配，不能只用上游commit替代本地源码归档。',
        '', 'policy为此前封存campaign的model_refined/policy.json，75个INT8/BF16混合投影、group32及保护组均不变。模型参数和payload哈希对应原文件；没有重新校准或训练。上一轮报告的0.998×属于不同实验，保持原样，不将多个实验倍率相乘形成新指标。',
        '', '## 从完整推理剖析确定开销',
        '', f'对已预热真实动作frame2记录cProfile及MUSA事件。量化Linear的解码stream elapsed合计 **{profile["decode_sum_ms"]:.3f}ms**，解码加原生GEMM合计 **{profile["linear_sum_ms"]:.3f}ms**。cProfile中候选树扩展约155ms、其中逐节点重建缓冲约93ms，总predict约397ms。',
        '', '这些数值来自单帧插桩诊断，事件、同步及cProfile会改变执行开销；stream elapsed也可能包含主机发射间隙。它们用来选择优化方向，不当作141帧平均阶段耗时，也不将嵌套函数累计耗时相加。',
        '', '原路径在设备端逐节点设置祖先mask、读取深度和token，并反复拼接新分支。新路径一次把必要的小整数数据送到CPU，按相同顺序生成前缀及树结构，再汇总传回设备，减少大量小GPU发射和主机同步。',
        '', '## 实际改动与数学行为',
        '', '新增run_calibrated_tree_libero.py及run_musa_calibrated_tree.sh，复用已有_augment_tree_with_kf_branches_batched和_rebuild_tree_buffers_batched，通过配置显式启用。新增真实命中审计、同输入树对照、MUSA事件剖析、两槽解码管线及回归。',
        '', '关键配置为tree_builder.enabled=true、static_runtime=false、kalman_batch=false。原native树函数本来逐分支调用标量Kalman，虽然其批量开关可能显示true，也不会在原native分支触发批量计算；新入口明确false以继续该数值路径。没有开启C++静态树、节点裁剪、Linear融合、attention/RoPE替换或Graph。',
        '', '现有通用tree_builder manifest会按verify_max_paths=48将semantic_mode标成profile_pruned，但本轮static_runtime=false，不进入消费该限制的静态规划分支。该标签不能替代实际执行判断；本轮以源码分支、28次四buffer对照及真实命中审计确认未剪枝，保留原manifest不重写。',
        '', '这次收益来自将兼容的既有FlagOS控制流优化接到量化部署路径，不把它写成新INT8矩阵乘或上游已合并的新功能。实际乘法仍为BF16，INT8用于权重存储。',
        '', '## 提前解码候选及筛选',
        '', '另实现OrderedWeightPrefetch：两块BF16槽位循环使用，解码在侧流，原生GEMM留在计算流；ready/free事件约束消费者读取和槽位复用。75投影槽位容量180355072bytes，没有整模型BF16常驻展开。160次流复用压力测试以及75个真实权重的同输入数值检查通过；完整模型部署门禁未完成，不推荐此候选。',
        '', '| 行数M | row解码75投影总ms中位数 | 双槽prefetch总ms中位数 | 倍率 |',
        '| --- | ---: | ---: | ---: |']
    for item in prefetch['cases']:
        lines.append(f'| {item["rows"]} | {item["median_ms"]["rows"]:.4f} | {item["median_ms"]["prefetch"]:.4f} | {item["speedup"]:.4f}× |')
    lines += ['', '该基准使用全部真实权重和固定seed合成激活，按75投影顺序轮换；不是完整Transformer。没有测得收益，因此未把侧流方案叠入主候选，也没有将其额外显存写进主方案的资源指标。',
        '', '## 三轮完整固定输入回放',
        '', '每个worker先完整预热141帧，再按AB/BA/AB串行计时。每帧恢复相同原始观测、raw动作历史和Kalman token前缀，框架seed17。两进程的原生参数哈希、冻结policy、75个实际解码权重、观测哈希逐项核验。profiler、索引guard和运行时逐步日志不进入计时。',
        '', '两个模型同时驻留但不同时计算，回放显存不用于单模型冷峰值。预热frame77有调试tensor导出；三轮指标只用无此插桩的measured对measured。',
        '', '动作一致数按整帧7维token/动作向量全部相同计数，不是逐token命中数。每轮raw MAE对141×7个输出维度求绝对差平均，最大差异为其中最大值；直接使用checkpoint动作输出，不额外换算毫米/弧度。夹爪按二值化并翻转符号后的执行值比较。',
        '', '| 轮次 | 参考模型均值ms | 新模型均值ms | 模型倍率 | pipeline倍率 | 动作一致 | raw MAE | raw最大差异 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for item in replay['rounds']:
        t=item['times'];lines.append(f'| {item["round"]+1} | {t["quant_reference"]["model_seconds"]["mean"]*1000:.3f} | {t["quant_kernel"]["model_seconds"]["mean"]*1000:.3f} | {item["model_speedup"]:.4f}× | {item["pipeline_speedup"]:.4f}× | {item["exact_action_tokens"]}/141 | {item["raw_action_error"]["mean"]:.9g} | {item["max_abs"]:.9g} |')
    lines += ['', f'合并倍率使用全423帧总耗时之比，非逐帧倍率的平均。各自第0轮对后续轮的token一致数：`{check["within_profile_round0_token_matches"]}`。这些重复性差异不从结果中剔除，也不能据此把所有跨worker差异归为量化优化造成或全部归为基线。',
        '', f'具体分歧帧：同profile对第0轮为`{repeatability}`；跨profile为`{cross}`。逐帧原始动作及误差完整保留在回放文件中。',
        '', '## 冷启动部署与资源',
        '', '同task0/state0/seed7，规划时域220，各新进程执行8个策略动作。用于检查真实部署、早期显存和命中，不能计算完整SR或当作220步峰值。',
        '', '| 配置 | 执行状态 | 步数 | allocated峰值bytes | reserved峰值bytes |',
        '| --- | --- | ---: | ---: | ---: |']
    for name in ['rows','tree_rows']:
        s=smoke[name];lines.append(f'| {name} | {s["execution"]} | {s["steps"]} | {s["peak_allocated_bytes"]} | {s["peak_reserved_bytes"]} |')
    lines += ['', f'冷启动前8条动作一致{smoke["exact_tokens"]}/8，观测一致{smoke["exact_observations"]}/8。主候选没有BF16预取池，只保留row解码的60000bytes查询表；静态量化权重节省预算与上一轮相同。原报告中相对BF16完整回合的GB节省属于历史测量，未用本轮8步冒烟替代。',
        '', '## 异常、覆盖修正与保留风险',
        '', '首个真实树诊断的挂钩只修改modeling_speculation的重导出符号，没有覆盖utils.initialize_tree实际使用的全局函数，因此v1/v2的树检查次数为0，不作为通过证据。v3修正定义模块及重导出两个位置，才获得上列28次真实树对照。所有版本结果与失败保留。',
        '', 'v1在原row路径（尚未命中新树路径）完成预热后，触发MUSA IndexSelectVectorKernel非法地址。dump已保留；只终止本次自有进程，没有重置GPU或驱动。带Embedding索引检查的后续诊断通过，但同步检查会改变调度，不能宣称已修复底层问题，根因仍未定位。计时回放不带该guard。',
        '', '局部树缓冲和权重逐位一致并不保证独立worker逐动作完全一致。旧校准policy相对BF16曾为111/141动作一致、任务未完成；本轮不重新判为低精度损失方案，也未重跑完整质量矩阵。未验证跨芯片、跨平台或新的真实机器人任务。',
        '', '141帧来自已多次用于研发的既有capture，前8帧又用于本轮问题定位；不把它们称作新的未见质量测试集。',
        '', '## 文件与复现',
        '', '- [操作指南](../../src/docs/MUSA_QUANTIZED_PIPELINE.md)，主脚本`src/scripts/run_musa_calibrated_tree.sh`。',
        '- `diagnostics/profile_row_backend/`：cProfile、MUSA事件、原始动作；`tree_same_input/`、`tree_same_input_v2/`、`tree_same_input_v3/`：覆盖修正和异常证据。',
        '- `replays/same_policy_tree_v1/`：三轮逐帧结果、命令、源码清单、SHA审计及独立复算。',
        '- `runs/smoke_*_s7/`：冷进程动作、显存、录像和实际命中。',
        '- `prefetch_rotation75/`与`operations/`：包括未选用的候选、失败及全部测试日志。',
        '- `reports/local_results.json`为机器可读结果；最终源码归档、依赖清单、交付manifest及checksums另存campaign中。',
        '', '旧campaign保持封存；跨机器交付须同时携带原checkpoint与75个量化payload，源码归档不包含这些大权重。证据完整性PASS表示能复算，不表示完整任务书或量化质量验收PASS。','']
    text='\n'.join(lines);report=r/'docs/acceptance/MOORE_QUANTIZED_PIPELINE_REPORT_20261027.md';report.write_text(text)
    (c/'summary.md').write_text(text);print(report)

if __name__=='__main__':main()
