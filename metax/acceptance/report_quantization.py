"""Render the completed quantized memory/quality pilot and its limitations."""
import argparse,json,math,re,subprocess,sys
from pathlib import Path


def wilson(k,n):
    z=1.959963984540054;p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [center-half,center+half]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign-dir',required=True,type=Path);a=p.parse_args()
    root=a.campaign_dir.resolve()
    subprocess.run([sys.executable,'-m','acceptance.aggregate_quantization','--campaign-dir',str(root)],check=True)
    report=json.loads((root/'reports/quantization_results.json').read_text());runs={x['run_id']:x for x in report['runs']}
    bf=runs['bf16_goal10'];q=runs['int8_middle18_goal10']
    assert bf['execution_status']==q['execution_status']=='PASS'
    assert bf['full_episodes']==q['full_episodes']==10
    comparison=report['comparisons']['middle18_goal10_pilot']
    confidence={'method':'95% Wilson per-profile descriptive binomial interval; does not model heterogeneous tasks or tuning bias',
        'bf16':wilson(bf['successes'],10),'int8_middle18':wilson(q['successes'],10)}
    pairs=list(zip(bf['episode_results'],q['episode_results']))
    heldout=[(b,i) for b,i in pairs if b['task_id']!=0]
    quality={'bf16_successes':bf['successes'],'quant_successes':q['successes'],'episodes_per_profile':10,
        'observed_loss_percentage_points':100*(bf['success_rate']-q['success_rate']),
        'relative_success_rate_loss':comparison['relative_success_rate_loss'],
        'task0_was_used_for_variant_selection':True,
        'remaining_9_tasks':{'bf16_success':sum(b['success'] for b,i in heldout),'int8_success':sum(i['success'] for b,i in heldout),'total':len(heldout)},
        'confidence_intervals':confidence,'formal_less_than_5_percent_loss_proven':False}
    quality['bf16_only_success_tasks']=[b['task_id'] for b,i in pairs if b['success'] and not i['success']]
    quality['int8_only_success_tasks']=[i['task_id'] for b,i in pairs if i['success'] and not b['success']]
    task_names=re.findall(r'^Task: (.+)$',(root/'runs/bf16_goal10/stdout.log').read_text(),re.MULTILINE)
    assert len(task_names)==10
    (root/'reports/quality_scope.json').write_text(json.dumps(quality,indent=2)+'\n')
    text=['# KERV 昇腾 INT8 显存优化验证报告','',
        '日期：2026-10-26。范围：当前昇腾，OpenVLA + SpecVLA Goal drafter，经FlagScale运行的量化资源与质量试验。',
        '', '## 结论','',
        f"- 中间18层MLP量化方案，在本轮10任务全程allocated峰值上节省 **{comparison['allocated_peak_saving_GB']:.3f} GB（{comparison['allocated_peak_saving_GiB']:.3f} GiB）**。包含模型加载和首次执行，未reset峰值。",
        f"- 10个Goal任务、各首个官方初始状态、seed7：BF16 **{bf['successes']}/10**，INT8 **{q['successes']}/10**。观察到的成功率变化为{comparison['success_rate_change_percentage_points']:+.1f}个百分点。",
        '- 项目已明确允许使用量化；本轮资源收益可以作为量化方案的实测结果。它比较KERV优化版BF16/INT8，不冒充真正Spec-VLA B0的验收对照。',
        '- 量化会改变动作和完成步数；质量门槛为有效动作和任务结果，未要求逐token等同BF16。小规模试验不能证明完整评测下成功率损失严格小于5%。',
        '', '## 方法与变更','',
        '- 仅转换verifier Transformer零起始第7至24层的gate/up/down投影，共54个Linear；其余层MLP、attention、vision、drafter、embedding、head仍为BF16。',
        '- 权重INT8、激活BF16；每输出通道、K方向group128，对称量化，scale为BF16。',
        '- 使用已安装torch_npu的npu_weight_quant_batchmatmul；不需要CUDA TorchAO。',
        '- 在CPU加载阶段完成量化，再将模型送到NPU；原checkpoint只读且逐文件哈希与此前一致。没有先让完整BF16模型驻留NPU再量化，也没有保留MLP的BF16影子权重。',
        '- 保留KERV原树、全部路径、KF、动态阈值、seven动作协议和已有工作区/KV/批量提交优化；默认入口仍保留BF16参考。',
        '', '## 单任务调优与候选选择','',
        '| 配置 | allocated峰值GB | 任务结果 | 动作数 | 单步模型均值ms |',
        '|---|---:|---|---:|---:|']
    for name,label in [('bf16_reference','BF16参考'),('int8_candidate','全32层MLP INT8'),('int8_middle18','中间18层MLP INT8')]:
        r=runs[name];ep=r['episode_results'][0]
        text.append(f"| {label} | {r['allocated_peak_bytes']/1e9:.3f} | {'成功' if ep['success'] else '失败'} | {ep['steps']} | {r['model_seconds']['mean']*1000:.3f} |")
    text += ['',
        '全32层与BF16单任务在物理chip4串行；中间18层单任务在chip6，不能将跨芯片与不同轨迹的均值当作独立kernel速度收益。',
        '全32层方案省4.155GB，但动作数163；中间18层省2.340GB且动作数122，相对BF16的115动作更接近，因此在看到量化Goal10结果前选定中间18层。原候选记录保留，没有删除失败或较慢结果。',
        '', '## 10任务质量试验','',
        '| Task ID / 任务 | BF16成功 | BF16动作数 | INT8成功 | INT8动作数 |',
        '|---|---|---:|---|---:|']
    for b,i in pairs:
        text.append(f"| {b['task_id']} / {task_names[b['task_id']]} | {'是' if b['success'] else '否'} | {b['steps']} | {'是' if i['success'] else '否'} | {i['steps']} |")
    text += ['',
        f"task0参与过候选选择。其余9任务：BF16 {quality['remaining_9_tasks']['bf16_success']}/9，INT8 {quality['remaining_9_tasks']['int8_success']}/9。",
        f"BF16独有成功任务{quality['bf16_only_success_tasks']}，INT8独有成功任务{quality['int8_only_success_tasks']}；总成功率相同不表示逐任务行为相同。",
        f"95% Wilson描述区间：BF16 [{confidence['bf16'][0]:.3f},{confidence['bf16'][1]:.3f}]，INT8 [{confidence['int8_middle18'][0]:.3f},{confidence['int8_middle18'][1]:.3f}]；没有据此证明严格5%损失边界。",
        f"相对成功率损失（BF16为分母）：{quality['relative_success_rate_loss'] if quality['relative_success_rate_loss'] is not None else '未定义'}；百分点损失另见quality_scope.json。",
        '所有超步数与策略失败均保留在10回合分母；不是300回合、多种子、多初始状态的正式评测。',
        '', '## 10任务全程显存','',
        '| 指标 | BF16 bytes | INT8 bytes | BF16−INT8 bytes |',
        '|---|---:|---:|---:|',
        f"| allocated峰值 | {bf['allocated_peak_bytes']:,} | {q['allocated_peak_bytes']:,} | {comparison['allocated_peak_saving_bytes']:,} |",
        f"| reserved峰值 | {bf['reserved_peak_bytes']:,} | {q['reserved_peak_bytes']:,} | {comparison['reserved_peak_saving_bytes']:,} |",
        '',
        f"达到十进制2GB：{comparison['saving_at_least_2GB']}；达到2GiB：{comparison['saving_at_least_2GiB']}。两种单位同时保留，D06正式口径仍单列。",
        'BF16 Goal10在chip4，INT8 Goal10在chip6；独立进程、相同运行栈和用例。完整同芯片的全32层量化对照另有记录。框架峰值约100ms轮询，记录框架全程峰值计数；vendor设备采样约10秒，不能当作100ms进程峰值。',
        'allocated和reserved分别报告，不将保留池变化冒充活跃张量节省。CPU量化冷启动增加的时间包含在load_seconds；未把它藏入稳态内存口径。',
        '', '## 时间与执行证据','',
        f"Goal10观察到的单步模型均值：BF16 {bf['model_seconds']['mean']*1000:.3f}ms，INT8 {q['model_seconds']['mean']*1000:.3f}ms。跨芯片且轨迹不同，该值只描述本轮成本，不声称量化额外加速。",
        f"INT8真实调用{q['quantization']['forward_calls']:,}次，转换模块{q['quantization']['module_count']}个，静态模型存储节省{q['quantization']['unique_storage_saving_bytes']:,}bytes。",
        '真实MLP形状单元检查覆盖M=8/279、K/N=4096/11008；输出有限、BF16，权重/scale存储检查通过；相对原BF16 Linear的L2误差约0.41%，该误差不能当作任务成功率损失。',
        '', '## 重现入口','', '```bash', 'cd /root/KERV',
        'KERV_NPU_ID=4 KERV_NPU_QUANT=mlp_int8 KERV_NPU_QUANT_LAYERS=7:25 \\',
        '  KERV_NPU_QUANT_GROUP_SIZE=128 scripts/run_kerv_moore_ascend.sh \\',
        '  --output-dir outputs/quant_middle18_reproduce --max-tasks 1 --trials 1 \\',
        '  --max-episode-steps 220', '```','',
        '运行前确认目标物理chip空闲，输出目录使用新名字。设KERV_NPU_QUANT=none恢复BF16。不要与旧的late TorchAO转换开关叠加。',
        '', '## 产物与范围','',
        '- reports/quantization_results.json/csv：从原始数据复算的资源、时间和任务结果。',
        '- reports/quality_scope.json：质量比较、调优样本和不确定性说明。',
        '- runs/*/metrics：逐步动作、token、验证事件、显存、加载时间和最终quantization.json命中。',
        '- inventory：变更前源码、同步哈希、原checkpoint哈希、环境、实际kernel schema和真实形状测试。',
        '- cases*.json、plan.md、selection.md：运行前登记及候选选择依据。',
        '- source code: ports/moore_ascend/openvla/experiments/robot/ascend_quantization.py。',
        '', '本轮没有证明完整课题验收，也没有把量化资源收益当作原BF16无损运行时收益；真实Spec-VLA B0、其他模型/平台和正式成功率协议仍需补齐。']
    (root/'summary.md').write_text('\n'.join(text)+'\n')
    print('Report:',root/'summary.md')

if __name__=='__main__':main()
