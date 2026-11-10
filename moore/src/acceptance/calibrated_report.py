"""Recompute calibrated W8 pilot precision/storage/runtime evidence."""
import argparse
import csv
from pathlib import Path
from acceptance.aggregate import aggregate
from acceptance.common import dump,load,rows,sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--campaign',type=Path,required=True)
    c=parser.parse_args().campaign;result=aggregate(c);proof=[];comparisons=[]
    policy=load(c/'model_quant/policy.json')
    for run in result['runs']:
        run['variant']=Path(run['policy_path']).parent.name if run['profile']=='calibrated' else run['profile']
        run['model_load_seconds']=next((x['elapsed_since_collector_start_seconds'] for x in rows(c/'runs'/run['run_id']/'phases.jsonl') if x['phase']=='model_loaded'),None)
        if run['profile']!='calibrated' or run['status']!='PASS':continue
        events=rows(c/'runs'/run['run_id']/'calibrated_quantization.jsonl')
        loaded=next(x for x in events if x['phase']=='loaded');final=next(x for x in events if x['phase']=='final')
        run_policy_path=Path(run['policy_path'])
        if not run_policy_path.is_absolute():run_policy_path=Path('/root/KERV')/run_policy_path
        run_policy=load(run_policy_path)
        expected={x['name'] for x in run_policy['modules']};actual={x['name'] for x in loaded['modules']};calls={x['name']:x['forward_calls'] for x in final['modules']}
        ok=(expected==actual==set(calls) and loaded['policy_sha256']==sha256(run_policy_path)
            and all(x>0 for x in calls.values()) and all(x['cpu_gpu_decode_bit_exact'] for x in loaded['modules'])
            and all(x['buffers']['qweight']['dtype']=='torch.int8' and x['buffers']['qweight']['device']=='musa:0'
                    and x['buffers']['protected_values']['dtype']=='torch.bfloat16' for x in loaded['modules']))
        proof.append({'run_id':run['run_id'],'variant':run['variant'],'status':'PASS' if ok else 'FAIL','module_count':len(actual),
            'minimum_forward_calls':min(calls.values()),'protected_groups':sum(x['protected_group_count'] for x in loaded['modules'])})
        if not ok:raise RuntimeError('Calibrated execution evidence failed: '+run['run_id'])
    full=[x for x in result['runs'] if x['kind']=='full']
    for q in [x for x in full if x['profile']=='calibrated' and x['status']=='PASS']:
        for b in [x for x in full if x['profile']=='native' and x['status']=='PASS' and all(x[k]==q[k] for k in ('task_id','init_state_id','seed','repeat_id'))]:
            left,right=[rows(c/'runs'/x['run_id']/'steps.jsonl') for x in [b,q]]
            comparisons.append({'baseline':b['run_id'],'calibrated':q['run_id'],'variant':q['variant'],'baseline_success':b['task_success'],'calibrated_success':q['task_success'],
                'same_initial_observation':left[0]['observation_hash']==right[0]['observation_hash'],
                'allocated_peak_saving_bytes':b['maximum_peak_allocated_bytes']-q['maximum_peak_allocated_bytes'],
                'reserved_peak_saving_bytes':b['maximum_peak_reserved_bytes']-q['maximum_peak_reserved_bytes'],
                'same_step_count':len(left)==len(right),'overlap_token_matches':sum(a['action_tokens']==z['action_tokens'] for a,z in zip(left,right)),
                'all_actions_and_observations_exact':len(left)==len(right) and all(a[k]==z[k] for a,z in zip(left,right) for k in ['observation_hash','action_tokens','raw_action','executed_action']),
                'scope':'Separately evolving trajectories; same work only if all actions/observations match'})
    result['variant_groups']=[]
    for variant in sorted({x['variant'] for x in result['runs']}):
        group=[x for x in result['runs'] if x['kind']=='full' and x['variant']==variant]
        executed=[x for x in group if x['status'] in ('PASS','FAIL')]
        result['variant_groups'].append({'variant':variant,'registered_full_cases':len(group),'executed':len(executed),
            'task_successes':sum(x['task_success'] is True for x in executed),
            'not_started':sum(x['status']=='NOT_STARTED' for x in group),
            'distinct_task_states':len({(x['task_id'],x['init_state_id']) for x in executed}),
            'scope':'registered pilot cases; repeats are not independent task/state coverage'})
    for group in result['groups']:
        if group['profile']=='calibrated':group.update(candidate_quality_usable=False,note='profile pools policies; use variant_groups and per-policy decisions')
    dump(c/'reports/local_results.json',result);dump(c/'reports/calibrated_execution_proof.json',proof);dump(c/'reports/closed_loop_comparisons.json',comparisons)
    columns=['run_id','kind','profile','variant','task_id','init_state_id','seed','status','task_success','steps','model_load_seconds','first_model_seconds',
             'maximum_peak_allocated_bytes','maximum_peak_reserved_bytes','vendor_sampled_device_peak_bytes']
    with (c/'reports/results.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(result['runs'])
    pilot=load(c/'block_pilot_v2/decision.json');action=load(c/'block_pilot_v2/action_replay.json');teacher=load(c/'model_quant/teacher_forcing.json')
    text=['# 摩尔S5000激活校准及GPTQ混合量化试验报告','', '日期：2026-10-26。Campaign：`'+c.name+'`。本机物理GPU0 MTT S5000、既有MUSA容器，FlagScale启动本地KERV；本地新增后端，不声称上游已接收。','',
          '## 本轮完成与边界','',
          '已实现并实际执行：激活校准、GPTQ式误差补偿、投影内BF16保护组、逐层候选/原BF16教师双路径、CPU导出与重新部署、MUSA实际解码/GEMM。量化是有损W8A16；没有将BF16无损存储方案改称量化通过。默认配置保留。','',
          '这是先验证机制的小规模试验。已采集task0/1/2各state3/seed7的前48个动作；采集截断不计SR。校准用三个任务的144帧，诊断选择用历史task0的36帧；后者已在既往研发中观察过，不是最终未见测试。没有执行设计文档中的60/45/300回合完整协议，不能据此证明成功率下降≤1个百分点。','',
          '## 单block对照','',
          '第15层MLP，RTN32输出相对RMSE '+format(pilot['rtn32_relative_output_rmse'],'.8f')+'；选中的 '+pilot['chosen']+' 为 '+format(pilot['chosen_relative_output_rmse'],'.8f')+'，变化 '+format((pilot['relative_rmse_ratio_to_rtn']-1)*100,'.2f')+'%。第一阶段每个C轨迹取24帧，合计72帧；独立于本次C的历史诊断轨迹取36帧。prompt/实际树与decode输入采样记录在sampling.json；此阶段没有另加teacher-forcing行。','',
          '该block替换后36帧固定动作回放：token向量完全相同 '+str(action['exact_action_tokens'])+'/36，raw MAE '+str(action['raw_action_mae'])+'，最大 '+str(action['maximum_raw_action_error'])+'，夹爪相同 '+str(action['gripper_matches'])+'/36。重建误差下降不代表动作精度通过。AWQ alpha=.5组合在该block未优于alpha=0的GPTQ+保护组，因此后续选择实际更好的配置，不强行叠加缩放。','',
          '## 全模型顺序校准','',
          '每个样本使用实际图像预处理、任务提示、BF16视觉embedding及捕获的7维动作token构成teacher-forced多模态上下文。原BF16教师状态与量化候选状态分别逐层推进；下一层统计来自当前候选，MLP内部down_proj使用实际量化gate/up的输入。原BF16教师输出用于诊断及V上的累积误差门槛，未实现额外的全block梯度重构训练。','',
          '每帧采样7个prompt位置及7个动作预测位置的3倍权重，即25/75采样质量；各轨迹和完整来源在模型校准protocol中记录。此全模型pilot没有额外混入验证树分支样本，区别于设计的25/50/25完整采样，也区别于第一阶段原生KERV激活采样。','',
          '前2/后4层、attention、视觉、drafter、embedding、动作头、norm及KV保持原配置。中间第2–27层的MLP候选按group32处理，保护比例约5%并按组向上取整；保护组先冻结，用原BF16值覆盖，GPTQ仅更新未保护列。分组风险采用激活能量加权的误差对角近似，不声称已实现完整Fisher或完整组Hessian风险搜索。','',
          'Hessian在CPU用FP64 Cholesky稳定分解，MUSA用FP32执行误差补偿，计入最终实际BF16解码舍入。alpha='+str(policy['alpha'])+'。最终量化 '+str(policy['module_count'])+' 个投影，静态节省 '+format(policy['estimated_static_saving_bytes']/1e9,'.6f')+' GB；逐层是否回退、局部/累计误差及CPU/GPU校准时间在model_quant/layers.json。','',
          'teacher-forcing诊断：动作预测位置top1相同 '+str(teacher['same_top1'])+'/'+str(teacher['prediction_positions'])+'，mean KL '+str(teacher['mean_KL'])+'，最大logit误差 '+str(teacher['maximum_logit_error'])+'。这是固定前缀的诊断，不是KERV实际动作回放，也不是闭环SR。','',
          '## 完整闭环','', '| Run | task/state | 执行 | 成功 | 动作数 | allocated峰值GB | reserved峰值GB |','| --- | --- | --- | --- | ---: | ---: | ---: |']
    for x in full:
        text.append('| '+x['run_id']+' | '+x['task_id']+'/'+x['init_state_id']+' | '+x['status']+' | '+str(x['task_success'])+' | '+str(x.get('steps','—'))+' | '+(format(x['maximum_peak_allocated_bytes']/1e9,'.6f') if x['maximum_peak_allocated_bytes'] is not None else '—')+' | '+(format(x['maximum_peak_reserved_bytes']/1e9,'.6f') if x['maximum_peak_reserved_bytes'] is not None else '—')+' |')
    text+=['','初始state0失败则按登记规则停止task1/state10、task6/state0的扩展，未执行项保持NOT_STARTED；失败或低成功率不能删除。state0曾出现在诊断数据中，不称其为最终独立任务。执行PASS只表示采集完整，任务成功另列。','',
           '## 显存对照','', '| native → 校准量化 | allocated节省GB | reserved节省GB | 全部动作/观测相同 |','| --- | ---: | ---: | --- |']
    for p in comparisons:text.append('| '+p['baseline']+' → '+p['calibrated']+' | '+format(p['allocated_peak_saving_bytes']/1e9,'.6f')+' | '+format(p['reserved_peak_saving_bytes']/1e9,'.6f')+' | '+str(p['all_actions_and_observations_exact'])+' |')
    text+=['','加载前起采，初始化与首步/稳态峰值均保留；CPU校准后上卡，没有先把完整BF16 verifier上卡再删除。实际显存包含解码临时权重、scale、BF16保护组和运行时工作区。与native基线对照不是SpecVLA B0验收对照；不把不同轨迹的耗时差直接称作同工作量加速。','']
    replay=c/'replays/quantized_fixed_input_v1/results.json'
    if replay.exists():
        r=load(replay);text+=['## 全模型固定输入回放','', '原生BF16与校准量化：同输入/历史 '+str(r['same_observation_hashes'])+' 帧，完整动作/token一致 '+str(r['exact_action_token_matches'])+'/'+str(r['protocol']['frames'])+'，raw MAE '+str(r['raw_action_absolute_error']['mean'])+'、最大 '+str(r['maximum_raw_action_absolute_error'])+'；执行夹爪相同 '+str(r['executed_gripper_matches'])+'。','',
             '每组完整预热141帧，再测141帧，全部帧保留。模型均值BF16/量化比 '+format(r['model_mean_speedup'],'.6f')+'×，pipeline比 '+format(r['pipeline_mean_speedup'],'.6f')+'×；加载、CPU校准及哈希不混入动作计时。两个worker同卡驻留、串行计算，回放显存不参与冷进程单模型峰值表。','']
    refined=c/'model_refined/decision.json'
    if refined.exists():
        rd=load(refined);selected=rd['selected'];rpol=load(c/'model_refined/policy.json')
        text+=['## 动作输出敏感性追加筛选','',
            '初始质量不足时，以同一已见V36帧分别恢复每个候选block为原BF16，按执行夹爪变化、6维动作bin绝对/最大误差、全部top1差异及KL排序，再检查恢复0/1/2/4/6个block的组合。只接受静态节省≥2GB的组合；这是预算内的研发选择，不是独立测试。', '',
            '原始teacher-forcing差异 '+str(rd['base']['wrong_top1_positions'])+'/252，选中后 '+str(selected['wrong_top1_positions'])+'/252；恢复BF16层 '+str(selected['restored_layers'])+'，保留INT8投影 '+str(rpol['module_count'])+'，静态节省 '+format(rpol['estimated_static_saving_bytes']/1e9,'.6f')+' GB。具体组合与所有单层试验均保留。', '',
            'BF16重复参考的top1差异 '+str(rd['bf16_repeat_different_top1_positions'])+'；量化后仍需按新ID完整闭环和同输入回放，不将V选择分数当成SR。']
    refined_replay=c/'replays/calibrated_refined_fixed_input_v1/results.json'
    if refined_replay.exists():
        rr=load(refined_replay)
        text+=['', '## 追加候选的固定回放', '',
             '独立冷进程加载所选policy后，动作/token一致 '+str(rr['exact_action_token_matches'])+'/141，raw MAE '+str(rr['raw_action_absolute_error']['mean'])+'，最大 '+str(rr['maximum_raw_action_absolute_error'])+'，执行夹爪相同 '+str(rr['executed_gripper_matches'])+'/141。模型均值BF16/量化比 '+format(rr['model_mean_speedup'],'.6f')+'×、pipeline比 '+format(rr['pipeline_mean_speedup'],'.6f')+'×；预热和计时口径与初始回放相同。']
    drift=c/'reports/baseline_drift.json'
    if drift.exists():
        dd=load(drift)
        text+=['', '## BF16基线重复性边界', '',
             '本轮native BF16在task0/state0/seed7也未完成，之前相同配置170步成功；首次动作分歧在索引'+str(dd['first_different_step_index'])+'（从0计），当步观测hash相同。对比per-run source manifest，原有推理源码无变化。该现象提示基线重复性问题，当前未定位具体算子或硬件原因，不能把闭环失败单独归因量化，也不能从一次基线失败和一次量化成功推断量化提升SR。失败与后续重复控制分别登记，全部保留。']
    fixed_baseline=c/'reports/fixed_input_baseline_repeatability.json'
    if fixed_baseline.exists():
        fb=load(fixed_baseline)
        text+=['', '## 相同输入的BF16参考重复性', '',
            '两个独立native BF16 worker各自的正式计时141帧，动作/token相同 '+str(fb['cross_process_measured_matches'])+'/141，差异帧 '+str(fb.get('cross_process_different_frames'))+'；实际加载参数哈希完全相同='+str(fb.get('parameter_hashes_equal'))+'。这与闭环轨迹发散分开，是相同输入/固定历史下的参考差异；根因尚未定位。', '',
            '同一worker预热/计时动作相同 '+str(fb['same_process_warmup_measured_matches'])+'/141，但预热额外保存frame77调试张量，观察器开销不同，不能把该比较说成完全相同执行条件的重复实验。正式跨worker计时对照没有这项观察器差异。', '',
            '初始量化与BF16有34帧动作差异，明显多于此BF16参考对照的一帧差异，不能用基线问题将所有量化差异解释掉；同样也不能用单次闭环失败孤立推断量化的套件SR损失。']
    text+=['## 复现与留档','', '专用入口：src/scripts/run_musa_calibrated.sh；必须显式提供容器内的冻结policy.json路径，缺失或hash/shape不匹配即退出。CPU payload与MUSA解码哈希一致性、INT8/BF16实际buffer和真实forward次数见calibrated_quantization.jsonl及reports/calibrated_execution_proof.json。校准phase的双路径内存及耗时均不写成部署收益。','',
           '首次block_pilot因prompt/验证树输入类别识别错误而在样本拼接处退出，未进入量化；修正后用新ID block_pilot_v2运行，原日志与原源码SHA实体保留。未覆盖旧campaign或既有量化失败结果。','',
           '读取model_quant/*.pt为本轮生成的可信本地量化张量，模型原checkpoint只读。精度、性能、显存、数据完整性分别判断；单任务成功不能当作完整LIBERO SR，也不能拿本机结果替代A100、昇腾或其他模型/平台的验收。']
    (c/'summary.md').write_text('\n'.join(text)+'\n');print(c/'summary.md')

if __name__=='__main__':main()
