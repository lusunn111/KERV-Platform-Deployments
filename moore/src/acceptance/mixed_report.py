"""Recompute mixed policy execution, memory and small-subset quality evidence."""
import argparse
import csv
from pathlib import Path
from acceptance.aggregate import aggregate
from acceptance.common import dump, load, rows, sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    args=parser.parse_args();c=args.campaign;root=Path('/root/KERV')
    result=aggregate(c);proof=[];comparisons=[]
    for item in result['runs']:
        item['model_load_seconds']=next((x['elapsed_since_collector_start_seconds'] for x in rows(c/'runs'/item['run_id']/'phases.jsonl') if x['phase']=='model_loaded'),None)
        item['variant'] = (Path(item['policy_path']).parent.name if item['profile']=='mixed' else item['profile'])
        if item['profile'].startswith('lossless_bf16_12') and item['status']=='PASS':
            events=rows(c/'runs'/item['run_id']/'compression.jsonl')
            loaded=next(x for x in events if x['phase']=='loaded');ended=next(x for x in events if x['phase']=='final')
            actual=loaded['modules'];used=ended['modules']
            ok=(loaded['lossy_quantization'] is False and loaded['gpu_bit_exact_verified_modules']==192
                and len(actual)==len(used)==192 and all(x['bit_exact'] and x['gpu_decoded_sha256']==x['original_sha256'] for x in actual)
                and all(x['forward_calls']>0 for x in used)
                and all(b['device']=='musa:0' for x in actual for b in x['buffers'].values()))
            proof.append({'run_id':item['run_id'],'status':'PASS' if ok else 'FAIL','lossy_quantization':False,
                'gpu_bit_exact_verified_modules':len(actual),'minimum_forward_calls':min(x['forward_calls'] for x in used),
                'static_saving_bytes':sum(x['weight_bytes_before']-x['weight_bytes_after'] for x in actual)})
            if not ok:raise ValueError('Lossless execution proof failed: '+item['run_id'])
        if item['profile']!='mixed' or item['status']!='PASS':continue
        policy_path=Path(item['policy_path']);policy_path=policy_path if policy_path.is_absolute() else root/policy_path
        policy=load(policy_path)
        events=rows(c/'runs'/item['run_id']/'quantization.jsonl')
        loaded=next(x for x in events if x['phase']=='after_model_load')
        ended=next(x for x in events if x['phase']=='process_exit')
        manifest=loaded['manifest'];buffers=loaded['execution_buffers'];retained=loaded['retained_bf16_execution_weights']
        names=set(policy['int8_modules']);bf16=set(policy['bf16_modules']);calls=ended['module_forward_calls']
        ok=(manifest.get('mixed_precision') and not manifest['fallback'] and manifest['group_size']==policy['group_size']
            and manifest['policy_sha256']==sha256(policy_path) and {x['name'] for x in buffers}==names
            and {x['name'] for x in retained}==bf16 and set(calls)==names and all(x>0 for x in calls.values())
            and all(x['weight_device']=='musa:0' and x['weight_dtype']=='torch.int8' for x in buffers)
            and all(x['device']=='musa:0' and x['dtype']=='torch.bfloat16' for x in retained)
            and len(names)+len(bf16)==96)
        proof.append({'run_id':item['run_id'],'status':'PASS' if ok else 'FAIL','int8_projections':len(names),
            'bf16_mlp_projections':len(bf16),'minimum_forward_calls':min(calls.values()),
            'policy_sha256':manifest['policy_sha256'],'group_size':manifest['group_size']})
        if not ok:raise ValueError('Mixed weights or execution disagree with frozen policy: '+item['run_id'])
        item['int8_projections']=len(names);item['bf16_mlp_projections']=len(bf16)
        item['policy_sha256']=manifest['policy_sha256']
    full=[x for x in result['runs'] if x['kind']=='full']
    for mixed in [x for x in full if (x['profile']=='mixed' or x['profile'].startswith('lossless_bf16_12')) and x['status']=='PASS']:
        for base in [x for x in full if x['profile'] in ('native','optimized_unfused','specvla') and x['status']=='PASS'
                     and all(x[k]==mixed[k] for k in ('task_id','init_state_id','seed'))]:
            a,b=[rows(c/'runs'/x['run_id']/'steps.jsonl') for x in (base,mixed)]
            comparisons.append({'baseline':base['run_id'],'mixed':mixed['run_id'],'variant':mixed['variant'],
                'baseline_task_success':base['task_success'],'mixed_task_success':mixed['task_success'],
                'same_initial_observation_hash':a[0]['observation_hash']==b[0]['observation_hash'],
                'allocated_peak_saving_bytes':base['maximum_peak_allocated_bytes']-mixed['maximum_peak_allocated_bytes'],
                'reserved_peak_saving_bytes':base['maximum_peak_reserved_bytes']-mixed['maximum_peak_reserved_bytes'],
                'token_equal_records_in_overlap':sum(x['action_tokens']==y['action_tokens'] for x,y in zip(a,b)),
                'same_step_count':len(a)==len(b),
                'same_all_observation_hashes':len(a)==len(b) and all(x['observation_hash']==y['observation_hash'] for x,y in zip(a,b)),
                'all_actions_exact':len(a)==len(b) and all(x[k]==y[k] for x,y in zip(a,b) for k in ('raw_action','executed_action','action_tokens')),'scope':'Independent evolving trajectories; equal initial state does not imply identical work'})
    dump(c/'reports/local_results.json',result);dump(c/'reports/mixed_execution_proof.json',proof)
    dump(c/'reports/mixed_comparisons.json',comparisons)
    columns=['run_id','variant','profile','task_id','init_state_id','seed','status','task_success','steps',
             'int8_projections','bf16_mlp_projections','policy_sha256','model_load_seconds','first_model_seconds','maximum_peak_allocated_bytes',
             'maximum_peak_reserved_bytes','steady_peak_allocated_bytes','steady_peak_reserved_bytes']
    with (c/'reports/mixed_results.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(result['runs'])
    text=['# 摩尔 BF16 / INT8 混合精度试验','', 'Campaign：`'+c.name+'`。仅本机S5000自测。校准成功不等于完整任务成功；不同策略版本不合并成功率。','',
          '## 校准与选择','', '| 策略 | group | INT8投影 | BF16 MLP投影 | 静态节省(GB) | 校准帧 | 全回放token相同 | raw动作MAE | raw动作最大误差 |',
          '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for folder in sorted(c.glob('calibration*')):
        if not (folder/'summary.json').exists():continue
        summary=load(folder/'summary.json');policy=load(folder/'policy.json')
        text.append('| '+folder.name+' | '+str(policy['group_size'])+' | '+str(summary['int8_module_count'])+' | '+str(summary['bf16_mlp_module_count'])+' | '+format(summary['static_saving_bytes']/1e9,'.6f')+' | '+str(summary['calibration_frames'])+' | '+str(summary['all_token_matches'])+'/141 | '+format(summary['raw_action_absolute_error']['mean'],'.8f')+' | '+format(summary['maximum_raw_action_absolute_error'],'.8f')+' |')
    text+=['','校准以同一141帧观测及固定历史为输入，主模型/动作头/视觉/attention/drafter保持BF16，仅在verifier MLP的96个投影中选择INT8。第一轮8帧校准，第二轮将第一轮最差误差帧加入，并改用group32。第二轮使用了已观察过的回放数据，不将剩余帧说成严格未见测试集；独立闭环另列。','',
           '校准期间同时保留BF16备份与INT8候选，显存不用于部署比较。真实部署从只读checkpoint在CPU按冻结JSON策略转换后上卡，未选中的投影保留原BF16权重；实际dtype/device、模块名与前向次数通过mixed_execution_proof.json核对。','',
           '## 完整闭环（不含8步冒烟）','', '| Run | 策略 | task/state | 执行 | 成功 | 动作数 | allocated峰值(GB) | reserved峰值(GB) |','| --- | --- | --- | --- | --- | ---: | ---: | ---: |']
    for x in full:
        text.append('| '+x['run_id']+' | '+x['variant']+' | '+x['task_id']+'/'+x['init_state_id']+' | '+x['status']+' | '+str(x['task_success'])+' | '+str(x.get('steps','—'))+' | '+(format(x['maximum_peak_allocated_bytes']/1e9,'.6f') if x['maximum_peak_allocated_bytes'] is not None else '—')+' | '+(format(x['maximum_peak_reserved_bytes']/1e9,'.6f') if x['maximum_peak_reserved_bytes'] is not None else '—')+' |')
    text+=['','## 与真实基线的峰值差（正数表示节省）','', '| 基线 → 混合 | allocated节省(GB) | reserved节省(GB) | 初始输入相同 |','| --- | ---: | ---: | --- |']
    for p in comparisons:text.append('| '+p['baseline']+' → '+p['mixed']+' | '+format(p['allocated_peak_saving_bytes']/1e9,'.6f')+' | '+format(p['reserved_peak_saving_bytes']/1e9,'.6f')+' | '+str(p['same_initial_observation_hash'])+' |')
    text+=['','峰值从加载前开始采集，取加载/首步及首步后窗口最大值。allocated与reserved不混算；静态权重字节账不代替GPU实测。闭环动作数或轨迹不同，也会改变临时工作量，原始逐步/验证/设备采样均保留。','',
           '## 精度与范围','', '只报告登记子集的任务完成情况与动作误差。校准/回放token相同不等于全部LIBERO SR，单任务成功也不证明损失<5%。既有BF16跨进程敏感性尚未在本轮修复，所有失败与停止扩展项均保留。', '',
           '原始策略、每次投影接受/拒绝原因、校准来源哈希与版本在calibration*/及inventory/；真实命令、配置、动作、录像和退出码在runs/、operations/。正式模型/平台/芯片矩阵与第三方验证边界沿用原任务大纲，本轮不宣称整体验收通过。']
    codec=[x for x in proof if x.get('lossy_quantization') is False]
    if codec:
        text += ['', '## BF16无损存储备选方案', '',
            '该方案属于存储压缩，不是FP12/INT8有损量化。每128个权重共享8位指数基值，每个值保留符号/7位尾数与4位指数差，少数超范围值使用int32位置+原BF16值。每层计算前在MUSA解码，再调用原生BF16 GEMM；不缓存全模型BF16副本。q_proj、视觉、drafter、embedding、lm_head及激活/KV保持原BF16配置。', '',
            '逐run核对192个压缩投影的CPU原权重SHA256与MUSA解码回CPU后的SHA256完全相同，并检查全部模块真实前向次数，见reports/mixed_execution_proof.json。静态压缩节省 '+format(codec[0]['static_saving_bytes']/1e9,'.6f')+' GB；质量以独立闭环和同输入回放分别判定，不以位精确权重直接代替整套任务SR。']
    replay=c/'replays/quantized_fixed_input_v1/results.json'
    if replay.exists():
        rr=load(replay)
        text += ['', '## 正式固定输入回放', '', '同输入及固定历史 '+str(rr['same_observation_hashes'])+' 帧；action token完全相同 '+str(rr['exact_action_token_matches'])+'；raw action完全相同 '+str(rr['exact_raw_action_matches'])+'；raw MAE '+str(rr['raw_action_absolute_error']['mean'])+'，最大误差 '+str(rr['maximum_raw_action_absolute_error'])+'。', '',
                 'BF16/压缩版模型均值比 '+format(rr['model_mean_speedup'],'.6f')+'×，pipeline均值比 '+format(rr['pipeline_mean_speedup'],'.6f')+'×。先各预热141帧再测141帧，两个worker同卡驻留、串行计算；该回放显存不计入冷进程单模型峰值比较。']
    (c/'summary.md').write_text('\n'.join(text)+'\n');print(c/'summary.md')


if __name__=='__main__':main()
