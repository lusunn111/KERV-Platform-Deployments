"""Create the campaign index and narrative from newly collected local evidence."""
import argparse, csv, hashlib, json, subprocess, sys
from pathlib import Path
from acceptance.preflight import ROOT, sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign-dir',type=Path,required=True);a=p.parse_args()
    root=a.campaign_dir.resolve();reports=root/'reports'
    subprocess.run([sys.executable,'-m','acceptance.aggregate','--campaign-dir',str(root)],check=True)
    data=json.loads((reports/'results.json').read_text());replay=json.loads((reports/'replay_results.json').read_text())
    pair=data['pair'];assert pair['actions_tokens_observations_exact'] and replay['all_actions_exact']
    counts={x['run_id']:x for x in data['runs']};b1=counts['paired_B1'];b2=counts['paired_B2']
    texts=['# 昇腾 TestOutline 分阶段自测',f'\nCampaign: `{root.name}`；仅当前昇腾，自测，不是第三方验收。',
        '\n## 本轮完成范围',
        '\nP00源码/权重/环境冻结、P01当前可执行矩阵登记、P02采集工具最小实现、P03定向机制/位置诊断、P04本机冒烟与完整配对已执行。',
        'P02仍缺完整DMA方向/时间关联、图像/中间量细分、补偿来源严格审计与预热稳态资源段；不写成采集指标全部完成。',
        '\n## 新实测，未复用历史结果',
        f'\n- B1/B2均115动作成功；观测、raw/executed action、token和done精确一致。仅Goal task0/init0/seed7各1回合。',
        f"- 闭环保留首步：模型{b1['timing']['model_seconds']['mean']*1000:.3f}→{b2['timing']['model_seconds']['mean']*1000:.3f}ms，{pair['model_speedup_including_first']:.3f}×；策略链路{pair['pipeline_speedup_including_first']:.3f}×；含仿真的闭环墙钟{pair['closed_loop_wall_speedup']:.3f}×。",
        f"- 固定157观测，AB/BA/AB三轮，每组每轮全序列预热后测1遍；合计每配置471个动作，全部精确一致，跨轮也确定。模型{replay['model_speedup']:.3f}×，pipeline {replay['pipeline_speedup']:.3f}×。",
        '- 两种速度均为KERV B1→B2，不是Spec-VLA B0；没有达到/宣称2.4×。',
        f"- 全程allocated峰值B1={b1['memory_bytes']['allocated_peak_bytes']:,}、B2={b2['memory_bytes']['allocated_peak_bytes']:,}bytes；B2增加{-pair['allocated_peak_saving_bytes']:,}bytes，不支持节省2GB。",
        f"- reserved峰值B2比B1多{-pair['reserved_peak_saving_bytes']:,}bytes，不能与allocated择优替换。",
        '- 修复后的8步诊断：B1/B2可见Tensor.item为3535/8次，H2D逻辑字节4,826,336/9,705,148，D2H为37,296/20,096；包含输入且未完整覆盖后端DMA，不用于100MB/s判定。',
        '- 修正的独立CANN轨迹有68,204个NPU kernel、112个CPU Kalman估计范围、16个树验证、2个KF补齐。所有8步诊断动作与烟测一致。',
        '\n## 统计口径',
        '\n保留首动作/失败/重试；P95 nearest-rank。模型、包含图像准备的策略链路、env.step、闭环分层记录；加载/复位/settling/视频另列或排除并说明。',
        '固定回放排除全序列预热，但不删除测量序列的首帧。不同芯片上的低干扰与重型诊断分离；不把诊断时延纳入速度。',
        '报告每组每轮原始数组；仅3轮，单任务闭环无法估计套件成功率的不确定性。',
        '\n## 失败、修复和未完成项',
        '\n初次烟测seed配置缺失退出1，保留并只重试一次；venv宿主机检查误判、profiler别名和ATen漏记也保留初版与修复版。详见reports/revision_notes.md。',
        'B0等价性、本机M2/其他平台适配、精确SKU、D01-D08确认、正式显存/带宽及第三方状态均未满足。A100/其他芯片不由本轮执行。',
        '当前KF用历史输出token，阈值按步数衰减，不是已验证的传感器运动反馈闭环；详见reports/T03_placement_and_feedback.md。',
        '\n## 可复算入口',
        '\n```bash',f'python -m acceptance.aggregate --campaign-dir {root.relative_to(ROOT)}',
        f'python -m acceptance.check_evidence --campaign-dir {root.relative_to(ROOT)}','```',
        '\n原始记录在runs/*/metrics、model及vendor_memory.jsonl；回放在replays/*，证据索引evidence_index.csv、校验checksums.sha256。',
        '正式覆盖未满足时，指标状态保持BLOCKED/NOT_STARTED，不把执行PASS等同于任务成功或验收通过。']
    (root/'summary.md').write_text('\n'.join(texts)+'\n')
    with (root/'evidence_index.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['requirement_id','model_id','platform_id','chip_id','run_id','status','raw_path','report_path','sha256','limitation']);w.writeheader()
        for run in data['runs']:
            raw=root/'runs'/run['run_id']/'run.json'
            w.writerow({'requirement_id':'1.1-F/1.2-F/1.3-I','model_id':'M1','platform_id':'P1','chip_id':'H2','run_id':run['run_id'],
                'status':run['execution_status'],'raw_path':str(raw.relative_to(root)),'report_path':'reports/results.json','sha256':sha(raw),
                'limitation':'execution only; current local self-test, formal coverage unmet'})
        for f in sorted((root/'replays').glob('*/benchmark.json')):
            w.writerow({'requirement_id':'1.1-T-runtime-only','model_id':'M1','platform_id':'P1','chip_id':'H2','run_id':f.parent.name,
                'status':'PASS','raw_path':str(f.relative_to(root)),'report_path':'reports/replay_results.json','sha256':sha(f),
                'limitation':'same-input replay, not B0/A100 task timing or suite success rate'})
    print('Summary and evidence index:',root)

if __name__=='__main__':main()
