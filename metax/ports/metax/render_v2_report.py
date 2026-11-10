"""Render round-two tables from retained raw data; never edit round-one outputs."""
import argparse,json,datetime
from pathlib import Path
from ports.metax.summarize import summarize
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('outputs/metax_v2_20261104'));p.add_argument('--v2-run',default='pointwise_goal3_init01');p.add_argument('--compression-run',default='compression_goal3_init01');p.add_argument('--series',nargs='*',default=['selected8','selected220','compression8','compression220']);p.add_argument('--markdown',type=Path,default=Path('docs/METAX_V2_PERFORMANCE.md'));a=p.parse_args()
paths=[('B1 native',Path('outputs/metax_native_goal3_init01_02')),('B2 第一轮',Path('outputs/metax_optimized_goal3_init01_02')),('Q2 第一轮无损',Path('outputs/metax_bf16_lossless_goal3_init01')),('B2 v2',a.root/a.v2_run),('Q2 v2',a.root/a.compression_run)]
runs={label:summarize(path) for label,path in paths if (path/'exit_code.txt').exists()}
replays={name:json.loads((a.root/f'{name}-summary.json').read_text()) for name in a.series if (a.root/f'{name}-summary.json').exists()}
report=dict(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),runs=runs,replays=replays,B0='independent unresolved SpecVLA boundary definition; no B0 performance ratios')
(a.root/'performance-results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
lines=['# C550 第二轮性能改进表','',f'更新时间：{report["updated_utc"]}。所有数字从保留的原始记录重算；第一轮输出不改写。','', '## 完整3任务×2初态','', '| 配置 | 成功 | 动作/验证 | 模型s | 策略s | 闭环s | native总模型时间/本配置 | allocated原始B（GB / GiB） | reserved原始B | 实际接受前缀 |','|---|---:|---:|---:|---:|---:|---:|---|---:|---:|']
base=runs['B1 native']['model']['sum_seconds']
for label,d in runs.items():
 m=d['allocated_peak'];v=m['bytes']
 lines.append(f'| {label} | {d["successful_episodes"]}/{d["complete_episodes"]} | {d["policy_steps"]}/{d["verify_calls"]} | {d["model"]["sum_seconds"]:.6f} | {d["policy_pipeline"]["sum_seconds"]:.6f} | {d["closed_loop_seconds"]:.6f} | {base/d["model"]["sum_seconds"]:.6f}× | {v:,}（{v/1e9:.6f} / {v/2**30:.6f}） | {d["reserved_peak"]["bytes"]:,} | {d["mean_accepted_prefix"]:.9f} |')
lines+=['','含每回合首动作与冷启动；模型、策略、闭环分别计时，不相加CPU/GPU。任务集总时间倍率是端到端任务集结果；轨迹变化时不视为相同模型输入的微观加速比。v2归一化存在数值/动作差异，第一轮696步精确轨迹结论不适用。设备观测峰值、root/bonus/KF/零接受统计分别保存在performance-results.json及原始vendor-memory.log。','', '## 预热后同输入回放','', '冻结图像、参考raw_action与token历史；每profile完整序列预热，正反顺序合并。这里的模型/策略毫秒没有仿真。','']
for name,d in replays.items():
 lines += [f'### {name}','',f'历史模式：{d.get("history_mode")}。动作精确一致：{d["all_actions_exact"]}。','', '| 配置 | 模型均值ms | 中位ms | P95ms | 策略均值ms | 模型参考/本配置 |','|---|---:|---:|---:|---:|---:|']
 ref='native' if 'native' in d['profiles'] else 'v2'
 for label,v in d['profiles'].items():
  m=v['model'];ratio=d['profiles'][ref]['model']['mean_ms']/m['mean_ms']
  lines.append(f'| {label} | {m["mean_ms"]:.6f} | {m["median_ms"]:.6f} | {m["p95_ms"]:.6f} | {v["pipeline"]["mean_ms"]:.6f} | {ratio:.6f}× |')
 lines += ['',f'完整原始计时数组与逐轮allocated/reserved峰值：`../{a.root}/{name}-summary.json`。','']
lines+=['## 质量和限制','','B1/B2第一轮5/6。每个v2候选分别计算成功数/分母，不把改变的轨迹称为完全一致；小样本不能证明总体质量稳定。无损指权重BF16位模式保留；与含数值融合的v2基线比较时应明确参照配置。省显存与速度必须来自同一配置，峰值从加载前统计，图池/工作区/缓存全部包含。B0保持独立待决，无B0加速比。','', '复跑、算法/数值边界与淘汰方案见 [第二轮技术说明](METAX_V2_TECHNICAL_REPORT.md)。']
a.markdown.write_text('\n'.join(lines)+'\n');print(a.markdown)
