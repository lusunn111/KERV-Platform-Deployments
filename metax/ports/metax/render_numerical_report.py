"""Rebuild the numerical-fix report from real artifacts, including cold terminal steps."""
import argparse,datetime,json,subprocess
from pathlib import Path
from ports.metax.summarize import summarize
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('outputs/metax_numerical_20261105'));a=p.parse_args();r=a.root
paths={'原生 B1':Path('outputs/metax_native_goal3_init01_02'),'第一轮 optimized':Path('outputs/metax_optimized_goal3_init01_02'),'旧 v2（数值不同）':Path('outputs/metax_v2_20261104/pointwise_goal3_init01'),'原生 RMS 回退':r/'native_rms_goal3_init01','修复版':r/'fixed_goal3_init01','修复版独立复跑':r/'fixed_goal3_init01_repeat'}
runs={k:summarize(path) for k,path in paths.items()};assert all(x['exit_code']=='0' for x in runs.values())
subprocess.run(['python3','ports/metax/episode_stats.py',*[str(x) for x in paths.values()],'--output',str(r/'episode-stats-final.json')],check=True,capture_output=True,text=True)
episodes=json.loads((r/'episode-stats-final.json').read_text());replay=json.loads((r/'fixed220-summary.json').read_text());diag=json.loads((r/'all-layer-summary-final.json').read_text());validation=json.loads((r/'numerical-validation.json').read_text())
assert validation['status']=='passed'
report=dict(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),runs=runs,episode_statistics=episodes,replay=replay,diagnostic=diag,validation='numerical-validation.json')
(r/'numerical-results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
factors=json.loads((r/'root-factor-substitution.json').read_text());first=json.loads((r/'all_layer_diag02/first-divergence.json').read_text())
text=f'''# KERV 沐曦数值修复

更新时间：{report['updated_utc']}。起点5b7de3c，稳定参考bf179f6及第一轮原生/optimized的696步。原始数据全部保留，本轮独立目录`outputs/metax_numerical_20261105/`，没有扩大压缩或B0工作。

修复版已完成两次独立进程的3任务×2初态闭环，两次均5/6、696步，与原生/第一轮的观测、image hash、token、raw/executed、done、success、阈值及终止原因精确一致。1928次验证的原始/实际接受前缀、候选token、补偿与边界也一致。不是仅重新命名配置。

## 首次差异与实测根因

1. 首次是step0、`model.layers.0.input_layernorm`，输入[1,279,4096]，eps=1e-6。flat90192（row22,col80）：原生{first['reference']}，旧v2 {first['old']}，共38个BF16输出不同。
2. 最早FP32阶段差发生在mean（155/279行），但不能把首次38个BF16差都归到归约：受控替换仅改变1/sqrt公式就复现38处，只改变该次8warp mean时BF16输出差0；两者合用与实际旧kernel输出完全一致。该次公式组合是首次BF16差的充分原因。
3. rsqrt试验首层9形状一致不能推广：step0第2层input_layernorm已有21个非零数值输出差，max_abs0.001953125。第一批全层原生mean替换后，保留相同rsqrt及FP32提升末乘，仅剩373处符号零差；归约路径差异是后续问题，单换rsqrt不足。
4. 输入+0、负BF16权重时，BF16乘给-0，升FP32相乘路径给+0。实测FP32中间结果已是bits0，单独把构造的FP32 -0转换BF16仍保留bits32768，因此不能归咎BF16转换。生成LLVM也显示：FP32提升末乘被lower为`llvm.mxc.pk.fma.f32(...,+0)`，BF16类型末乘为scalar fmul再cvt；即使`enable_fp_fusion=False`仍可有此后端FMA(+0)路径。修复保留原生BF16乘法类型。
5. 原始RMS阶段FMA开/关在采样全部阶段差0；这个开关实验与上述类型触发的低级lowering必须区分。原生mean作为输入时，Triton/native的epsilon、sqrt、reciprocal、rsqrt均0位差，本批证据不支持把差异笼统归为rsqrt近似指令。

原生路径是在独立venv中实际使用的Transformers4.40.1 `LlamaRMSNorm.forward`，Torch2.10.0+metax3.8.1.0：FP32 cast→pow(2)→原生mean→epsilon→rsqrt→normalize→BF16 cast→BF16权重乘。`paths.json`保存实际源码，native-rms-trace.json保存厂商GPU kernel轨迹。drafter使用cnets的F.rms_norm，其舍入契约不同，保持原状。

完整首差的平方、mean、epsilon、sqrt/rsqrt、normalize、BF16及乘权重中间张量存为`first-rms-stage-tensors.pt`（约45.7MB）；`first-rms-stage-values.json`保存row22/col80的原始数值和位模式及文件SHA，可用export_numerical_evidence.py复算。其余全层调用保存各阶段逐元素位差计数与最大误差，避免将数十亿元素全部写盘。

## 改动与覆盖范围

新`numerical_fixed`保留原生FP32 pow/mean，将epsilon/同后端rsqrt、normalize、BF16转换与BF16权重乘合入尾核。所有65个验证器RMS层统一处理，无测试动作条件、逐层择优或按结果选策略；只按设备、BF16、4096维与连续存储支持，否则走原生。`numerical_native_rms`仅rope+vision_graph，是完整原生RMS回退。

在原生轨迹相同真实输入上，诊断始终返回原生输出，不让误差污染下游输入；额外并排计算旧v2、rsqrt与修复候选。共{diag['calls']}次、{diag['layers']}个层、{len(diag['shapes'])}种真实形状，涵盖19个早晚步骤、短验证、6验证循环长路径与周期KF。修复候选在该批RMS输出及native-mean后的inverse阶段位差均0。原始数据在all_layer_diag02及all_layer_long_diag03，不能据此宣称所有可能输入都位精确。

视觉图输出在参考计算前立即clone检查异步依赖与复用；采样vision/verifier RoPE/draft RoPE/draft原生RMS数据位差均0，global函数恢复identity通过。一次draft T=1 singleton stride不同，数据相同且完整轨迹恢复，不将其归为数值根因。旧release缺少实例RMS还原，现补齐并记录restore_audit；历史每组重新加载模型、原生入口断言65层原生，因此不把生命周期缺口错当历史696→722的原因。

## 相同输入、无诊断正式测速

固定完整220步图像和原始参考token/raw动作历史，seed7、完整树、阈值和规划时域220不变，每组完整序列预热后测量，AB/BA反序独立进程；原生/第一轮/旧v2/修复同场。诊断在独立进程且同卡锁串行，正式测量关闭诊断。各组加载前计峰值，释放后分配为0；工作区清理发生在模型之间，没有排除加载峰值。

| 配置 | 模型均值ms | 中位ms | P95ms | 策略均值ms | 对原生模型倍率 | 行为 |
|---|---:|---:|---:|---:|---:|---|
'''
labels={'native':'原生','runtime':'第一轮 optimized','old_v2':'旧v2实验','v2':'数值修复'}
for k,v in replay['profiles'].items():
 m=v['model'];ratio=replay['profiles']['native']['model']['mean_ms']/m['mean_ms']
 text+=f'| {labels[k]} | {m["mean_ms"]:.6f} | {m["median_ms"]:.6f} | {m["p95_ms"]:.6f} | {v["pipeline"]["mean_ms"]:.6f} | {ratio:.6f}× | '+('不同' if k=='old_v2' else '与稳定参考一致')+' |\n'
text+='\n修复版较同场第一轮模型吞吐'+f"{replay['pairwise_ratios']['runtime/v2']['model']:.6f}×，对原生策略倍率{replay['pairwise_ratios']['native/v2']['pipeline']:.6f}×"+'。不为维持2×调整阈值/候选树/精度。8步四组AB另存fixed8_AB，不与220步数字拼接。\n\n## 完整真实闭环与显存\n\n| 配置 | 成功 | 动作 | 模型s | 策略s | 闭环s | allocated B | reserved B |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
for name,d in runs.items():text+=f'| {name} | {d["successful_episodes"]}/{d["complete_episodes"]} | {d["policy_steps"]} | {d["model"]["sum_seconds"]:.6f} | {d["policy_pipeline"]["sum_seconds"]:.6f} | {d["closed_loop_seconds"]:.6f} | {d["allocated_peak"]["bytes"]:,} | {d["reserved_peak"]["bytes"]:,} |\n'
text+='\n包含每回合首动作、失败回合及成功终止步；旧v2为不同722步轨迹，闭环时间不能当作相同模型输入的纯引擎倍率。修复与原生相同696步，接受前缀2089/1928=1.083506224，初始root696、bonus1232、KF855、padding0，包含零接受。成功数差0/6、百分点差0、相对差0%；六回合不能证明总体分布无质量损失。\n\n原始allocated/reserved字节、GB/GiB和设备/进程1秒观测峰值分别保存在numerical-results.json、raw/memory.jsonl与vendor-memory.log。采样峰值不等于瞬时真实峰值，不能相加CPU/GPU重叠时间。\n\n## 每回合排除首步统计\n\n数值依次是均值/中位/P95（nearest-rank），毫秒；每回合只去step0，保留成功终止动作。全部含首步/去首步原始统计可由episode_stats.py重算。\n\n| 配置 | 回合 | 去首步n | 模型均值/中位/P95 | 策略均值/中位/P95 | 首步模型ms |\n|---|---|---:|---|---|---:|\n'
for (label,_),data in zip(paths.items(),episodes):
 for e in data['episodes']:
  m=e['excluding_first']['model_seconds'];q=e['excluding_first']['pipeline_seconds']
  text+=f'| {label} | {e["episode"]} | {m["n"]} | {m["mean_ms"]:.3f}/{m["median_ms"]:.3f}/{m["p95_ms"]:.3f} | {q["mean_ms"]:.3f}/{q["median_ms"]:.3f}/{q["p95_ms"]:.3f} | {e["first_model_ms"]:.3f} |\n'
text+='''
## 复跑与审计

```bash
# 推荐修复入口（不要覆盖已有输出）
scripts/run_kerv_metax.sh --profile numerical_fixed --max-tasks 3 --trials 2 --output-dir outputs/your_numerical_fixed
# 原生RMS回退
scripts/run_kerv_metax.sh --profile numerical_native_rms --max-tasks 3 --trials 2 --output-dir outputs/your_native_rms
# 四组220步同输入回放；反序改BA
scripts/benchmark_kerv_metax.sh --comparison numerical_fix --v2-features rms_native_mean,rope,vision_graph --frozen-history --samples 220 --repeats 1 --profile-order AB --input-dir outputs/metax_v2_20261104/representative_capture/observations --output outputs/your_numerical_replay_AB
python3 -m ports.metax.validate_numerical
python3 -m ports.metax.render_numerical_report
```

仍通过真实FlagScale链启动，实际模型引擎是厂商PyTorch/MACA，物理GPU0→cuda:0，/tmp/kerv-metax-card0.lock。联网仅命令级baai-proxy，本轮无新增下载/依赖/驱动修改。旧optimized_v2/rms/rms_rsqrt保留为实验；第一轮bf179f6与全部原始结果保留。不推广首层相同或随机张量结果为全模型保证；当前结论是所采集全层位精确、两次完整696步轨迹恢复，未证明所有可能输入普遍位精确。
'''
Path('docs/METAX_NUMERICAL_FIX.md').write_text(text);print('docs/METAX_NUMERICAL_FIX.md')
