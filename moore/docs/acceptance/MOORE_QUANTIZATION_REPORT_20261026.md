# KERV摩尔MTT S5000量化补充报告

Campaign：`20261026T021321Z_w8a16`。根据用户本轮确认，将量化纳入允许的优化范围；本轮使用独立配置，不改变默认BF16入口。

## 交付结论（2026-10-26）

本轮已按用户确认接入并实测MUSA W8A16权重压缩，显存收益成立；**当前两个INT8候选的任务质量未通过，不能替换默认BF16配置，也不能据此宣布整体验收通过。** 该结论只针对本轮无校准分组量化候选，不表示所有量化方法均不可行。

- 与同样关闭Linear融合的BF16控制相比，group128的allocated/reserved峰值分别少 **4.070GB / 3.483GB**。控制回合在170步成功，量化版到220步仍未完成。
- 相对真实SpecVLA B0，group128的allocated峰值少 **3.421GB**，reserved少 **1.793GB**；allocated达到2GB，reserved未达到2GB。两种口径同时保留。
- 三个官方初始状态中，BF16 optimized成功 **1/3**，group128成功 **0/3**。group32在state0也未完成；按登记规则停止其state1/2扩展，保留两项NOT_STARTED，不混入已执行成功率分母。
- 固定输入141帧：动作/token完全相同 **109/141**，夹爪相同 **141/141**。模型耗时0.202829→0.234629秒，增加约 **15.68%**；pipeline增加约 **15.00%**。此路径当前有显存收益，同时存在时延及质量代价。
- 最终37项回归：**36通过、1项Graph因驱动限制跳过**，0失败。数据完整性、完整权重哈希及固定回放独立复算通过，完整性PASS不等于策略质量PASS。

本轮不再扩展新候选。后续应基于校准数据和分层精度选择继续优化，并用独立任务/状态检验质量；本次没有将这些未执行方法写成已完成结果。

## 环境与操作材料

硬件为本机物理GPU0的MTT S5000 80GiB，torch/torch_musa 2.9.1、Transformers4.40.1、Triton MUSA3.2，驱动3.3.5-server；复用现有`kerv-musa-runtime`镜像`sha256:d119ee0e1e115a26cf32a29b7d7c10e85a208d9251d5576c57e9b131560f0ca1`。未安装TorchAO/bitsandbytes，未升级驱动或运行时。期间只调度本机GPU0，不停止其他用户作业。

Verifier固定revision `fa5ae1e7509348889295bba8e08621d8b55e9baf`，drafter SHA256 `d3f62ae500f27b356083f997b24fa7a67e5dda86e5083fa15985f45250009aae`。完整权重校验见`inventory/weight_manifest.json`，模型与平台沿用OpenVLA LIBERO-Goal / LIBERO。源码HEAD为KERV `4abb3dfdbe24783a738a23a893a6bd374beca0a3`加本地适配；实际新增代码、未提交差异及最终完整源码包分别保存，不能只检出上游commit复现。

W8存储/反量化为本地`KERVRuntimeOptimization`新增实现，经FlagScale配置及桥接启动；不是调用一个已经验证的FlagGems量化GEMM，也不声称已被公开上游接收。部署入口、逐层算法与限制见[量化操作指南](../../src/docs/MUSA_QUANTIZATION.md)，此前未量化结论见[2026-10-19阶段报告](MOORE_TEST_REPORT_20261019.md)。

## 配置与执行事实

W8：INT8分组对称权重，主候选group_size=128，追加候选group_size=32，FP32 scale；A16：BF16激活/GEMM。仅verifier的96个MLP投影量化，attention、vision、drafter、embedding、lm_head、KV保持原精度。CPU上转换后才搬到MUSA；每层前向临时反量化，不保存全模型BF16副本。这是显存优化路径，不宣称原生INT8 GEMM加速。

group128选中权重存储从8,657,043,456降为4,463,788,032 bytes，静态节省4,193,255,424 bytes；group32为4,869,586,944 bytes，节省3,787,456,512 bytes（均含FP32 scale，不含未改变的bias）。实际全模型显存以实测峰值为准。group32在观察到group128任务失败后追加，失败与新增用例分别保留，不作为原结果的覆盖重试。

量化开启时BF16投影融合不叠加，树/Kalman/阈值设置保留。通过专用`quantized` profile启动；错误严格退出，不允许将BF16回退写成量化成功。

## 全部完整闭环

| Run | 官方state | 执行 | 成功 | 动作数 | 排除首步模型均值(s) | pipeline均值(s) | allocated峰值(GB) | reserved峰值(GB) |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| full_specvla_s7_state0 | 0 | PASS | 是 | 162 | 0.243195 | 0.316124 | 16.235267 | 16.418603 |
| full_optimized_s7_state0 | 0 | PASS | 是 | 170 | 0.225627 | 0.29024 | 16.890852 | 26.392658 |
| full_quantized_s7_state0 | 0 | PASS | 否 | 220 | 0.260831 | 0.320705 | 12.814405 | 14.625538 |
| full_quantized_s7_state1 | 1 | PASS | 否 | 220 | 0.274542 | 0.333485 | 12.813254 | 13.765706 |
| full_optimized_s7_state1 | 1 | PASS | 否 | 220 | 0.230812 | 0.282874 | 16.888786 | 26.440892 |
| full_optimized_s7_state2 | 2 | PASS | 否 | 220 | 0.237733 | 0.290797 | 16.891359 | 26.153583 |
| full_quantized_s7_state2 | 2 | PASS | 否 | 220 | 0.279018 | 0.3391 | 12.822646 | 14.375977 |
| full_quantized_g32_s7_state0 | 0 | PASS | 否 | 220 | 0.268176 | 0.328949 | 13.218111 | 14.663287 |
| full_quantized_g32_s7_state1 | 1 | NOT_STARTED | 未完成 | 未完成 | 未完成 | 未完成 | 未完成 | 未完成 |
| full_quantized_g32_s7_state2 | 2 | NOT_STARTED | 未完成 | 未完成 | 未完成 | 未完成 | 未完成 | 未完成 |
| full_optimized_unfused_s7_state0 | 0 | PASS | 是 | 170 | 0.224653 | 0.297055 | 16.884735 | 18.108908 |

task0，官方state0/1/2，框架seed7；env.seed仍固定0，但三个官方初始状态分别注入。最多220策略步，稳定化10步另计。每种状态的BF16/量化顺序为AB、BA、AB；B0另做state0。冒烟8步不算任务失败，也不进入SR。执行PASS仅指运行与采集完成，任务成功另列。

## 峰值显存：基线减量化，正数表示节省

| 基线 → 量化 | allocated节省(GB) | reserved节省(GB) | allocated≥2GB/2GiB | reserved≥2GB/2GiB | 初始观测相同 |
| --- | ---: | ---: | --- | --- | --- |
| full_specvla_s7_state0 → full_quantized_s7_state0 | 3.420862 | 1.793065 | 是/是 | 否/否 | 是 |
| full_optimized_s7_state0 → full_quantized_s7_state0 | 4.076447 | 11.76712 | 是/是 | 是/是 | 是 |
| full_optimized_unfused_s7_state0 → full_quantized_s7_state0 | 4.07033 | 3.483369 | 是/是 | 是/是 | 是 |
| full_optimized_s7_state1 → full_quantized_s7_state1 | 4.075533 | 12.675187 | 是/是 | 是/是 | 是 |
| full_optimized_s7_state2 → full_quantized_s7_state2 | 4.068713 | 11.777606 | 是/是 | 是/是 | 是 |
| full_specvla_s7_state0 → full_quantized_g32_s7_state0 | 3.017156 | 1.755316 | 是/是 | 否/否 | 是 |
| full_optimized_s7_state0 → full_quantized_g32_s7_state0 | 3.672741 | 11.729371 | 是/是 | 是/是 | 是 |
| full_optimized_unfused_s7_state0 → full_quantized_g32_s7_state0 | 3.666625 | 3.445621 | 是/是 | 是/是 | 是 |

峰值从模型加载前开始采集，首动作后另建稳态窗口，取两窗口最大值；没有排除BF16整模型上卡峰值来人为降低量化峰值，量化是在CPU准备后才上卡。allocated与reserved独立比较；整卡vendor采样峰值、实际采样间隔及冷/稳态分布见JSON。不同轨迹/动作数造成的工作量差异也需保留，不能将此表解释成相同动作工作量下的全部收益。

## 任务质量及部署状态

- optimized：成功 1/3 个已执行完整case；执行故障 0；未执行 0。
- optimized_unfused：成功 1/1 个已执行完整case；执行故障 0；未执行 0。
- quantized：成功 0/3 个已执行完整case；执行故障 0；未执行 0。
- quantized_g32：成功 0/1 个已执行完整case；执行故障 0；未执行 2。

本地三个初始状态只能说明该子集的任务完成情况，不能证明完整LIBERO SR或损失<5%。量化是有损变换，不要求/宣称与BF16逐token相同；同输入的动作误差另以固定回放记录。原有BF16跨seed/跨进程不一致问题没有在本轮被修复，不把量化作为其修复。BF16 optimized与量化版还存在Linear融合配置差异，追加optimized_unfused控制回合以免将所有退化归因于INT8；各控制仍为小样本，不能单独确立一般因果。

量化实际执行证明见`reports/quantization_evidence.json`：逐run检查96个INT8 MUSA buffer、FP32 scale、非fallback manifest及所有量化模块的真实forward次数。若尚未完成，保留INCOMPLETE_OR_FAIL。

模型加载时间（含CPU转换）、首步模型耗时及全部冷/稳态数值另列在`reports/results.csv`；CPU量化耗时不混入逐动作计时，也不从初始化显存采集中排除。

## 同输入固定回放

每组141帧，先完整预热一次，再计时一次。输入与历史逐帧固定；观测hash相同141，token完全相同109；执行夹爪相同141。

7维raw action绝对误差均值 0.004072956765452544，P95 0.023642856876055413，最大 0.22119705903179504；逐维MAE、每帧L2和全部记录保留。该误差为checkpoint动作空间数值，不额外假定物理单位；回放不产生SR。

BF16/量化模型均值比 0.8644654663358645×，pipeline均值比 0.8695510801611978×；不同算法输出、量化误差及样本范围均披露，不称为无损加速。两个worker同卡驻留、串行计算，回放显存不参与单模型显存表。

换算为耗时变化：量化版模型均值增加 15.68%，pipeline均值增加 15.00%。本轮不能声称量化带来时延收益。

## 直接运行及复算

以下是实验入口，不代表量化候选已通过任务质量门禁；默认入口仍为BF16 optimized。

```bash
cd /root/KERV
KERV_MUSA_PROFILE=quantized src/scripts/run_musa.sh --max-tasks 1 --trials 1 --max-episode-steps 220

# group32追加候选：
KERV_MUSA_PROFILE=quantized src/scripts/run_musa.sh --max-tasks 1 --trials 1 --max-episode-steps 220 \
  --override inference.runtime.w8a16.group_size=32

PYTHONPATH=src python3 -m acceptance.quantization_report --campaign outputs/quantization/20261026T021321Z_w8a16
```

不要使用`--load_in_8bit`/`--load_in_4bit`代替此入口：它们是另一个checkpoint加载接口，当前MUSA仍明确拒绝。专用profile为INT8权重存储+BF16逐层计算，当前只验证verifier MLP组合。

## 证据与正式范围

配置、命令、退出码、视频、逐步动作/接受/时延/内存及量化manifest均在`runs/`；预登记与环境/权重哈希在`inventory/`；固定回放在`replays/`。历史2026-10-19报告和原始证据保持原样，本轮作为独立版本与独立测试记录。

正式验收仍需既定模型/平台/芯片覆盖、SR统计和相关口径材料；本机量化允许范围已经获得用户确认，不再重复征求许可。本轮结果不代替A100实验、第三方验证或项目签认。

## 补充判定与证据保存

BF16 optimized与量化配置之间同时存在权重位宽和Linear融合变化。新增`optimized_unfused`控制回合在相同task/state/seed下成功，消除了“只要禁融合就必然失败”这一解释；由于既有跨进程敏感性及小样本，仍不能从单回合建立一般因果或套件SR保证。三状态BF16本身仅1/3成功也如实保留。

reserved差异包含融合打包和allocator行为，不能把相对BF16 optimized约11GB的reserved下降全部解释成INT8存储压缩。静态权重账和实际峰值分开报告；CPU在加载时准备量化权重，推理时所有INT8权重及scale驻留MUSA。当前checkpoint每次启动均从只读BF16文件转换，未实现量化文件格式的导出/快速加载。

完整闭环期间核心量化/推理数学代码未修改；统计、调度、未被标准闭环调用的回放工具及文档曾演进，各run保存独立源文件哈希。关键推理版本实体已归档；部分中间分析/调度版本未保存实体，逐项见`inventory/source_object_coverage.json`，不声称所有历史版本均可完整重建。

所有失败用例及停止扩展决定均保留。辅助预检首次因输出父目录未建立而失败，创建目录后用新operation ID执行通过；原日志在`operations/final_preflight/`，成功日志在`operations/final_preflight_output_directory/`。这是工具准备错误，不计为模型失败，也没有重跑或覆盖失败任务。

交付索引为campaign内`evidence_index.csv`，原始动作、时延、接受、内存、采样和录像在`runs/`；固定回放原始输入副本、哈希、逐帧输出和误差在`inputs/`及`replays/`；完整性检查为`reports/evidence_checks.json`和`reports/replay_evidence_checks.json`。本次源代码与文档包为`inventory/source_final.tar.gz`，不重复打包模型权重或导出容器镜像。`checksums.sha256`覆盖campaign交付文件。

已有结果的复算不会再次运行模型：

```bash
cd /root/KERV
PYTHONPATH=src python3 -m acceptance.quantization_report \
  --campaign outputs/quantization/20261026T021321Z_w8a16
PYTHONPATH=src python3 -m acceptance.check_quant_replay \
  --campaign outputs/quantization/20261026T021321Z_w8a16
cd outputs/quantization/20261026T021321Z_w8a16
sha256sum -c checksums.sha256
```

复算后如派生文件有变化，应在交付最后重新生成校验清单。2026-10-19原始证据包保持原样；两个campaign不合并成功率，不拼接加速倍数。
