# KERV摩尔S5000混合精度及保精度存储压缩报告

## 本轮结论（2026-10-26）

按用户要求实现并实测了 **BF16/INT8按投影混合量化**。两轮校准均不能保证任务精度：第1轮57个INT8投影，第2轮提高精度门槛后只剩3个；两者在task0/state0均到220步仍未完成，而匹配BF16控制170步成功。**这些INT8策略未通过质量门槛，不替换默认配置。**

为继续满足“降显存、保精度”，本轮追加了 **BF16无损存储压缩**：12-bit码加分组指数基值，超范围值保留原BF16；逐层解码为原BF16后计算。它不是FP12或INT8有损量化，不能以“混合INT8精度已解决”对外描述。

- **native+无损存储的追加控制已在170步成功**，与native BF16的170条动作、token及全过程观测全部一致；allocated/reserved峰值减少**2.897035/2.894070 GB**。模型均值耗时增加**14.96%**，加载约**91.12秒**。仅限task0/state0/seed7，默认不变。
- 真实模型192个投影，CPU原权重与MUSA解码结果逐投影SHA256一致，全部实际参与前向；静态减少 **2.912395 GB**。但优化runtime下首个完整任务仍失败，端到端精度未通过，不能把权重无损写成动作无损。
- task0/state0相对匹配的BF16 optimized_unfused：allocated峰值减少 **2.890168 GB**，reserved峰值减少 **2.770338 GB**；动作完全相同：**False**，全过程观测相同：**False**。
- 相对真实隔离SpecVLA B0，同一case allocated/reserved峰值分别减少 **2.240347 / 1.080033 GB**。两类基线不混用，GB=10⁹ bytes、GiB=2³⁰ bytes；≥2GB与≥2GiB另在判定JSON中独立列出。
- 优化runtime已执行子集（state0）：匹配BF16成功 **1/1**，无损存储成功 **0/1**；逐case质量和动作差异在下面表格/JSON中完整保留。这不是完整LIBERO套件SR，也不能推出性能损失<5%的统计结论。
- 正式固定输入回放：**139/141** 条action token向量相同，raw action完全相同 **139/141**；raw MAE **0.0000206891**，最大误差 **0.0130672268**。
- 固定回放模型均值比BF16/压缩版为 **0.6843×**，pipeline为 **0.6947×**；压缩版耗时变化分别 **+46.14% / +43.94%**。压缩减少存储，同时有每层解码开销，不宣称INT8 GEMM加速。
- 44项回归：43通过、1项Graph因驱动能力跳过；证据流、权重哈希及回放独立复算通过。默认BF16配置保留，新增压缩入口可显式选择。

## 操作入口与环境

操作步骤、参数含义、两种实现、核验文件及回退见[混合精度操作指南](../../src/docs/MUSA_MIXED_QUANTIZATION.md)。既有W8A16全量MLP结果见[上轮量化报告](MOORE_QUANTIZATION_REPORT_20261026.md)，BF16原始优化结果见[10月19日阶段报告](MOORE_TEST_REPORT_20261019.md)。各campaign独立封存，不拼接成功率或加速倍数。

本轮复用物理GPU0 MTT S5000 80GiB，驱动3.3.5-server；torch/torch_musa2.9.1、Transformers4.40.1、Triton MUSA3.2。容器`kerv-musa-runtime`镜像ID `sha256:d119ee0e1e115a26cf32a29b7d7c10e85a208d9251d5576c57e9b131560f0ca1`。没有升级运行时、安装TorchAO/bitsandbytes或使用其他卡。实际推理由FlagScale启动本地KERV/FlagOS适配和torch_musa执行，压缩内核为本地Triton MUSA实现；不声称公开上游FlagScale已有此功能。

Verifier固定revision `fa5ae1e7509348889295bba8e08621d8b55e9baf`，drafter SHA256 `d3f62ae500f27b356083f997b24fa7a67e5dda86e5083fa15985f45250009aae`。原始checkpoint只读，完整文件哈希在inventory/weight_manifest.json。KERV基于HEAD `4abb3dfdbe24783a738a23a893a6bd374beca0a3`加本地适配，不能仅检出上游HEAD复现；本轮归档实际源码与依赖、每次命令和配置。

## 无损存储算法与实现细节

BF16含1位符号、8位指数、7位尾数。对每128个展平权重，取指数最大值减15并截断到0，作为该组uint8指数基值。常见值用8位符号/尾数与4位指数差组成12-bit码，相邻两个码打包为3个uint8；指数低于基值的少数权重另存int32位置及原BF16值。解码恢复原始16位，异常值再原样补回；CPU及MUSA哈希检查失败即退出。

实际选中5,939,134,464个权重，原始11,878,268,928 bytes：打包码8,908,701,696、分组基值46,399,488、1,795,427个稀疏例外合计10,772,562，共8,965,873,746 bytes，节省2,912,395,182 bytes。元数据与例外均已计入；平均实际存储约12.08 bit/weight，不将其误写成固定12 bit的总存储。未改变bias另计，当前选中投影无bias。

保留32个q_proj为原nn.Linear，以兼容原模型直接访问weight dtype/device；96个MLP和96个attention k/v/o投影采用压缩模块。视觉、drafter、embedding、动作头、激活和KV维持原BF16方案。每层按需生成一个BF16临时权重并调用原生GEMM，不保留全模型BF16 GPU副本。量化/压缩互斥，不叠加原BF16 Linear融合。树、动态阈值、Kalman、缓存和动作后处理未为提高成功率而修改。

专用入口在CPU加载阶段注册后端。为兼容既有桥接，使用w8a16.enabled字段触发CPU准备、互斥Linear融合，并在历史w8a16.json文件中写实际压缩manifest；`implementation`为`musa_lossless_bf16_exp_delta12_native_bf16_gemm`、`lossy_quantization=false`。这是字段复用，绝不代表本路径实际使用INT8或FP12 GEMM。

初始化包含CPU编码、往返校验和每个设备投影解码哈希检查。加载时间、首步时间以及初始化/稳态内存均保存；哈希不混入逐动作推理计时，临时设备解码内存计入初始化峰值。全闭环内核调用数在compression.jsonl最终事件中可查。实际optimized无损存储模型加载约90.216秒，匹配BF16约9.896秒；初始化耗时显著增加，首次动作的约10秒另计。全部case的加载与首步时间见reports/mixed_results.csv。压缩模块是推理专用，不支持训练或直接作为标准HF checkpoint导出；重启时从原BF16 checkpoint重新编码。

## 追加保守运行时对照

两臂都使用native KERV，关闭optimized运行时，不更改KERV树/Kalman算法及动作协议。BF16成功=True、动作数=170；native+无损存储成功=True、动作数=170。初始观测相同=True，全过程观测相同=True，全部动作相同=True。

相对真实SpecVLA B0，native+无损存储的allocated/reserved峰值分别减少2.253002/2.088763 GB；两者均≥2GB，但reserved尚小于2GiB。B0与native KERV动作生成算法及轨迹不同，该对照只报告实际峰值和任务结果，不作逐动作精度相等的声明。

相对该匹配native BF16，allocated/reserved峰值减少2.897035/2.894070 GB。排除首步的模型均值0.516231→0.593456秒，pipeline均值0.586068→0.662415秒；仅当全部观测/动作一致时才视为相同闭环工作量的时间比较，不能与optimized路径拼接成加速倍数。这是单任务控制，不是整套SR或泛化精度结论。

## 差异排查

与实际BF16回放模型核对：800个未压缩参数及192个压缩投影共992个权重实体全部一致，见reference_weight_equivalence.json。使用真实第0层七种投影和7种行数（1/4/8/232/256/280/320）做49组GPU测试，解码权重、投影输出及独立BF16权重克隆输出均逐bit一致，见large_gemm_diagnostic.json。小型Transformer首轮及KV cache续推也完全相同。

这些检查排除了所测权重/单投影解码差异，不能直接定位完整VLA链路中细微差异的根因。优化runtime固定回放在frame11与104发生小幅动作差异；完整闭环在索引32（第33步）首次分歧，随后轨迹发散。当前没有证明是某个特定融合、cache、allocator或硬件算子导致，不将推测写成已修复原因。逐bit原权重不等于完整运行链路的确定性保证。


Campaign：`20261026T025710Z_mixed`。仅本机S5000自测。校准成功不等于完整任务成功；不同策略版本不合并成功率。

## 校准与选择

| 策略 | group | INT8投影 | BF16 MLP投影 | 静态节省(GB) | 校准帧 | 全回放token相同 | raw动作MAE | raw动作最大误差 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| calibration | 128 | 57 | 39 | 2.489745 | 8 | 105/141 | 0.00477028 | 0.47004202 |
| calibration_refined | 32 | 3 | 93 | 0.118358 | 16 | 122/141 | 0.00258785 | 0.48529412 |

校准以同一141帧观测及固定历史为输入，主模型/动作头/视觉/attention/drafter保持BF16，仅在verifier MLP的96个投影中选择INT8。第一轮8帧校准，第二轮将第一轮最差误差帧加入，并改用group32。第二轮使用了已观察过的回放数据，不将剩余帧说成严格未见测试集；独立闭环另列。

校准期间同时保留BF16备份与INT8候选，显存不用于部署比较。真实部署从只读checkpoint在CPU按冻结JSON策略转换后上卡，未选中的投影保留原BF16权重；实际dtype/device、模块名与前向次数通过mixed_execution_proof.json核对。

## 完整闭环（不含8步冒烟）

| Run | 策略 | task/state | 执行 | 成功 | 动作数 | allocated峰值(GB) | reserved峰值(GB) |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| full_specvla_s7_task0_state0 | specvla | 0/0 | PASS | True | 162 | 16.235267 | 16.418603 |
| full_optimized_unfused_s7_task0_state0 | optimized_unfused | 0/0 | PASS | True | 170 | 16.885088 | 18.108908 |
| full_mixed_s7_task0_state0 | calibration | 0/0 | PASS | False | 220 | 14.475749 | 15.974007 |
| full_mixed_s7_task0_state1 | calibration | 0/1 | NOT_STARTED | None | — | — | — |
| full_optimized_unfused_s7_task0_state1 | optimized_unfused | 0/1 | NOT_STARTED | None | — | — | — |
| full_optimized_unfused_s7_task0_state2 | optimized_unfused | 0/2 | NOT_STARTED | None | — | — | — |
| full_mixed_s7_task0_state2 | calibration | 0/2 | NOT_STARTED | None | — | — | — |
| full_mixed_s7_task1_state0 | calibration | 1/0 | NOT_STARTED | None | — | — | — |
| full_optimized_unfused_s7_task1_state0 | optimized_unfused | 1/0 | NOT_STARTED | None | — | — | — |
| full_mixed32_s7_task0_state0 | calibration_refined | 0/0 | PASS | False | 220 | 16.770847 | 18.524144 |
| full_mixed32_s7_task0_state1 | calibration_refined | 0/1 | NOT_STARTED | None | — | — | — |
| full_mixed32_s7_task0_state2 | calibration_refined | 0/2 | NOT_STARTED | None | — | — | — |
| full_mixed32_s7_task1_state0 | calibration_refined | 1/0 | NOT_STARTED | None | — | — | — |
| full_lossless_s7_task0_state0 | lossless_bf16_12 | 0/0 | PASS | False | 220 | 13.994920 | 15.338570 |
| full_lossless_s7_task0_state1 | lossless_bf16_12 | 0/1 | NOT_STARTED | None | — | — | — |
| full_lossless_s7_task0_state2 | lossless_bf16_12 | 0/2 | NOT_STARTED | None | — | — | — |
| full_lossless_s7_task1_state0 | lossless_bf16_12 | 1/0 | NOT_STARTED | None | — | — | — |
| full_native_control_s7_task0_state0 | native | 0/0 | PASS | True | 170 | 16.879300 | 17.223909 |
| full_lossless_bf16_12_native_control_s7_task0_state0 | lossless_bf16_12_native | 0/0 | PASS | True | 170 | 13.982265 | 14.329840 |

## 与真实基线的峰值差（正数表示节省）

| 基线 → 混合 | allocated节省(GB) | reserved节省(GB) | 初始输入相同 |
| --- | ---: | ---: | --- |
| full_specvla_s7_task0_state0 → full_mixed_s7_task0_state0 | 1.759518 | 0.444596 | True |
| full_optimized_unfused_s7_task0_state0 → full_mixed_s7_task0_state0 | 2.409339 | 2.134901 | True |
| full_native_control_s7_task0_state0 → full_mixed_s7_task0_state0 | 2.403551 | 1.249903 | True |
| full_specvla_s7_task0_state0 → full_mixed32_s7_task0_state0 | -0.535580 | -2.105541 | True |
| full_optimized_unfused_s7_task0_state0 → full_mixed32_s7_task0_state0 | 0.114242 | -0.415236 | True |
| full_native_control_s7_task0_state0 → full_mixed32_s7_task0_state0 | 0.108453 | -1.300234 | True |
| full_specvla_s7_task0_state0 → full_lossless_s7_task0_state0 | 2.240347 | 1.080033 | True |
| full_optimized_unfused_s7_task0_state0 → full_lossless_s7_task0_state0 | 2.890168 | 2.770338 | True |
| full_native_control_s7_task0_state0 → full_lossless_s7_task0_state0 | 2.884380 | 1.885340 | True |
| full_specvla_s7_task0_state0 → full_lossless_bf16_12_native_control_s7_task0_state0 | 2.253002 | 2.088763 | True |
| full_optimized_unfused_s7_task0_state0 → full_lossless_bf16_12_native_control_s7_task0_state0 | 2.902823 | 3.779068 | True |
| full_native_control_s7_task0_state0 → full_lossless_bf16_12_native_control_s7_task0_state0 | 2.897035 | 2.894070 | True |

峰值从加载前开始采集，取加载/首步及首步后窗口最大值。allocated与reserved不混算；静态权重字节账不代替GPU实测。闭环动作数或轨迹不同，也会改变临时工作量，原始逐步/验证/设备采样均保留。

## 精度与范围

只报告登记子集的任务完成情况与动作误差。校准/回放token相同不等于全部LIBERO SR，单任务成功也不证明损失<5%。既有BF16跨进程敏感性尚未在本轮修复，所有失败与停止扩展项均保留。

原始策略、每次投影接受/拒绝原因、校准来源哈希与版本在calibration*/及inventory/；真实命令、配置、动作、录像和退出码在runs/、operations/。正式模型/平台/芯片矩阵与第三方验证边界沿用原任务大纲，本轮不宣称整体验收通过。

## BF16无损存储备选方案

该方案属于存储压缩，不是FP12/INT8有损量化。每128个权重共享8位指数基值，每个值保留符号/7位尾数与4位指数差，少数超范围值使用int32位置+原BF16值。每层计算前在MUSA解码，再调用原生BF16 GEMM；不缓存全模型BF16副本。q_proj、视觉、drafter、embedding、lm_head及激活/KV保持原BF16配置。

逐run核对192个压缩投影的CPU原权重SHA256与MUSA解码回CPU后的SHA256完全相同，并检查全部模块真实前向次数，见reports/mixed_execution_proof.json。静态压缩节省 2.912395 GB；质量以独立闭环和同输入回放分别判定，不以位精确权重直接代替整套任务SR。

## 正式固定输入回放

同输入及固定历史 141 帧；action token完全相同 139；raw action完全相同 139；raw MAE 2.0689126606253904e-05，最大误差 0.013067226783902042。

BF16/压缩版模型均值比 0.684253×，pipeline均值比 0.694744×。先各预热141帧再测141帧，两个worker同卡驻留、串行计算；该回放显存不计入冷进程单模型峰值比较。

## 测试方法、边界与证据交付

全闭环为单case冷进程，按相同官方task/state与框架seed7配对。env.seed固定0，实际官方初始状态逐case注入；最多220个策略动作，初始稳定化10步另计。冒烟8步只验证启动/执行，不计为任务失败或SR。执行PASS表示完整运行与采集；task_success单独反映任务是否完成。第1轮和第2轮INT8在state0失败后按事先登记规则停止扩展，未执行项保留NOT_STARTED，未将其删去或算成成功。

优化runtime无损存储在state0失败后，停止其state1/2及task1扩展，未见数据不能填为已通过。随后单独登记native KERV与native+无损存储一对保守运行时控制，作为新配置验证，原失败保留。不同候选分别记账；校准期间同时保存原BF16与INT8副本，其显存不能用于部署指标。所有部署从只读权重重新加载，不复用校准中GPU全模型副本。

正式回放使用141帧捕获观测及逐帧固定raw action/Kalman token历史，两组先各预热141帧再测141帧；逐帧校验图像和proprio hash、shape、dtype。两个模型同卡驻留、串行计算，回放时的双模型显存不进入单模型峰值表。输出比较包含全部帧，不剔除异常帧。校准脚本为提高搜索效率缓存CPU图像处理，其时间不与正式pipeline混算。

权重逐bit一致只排除了压缩引入的权重舍入误差，不能证明整个BF16运行链路对所有输入/进程都确定；既有BF16跨进程敏感性仍需按实测结果披露。有限任务、状态和单轮回放也不等于完整LIBERO SR、泛化精度或三轮无损结论，更不能将其他机器的471动作结果移作本机指标。

整体验收仍需TestOutline所列其他硬件、模型/平台、对照和正式指标确认；本轮仅本机摩尔卡自测，不填补未执行的A100/昇腾等矩阵，也不声称第三方独立验证。

证据根目录：`outputs/mixed_quantization/20261026T025710Z_mixed`。原始动作、接受长度、耗时、内存、设备采样和录像在runs/；校准策略、每个投影试验和重复参考在calibration*/；正式同输入回放在replays/；可复算数据、候选执行证明和完整性检查在reports/。cases*.csv、protocol及追加理由在inventory/，所有失败原样保存。operations/lossless_extended_and_replay的exit_code=1为初始任务失败后触发停止门槛，未进入扩展或该脚本的回放阶段；固定输入回放后续使用单独的lossless_replay_diagnostic operation执行，未覆盖该停止记录。

CPU/设备推理关键版本实体均归档到inventory/source_objects/，并以每run source_manifest.sha256对应。分析、调度、非本次闭环消费的回放工具和文档在实验期间演进，部分中间分析版本实体未保存，逐项见source_object_coverage.json；不声称全部历史版本可完整重建。最终源码包source_final.tar.gz包含本地实际代码、依赖适配和文档，不重复打包权重或导出容器。

复算：

```bash
cd /root/KERV
export PYTHONPATH=src
C=outputs/mixed_quantization/20261026T025710Z_mixed
python3 -m acceptance.mixed_report --campaign "$C"
python3 -m acceptance.check_evidence --campaign "$C" --check-source-hashes --check-weight-hashes
python3 -m acceptance.check_quant_replay --campaign "$C"
cd "$C"
sha256sum -c checksums.sha256
```

复算如改写派生文件，交付前须重建校验清单。已完成run ID禁止追加重跑；新实验使用新campaign或先登记新追加清单及新ID。
