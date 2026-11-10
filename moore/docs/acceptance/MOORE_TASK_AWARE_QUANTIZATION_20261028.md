# 摩尔 S5000：任务成功率优先的混合量化实验

日期：2026-10-28；实验批次：`20261028T022212Z`。

甲方提交用的完整说明见[适配、推理优化与保精度压缩技术报告](MOORE_TECHNICAL_REPORT_20261028.md)，包含架构、量化机制、指标口径、验收对照及证据索引。本文保留实验明细；本次仅增加导航，封存目录内的原报告及数据不变。

用户允许动作差异，本轮以实际闭环任务成功作为主要质量依据。动作 token 一致率和权重 RMSE 仅作诊断，不能代替任务成功率。

当前选择状态：**RETAIN_VERIFIED_EXACT_BF16**。

**本轮没有找到通过任务门禁的更快有损配置，保留原96投影无损BF16存储配置。** 七类有损候选均在抽屉试点失败；不因权重误差小、动作差异被允许或某个任务成功而放行。新增解码原型也未取得稳定Linear收益。

本轮重新启动的BF16与推荐配置均在170步成功，动作、raw、执行动作及观测全部一致；96个投影实际MUSA解码SHA均等于原BF16。静态权重节省2,122,448,562 bytes，allocated峰值节省2,107,057,664 bytes（2.107058 GB），reserved峰值节省2,092,957,696 bytes。

代价仍在：本轮相同轨迹排除首步的模型均值为261.860→312.572 ms，增加19.37%。这是一次冷进程配对观测，未宣称本轮新增加速。上一批两对成功及262条一致动作是历史记录；本轮task0是重复复测，不增加独立任务/状态覆盖。

推荐policy：`recommended_exact_mlp/deployment_policy.json`，SHA256：`c36da97d0e13c2d69c3926014f2962831ef7b5271982de22849866187b8bc475`。它与本轮实际运行的上一批推荐policy逐字节一致，没有预制有损payload。

| 有损候选族 | 本次完成任务/已执行episode | 结果 |
| --- | ---: | --- |
| pilot_int8 | 1/2 | task0失败，不推荐 |
| pilot_nearzero | 1/2 | task0失败，不推荐 |
| pilot_tree_gptq_up_hybrid | 0/1 | task0失败，不推荐 |
| pilot_tree_gptq_w10 | 0/1 | task0失败，不推荐 |
| pilot_w10_pow2 | 1/2 | task0失败，不推荐 |
| pilot_w10 | 0/1 | task0失败，不推荐 |
| pilot_w12_up8 | 0/2 | task0失败，不推荐 |

候选经过自适应研发筛选，按具体policy分别列分母，不把不同候选混合成一个模型的成功率。任务失败也保留完整episode；未启动的task1不计分母。

最终针对性回归：**37项通过**。先前编译/测试夹具失败与修复记录仍在operations/中。

以下逐个列出实际启动的 run；执行 PASS 仅表示流程和采集完整，任务成功单列。未启动的登记项不计任务分母。

| run | task/state | 执行 | 任务成功 | 动作数 | 首步 ms | 其余步模型均值 ms | allocated峰值 GB | reserved峰值 GB |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| comparison_old_exact_mlp_task0_state0_s7 | 0/0 | PASS | True | 170 | 8620.541 | 312.572 | 14.772259 | 15.130952 |
| control_exact_packed8_task0_state0_s7 | 0/0 | PASS | True | 170 | 8641.837 | 316.361 | 15.502678 | 15.850275 |
| control_fresh_bf16_task0_state0_s7 | 0/0 | PASS | True | 170 | 9100.771 | 261.860 | 16.879316 | 17.223909 |
| pilot_int8_task0_state0_s7 | 0/0 | PASS | False | 220 | 8713.424 | 310.555 | 14.372642 | 14.677967 |
| pilot_int8_task1_state10_s7 | 1/10 | PASS | True | 89 | 8861.417 | 273.810 | 14.378856 | 14.608761 |
| pilot_nearzero_task0_state0_s7 | 0/0 | PASS | False | 220 | 8689.125 | 303.081 | 14.774471 | 15.195963 |
| pilot_nearzero_task1_state10_s7 | 1/10 | PASS | True | 87 | 8698.557 | 263.459 | 14.772332 | 15.156118 |
| pilot_tree_gptq_up_hybrid_task0_state0_s7 | 0/0 | PASS | False | 220 | 8733.396 | 313.886 | 14.648174 | 14.910751 |
| pilot_tree_gptq_w10_task0_state0_s7 | 0/0 | PASS | False | 220 | 8766.460 | 304.719 | 14.845381 | 15.091106 |
| pilot_w10_pow2_task0_state0_s7 | 0/0 | PASS | False | 220 | 8844.056 | 295.234 | 14.840214 | 15.099494 |
| pilot_w10_pow2_task1_state10_s7 | 1/10 | PASS | True | 94 | 8768.908 | 278.439 | 14.842092 | 15.059649 |
| pilot_w10_task0_state0_s7 | 0/0 | PASS | False | 220 | 8819.892 | 298.236 | 15.176562 | 15.432942 |
| pilot_w12_up8_task0_state0_s7 | 0/0 | PASS | False | 220 | 8686.041 | 316.129 | 14.822261 | 15.145632 |
| pilot_w12_up8_task1_state10_s7 | 1/10 | PASS | False | 220 | 8821.891 | 290.408 | 14.817461 | 15.195963 |

GB=10⁹ bytes。内存为独立冷进程从模型加载至退出期间 framework allocator 峰值；allocated 与 reserved 分列，不能与静态权重节省混为一谈。首步保留且单列，其余步统计排除各 episode 首步。

最初两个试点使用上一批已封存 BF16 成功记录作为筛选参考：task0/state0/seed7 为170步成功，task1/state10/seed7为92步成功。历史参考分别为 `outputs/precision_refinement/20261027T084306Z/runs/full_bf16_task0_state0_s7` 与 `full_bf16_task1_state10_s7`。这些历史数据不作本轮新鲜配对性能结论。

通过试点后才执行预先登记的三个新状态配对：task0/state10、task1/state20、task2/state10，seed7、上限220策略动作。新配对使用相同 guarded Softmax、host tree builder、共享原标量 Kalman 历史、原生 BF16 GEMM；每次独立冷进程、GPU0串行执行。若候选在BF16成功的case失败，不推荐该候选。

| 新配对 reference / candidate | BF16成功 | 候选成功 | 完整 | allocated节省 GB | reserved节省 GB |
| --- | --- | --- | --- | ---: | ---: |

本轮没有新候选通过试点与性能门禁，因此三个预登记新状态未执行，清单完整保留；不填入成功率分母。

| 真实layer15投影热微基准 | 解码倍率范围 | 解码+原生GEMM倍率范围 | 推荐 |
| --- | ---: | ---: | --- |
| CSR融合初版 | 0.8275–0.8320× | 0.8379–0.9014× | 否 |
| CSR静态展开 | 0.6672–0.7643× | 0.7296–0.8532× | 否 |
| 两级字典，1024/4 | 0.3513–0.3562× | 0.4275–0.5037× | 否 |
| 两字节拆位，1024/4 | 1.0016–1.0141× | 0.9771–1.0034× | 否 |

微基准使用同一层gate/up/down原始BF16权重，GPU解码逐位核验；M=1/8/279，4次预热后8组、每组20次，按AB/BA顺序交替，保留全部同步wall-clock批次样本。倍率分母为候选，参考为原无损codec。它们是热算子结果，不能替代模型或任务速度；其余线程配置见各results.json，没有删除更慢配置。

发生动作差异后，两端后续观测和计算工作量可能不同，因此不计算错位轨迹的逐步动作 MAE，也不把闭环平均延迟之比称为同工作量算子加速。小样本未验证TestOutline要求的套件成功率损失<5%，也未证明更严格的≤1个百分点目标。量化/压缩显存收益不能代替TestOutline中同精度异构部署的P06验收。

实现包含打包W10/W12解码、power-of-two scale的int8指数存储，以及覆盖实际BF16解码舍入的GPTQ误差反馈；运行时解码到BF16后调用原生GEMM，不是硬件原生INT10矩阵乘。保护组保存原始BF16，attention、vision、drafter、head和KV保持BF16。具体模块数和scale类型以每个policy为准。

RTN W10/fp32和W10/pow2复用旧75投影的保护组，没有重新激活校准。真实树GPTQ则在三个C轨迹共72帧收集实际提示词及推测验证树激活，每层分别抽样；36帧V来自已见诊断轨迹。各MLP独立使用原始BF16上游输入，down投影使用本块量化gate/up输出，未做跨块误差传播。局部MLP输出相对RMSE门限1%，超限保持BF16；局部门限不是闭环成功证据。

完整75投影GPTQ仍在task0/state0达到220步上限而失败，因此停止该候选的后续试点，没有把未执行的task1计入分母。另登记up-only混合候选：仅25个up投影使用新GPTQ W10，其余71个MLP投影无损BF16存储，attention仍原生BF16。该布局复用本轮BF16上游独立校准的up权重；不把完整75投影的局部重建误差直接当作up-only的闭环精度。

25个up投影的10-bit混合方案也未完成抽屉任务。随后登记8个up投影的W12/pow2候选，其余88个MLP投影无损存储；这8个模块沿用上一批低敏感度模块名单，直接从原始BF16做RTN，保护组保持原样，没有做12-bit GPTQ。其权重相对RMSE为0.000207508，是重建诊断而非任务成功率。注意这里8个有损整数W12与其余模块的无损BF16指数编码是两种不同机制。

新鲜BF16与全保护组打包控制均170步成功，170条动作、raw、执行动作和观测一致。控制的96个投影实际MUSA解码SHA均等于原始BF16；它仅隔离打包路径，冗余存储不作部署收益。这支持当前场景下打包路径保真，不能据此断言所有运行时问题均被排除，也未定位有损误差如何触发具体KERV决策。

追加近零量化：96个MLP投影使用12-bit指数差编码，仅把每组指数范围下方及有符号边界值映射为有符号零，其余权重位精确；每个权重绝对误差不超过所在组原始最大绝对值/32768。GPU单次解码取消异常值scatter；注意力等模块BF16。此机制不使用整数GEMM、激活校准或GPTQ，局部数学误差界不能代替任务验证。

近零量化实测：4,328,521,728个权重中18,963个非零值被置零（约0.000438%），整体权重相对RMSE约1.48745e-7。抽屉试点仍失败，第二个任务成功；按用户任务成功优先的标准拒绝该候选。不能从极小权重RMSE推断闭环任务成功率。

解码算子试验保留：CSR融合恢复初版在layer15三个真实投影的解码微基准约0.83倍，静态展开版更慢；两级异常字典在本机同样更慢，均未进入完整任务推荐测试。原始AB/BA样本见inline_operator_benchmark*/及hash_operator_benchmark/。这只说明这些具体实现的实测结果，不将性能归咎于未证实的硬件或库问题。

字典测试曾因测试数据在一级表中恰好无冲突而错误要求二级表非空；已改用随机碰撞和强制二级命中场景，CPU和设备检查通过。两字节拆位内核初次编译检测到32/16位bitcast不匹配，显式转换uint16后通过。相关失败日志和修复后命令均保留，不计为任务失败。

校准采集入口使用native树路径并补上guarded Softmax；部署和新配对使用host tree/shared history。这一差异记录于校准启动命令；校准不参与部署速度和峰值显存统计。

一次RTN队列在第二个case启动前被GPU空闲检查拦截：并行的CPU权重构建进程持有2MiB MUSA上下文。该case未开始episode，原日志保留，不计为任务失败。后续构建和设备测试串行执行，没有重置GPU或终止其他任务。

源码快照见source_w10/、source_tree_gptq/、source_nearzero/、source_inline_exact*/和各run/source_manifest.sha256；最终完整源码及本轮文档归档见inventory/source_final.tar.gz，最终校验见checksums.sha256；操作命令、退出码和日志见operations/；case清单见inventory/cases*.csv；权重校验见inventory/weight_manifest.json。依赖版本与上一批相同，未升级驱动、muDNN或框架，原始checkpoint未修改。

操作指南：[任务优先量化与推荐配置](../../src/docs/MUSA_TASK_AWARE_QUANTIZATION.md)。JSON复算：`python3 -m acceptance.task_aware_quantization_report --campaign <目录>`。封存后在副本中复算，避免改动校验清单中的派生文件。
