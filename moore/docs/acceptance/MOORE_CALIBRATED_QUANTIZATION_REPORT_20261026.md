# KERV摩尔S5000激活校准量化试验交付报告

## 实测结论（2026-10-26）

已实现并跑完小规模试验：真实多模态激活校准、GPTQ式误差补偿、投影内原BF16保护组、顺序量化、CPU导出和MUSA部署。**当前结果不能证明“精度降低很小”，默认BF16配置未替换。** 本轮不是仅做静态预算，也没有把权重无损存储压缩写成量化精度通过。

- 单MLP block的输出相对RMSE：RTN32为0.00907757，GPTQ+保护组为0.00884085，降低约2.61%；局部改善有限，不能外推完整策略精度。
- 首候选实际量化78个MLP投影，alpha=0、group32、约5%连续输入列组保留BF16；静态节省2.696183GB。相对本轮native控制，allocated/reserved冷进程峰值分别减少 **2.607497/2.654994 GB**。
- 首候选完整回放141帧，动作/token一致 **107/141**，raw动作MAE **0.00647197**，最大 **1.23920170**；执行夹爪相同141/141。首个完整任务未完成，按规则停止扩展，不删除失败。
- 本轮BF16基线同样出现失败，且与此前成功case的推理源码、task/state/seed配置一致。后续另做2个冷进程重复控制，合计成功 **1/3**，动作数[220, 152, 220]。该重复性问题未定位根因，不能把闭环失败单独归因量化。
- 50项回归：49通过、1项Graph因当前驱动能力跳过；原始证据、权重哈希和回放复算通过。单元/完整性PASS不等于策略质量PASS。

追加输出敏感性候选恢复BF16层[23]，保留75个INT8投影，静态节省2.592483GB；teacher-forcing差异从13/252变为9/252。重新部署后的141帧动作一致 **111/141**，raw MAE **0.00481229**，最大 **1.23920170**，完整case成功 **0/1**。该候选状态为 **NOT_PASSED**，选择数据不是最终未见测试；不能只展示变好的teacher-forcing分数。

## 环境、实施材料及计量范围

操作指南见[校准量化操作指南](../../src/docs/MUSA_CALIBRATED_QUANTIZATION.md)，设计见[机制V2](QUANTIZATION_DESIGN_V2_20261026.md)，此前试验见[混合量化/无损存储报告](MOORE_MIXED_QUANTIZATION_REPORT_20261026.md)。旧原始campaign保持封存，未覆盖失败或重写历史结果。

复用物理GPU0 MTT S5000 80GiB、驱动3.3.5-server，torch/torch_musa2.9.1、Transformers4.40.1、Triton MUSA3.2。FlagScale启动本地KERV和本地新增量化后端，实际执行为torch_musa及Triton MUSA；并非vLLM执行OpenVLA，也不声称上游FlagScale/FlagGems已接收此量化实现。没有安装TorchAO/bitsandbytes或升级运行时。

容器镜像sha256:d119ee0e1e115a26cf32a29b7d7c10e85a208d9251d5576c57e9b131560f0ca1；Verifier revision fa5ae1e7509348889295bba8e08621d8b55e9baf，drafter SHA256 d3f62ae500f27b356083f997b24fa7a67e5dda86e5083fa15985f45250009aae。KERV基于HEAD4abb3dfdbe24783a738a23a893a6bd374beca0a3加本地适配，实际源码与未提交差异已保存。原BF16 checkpoint只读，文件级SHA复核通过。

GB按10⁹ bytes，GiB按2³⁰ bytes，静态字节账和framework实际峰值分开。动作raw数值为checkpoint动作空间，不额外假定毫米/弧度。未将原optimized加速倍数与本轮native量化显存结果拼接，也不拿KERV native替代SpecVLA B0做正式验收。


日期：2026-10-26。Campaign：`20261026T045030Z_v2`。本机物理GPU0 MTT S5000、既有MUSA容器，FlagScale启动本地KERV；本地新增后端，不声称上游已接收。

## 本轮完成与边界

已实现并实际执行：激活校准、GPTQ式误差补偿、投影内BF16保护组、逐层候选/原BF16教师双路径、CPU导出与重新部署、MUSA实际解码/GEMM。量化是有损W8A16；没有将BF16无损存储方案改称量化通过。默认配置保留。

这是先验证机制的小规模试验。已采集task0/1/2各state3/seed7的前48个动作；采集截断不计SR。校准用三个任务的144帧，诊断选择用历史task0的36帧；后者已在既往研发中观察过，不是最终未见测试。没有执行设计文档中的60/45/300回合完整协议，不能据此证明成功率下降≤1个百分点。

## 单block对照

第15层MLP，RTN32输出相对RMSE 0.00907757；选中的 gptq32_p5 为 0.00884085，变化 -2.61%。第一阶段每个C轨迹取24帧，合计72帧；独立于本次C的历史诊断轨迹取36帧。prompt/实际树与decode输入采样记录在sampling.json；此阶段没有另加teacher-forcing行。

该block替换后36帧固定动作回放：token向量完全相同 29/36，raw MAE 0.0016200093388501777，最大 0.11107142766316724，夹爪相同 36/36。重建误差下降不代表动作精度通过。AWQ alpha=.5组合在该block未优于alpha=0的GPTQ+保护组，因此后续选择实际更好的配置，不强行叠加缩放。

## 全模型顺序校准

每个样本使用实际图像预处理、任务提示、BF16视觉embedding及捕获的7维动作token构成teacher-forced多模态上下文。原BF16教师状态与量化候选状态分别逐层推进；下一层统计来自当前候选，MLP内部down_proj使用实际量化gate/up的输入。原BF16教师输出用于诊断及V上的累积误差门槛，未实现额外的全block梯度重构训练。

每帧采样7个prompt位置及7个动作预测位置的3倍权重，即25/75采样质量；各轨迹和完整来源在模型校准protocol中记录。此全模型pilot没有额外混入验证树分支样本，区别于设计的25/50/25完整采样，也区别于第一阶段原生KERV激活采样。

前2/后4层、attention、视觉、drafter、embedding、动作头、norm及KV保持原配置。中间第2–27层的MLP候选按group32处理，保护比例约5%并按组向上取整；保护组先冻结，用原BF16值覆盖，GPTQ仅更新未保护列。分组风险采用激活能量加权的误差对角近似，不声称已实现完整Fisher或完整组Hessian风险搜索。

Hessian在CPU用FP64 Cholesky稳定分解，MUSA用FP32执行误差补偿，计入最终实际BF16解码舍入。alpha=0.0。最终量化 78 个投影，静态节省 2.696183 GB；逐层是否回退、局部/累计误差及CPU/GPU校准时间在model_quant/layers.json。

teacher-forcing诊断：动作预测位置top1相同 239/252，mean KL 0.011236340282410974，最大logit误差 1.5625。这是固定前缀的诊断，不是KERV实际动作回放，也不是闭环SR。

## 完整闭环

| Run | task/state | 执行 | 成功 | 动作数 | allocated峰值GB | reserved峰值GB |
| --- | --- | --- | --- | ---: | ---: | ---: |
| full_native_s7_task0_state0 | 0/0 | PASS | False | 220 | 16.880072 | 17.179869 |
| full_calibrated_s7_task0_state0 | 0/0 | PASS | False | 220 | 14.272575 | 14.524875 |
| full_calibrated_s7_task1_state10 | 1/10 | NOT_STARTED | None | — | — | — |
| full_native_s7_task1_state10 | 1/10 | NOT_STARTED | None | — | — | — |
| full_native_s7_task6_state0 | 6/0 | NOT_STARTED | None | — | — | — |
| full_calibrated_s7_task6_state0 | 6/0 | NOT_STARTED | None | — | — | — |
| full_calibrated_refined_s7_task0_state0 | 0/0 | PASS | False | 220 | 14.372564 | 14.677967 |
| full_calibrated_refined_s7_task1_state10 | 1/10 | NOT_STARTED | None | — | — | — |
| full_calibrated_refined_s7_task6_state0 | 6/0 | NOT_STARTED | None | — | — | — |
| full_native_repeat1_s7_task0_state0 | 0/0 | PASS | True | 152 | 16.878614 | 17.179869 |
| full_native_repeat2_s7_task0_state0 | 0/0 | PASS | False | 220 | 16.879305 | 17.167286 |

初始state0失败则按登记规则停止task1/state10、task6/state0的扩展，未执行项保持NOT_STARTED；失败或低成功率不能删除。state0曾出现在诊断数据中，不称其为最终独立任务。执行PASS只表示采集完整，任务成功另列。

## 显存对照

| native → 校准量化 | allocated节省GB | reserved节省GB | 全部动作/观测相同 |
| --- | ---: | ---: | --- |
| full_native_s7_task0_state0 → full_calibrated_s7_task0_state0 | 2.607497 | 2.654994 | False |
| full_native_s7_task0_state0 → full_calibrated_refined_s7_task0_state0 | 2.507508 | 2.501902 | False |

加载前起采，初始化与首步/稳态峰值均保留；CPU校准后上卡，没有先把完整BF16 verifier上卡再删除。实际显存包含解码临时权重、scale、BF16保护组和运行时工作区。与native基线对照不是SpecVLA B0验收对照；不把不同轨迹的耗时差直接称作同工作量加速。

## 全模型固定输入回放

原生BF16与校准量化：同输入/历史 141 帧，完整动作/token一致 107/141，raw MAE 0.006471968439086216、最大 1.2392016957787906；执行夹爪相同 141。

每组完整预热141帧，再测141帧，全部帧保留。模型均值BF16/量化比 0.946590×，pipeline比 0.947356×；加载、CPU校准及哈希不混入动作计时。两个worker同卡驻留、串行计算，回放显存不参与冷进程单模型峰值表。

## 动作输出敏感性追加筛选

初始质量不足时，以同一已见V36帧分别恢复每个候选block为原BF16，按执行夹爪变化、6维动作bin绝对/最大误差、全部top1差异及KL排序，再检查恢复0/1/2/4/6个block的组合。只接受静态节省≥2GB的组合；这是预算内的研发选择，不是独立测试。

原始teacher-forcing差异 13/252，选中后 9/252；恢复BF16层 [23]，保留INT8投影 75，静态节省 2.592483 GB。具体组合与所有单层试验均保留。

BF16重复参考的top1差异 0；量化后仍需按新ID完整闭环和同输入回放，不将V选择分数当成SR。

## 追加候选的固定回放

独立冷进程加载所选policy后，动作/token一致 111/141，raw MAE 0.004812289456216201，最大 1.2392016957787906，执行夹爪相同 141/141。模型均值BF16/量化比 0.919066×、pipeline比 0.921145×；预热和计时口径与初始回放相同。

## BF16基线重复性边界

本轮native BF16在task0/state0/seed7也未完成，之前相同配置170步成功；首次动作分歧在索引17（从0计），当步观测hash相同。对比per-run source manifest，原有推理源码无变化。该现象提示基线重复性问题，当前未定位具体算子或硬件原因，不能把闭环失败单独归因量化，也不能从一次基线失败和一次量化成功推断量化提升SR。失败与后续重复控制分别登记，全部保留。

## 相同输入的BF16参考重复性

两个独立native BF16 worker各自的正式计时141帧，动作/token相同 140/141，差异帧 [85]；实际加载参数哈希完全相同=True。这与闭环轨迹发散分开，是相同输入/固定历史下的参考差异；根因尚未定位。

同一worker预热/计时动作相同 140/141，但预热额外保存frame77调试张量，观察器开销不同，不能把该比较说成完全相同执行条件的重复实验。正式跨worker计时对照没有这项观察器差异。

初始量化与BF16有34帧动作差异，明显多于此BF16参考对照的一帧差异，不能用基线问题将所有量化差异解释掉；同样也不能用单次闭环失败孤立推断量化的套件SR损失。
## 复现与留档

专用入口：src/scripts/run_musa_calibrated.sh；必须显式提供容器内的冻结policy.json路径，缺失或hash/shape不匹配即退出。CPU payload与MUSA解码哈希一致性、INT8/BF16实际buffer和真实forward次数见calibrated_quantization.jsonl及reports/calibrated_execution_proof.json。校准phase的双路径内存及耗时均不写成部署收益。

首次block_pilot因prompt/验证树输入类别识别错误而在样本拼接处退出，未进入量化；修正后用新ID block_pilot_v2运行，原日志与原源码SHA实体保留。未覆盖旧campaign或既有量化失败结果。

读取model_quant/*.pt为本轮生成的可信本地量化张量，模型原checkpoint只读。精度、性能、显存、数据完整性分别判断；单任务成功不能当作完整LIBERO SR，也不能拿本机结果替代A100、昇腾或其他模型/平台的验收。

## 交付文件与后续判断

主证据目录是`outputs/calibrated_quantization/20261026T045030Z_v2`。`evidence_index.csv`列出原始动作、验证、内存、设备采样、录像、校准、模型产物和复算文件；`checksums.sha256`核验交付文件。`delivery_manifest.json`记载报告和最终源码归档哈希。

最终源码`inventory/source_final.tar.gz`包含本地代码、依赖适配与文档，不重复打包原BF16权重或导出容器镜像。新量化payload已保留在campaign的model_quant/，追加policy引用同一份tensor文件，校准来源和生成参数可追溯。仅凭上游commit不能复现本地适配。

每个run和校准phase都保存源码哈希；关键推理实体有SHA对应归档。实验期间分析、调度、测试及不被普通闭环消费的校准/回放工具曾演进，部分中间实体无法完整恢复，逐项见source_object_coverage.json，不声称所有历史版本都可完整重建。首次采样分类错误的原入口源码实体和失败日志均保留。

当前不足包括：校准只覆盖少量任务/状态和早期动作，最终300对回合未执行；BF16基线存在跨进程/运行差异；隐藏状态重建与teacher-forcing改善未必转化为KERV接受/补偿链路及闭环稳定性。不能据小样本宣称成功率损失≤1个百分点，更不能据此宣称任务书整体验收通过。

后续应先定位BF16重复性，再增加接触、夹爪切换、失败恢复及真实验证树分支的校准覆盖，并评估动作决策约束或更高位宽。不应仅降低层输出MSE就继续扩大INT8比例，也不能为满足显存目标放宽精度结论。

复算：

```bash
cd /root/KERV
export PYTHONPATH=src
C=outputs/calibrated_quantization/20261026T045030Z_v2
python3 -m acceptance.calibrated_report --campaign "$C"
python3 -m acceptance.check_evidence --campaign "$C" --check-source-hashes --check-weight-hashes
python3 -m acceptance.check_quant_replay --campaign "$C"
python3 -m acceptance.check_quant_replay --campaign "$C" --experiment-name calibrated_refined_fixed_input_v1
cd "$C"
sha256sum -c checksums.sha256
```

复算如改写派生报告，交付前须重建校验清单。已完成run禁止追加重跑，新实验需新ID和预登记。
