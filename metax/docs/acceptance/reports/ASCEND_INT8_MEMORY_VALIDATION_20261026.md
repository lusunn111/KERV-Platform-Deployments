# KERV 昇腾 INT8 显存优化验证报告

日期：2026-10-26。范围：当前昇腾，OpenVLA + SpecVLA Goal drafter，经FlagScale运行的量化资源与质量试验。

后续[变慢原因核查](ASCEND_INT8_SLOWDOWN_DIAGNOSIS_20261026.md)已补同卡、固定观测、逐帧交替测试：
本轮时延增加3.59%，量化矩阵乘kernel更慢，平均验证轮数未增加。下文14.4%的闭环均值差不能当作固定量化代价。

## 结论

- 中间18层MLP量化方案，在本轮10任务全程allocated峰值上节省 **2.339 GB（2.179 GiB）**。包含模型加载和首次执行，未reset峰值。
- 10个Goal任务、各首个官方初始状态、seed7：BF16 **7/10**，INT8 **7/10**。观察到的成功率变化为+0.0个百分点。
- 项目已明确允许使用量化；本轮资源收益可以作为量化方案的实测结果。它比较KERV优化版BF16/INT8，不冒充真正Spec-VLA B0的验收对照。
- 量化会改变动作和完成步数；质量门槛为有效动作和任务结果，未要求逐token等同BF16。小规模试验不能证明完整评测下成功率损失严格小于5%。

## 方法与变更

- 仅转换verifier Transformer零起始第7至24层的gate/up/down投影，共54个Linear；其余层MLP、attention、vision、drafter、embedding、head仍为BF16。
- 权重INT8、激活BF16；每输出通道、K方向group128，对称量化，scale为BF16。
- 使用已安装torch_npu的npu_weight_quant_batchmatmul；不需要CUDA TorchAO。
- 在CPU加载阶段完成量化，再将模型送到NPU；原checkpoint只读且逐文件哈希与此前一致。没有先让完整BF16模型驻留NPU再量化，也没有保留MLP的BF16影子权重。
- 保留KERV原树、全部路径、KF、动态阈值、seven动作协议和已有工作区/KV/批量提交优化；默认入口仍保留BF16参考。

## 单任务调优与候选选择

| 配置 | allocated峰值GB | 任务结果 | 动作数 | 单步模型均值ms |
|---|---:|---|---:|---:|
| BF16参考 | 17.032 | 成功 | 115 | 277.913 |
| 全32层MLP INT8 | 12.877 | 成功 | 163 | 311.949 |
| 中间18层MLP INT8 | 14.691 | 成功 | 122 | 300.233 |

全32层与BF16单任务在物理chip4串行；中间18层单任务在chip6，不能将跨芯片与不同轨迹的均值当作独立kernel速度收益。
全32层方案省4.155GB，但动作数163；中间18层省2.340GB且动作数122，相对BF16的115动作更接近，因此在看到量化Goal10结果前选定中间18层。原候选记录保留，没有删除失败或较慢结果。

## 10任务质量试验

| Task ID / 任务 | BF16成功 | BF16动作数 | INT8成功 | INT8动作数 |
|---|---|---:|---|---:|
| 0 / open the middle drawer of the cabinet | 是 | 115 | 是 | 122 |
| 1 / put the bowl on the stove | 是 | 76 | 是 | 88 |
| 2 / put the wine bottle on top of the cabinet | 是 | 102 | 是 | 109 |
| 3 / open the top drawer and put the bowl inside | 是 | 186 | 否 | 220 |
| 4 / put the bowl on top of the cabinet | 否 | 220 | 是 | 120 |
| 5 / push the plate to the front of the stove | 是 | 132 | 是 | 133 |
| 6 / put the cream cheese in the bowl | 否 | 220 | 否 | 220 |
| 7 / turn on the stove | 是 | 75 | 是 | 91 |
| 8 / put the bowl on the plate | 是 | 80 | 否 | 220 |
| 9 / put the wine bottle on the rack | 否 | 220 | 是 | 175 |

task0参与过候选选择。其余9任务：BF16 6/9，INT8 6/9。
BF16独有成功任务[3, 8]，INT8独有成功任务[4, 9]；总成功率相同不表示逐任务行为相同。
95% Wilson描述区间：BF16 [0.397,0.892]，INT8 [0.397,0.892]；没有据此证明严格5%损失边界。
相对成功率损失（BF16为分母）：0.0；百分点损失另见quality_scope.json。
所有超步数与策略失败均保留在10回合分母；不是300回合、多种子、多初始状态的正式评测。

## 10任务全程显存

| 指标 | BF16 bytes | INT8 bytes | BF16−INT8 bytes |
|---|---:|---:|---:|
| allocated峰值 | 17,044,288,512 | 14,704,834,560 | 2,339,453,952 |
| reserved峰值 | 20,946,354,176 | 18,738,053,120 | 2,208,301,056 |

达到十进制2GB：True；达到2GiB：True。两种单位同时保留，D06正式口径仍单列。
BF16 Goal10在chip4，INT8 Goal10在chip6；独立进程、相同运行栈和用例。完整同芯片的全32层量化对照另有记录。框架峰值约100ms轮询，记录框架全程峰值计数；vendor设备采样约10秒，不能当作100ms进程峰值。
allocated和reserved分别报告，不将保留池变化冒充活跃张量节省。CPU量化冷启动增加的时间包含在load_seconds；未把它藏入稳态内存口径。

## 时间与执行证据

Goal10观察到的单步模型均值：BF16 286.898ms，INT8 328.289ms。跨芯片且轨迹不同，该值只描述本轮成本，不声称量化额外加速。
INT8真实调用294,408次，转换模块54个，静态模型存储节省2,396,749,824bytes。
真实MLP形状单元检查覆盖M=8/279、K/N=4096/11008；输出有限、BF16，权重/scale存储检查通过；相对原BF16 Linear的L2误差约0.41%，该误差不能当作任务成功率损失。

## 重现入口

```bash
cd /root/KERV
KERV_NPU_ID=4 KERV_NPU_QUANT=mlp_int8 KERV_NPU_QUANT_LAYERS=7:25 \
  KERV_NPU_QUANT_GROUP_SIZE=128 scripts/run_kerv_moore_ascend.sh \
  --output-dir outputs/quant_middle18_reproduce --max-tasks 1 --trials 1 \
  --max-episode-steps 220
```

运行前确认目标物理chip空闲，输出目录使用新名字。设KERV_NPU_QUANT=none恢复BF16。不要与旧的late TorchAO转换开关叠加。

## 产物与范围

- reports/quantization_results.json/csv：从原始数据复算的资源、时间和任务结果。
- reports/quality_scope.json：质量比较、调优样本和不确定性说明。
- runs/*/metrics：逐步动作、token、验证事件、显存、加载时间和最终quantization.json命中。
- inventory：变更前源码、同步哈希、原checkpoint哈希、环境、实际kernel schema和真实形状测试。
- cases*.json、plan.md、selection.md：运行前登记及候选选择依据。
- source code: ports/moore_ascend/openvla/experiments/robot/ascend_quantization.py。

本轮没有证明完整课题验收，也没有把量化资源收益当作原BF16无损运行时收益；真实Spec-VLA B0、其他模型/平台和正式成功率协议仍需补齐。


## 证据链接

- [完整原始目录说明](../../../outputs/acceptance/quant_20261026T0215Z/summary.md)。
- [机器可读资源与质量报告](../../../outputs/acceptance/quant_20261026T0215Z/reports/quantization_results.json)。
- [质量范围与调优样本](../../../outputs/acceptance/quant_20261026T0215Z/reports/quality_scope.json)。
- [昇腾量化实现](../../../ports/moore_ascend/openvla/experiments/robot/ascend_quantization.py)。
- [项目允许量化的记录](../decisions/D09_quantization_20261026.md)。
