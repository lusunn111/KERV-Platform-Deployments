# 昇腾 W8A8：省内存与时延优化

实验日期：2026-10-26，汇总更新：2026-10-27。范围：本机昇腾、OpenVLA LIBERO Goal、SpecVLA drafter、FlagScale KERV adapter。

## 最终结果：80个闭环回合，2026-10-27汇总

本轮内存/平均时延优先候选为**融合W8A8，保留第15、16、19、20层down投影的BF16权重**。
中间18层共54个原始MLP投影，50个量化、4个保留BF16；gate/up合并后为32个量化矩阵乘模块。
配置仍显式选择，原BF16默认配置未更改。

| 指标 | BF16 | 保守混合配置 |
|---|---:|---:|
| 同卡单模型全程allocated峰值GB | 17.040 | 14.802 |
| 完整157帧同卡交替模型均值ms | 296.218 | 289.865 |
| 模型时延中位数ms | 295.213 | 294.894 |
| 模型时延P95 ms | 514.555 | 481.520 |
| Goal10/init2成功数 | 7/10 | 7/10 |
| Goal10/init1成功数 | 7/10 | 6/10 |
| 两初态合计成功数 | 14/20 | 13/20 |

**显存节省2.237GB（2.084GiB）；平均模型时延降低2.14%，P95降低6.42%。**
三遍平均时延分别降低1.92%、2.24%、2.28%，每精度共471次测量；中位数降低约0.11%。
完整回放的验证轮数1416 → 1422，各精度保留自己的历史和验证树，不强制相同内部数值或工作量。
这些是本机固定回放的结果，不表示每个输入、任意负载或完整闭环完成时间都更快。

质量仍有代价：init1新增成功为任务5，失去任务4、9的两次成功；init2新增成功为任务3，失去任务0的一次成功。
两初态总成功数减少1次，观察成功率70% → 65%；**不能宣称不降精度**。
两个初态共享同10个任务，不是20个独立任务，小样本不足以证明总体成功率损失边界。
保留BF16配置和此前的W8A16/group32候选，便于按质量/资源需求选择。

### 算子优化过程

- 未合并的动态W8A8：init2为5/10，同卡64帧平均时延增加2.48%，不推荐。
- gate/up合并：init2为8/10，同卡64帧平均时延增加0.55%。
- 直接MLP入口与上述合并版在完整init2的1303个动作/token精确一致；单独去除Python包装没有得到稳定收益。
- 原生SwiGLU/量化融合：真实MLP微基准约298 → 262微秒；完整157帧平均时延降低1.46%，省2.413GB。
  该版init2为8/10、init1为5/10，合计13/20。
- 保守混合配置：按init0局部误差选择4个down保留BF16，init2为7/10、init1为6/10，
  相同总成功数下减少单初态的退化幅度，仍省超过2GiB且本轮均值/P95均未增加。

保护局部误差最大的投影没有恢复全部闭环成功率，局部重建误差不能代替任务质量。
所有尝试和失败保留，未将旧KERV/native加速比与本轮量化时延拼接。

## 1. 目标和参考

用户目标是省内存、精度损失小、运行不变慢。
此前W8A16/group32已经省2.218GB，但同卡模型时延增加2.29%；因此继续优化实际执行路径。
本轮不修改KERV接受阈值、Kalman、seven动作协议、原TensorFlow预处理或220步规划长度。

## 2. 实际实现

采用现有昇腾算子，将verifier第7至24层的MLP权重、激活都量化为INT8，输出仍为BF16：

1. 使用上一轮BF16 init0轨迹采集的54个投影输入，每投影216行，覆盖10个Goal任务。
2. 搜索SmoothQuant alpha=0/0.25/0.5/0.75，以实际NPU输出局部平方误差选择参数。
   gate/up合并校准，使用共同的平滑向量；down单独校准。
3. 权重逐输出通道对称INT8，在CPU准备后转移到NPU；不保留量化MLP的BF16影子权重。
4. gate/up共享一次`npu_dynamic_quant`和一次`npu_quant_matmul`。
5. 量化down使用`npu_dequant_swiglu_quant`融合SwiGLU和动态量化，再调用INT8矩阵乘；
   保留BF16的4个down使用原SiLU/mul与BF16 Linear。
   本环境要求该融合算子的平滑系数为FP32；首次BF16参数失败与修正记录均保留。
6. 第15、16、19、20层down按init0局部重建误差排名选择，保持原始BF16权重。
7. 根据真实形状的测速选择ND权重布局：gate/up使用K×N连续存储，down使用转置视图。

该融合改变中间舍入，与分开的BF16 SiLU/mul不逐位一致，因此任务质量独立验证。
校准policy包含原始权重SHA256，加载前核对身份，防止误用于其他权重。
新路径显式选择`mlp_dynamic_int8`，未修改原BF16默认配置。

## 3. 对照和统计范围

- 单模型显存：全程allocated峰值，含模型加载和首次执行，不reset峰值。
- 时延：物理chip14上两个模型临时同时驻留，逐帧AB/BA交替；每精度独立预热、独立维护历史和验证树。
- 64帧：预热64次、测量4遍，共256次动作/精度。
- 完整157帧：预热157次、测量3遍，共471次动作/精度。
- 完整任务：Goal10，seed7，每回合最多220步；新候选分别测试官方init1和init2。
  init2 BF16是本轮新跑，init1 BF16质量引用上一轮同协议记录，历史时延不参与新加速比。
- 每配置每初态只有10回合，两个初态共享任务身份。样本不能证明总体精度损失边界。
- 所有失败回合保留在分母；局部张量误差不冒充任务准确率。
- 不将旧KERV/native加速比与本轮量化时延拼接计算新总加速比。

## 4. 已排查的路线

- 单纯更换W8A16权重转置布局仍慢于BF16。
- 显式开启internal format后，W8A16 NZ单算子路径报错；未将失败探针当作可用方案。
- W8A8 NZ布局没有获得优于所选ND布局的微基准结果。
- 单独TorchScript包装W8A16收益不足，未进入部署配置。
- 未合并的动态W8A8在init2为5/10成功，低于BF16的7/10，且同卡慢2.48%，不推荐。
- gate/up合并版init2为8/10成功，同卡慢0.55%；进一步评估原生激活/量化融合以消除剩余开销。

## 5. 复现（保守混合配置）

候选policy路径是容器内路径，使用前检查chip空闲和输出目录未使用：

```bash
KERV_NPU_ID=14 KERV_NPU_QUANT=mlp_dynamic_int8 \
  KERV_NPU_QUANT_POLICY=/data/data/KERV/outputs/acceptance/quant_noslow_20261026T054934Z/calibrated_hybrid4/policy.pt \
  KERV_LIBERO_INITIAL_STATE_OFFSET=2 \
  scripts/run_kerv_moore_ascend.sh \
  --output-dir outputs/w8a8_hybrid4_new --max-tasks 10 --trials 1 \
  --max-episode-steps 220
```

`KERV_NPU_QUANT=none`选择BF16参考。原checkpoint保持不变。

## 6. 实现和证据

- [原生W8A8模块与融合](../../../ports/moore_ascend/openvla/experiments/robot/ascend_dynamic_quantization.py)。
- [校准](../../../acceptance/calibrate_dynamic_int8.py)、[同卡回放](../../../acceptance/interleave_quantization.py)、[结果汇总](../../../acceptance/report_dynamic_int8.py)。
- [实验目录](../../../outputs/acceptance/quant_noslow_20261026T054934Z/)。
- [64帧同卡结果](../../../outputs/acceptance/quant_noslow_20261026T054934Z/timing_swiglu/results.json)。
- [最终候选policy](../../../outputs/acceptance/quant_noslow_20261026T054934Z/calibrated_hybrid4/policy.pt)。

每阶段源码快照、policy、真实输入探针、失败日志、单元检查、任务日志和视频分别保存在实验目录。

## 7. 校准复现

以下在容器内项目根目录执行，设备和新输出目录按实际空闲情况选择：

```bash
ASCEND_RT_VISIBLE_DEVICES=14 OMP_NUM_THREADS=16 \
  PYTHONPATH=ports/moore_ascend/openvla .venv/bin/python acceptance/calibrate_dynamic_int8.py \
  --inputs outputs/acceptance/quant_accuracy_20261026T045119Z/runs/bf16_calibration_init0/calibration/inputs.pt \
  --checkpoint checkpoints/openvla-libero-goal --output outputs/new_w8a8_policy \
  --fuse-gate-up --fused-swiglu-quant --preserve-down 4
```

该命令使用既有BF16校准输入重新导出policy；输入采集过程见上一轮精度优先量化报告。

## 8. 最终结果和验证

- [完整混合配置结果](../../../outputs/acceptance/quant_noslow_20261026T054934Z/reports/hybrid4_results.json)。
- [完整157帧原始计时](../../../outputs/acceptance/quant_noslow_20261026T054934Z/timing_hybrid4_full157/results.json)。
- [此前全量W8A8融合结果](../../../outputs/acceptance/quant_noslow_20261026T054934Z/reports/swiglu_results.json)。
- 4项NPU检查通过，包含policy身份、原生融合前向、请求保留的BF16权重、缓存和直接入口数值检查。
- FlagScale KERV启动路径和原有树构建、接受控制、KV提交算子继续执行，日志位于各run的model/manifests。
