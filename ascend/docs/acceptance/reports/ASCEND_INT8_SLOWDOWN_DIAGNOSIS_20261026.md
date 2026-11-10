# 昇腾 INT8 变慢原因核查

日期：2026-10-26。仅当前昇腾，物理 chip14。量化配置保持为中间18层MLP、W8A16、group128。

## 结论

**已确认当前量化矩阵乘路径在实际输入形状上比BF16慢；本轮验证轮数和原始接受长度没有恶化。**
不过，原先闭环均值的 **14.4%** 差异不能全部归为量化算子开销。固定观测、同卡逐帧交替运行后，
本轮模型均值为 **245.478 → 254.301 ms，增加3.59%（8.823 ms）**。
这说明当前实现存在量化执行代价，也说明原始比较混入了轨迹、执行顺序和共享环境的影响。
本轮没有证明原始14.4%的差异可以逐项分解，也不能把3.59%推广为全部LIBERO任务的固定代价。

## 1. 同卡、固定观测测试

输入取已有157帧记录的前64帧，原始TensorFlow预处理、seven动作协议、seed7、规划长度220。
图、linear fusion、RoPE fusion关闭；两组均启用相同的C++树构建、批量Kalman/KV提交和持久化工作区。
每配置先完整预热64帧，再测两遍，共128个动作。加载、预热和设备profiling均不计入下表。

| 执行方式 | BF16模型均值ms | INT8模型均值ms | INT8时延变化 |
|---|---:|---:|---:|
| 分组串行，BF16在前 | 230.676 | 264.229 | +14.55% |
| 分组串行，INT8在前 | 241.160 | 236.920 | -1.76% |
| **逐帧交替，优先使用的对照** | **245.478** | **254.301** | **+3.59%** |

串行顺序改变后结果波动，因此追加逐帧交替测试：两个模型同时驻留同一卡，每个观测分别执行两种精度，
AB/BA顺序随帧和重复轮次交替。该测试临时双模型驻留，**不用于部署显存比较**；省2.339GB的结论仍来自此前单模型完整Goal10记录。
共享服务器的其他CPU/NPU任务仍在运行，未宣称服务器独占，也未将差异全部归为某一项CPU或功耗因素。

三次运行的输入哈希一致，每种精度各自的128个动作、重复和跨运行结果精确一致。
量化与BF16之间允许动作不同；固定观测不会强制两组使用完全相同的历史或验证树。
40对验证结果与forward形状相同的动作，其模型均值为202.583 → 206.219ms，另作辅助证据；
这不等于两组内部所有张量逐位相同。

## 2. 排除平均验证轮数增多

| 逐帧交替测试，128动作/组 | BF16 | INT8 |
|---|---:|---:|
| 总验证轮数 | 316 | 314 |
| 原始posterior接受长度均值 | 1.8165 | 1.8471 |
| 每动作verifier输入token数均值，含prefill和验证 | 967.828 | 965.641 |

当前中间18层量化没有表现为平均接受长度降低、验证轮数增加或验证token总量增加。
此前Goal10日志也一致：每动作验证轮数2.695 → 2.640，原始接受长度1.472 → 1.535。
这些是运行时原始posterior统计，不冒充任务书中已审计padding/compensation的正式ΔL指标。
全32层量化单任务的接受情况较差，不能把它与当前中间18层配置混为一谈。

## 3. 已确认的算子性能代价

从真实BF16运行中取第7层的输入与原始权重，在同一卡对照普通BF16矩阵乘与当前量化矩阵乘。
量化只发生在CPU准备阶段，计时不含权重量化、H2D、设备trace或模型加载。
每个变体预热15次、连续测100次，四轮正反顺序交替，取轮均值的中位数。

| 真实矩阵，M=279 | BF16 µs | 当前group128 INT8 µs | INT8时延增加 |
|---|---:|---:|---:|
| gate投影，K4096/N11008 | 126.513 | 148.345 | **17.26%** |
| down投影，K11008/N4096 | 108.641 | 145.444 | **33.88%** |

输入数值和原始权重相同，INT8变体由该原始权重按生产配置量化。核对有限BF16输出，
相对BF16输出L2误差分别约0.51%和0.64%。这项kernel误差不是LIBERO质量通过证明。
实际forward的M包含约200～300行验证树token，不能仅因batch=1就当作M=1的单token矩阵乘。

独立设备trace确认：

- BF16实际执行`aclnnMatmul_MatMulCommon_MatMulV2`。
- INT8实际执行`aclnnWeightQuantBatchMatmulV2_WeightQuantBatchMatmulV2_WeightQuantBatchMatmulV2`。
- 微基准的gate设备kernel均值126.316 → 146.470µs，down为109.629 → 149.403µs，方向一致。
- 完整单动作trace中有162次量化kernel，即54个量化Linear × 3次forward；没有将该路径偷偷回退为CPU Linear。

**W8A16保留BF16激活，权重INT8储存；算子需要处理量化权重及缩放。省下权重存储不保证计算吞吐更高。**
本机、本形状下，该专用量化kernel实测更慢。已确认的是整体kernel路径的效率差异；
没有采集足以将其内部耗时进一步精确分成反量化、访存、流水线或tiling比例的底层计数器，不能将其中某项单独认定为主因。

量化包装还引入了额外host元数据/view操作：同一动作trace中，`aten::t`多162次、
`aten::reshape`多324次、`aten::view`多324次、`npu::get_npu_format`多162次。
两个完整trace各8526个device kernel，`aten::_to_copy`均877次；未见新增独立大权重拷贝kernel链。
额外host操作是可确认的执行差异，但profiling有自身开销，不能把其嵌套耗时相加后声称解释了全部8.823ms。

## 4. 简单消融结果

- **提前转置并连续化权重/scale**：gate约148.266 → 140.209µs，down约145.153 → 141.926µs；仍慢于对应BF16。
  这只是临时layout探针，生产量化实现没有修改，也未做新LIBERO质量验证。
- **缩放粒度**：group32/64/128及per-channel均测试。per-channel为gate147.974µs、down138.069µs，仍慢于BF16。
  因此不能把问题仅归为group128，也不能承诺改group就能恢复速度。
- **NZ预打包**：该可选探针在初始化时因进程未加入CANN的TBE Python路径报错，未获得有效数据。
  错误及输出保留。生产量化前向不走该初始化分支，不能将此错误当作生产变慢的原因或声称NZ无效。

## 5. 工程判断

当前方案已验证显存收益；同时存在量化Linear的执行代价。要同时保住显存和速度，下一步应针对实际M/K/N选择量化kernel、
布局与量化层，并用相同输入交替计时筛选，再做闭环质量验证。仅打开FlagScale启动入口或更换group大小尚不足以保证加速。
本次核查没有改动生产算子或默认精度，没有重做完整Goal10成功率，也没有计算新的KERV相对native总加速比。

## 6. 复现和证据

串行对照入口：

```bash
KERV_NPU_ID=14 scripts/diagnose_kerv_quantization.sh \
  --output /data/data/KERV/outputs/NEW_AB --order bf16 int8 --samples 64 --repeats 2
# 反向顺序使用新输出目录，并设置 --order int8 bf16。
```

逐帧交替入口是`acceptance/interleave_quantization.py`。与上述launcher使用相同容器、设备及PYTHONPATH，
传入新目录`--output`、`--samples 64 --repeats 2 --trace-model`。
`acceptance/profile_quant_kernels.py`和`acceptance/quant_group_probe.py`读取回放生成的`kernel_inputs.pt`，分别进行trace/layout探针和缩放粒度消融。

- [本轮汇总与一致性检查](../../../outputs/acceptance/quant_diagnosis_20261026T032238Z/diagnosis.json)。
- [逐帧交替原始计时、动作和验证记录](../../../outputs/acceptance/quant_diagnosis_20261026T032238Z/interleaved/results.json)。
- [量化kernel与缩放粒度探针](../../../outputs/acceptance/quant_diagnosis_20261026T032238Z/group_probe/results.json)。
- [先前闭环验证统计复核](../../../outputs/acceptance/quant_diagnosis_20261026T032238Z/prior_closed_loop_analysis.json)。
- [先前显存与Goal10报告](ASCEND_INT8_MEMORY_VALIDATION_20261026.md)。

完整日志、原始CANN trace、真实kernel输入/权重、源码快照及SHA256清单位于同一实验目录。
