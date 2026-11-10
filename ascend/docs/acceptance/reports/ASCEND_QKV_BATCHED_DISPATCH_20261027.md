# 昇腾QKV数值保持、形状选择与派发开销优化

日期：2026-10-27。范围：本机Ascend910_9382、Torch2.7.1/torch_npu2.7.1、固定KERV/SpecVLA和LIBERO Goal自测。

## 结果

未通过全部预设门槛，保留上一版W8A16/BF16混合为推荐候选，BMM＋缓存视图仅供显式实验。

| 候选 | Goal10/init1 | Goal10/init2 | 对上一版平均模型时延变化 | 对BF16平均模型时延变化 | 预设观察门槛 |
|---|---:|---:|---:|---:|---|
| 中间18层拼接QKV | 6/10 | 7/10 | -2.362% | -1.819% | 未通过 |
| 32层BMM，形状限制 | 8/10 | 8/10 | +0.253% | +0.679% | 未通过 |
| 32层BMM＋缓存视图 | 8/10 | 8/10 | -0.305% | +0.106% | 未通过 |

上一版混合量化为8/10、8/10，历史BF16为7/10、7/10。逐初态比较，不用合计数掩盖退化。
最终配置2649个闭环步骤与上一版逐步对比的结果为：全部一致；
动作一致性指相对上一版已量化候选，不代表量化模型与BF16逐值相同。

## 固定的量化机制

MLP policy SHA256为`363e245a3b8265d4165cf18e78b1491fecffa1e8e6019fe2a2506661db3c62d7`。
verifier中间18层7:25，50个投影W8A16/group32；down15/16/19/20保留BF16。
gate/up预打包融合，Ascend Triton SwiGLU保留BF16 SiLU中间舍入。其余MLP、attention、vision和drafter均不新增量化。
本轮没有修改动作协议、七个动作token、接受阈值或Kalman策略。

## 为什么换计算形式

上轮32层BF16 QKV拼接把三个N=4096的GEMM改为一个N=12288的GEMM，数学等价，但NPU结果不逐值相同。
本轮先固定中间18层7:25，避免使用任务标签选层。开发init0前8帧真实q_proj输入覆盖32层、3种行数202/279/295，共96项；
中间18层最大相对L2约7.49e-5，局部合计微基准快7.91%，但完整任务只有6/10、7/10，说明小局部误差不足以保证闭环成功率。

接着将Q/K/V作为三个batch，保持每个投影N=4096。开发真实输入中，202行既有输出差异又更慢；279/295行的32层均逐值一致。
因此进一步对257..320行逐一筛查：32层×64行数×两个真实输入变体，共4096项与原三个F.linear逐值一致。
128项局部计时的合计中位数下降6.87%。形状选择未使用init1/2的成功标签；输入变体由真实279/295行激活按行重复/截断构成，不是4096组独立机器人观测。
最终只允许257..320行使用BMM，其他行仍执行原三个F.linear。

初版BMM完整模型均值略高（对上一版+0.2535%、对BF16+0.6790%）。微基准收益不能直接替代整体时延。
随后缓存最终NPU权重的切片和转置视图，避免fallback每次narrow，并用一次输出拆分代替逐投影view；
这些视图共享唯一packed权重storage，9项NPU测试包含storage指针、原形状fallback和新输入缓存清理验证。
JIT派发探针没有稳定优于eager的局部收益，未作为部署选项；最终manifest的scripted_bmm_qkv_group_count=0。
没有用设备trace逐项分解整体时延变化，因此不把全部差值归因于某一种CPU开销。

QKV在CPU上预打包，原单独Q/K/V不取得NPU存储；仅一次显式搬运packed权重，推理期间没有CPU权重卸载。
这是固定BF16/NPU推理路径，通过现有FlagScale KERV bridge启用。

## 同卡计时

每组独立参考与候选同时驻留在同一chip，157个固定观测完整预热1遍，再测3遍，每帧AB/BA交替。
不计模型加载、预热、诊断捕获；双模型驻留内存不作为部署内存结果。

| 新测参考 | 参考均值ms | 最终候选均值ms | 变化 | P95 ms | verify cycles |
|---|---:|---:|---:|---:|---:|
| 上一版混合量化，chip14 | 294.984210 | 294.085879 | -0.3045% | 509.796 → 504.170 | 1398 → 1398 |
| BF16 runtime，chip10 | 299.885152 | 300.203855 | +0.1063% | 520.566 → 513.793 | 1416 → 1398 |


上一版混合量化三遍分别为：-0.5207%、-0.2675%、-0.1209%。


BF16 runtime三遍分别为：+0.1215%、-0.0611%、+0.2621%。


每组每profile测471个action。同上一版的471个action、verify记录和verifier形状逐值相同，避免树工作量变化掩盖比较。
BF16与量化候选动作/树工作量允许不同，其比较是完整模型收益，不是纯kernel收益。
完整pipeline均值也保存在JSON中。+0.106%的小差值不能单独证明存在显著变慢，也不能据此宣称严格无回退；门槛未在看到结果后放宽。本轮不重新声称相对AR或Naive SD的总加速比，不与历史1.970×相乘，也未证明2.4×。

## 单模型完整回合内存

峰值包含加载和10个完整回合。GB为1e9B，GiB为2^30B。参考为同task/init的历史BF16记录。

| 完整运行 | allocated GB | allocated节省GB / GiB | reserved GB | reserved节省GB | driver采样峰值MB 候选 / BF16 | driver同chip |
|---|---:|---:|---:|---:|---:|---|
| init1，chip0 | 14.950877 | 2.093458 / 1.949685 | 17.314087 | 2.497708 | 19901 / 22284 | 是 |
| init2，chip2 | 14.951180 | 2.088744 / 1.945295 | 17.628660 | 2.189427 | 20193 / 22291 | 否 |

allocated、reserved和驱动HBM是不同口径，不能将allocated省2GB等同于所有口径均省2GB；allocated仍未达到2GiB节省。
驱动HBM为npu-smi原始MB标签、10秒采样，可能漏掉短暂峰值。init2与历史BF16不在同一chip，其driver比较仅供参考。

## 闭环和限制

seed7、TF中心裁剪、seven协议、Goal10、初态偏移1/2、每回合最多220步，所有回合执行状态PASS与任务成功分别核验。
最终逐步检查episode/step、观测与图像哈希、action token、原始和执行动作、done、阈值；完整计数见JSON。
本轮3个有效候选共60个完整回合，BMM两轮是在同一20个task/init上的重复验证，不能作为60个独立质量样本。
这些初态已多次用于开发，不是未见验证集；未证明总体任务精度无损或总体时延不回退。
默认仍为BF16，实验失败记录保留。

初次完整入口把7:25当YAML时间解析为445，两次加载阶段失败、没有评测回合；bridge改用JSON字符串引用后修复，
保留原失败目录，独立重跑完整闭环。测速直接构造配置，不受这一CLI解析问题影响。

## 复现

先用`npu-smi info`选空闲chip，使用新输出目录。显式运行最终缓存BMM候选：

```bash
KERV_NPU_ID=14 KERV_QKV_BACKEND=bmm KERV_QKV_LAYERS=all \
  KERV_LIBERO_INITIAL_STATE_OFFSET=1 \
  scripts/run_kerv_quant_qkv_ascend.sh \
  --output-dir outputs/quant_bmm_new --max-tasks 10 --trials 1 \
  --max-episode-steps 220
```

`KERV_QKV_LAYERS=7:25`仅用于层选择实验，不是最终BMM配置；最终默认all32，形状仅257..320。
量化入口历史名称为`mlp_dynamic_int8`，此policy实际不使用INT8激活。
上一版量化配置直接用`run_kerv_moore_ascend.sh`并指定相同quant policy，见主README。

## 证据与源码

- [最终缓存版结果](../../../outputs/acceptance/qkv_bmm_cached_20261027/reports/qkv_results.json)
- [初版BMM结果](../../../outputs/acceptance/qkv_bmm_guarded_20261027/reports/qkv_results.json)
- [中间18层结果](../../../outputs/acceptance/qkv_middle18_fixed_20261027/reports/qkv_results.json)
- [真实输入筛查](../../../outputs/acceptance/qkv_middle18_20261027/screen/results.json)
- [4096项BMM形状筛查](../../../outputs/acceptance/qkv_middle18_fixed_20261027/bmm_sweep/results.json)
- `inventory/source.tar.gz`与`source_after_manifest.sha256`记录各阶段实际源码；交付源码含后续报告工具和文档，另存delivery_source，未覆盖旧归档。
- 运行目录保留完整CLI、source/weight manifest、逐步记录、operator命中、单模型内存和10s驱动采样；结束后释放本轮NPU任务。
