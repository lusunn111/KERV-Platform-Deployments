# 昇腾精度优先量化：原型试验结果

日期：2026-10-26。实验：`quant_accuracy_20261026T045119Z`。仅当前昇腾，通过FlagScale运行KERV。

## 结论

**本轮优先推荐中间18层MLP全部使用W8A16/group32。** 两组官方初始状态下，它均为9/10成功，
对应BF16均为7/10。独立初态的单模型allocated峰值节省 **2.218GB（2.066GiB）**；
另行同卡交替测速，模型时延 **245.002 → 250.610ms，增加2.29%**。

真实激活校准得到的混合group配置在独立初态为8/10，额外节省约69MB，但本轮质量未超过全group32。
本次实现并验证了逐投影policy、激活采集与分组/裁剪搜索；**没有宣称完整GPTQ已实现**。
每配置独立验证仅10回合，小样本不能证明总体成功率损失严格≤2个百分点或满足正式5%损失边界。

## 1. 试验顺序与隔离

运行前登记三组开发对照和四组独立初态验证，共70个完整回合，每回合最多220次策略动作。
同一10个Goal任务、seed7、原TensorFlow预处理、seven协议、相同KERV阈值/Kalman/树路径与运行时优化。
未改动作接受门槛以弥补量化误差。

| 阶段 | 初始状态 | 配置与物理chip |
|---|---|---|
| 校准/开发 | 各任务官方init0 | BF16采集chip0、原group128 chip2、group32 chip14 |
| 独立初态验证 | 各任务官方init1 | BF16 chip0、group128 chip2、group32 chip4、混合policy chip14 |
| 单独时延诊断 | 固定64观测 | chip14逐帧AB/BA交替，BF16/group32及BF16/混合各一组 |

独立集按初始状态/轨迹划分，任务身份与校准集相同；不能称为未见任务泛化。
混合配置只使用init0激活与预先登记的选择规则生成，没有根据init1结果调整。
BF16/init0有额外激活采集，该组时延仅为诊断，不用于加速比。
其1426个动作token与此前BF16记录精确一致；原group128的1498个动作token也全部复现。

服务器上的其他任务保持运行；没有宣称整机独占。模型作业均在开跑前检查目标chip空闲，未中止其他用户任务。

## 2. 任务结果

| 配置 | 开发init0 | 独立init1 | init1总动作数 |
|---|---:|---:|---:|
| BF16参考 | 7/10 | 7/10 | 1419 |
| 原group128，中间18层 | 7/10 | 7/10 | 1434 |
| **group32，中间18层** | **9/10** | **9/10** | **1233** |
| 校准混合，24个group32 + 30个group128 | 未运行该配置闭环 | 8/10 | 1308 |

### 独立init1逐任务结果

单元格为“任务结果/动作数”；失败均保留220步上限。

| Task | BF16 | group128 | group32 | 校准混合 |
|---|---|---|---|---|
| 0 中间抽屉 | 失败/220 | 失败/220 | 成功/130 | 成功/121 |
| 1 碗放炉上 | 成功/79 | 失败/220 | 成功/90 | 成功/87 |
| 2 酒瓶放柜顶 | 成功/89 | 成功/95 | 成功/99 | 成功/96 |
| 3 打开上层抽屉并放碗 | 成功/162 | 成功/164 | 成功/159 | 失败/220 |
| 4 碗放柜顶 | 成功/90 | 成功/88 | 成功/93 | 失败/220 |
| 5 推盘至炉前 | 失败/220 | 成功/150 | 成功/143 | 成功/168 |
| 6 奶油奶酪放碗中 | 失败/220 | 成功/120 | 成功/123 | 成功/86 |
| 7 开炉 | 成功/87 | 成功/69 | 成功/100 | 成功/72 |
| 8 碗放盘上 | 成功/80 | 成功/88 | 成功/76 | 成功/74 |
| 9 酒瓶放酒架 | 成功/172 | 失败/220 | 失败/220 | 成功/164 |

group32相对BF16在任务0/5/6新增成功，在任务9失去一次成功；净增加2次。
开发init0中group32唯一失败为task0。成功数改善不表示动作逐token相同，也不保证每个任务都改善。

## 3. 显存

以下来自独立init1的**单模型全程allocated峰值**，包含加载与首次执行，未reset峰值。
比较在本机不同chip上完成，使用相同软件、模型和用例协议；未将reserved池或时延诊断双模型内存当作部署收益。

| 配置 | allocated峰值bytes | 峰值GB | 对BF16节省GB / GiB |
|---|---:|---:|---:|
| BF16 | 17,044,335,616 | 17.044 | 参考 |
| group128 | 14,700,679,680 | 14.701 | 2.344 / 2.183 |
| group32 | 14,826,449,408 | 14.826 | **2.218 / 2.066** |
| 校准混合 | 14,757,175,296 | 14.757 | 2.287 / 2.130 |

group32超过2GiB节省的余量约67MiB，结论限于本轮模型、batch和运行范围；新增workspace需重新测量。
原checkpoint保持只读，CPU完成权重量化后再上NPU，没有保留量化MLP的BF16影子权重。

## 4. 同卡时延

chip14，固定64个已有观测，每组完整预热64次后测两遍，共128次动作。
两个模型临时同时驻留，逐观测交替AB/BA，排除加载、预热、激活采集与设备profiler。
各精度保留自己的历史与验证树；是固定观测对照，不强制相同内部数值或工作量。

| 同卡配对配置 | 本组BF16均值ms | INT8均值ms | INT8时延变化 | 验证轮数BF16 / INT8 |
|---|---:|---:|---:|---:|
| group32 | 245.002 | 250.610 | **+2.29%** | 316 / 314 |
| 校准混合 | 242.674 | 252.455 | +4.03% | 316 / 322 |

group32原始posterior接受长度1.8165 → 1.8344，混合为1.8165 → 1.8075。
两个时段分别与各自BF16配对，不拼接均值计算新的总加速比。
本轮没有重测KERV相对native的总加速比，闭环动作数减少也不等同于kernel加速。

## 5. 校准机制与实际选择

本地FlagScale已有GPTQ adapter，但当前环境未安装`llmcompressor`、`compressed_tensors`与`msmodelslim`。
本轮实现了可直接使用现有NPU算子的训练后量化原型，未修改依赖环境：

1. 在BF16的10条init0轨迹采集早期、中期和后期激活，每投影216行，共54个MLP投影。
2. 针对每投影，在真实NPU算子上搜索group128/32与clip比例1.0/0.99/0.975，用相同输入和原始权重计算相对输出平方误差。
3. 以误差降低为排序，最多24个投影升级group32；其余保留group128。每次升级的scale额外存储相同。
4. 导出的policy记录模块、group、clip与原BF16权重SHA256；部署加载时验证身份，并在CPU完成量化。

最终选择24个group32投影：down15、up8、gate1。所有投影的最佳clip均为1.0，裁剪没有带来额外收益。
平均局部相对平方误差由4.1789553e-5降至3.6019398e-5，相对原group128降低 **13.81%**。
这项局部改善不足以证明任务质量；本轮独立验证8/10，完整group32为9/10。

混合机制的下一步应增加动作token分布、夹爪与接受行为的敏感性评分，并验证多个初始状态；
当前显存目标下，全group32已经满足本轮资源门槛，因此本轮优先采用实测质量更好的group32候选。

## 6. 可用命令

先确认目标chip空闲，输出目录使用新名字。以下复现独立init1设置：

```bash
KERV_NPU_ID=14 KERV_NPU_QUANT=mlp_int8 \
  KERV_NPU_QUANT_LAYERS=7:25 KERV_NPU_QUANT_GROUP_SIZE=32 \
  KERV_LIBERO_INITIAL_STATE_OFFSET=1 \
  scripts/run_kerv_moore_ascend.sh \
  --output-dir outputs/quant_g32_new_run --max-tasks 10 --trials 1 \
  --max-episode-steps 220
```

`KERV_LIBERO_INITIAL_STATE_OFFSET=0`使用开发集初始状态。`KERV_NPU_QUANT=none`可运行BF16参考。
混合实验使用相同入口，增加下面的**容器内路径**：

```bash
export KERV_NPU_QUANT_POLICY=/data/data/KERV/outputs/acceptance/quant_accuracy_20261026T045119Z/calibrated/policy.json
```

policy逐模块覆盖group设置。每投影原始权重哈希不匹配会拒绝加载，不应将该policy直接用于其他模型权重。

## 7. 验证、实现与证据

- 70回合均完成程序执行，七维动作、有限值、token历史与动作协议通过结构检查；任务失败保留在分母。
- 两项新增NPU policy测试通过，包含部分投影选择、混合group、实际前向与权重身份错误拦截。
  首次小形状测试因K等于group不符合CANN合法范围失败，已修正测试形状并增加提前校验；原失败日志保留。
- 校准CPU加载路径通过实际权重加载检查。原BF16和group128的完整init0动作均复现此前记录。
- 新增逐投影policy、激活采集、校准搜索、独立初始状态偏移、配对计时和报告工具；没有使用CUDA量化后端。

主要实现：

- [量化与policy加载](../../../ports/moore_ascend/openvla/experiments/robot/ascend_quantization.py)。
- [真实激活采集](../../../acceptance/quant_calibration_capture.py)、[校准选择](../../../acceptance/calibrate_quant_policy.py)。
- [完整结果报告工具](../../../acceptance/report_accuracy_pilot.py)、[独立阶段调度](../../../acceptance/continue_accuracy_pilot.py)。

原始证据：

- [本轮自动报告](../../../outputs/acceptance/quant_accuracy_20261026T045119Z/summary.md)。
- [完整任务、显存与同卡时延JSON](../../../outputs/acceptance/quant_accuracy_20261026T045119Z/reports/results.json)。
- [冻结的混合policy](../../../outputs/acceptance/quant_accuracy_20261026T045119Z/calibrated/policy.json)、[校准统计](../../../outputs/acceptance/quant_accuracy_20261026T045119Z/calibrated/summary.json)。
- [实际命令与阶段记录](../../../outputs/acceptance/quant_accuracy_20261026T045119Z/orchestration.json)。
- [原参考策略精确复现](../../../outputs/acceptance/quant_accuracy_20261026T045119Z/inventory/previous_baseline_parity.json)。

同目录保存源码快照、模型哈希、原始日志、视频、逐步指标和SHA256清单。
