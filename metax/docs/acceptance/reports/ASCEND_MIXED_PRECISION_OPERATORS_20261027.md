# 昇腾混合量化与保留舍入的融合算子

## 本轮结论

新增60个完整闭环回合和四组完整157帧三遍回放后，当前推荐的显式省内存候选为
**中间18层MLP W8A16/group32，15/16/19/20层down保留BF16，预打包gate/up融合，加按行SwiGLU融合**。

| 最终候选 | BF16参考 | W8A16/BF16混合 |
|---|---:|---:|
| 全程allocated峰值GB，init2 | 17.039925 | 14.954441 |
| 全程reserved峰值GB，init2 | 19.818086 | 17.878221 |
| 同卡157帧×3遍平均模型时延ms | 293.300 | 290.277 |
| 同卡模型时延中位数ms | 278.449 | 274.302 |
| 同卡模型时延P95 ms | 504.565 | 489.535 |
| Goal10/init1成功数 | 7/10 | 8/10 |
| Goal10/init2成功数 | 7/10 | 8/10 |

**allocated峰值节省2.085483520GB（1.942257881GiB）；平均时延降低1.0306%，三遍均降低。**
达到本轮预先记录的2,000,000,000bytes目标；按2GiB口径则尚未达到。
两初态合计14/20→16/20，达到本轮“两个初态成功数分别不低”的观察性标准。
这不证明总体精度无损或成功率提高，任务与初态已经用于探索，默认仍保留BF16。
本轮没有重新验证总体2.4×目标，也不将量化收益与历史KERV加速比相乘。

## 实验目的与配置

目标是在当前昇腾机器上节省至少2GB的完整模型allocated峰值，尽量不增加模型时延，
同时检查LIBERO闭环成功率。BF16默认配置未改变。

本轮测试两种新配置，只处理verifier第7至24层的MLP，其他层、注意力、视觉编码器和drafter保持BF16：

| 配置 | gate/up | down | 激活 |
|---|---|---|---|
| mixed | W8A8，共享平滑参数和输入动态量化 | W8A16/group32 | down输入BF16 |
| weight_only_rounded | W8A16/group32 | W8A16/group32 | 全部BF16 |
| weight_only_hybrid4 | W8A16/group32 | 除15/16/19/20层BF16外，W8A16/group32 | 全部BF16 |

这两种配置与上一轮保留4个BF16 down投影的hybrid4分别记录，不把候选名称混用。
量化参数使用已冻结的init0激活校准，没有重新调整KERV接受阈值或Kalman策略。
第三种配置沿用上一轮敏感性选择的4层，不根据本轮任务成败重新选层。
init1/init2已经参与此前试验，因此本轮是探索性的配对评估，不能称为未见测试集。

## 算子实现

代码位于 `ports/moore_ascend/openvla/experiments/robot/`：

- `ascend_dynamic_quantization.py`：在CPU加载阶段完成量化和K,N布局预打包；gate/up共用一次矩阵乘；
  down可选择动态INT8、保留BF16或W8A16/group32。CPU转换后再搬到NPU，不在设备保留原BF16权重副本。
- `ascend_mlp_ops.py`：显式选择的Ascend Triton SwiGLU，按行分块，合并SiLU和逐元素乘。
  SiLU后先舍入为BF16，再做乘法、存储BF16输出，保留原图的中间舍入位置。
- 入口继续使用现有FlagScale KERV adapter和 `scripts/run_kerv_moore_ascend.sh`。
  这是本地适配层中的新算子，没有把它描述成上游FlagScale已经提供的量化功能。

当前环境实际安装了Ascend版Triton；新算子单独显式开启，没有批量启用未经验证的FlagGems替换。
原生W8A8部分调用 `npu_dynamic_quant`、`npu_quant_matmul`；
W8A16部分调用 `npu_weight_quant_batchmatmul`，没有在Python中每次完整反量化权重。
兼容入口 `KERV_NPU_QUANT=mlp_dynamic_int8` 表示加载逐投影policy；实际精度由policy决定，
并非该入口下所有投影都必须使用INT8激活。W8A16候选的manifest记录激活为BF16，
没有动态激活量化投影，也不调用原生SwiGLU动态量化路径。

## 数值与局部性能验证

- 六项NPU单元测试通过，包括混合精度加载、权重身份校验、共享gate/up的新输入处理、
  保留BF16投影，以及W8A16完整MLP在1、33、279行时自定义/原生激活的输出相等检查。
- 遍历65,280个有限BF16输入，以up=1检查SiLU：0处差异。
  该检查覆盖SiLU映射，不能单独证明所有乘法组合、NaN/Inf或整个模型逐位相等。
- layer7真实279行输入上，新SwiGLU与分离SiLU/乘法输出相等；MLP数值误差在融合前后保持相同。

同一微基准会话中，每项交替测6组，每组100次，预热15次，统计组均值的中位数：

| 完整MLP局部配置 | ms | 对BF16输出的relative L2 |
|---|---:|---:|
| BF16 | 0.432772 | 0 |
| mixed，原生分离激活 | 0.448946 | 0.017579 |
| mixed，按行融合激活 | 0.419104 | 0.017579 |
| W8A16，原生分离激活 | 0.459265 | 0.008135 |
| W8A16，按行融合激活 | 0.438361 | 0.008135 |

首版采用扁平索引和整除/取余，完整MLP约2.169ms，明显更慢，未部署到闭环。
按行分块消除了该版的访问和索引开销。以上是局部真实输入筛选，不能直接当作整模型加速比。

## 整模型评估口径

每候选在Goal10的init1/init2分别跑10个完整回合，seed7、seven动作协议、TensorFlow裁剪和220步上限。
历史BF16只用于同任务/同初态的质量对照，不拼接历史时延计算加速比。

时延独立采用完整157条固定观测，先预热一遍，再测三遍；同一芯片上BF16与候选逐观测交替AB/BA，
每个模型保留独立历史。每配置471个计时动作，包含模型路径内的同步，排除加载、预热与profiler。
量化会改变动作、树和验证轮数，因此报告的是该输入流下完整策略成本，不把全部变化归因于kernel吞吐。
平均值、中位数、P95和每遍变化均需检查。

时延诊断临时双模型驻留的显存不用于节省内存结论。内存取单模型完整闭环的allocated全程峰值，包含加载和首次执行；
allocated、reserved和驱动HBM占用是不同口径。GB为10^9 bytes，GiB为2^30 bytes。

## 结果

| 同卡157观测×3遍计时 | BF16均值ms | 候选均值ms | 均值变化 | BF16中位数ms | 候选中位数ms |
|---|---:|---:|---:|---:|---:|
| mixed，分离激活，chip14 | 296.474 | 293.268 | −1.08% | 294.230 | 306.046 |
| mixed，融合激活，chip8 | 298.616 | 297.098 | −0.51% | 296.985 | 308.451 |
| weight_only_rounded，chip14 | 288.967 | 290.644 | +0.58% | 293.285 | 299.552 |
| weight_only_hybrid4，chip10 | 293.300 | 290.277 | −1.03% | 278.449 | 274.302 |

weight_only_hybrid4三遍均值变化为−0.944%、−1.021%、−1.126%；
P95从504.565ms降到489.535ms（约−2.98%）。验证轮数1416→1398，
每动作verifier输入token均值1109.777→1101.134；工作量也有变化，不能把1.03%全部归因于融合kernel。

各行只使用该行自己的同卡BF16参考。分离/融合激活两行不是同一芯片同一时段的直接消融，
不能把两行差值当作kernel的固定代价。融合版本在完整471个动作中与原生mixed的动作、验证记录和输入形状均相同；
它保持了这组输入流的行为，但没有显示明确的整模型额外加速。

mixed完整闭环：init1为5/10，init2为8/10；BF16为7/10、7/10。
合计13/20低于14/20，不是精度保持配置。init2完整allocated峰值14,682,171,392bytes，
与BF16 init2的17,039,924,736bytes相比省2.358GB/2.196GiB；两者分别在chip2和chip0。
相同chip0的init1记录为BF16 17,044,335,616、mixed 14,682,378,752bytes，省2.362GB。
这些是完整单模型运行的框架峰值，不是临时双模型测速内存。

全W8A16融合版闭环完成：init1为9/10、init2为5/10，合计14/20。
虽然合计与BF16相同，init2退化2个任务，不能称为质量稳定。init2完整allocated峰值为14,786,812,928bytes，
比同初态BF16低2.253GB/2.098GiB；两者分别在chip6和chip0。

保留4个BF16 down投影的版本完成20个回合，init1/init2各8/10。
init1仅task5从失败变成功；init2的task3、task6从失败变成功，task5从成功变失败。
完整逐任务动作数和结果均保存在机器可读报告中，不能称为BF16轨迹等价。

该版本init2完整allocated峰值14,954,441,216bytes，在chip2；BF16 init2为17,039,924,736bytes，在chip0。
补充的同chip0、同init1对照为BF16 17,044,335,616、候选14,954,033,152bytes，省2.090302464GB。
该同卡记录的驱动HBM峰值采样也有下降，但10秒一次的采样不是连续精确峰值；最终省内存结论使用框架全程allocated峰值。
模型唯一权重/scale存储节省2,113,536,000bytes，与完整峰值的差异来自实际执行的中间张量和workspace。

最终policy SHA256：`363e245a3b8265d4165cf18e78b1491fecffa1e8e6019fe2a2506661db3c62d7`。
manifest确认50个W8A16投影、4个保留BF16投影、0个动态INT8激活投影，
实际使用 `triton_rounded` SwiGLU；不是量化未命中后的BF16回退。

| 全部新候选 | init1 | init2 | 完整峰值节省GB（init2） | 各自同卡平均时延变化 |
|---|---:|---:|---:|---:|
| mixed，原生激活 | 5/10 | 8/10 | 2.358 | −1.08% |
| 全W8A16，融合激活 | 9/10 | 5/10 | 2.253 | +0.58% |
| **W8A16 + 4 BF16 down，融合激活** | **8/10** | **8/10** | **2.085** | **−1.03%** |

mixed融合激活版本仅单独跑了回放，没有新增闭环成功率结论；回放471动作与原生mixed的记录全部相等。
这些比例是各自BF16对照下的完整路径成本，不能作为不同芯片候选的绝对时延排序。

## 复现

以下为当前本机环境。运行前检查物理chip空闲，并使用未存在的输出目录。
启动器负责进入 `kerv-ascend` 容器；policy环境变量使用容器内路径。

```bash
KERV_NPU_ID=14 KERV_NPU_QUANT=mlp_dynamic_int8 \
  KERV_NPU_QUANT_POLICY=/data/data/KERV/outputs/acceptance/weight_only_hybrid4_20261027/policy.pt \
  KERV_LIBERO_INITIAL_STATE_OFFSET=1 \
  scripts/run_kerv_moore_ascend.sh \
  --output-dir outputs/w8a16_hybrid4_new --max-tasks 10 --trials 1 \
  --max-episode-steps 220
```

`KERV_NPU_QUANT=none` 为BF16参考。上述命令是显式候选配置，不改变默认精度。
重新导出相同方案时，在容器内项目根目录执行，输出文件必须不存在：

```bash
.venv/bin/python -m acceptance.export_mixed_precision \
  --source outputs/acceptance/quant_noslow_20261026T054934Z/calibrated_swiglu/policy.pt \
  --sensitivity outputs/acceptance/quant_accuracy_20261026T045119Z/calibrated/sensitivity.json \
  --scheme weight_only --silu-backend triton_rounded \
  --preserve-down-layers 15 16 19 20 --output outputs/new_policy/policy.pt
```

省略 `--preserve-down-layers` 即全W8A16中间18层MLP候选；`--scheme mixed` 为W8A8 gate/up加W8A16 down。
导出器核对校准权重身份、group32裁剪策略，加载器在替换前核对每个原始权重SHA。
导出文件名等序列化细节可能改变文件SHA，逐投影内容及来源信息仍可核对。

## 证据位置

- `outputs/acceptance/mixed_precision_20261027/`：原生mixed闭环、原生/融合激活回放、真实输入微基准和有限BF16穷举检查。
- `outputs/acceptance/weight_only_fused_20261027/`：W8A16融合版本闭环、计时、policy和独立源码快照。
- `outputs/acceptance/weight_only_hybrid4_20261027/`：4个down投影保留BF16的W8A16融合版本。
- 三轮均保留原始失败回合和固定配置。新算子阶段另存源码快照，早期快照不被覆盖。

机器可读汇总：

- `mixed_precision_20261027/reports/mixed_native_results.json`
- `mixed_precision_20261027/rounded_replay_parity.json`
- `weight_only_fused_20261027/reports/weight_only_results.json`
- `weight_only_hybrid4_20261027/reports/hybrid4_results.json`

上述路径相对于 `outputs/acceptance/`。源码归档中的文件内容已逐一核对SHA，
各阶段使用自己的快照，不能用后续工作树覆盖早期实验的源码身份。
