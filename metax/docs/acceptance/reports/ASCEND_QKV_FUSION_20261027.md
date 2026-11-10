# 昇腾混合量化叠加 verifier QKV 融合

## 当前结论

QKV融合获得了额外速度收益，但缓存修复前的20回合为init1 **6/10**、init2 **9/10**，
合计15/20，上一版为8/10、8/10、合计16/20。init1退化，未通过本轮预先记录的质量门槛。
保留上一版W8A16/BF16混合作为较稳妥的已测候选；QKV启动器为显式实验入口。

排查并修复QKV打包的加载阶段驻留问题：单纯释放Torch缓存只能改善reserved，无法消除设备侧额外占用。
最终改为CPU预打包再搬到NPU，完整验证已完成。速度获得增量，但质量未通过门槛，未替换上一版推荐。

## 最终CPU预打包版

| 新测同卡对照 | 参考均值ms | 新版均值ms | 均值变化 | 参考P95 ms | 新版P95 ms |
|---|---:|---:|---:|---:|---:|
| 上一版量化→新版，chip14 | 294.275 | 281.860 | −4.219% | 521.006 | 497.806 |
| BF16→新版，chip10 | 297.516 | 285.368 | −4.083% | 515.398 | 495.207 |

相对上一版三遍−4.140%、−4.114%、−4.402%；相对BF16三遍−4.385%、−3.750%、−4.110%。
两组均值、中位数和P95都下降，分别使用本组同卡参考。

| 完整单模型峰值 | init1 | init2 |
|---|---:|---:|
| BF16 allocated bytes | 17,044,335,616 | 17,039,924,736 |
| 新版 allocated bytes | 14,953,822,208 | 14,951,961,600 |
| allocated节省GB / GiB | 2.090513 / 1.946942 | 2.087963 / 1.944567 |
| BF16 reserved bytes | 19,811,794,944 | 19,818,086,400 |
| 新版 reserved bytes | 17,930,649,600 | 17,320,378,368 |
| reserved节省GB | 1.881145 | 2.497708 |
| 驱动HBM采样峰值，BF16→新版，工具标注MB | 22,284→20,487 | 22,291→19,900 |

allocated全程峰值达到2e9bytes目标，但不足2GiB；reserved和驱动HBM也有下降，
init1的reserved节省不足2GB，不能宣称所有口径、所有轨迹均严格省2GB。
驱动峰值每10秒采样，有漏掉短暂峰值的可能；init1对照同chip0，init2为历史chip0对当前chip2。
内存包括加载和首次执行；临时双模型驻留测速内存没有用来计算节省。

最终20回合仍为6/10、9/10。CPU版与原32层QKV阶段的1539+1205=2744步，
观测hash、图像hash、token、raw/executed action、done和阈值全部相等；
两组回放各471动作的输出、验证记录和输入形状也全部相等。
加载优化没有引入新的质量变化，原QKV融合对上一版的退化仍然存在。
机器报告 `recommend_new_candidate=false`、`population_quality_loss_bound_proven=false`。

本轮共完成60个回合：原32层、释放Torch缓存、CPU预打包各20；后两轮是同一20条轨迹的重复，
不能作为60个独立质量样本。另有范围错误和NPU复制两组未完成尝试，均保留，未计入完成数。

以下保留此前阶段数字用于追溯，不作为最终CPU预打包版速度。

## 中间阶段：仅释放Torch缓存

| 缓存修复后的同卡对照 | 参考均值ms | 新版均值ms | 均值变化 | 参考P95 ms | 新版P95 ms |
|---|---:|---:|---:|---:|---:|
| 上一版量化→QKV叠加，chip14 | 299.620 | 288.711 | −3.641% | 526.618 | 511.455 |
| BF16→QKV叠加，chip10 | 304.285 | 294.071 | −3.357% | 520.333 | 514.873 |

相对上一版三遍分别−3.532%、−3.705%、−3.686%；相对BF16分别−3.110%、−3.634%、−3.332%。
两组中位数也下降。两组新版各471个动作的输出、验证记录和输入形状与修复前全部相等。
共享主机与不同实验时段有波动，每组只使用自己的同卡参考，不比较跨卡绝对ms，不挑选旧阶段较大的收益作为最终数字。

## 本轮目标

上一轮显式候选采用50个W8A16/group32投影、4个BF16 down、融合gate/up和保留BF16舍入的SwiGLU，
Goal10 init1/init2各8/10，allocated峰值节省约2.085GB。
本轮固定该policy，进一步复用FlagScale KERV的BF16 QKV融合接口。

预先记录的观察性标准：两个初态成功数分别不低于上一版8/10，完整峰值相对BF16省至少2e9bytes，
新做的同卡回放相对上一版和BF16都不增加平均时延。
同任务/初态已反复用于探索，不能据此证明总体精度无损。

## 实现

- `run_kerv_libero.py` 原先对量化模型关闭整组Linear融合。现允许显式的QKV-only配置，要求gate/up、
  SwiGLU和add-RMSNorm的额外融合allowlist均为空，量化MLP继续走其原实现。
- 调用既有 `KERVRuntimeOptimization.adaptive_linear_fusion`，将每层BF16 Q/K/V打包成一次矩阵乘；
  对象限定为 `model.base_model.language_model`，不通过隐藏维度推断模型归属。
- 限定输入4096、flattened rows 1至512；其他行数使用原生分投影计算。
  manifest和运行时断言确认覆盖恰好32个verifier attention模块，没有drafter模块。
- packing替代原Q/K/V权重，不在模型中额外保留一份BF16影子权重。完整allocated峰值仍需包含打包过程。
- NPU每层打包后显式释放旧Linear列表引用，再调用 `torch.npu.empty_cache()` 回收其缓存块。
  该操作仅在加载阶段执行；forward、权重数值和矩阵乘形状不变，不在动作循环中清缓存。
- 最终NPU入口在CPU上先替换原Q/K/V投影，打包后只将唯一packed权重搬到NPU，
  原分投影不会先占用设备内存。再次启用FlagScale融合接口时识别已有group，不重复打包。
  入口和回放均断言32层全部为CPU预打包；权重常驻NPU，动作循环没有CPU权重搬运。
- 新增显式启动器 `scripts/run_kerv_quant_qkv_ascend.sh`，默认引用上一版混合量化policy。
  原启动器默认BF16不变；使用新启动器时仍可显式指定其他量化policy。

量化policy SHA保持 `363e245a3b8265d4165cf18e78b1491fecffa1e8e6019fe2a2506661db3c62d7`。
QKV融合由独立配置控制，因此仅比对policy SHA不足以判断两次运行相同；同时保存Linear融合manifest、hits及源码快照。

## 数值与局部筛选

使用checkpoint中layer7真实QKV权重，以及已捕获的MLP激活作为形状/分布代理。
这不是实际attention输入捕获；整模型回放才使用真实模型路径。
每项交替测6组、每组100次、预热15次，组均值中位数如下：

| M | 分离QKV ms | 融合QKV ms | 输出不相等元素数 | 最大relative L2 |
|---|---:|---:|---:|---:|
| 1 | 0.063910 | 0.043314 | 0 | 0 |
| 8 | 0.064680 | 0.043712 | 13 | 0.00003993 |
| 232 | 0.123287 | 0.103302 | 437 | 0.00003513 |
| 279 | 0.135611 | 0.132348 | 594 | 0.00003562 |
| 320 | 0.126681 | 0.113172 | 630 | 0.00003577 |

融合矩阵乘的数值不完全相等，不能复用原闭环成功率；本轮重新执行20个完整回合。
六项NPU单元测试通过，新增检查确认BF16 attention可融合而量化MLP对象及输出保持原状。
新启动器shell语法及FlagScale dry-run通过，生成配置为512个行数、其余融合allowlist为空。

另外筛选SwiGLU的512/1024/2048/4096/8192/16384六种block，8/232/279/320行均无数值差异，
但同形状时延接近波动，未修改原4096分块参数。
另测 `npu_grouped_matmul`：在所测五个形状下都仍有输出差异（1至716个元素），
且320行时0.127662ms慢于分离投影0.126536ms；没有消除数值风险，因此未接入本轮模型。
保留三次原生Linear的TorchScript主机调度原型，在五个形状下输出相等，
但232/279/320行时约0.122/0.137/0.127ms，与原生相当，未接入模型。

## 首次范围错误与修复

首次按输入维度4096筛选时额外命中drafter的一层，回放的32层断言发现实际33层并终止。
两个闭环进程只终止本轮自有进程，保留部分日志，不作为完成回合或性能结果。
修复为明确传入verifier子模型并增加生产入口层数断言。

- 首次尝试：`outputs/acceptance/qkv_fusion_20261027/`。
- 修复后正式运行：`outputs/acceptance/qkv_scoped_20261027/`。
- 加载缓存修复复测：`outputs/acceptance/qkv_cachefix_20261027/`，另存源码快照和配置。
- 逐片NPU复制尝试：`outputs/acceptance/qkv_memcpy_20261027/`，诊断确认未消除设备占用差额后终止自有测试，保留部分记录，不作为完整回合。
- 最终CPU预打包：`outputs/acceptance/qkv_cpu_pack_20261027/`，独立源码快照、配置和完整复测。

## 整模型评估

质量：Goal10，init1/init2各10完整回合，seed7，seven协议，TensorFlow预处理，220步上限；
与冻结的上一版候选和BF16按相同任务/初态配对。保留所有失败，未调KERV接受阈值和Kalman参数。

时延：每组先预热完整157观测，再测3遍，逐观测AB/BA交替、独立模型历史。
分别在单芯片上测上一版→新版、BF16→新版，两组都有自己的新测参考，不拼接历史时延。
量化和QKV舍入可能改变动作及验证轮数，因此报告的是整个策略成本，不能把全部时延变化都归因于kernel吞吐。

回放JSON沿用旧key `bf16`/`int8`。若 `reference_kind=quantized_policy`，`bf16`实际表示上一版量化候选；
`display_names`和`reference_policy_sha256`给出其身份。旧的BF16报告器已增加断言，防止将此类回放误写为BF16参考。
专用报告器 `acceptance/report_qkv_fusion.py` 使用明确的参考身份。

## 缓存修复前的结果（保留追溯）

三遍回放完成，每配置471个计时动作：

| 新做的同卡对照 | 参考均值ms | 新版均值ms | 均值变化 | 参考P95 ms | 新版P95 ms |
|---|---:|---:|---:|---:|---:|
| 上一版量化→QKV叠加，chip14 | 299.681 | 288.216 | −3.825% | 539.993 | 507.953 |
| BF16→QKV叠加，chip10 | 297.067 | 283.769 | −4.476% | 513.401 | 487.745 |

相对上一版三遍分别−3.945%、−3.928%、−3.599%；相对BF16三遍分别−4.747%、−4.268%、−4.413%。
两组中位数也降低：297.378→289.655ms、294.048→286.901ms。
两行各有自己的同卡新测参考，不能将两行的绝对时延作跨卡排序。

相对上一版时验证轮数1398→1389；相对BF16为1416→1389。
上一版与新版471个动作中仅342个完全相等，验证记录396个完全相等；
即使固定观测相同，历史和验证树也发生了变化，不是逐位等价或严格固定工作量的加速。

20个正式闭环回合完成：init1 task3、task9由上一版成功变失败；init2 task5由失败变成功。
BF16同任务/初态为7/10、7/10，不能仅用合计15/20高于BF16的14/20来掩盖init1的下降。

初版init2 allocated峰值14,956,782,080bytes，比BF16 17,039,924,736低2.083GB；
但reserved峰值19,979,567,104bytes，高于BF16 19,818,086,400bytes。
因此继续修复缓存累积，没有只报告allocated收益后停止。

## 缓存修复验证

旧Q/K/V权重分别占用较小块，打包需要新的较大连续块；旧块释放后留在NPU缓存池，逐层累积。
在旧引用释放后逐层回收，避免这些闲置块与新packed权重同时长期占用reserved。
数学计算没有变化；六项NPU单元测试在修复后再次通过。

另跑相同init1/init2的20回合，观测、图像、token、raw/executed action、done与阈值逐步相等；
两组回放各471个动作也相等。质量仍为6/10、9/10，不能把重复轨迹当新的独立质量样本。

但驱动HBM仍然偏高，因此增加相同物理chip8的阶段诊断。下表中设备used来自
`torch.npu.mem_get_info()`的total−free，Torch reserved是另一口径，不能相互替代：

| 阶段 | Torch reserved GB | 设备used GB |
|---|---:|---:|
| NPU打包前的原分投影 | 14.028 | 14.489 |
| NPU cat打包后，尚未forward | 13.894 | 17.456 |
| NPU逐片copy打包后，尚未forward | 13.894 | 17.455 |
| **CPU预打包搬入后，尚未forward** | **13.894** | **14.353** |
| NPU cat路径首次action后 | 15.741 | 19.209 |
| **CPU预打包路径首次action后** | **15.741** | **16.352** |

连续NPU复制与cat都有缺口，不能简单归因为cat单算子问题；`empty_cache`也没有消除它。
数据把问题定位到设备上的临时权重替换生命周期。内部具体持有者尚未直接观测，不能称为已确认的驱动或ACLNN泄漏。
CPU预打包避免先搬原分投影，再在NPU重新分配packed权重；打包后设备used降低约3.10GB，首次action后降低约2.86GB。
这组对照中Torch计数相同，也说明仅报告Torch峰值会漏掉这部分驻留。

CPU预打包另做完整闭环、同卡计时和数值复现检查；七项NPU测试通过，包括打包权重CPU→NPU后逐值一致、
最终device转移和再次启用接口不重复打包。最终结果见本文开头及机器可读报告。

## 复现入口

确认物理chip空闲并选择新输出目录后，在本机项目根目录运行：

```bash
KERV_NPU_ID=14 KERV_LIBERO_INITIAL_STATE_OFFSET=1 \
  scripts/run_kerv_quant_qkv_ascend.sh \
  --output-dir outputs/qkv_quant_new --max-tasks 10 --trials 1 \
  --max-episode-steps 220
```

对照上一版时，用相同的 `KERV_NPU_QUANT`/`KERV_NPU_QUANT_POLICY` 环境变量，
改用 `scripts/run_kerv_moore_ascend.sh`，默认不启用QKV融合。

已有证据可在宿主机重新生成最终报告：

```bash
python -m acceptance.report_qkv_fusion \
  outputs/acceptance/qkv_cpu_pack_20261027 \
  --previous outputs/acceptance/weight_only_hybrid4_20261027 \
  --cachefix-reference outputs/acceptance/qkv_scoped_20261027 \
  --require-cpu-prepack
```

最终机器报告：`outputs/acceptance/qkv_cpu_pack_20261027/reports/qkv_results.json`。
阶段诊断原始数据：`qkv_cachefix_20261027/memory_stages{,_copy,_cpu}/stages.json`，
相对于 `outputs/acceptance/`。各阶段固定配置、源码归档、权重清单、失败日志与最终校验清单均保留。
