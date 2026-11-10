# 摩尔成功版与当前昇腾版的实现对照

日期：2026-10-15。来源为用户提供的
`imports/moore1-20261015-8d69709e/KERV-moore1-code-chats-20261015-moore1-8d69709e.tar.gz`。
SHA256 校验通过，解包到独立的 `extracted/` 目录，未覆盖归档中的工程或会话数据库。

## 1. 摩尔版本做到了什么

归档的工作区 README 和会话记录给出：MTT S5000、BF16、Goal task0、初始状态0、seed7、
220步规划与动作上限。原生与优化配置均在第170步成功，170条七维动作和最终动作token一致。
排除首步的模型动作预测均值：0.522852 → 0.230204秒，即 **2.2713×**。
包含首步的均值比为2.0465×；验证周期488/489，内部过程并非逐操作相同。

该2.2713×比较的是 **KERV原生MUSA配置 → KERV优化MUSA配置**，不等同于上游论文的
Naive VLA+SD基线。计时排除CPU图像预处理/processor和仿真；样本是同一条完整成功轨迹，
各配置一遍，排除首步。当前昇腾报告采用157个固定观测、全序列预热后重复两遍，并分别记录
model/pipeline，比较时必须标清这些区别。

归档排除了原始outputs和录像，因此这里的摩尔数值引用其交付记录；没有在昇腾机器上独立
重算摩尔日志。实现与配置本身可直接核对。

## 2. 真正启用的优化差异

| 项目 | 摩尔最终实现 | 上一版昇腾实现 |
|---|---|---|
| C++草稿树/验证树 | 补齐缺失源文件，两个构建路径实际启用 | 原源文件缺失，关闭 |
| 验证路径 | `verify_max_paths=0`，保留所有路径，包括Kalman分支 | 配置48条上限 |
| node buckets | 不强制固定bucket | 沿用224–320的bucket配置 |
| 验证树/草稿树同分 | 稳定排序保留祖先闭包 | 原非稳定topk |
| 批量Kalman | 启用，独立一致性检查 | 启用 |
| 批量KV提交 | 启用，记录447次 | 关闭 |
| 解码工作区 | 启用，记录170次 | system optimization关闭 |
| 持久化drafter KV | 启用，只分配1次，无fallback | A100条件阻断NPU，关闭 |
| 控制缓冲 | 设备/pinned CPU缓冲复用，489次 | NPU未接入ControlBuffer，只接了其他传输路径 |
| 三值fused control bundle | 关闭，使用pair缓冲 | 启用 |
| QKV/Gate-Up | 按MUSA实测行数选择，原生合并GEMM | NPU候选有动作差异，关闭 |
| RoPE | 自定义Triton，保留BF16逐乘法舍入 | 关闭，未移植该修正版 |
| Add-RMSNorm | 局部融合，限定行数 | 关闭 |
| 全局FlagGems RMSNorm | 关闭；单项快但完整任务失败 | 最终也关闭FlagGems候选，保留native verifier hooks |
| Graph | 驱动不支持，关闭 | NPU三阶段图启用 |
| 常驻verifier KV、量化、action head | 均关闭 | 均关闭 |

因此上一版昇腾还没有移植摩尔成功的整套运行时。接入FlagScale launcher、注册公共算子、
运行NPUGraph与移植这些缓存/批量/融合路径，是不同的工作。上一次的1.009×只能说明当时
所测的native算子+图配置基本持平，不能用于判断摩尔优化路线在昇腾上的潜力。

摩尔没有做所有优化项的完整闭环消融，因此不能把整体2.27×单独归因于某一项。

## 3. 算法参考也被我改动过，需要先对齐

| 项目 | 摩尔成功参考 | 上一版昇腾 |
|---|---|---|
| Drafter RMSNorm | 原`F.rms_norm` | NPU默认改为手写FP32 normalize→BF16→gamma |
| Drafter embedding | 丢弃checkpoint冗余embedding，共享verifier | 默认保留checkpoint embedding |
| KF生成/解码 | 保留原`root+7`，历史前7、解码后7的行为 | 改为`root+6`，历史/解码统一 |
| SD结束边界 | 保留原`>6`条件 | 改为`>= action_dim-1` |
| 强制KF | 原每4步条件 | 先检查enabled，周期默认4 |

其中KF开关修复对Naive SD基线有明确意义，但摩尔的对照本身都运行KERV并启用KF。
历史/动作切片不一致在代码上确实存在；摩尔明确选择保留它以保持原轨迹，外部仍输出七维动作。
我把统一7-token语义混入昇腾适配后，算法参考就变了。即使这样的改动有理由，也应另开实验，
不能将其结果直接视为摩尔成功算法的芯片迁移结果。

后续需要保留明确区分的参考配置：先复现摩尔成功版的数学和动作行为，再比较只改变运行时
的优化配置。之前的昇腾结果保留，作为其明确标注的配置结果，不冒充该对照。

## 4. 值得直接复用的代码

1. `csrc/kerv_tree_ops.cpp`：CPU拓扑构建，设备无关；token打包通过ATen分派。
2. `cnets._select_draft_tree_nodes`：BF16同分时稳定排序，先保留祖先。
3. `ControlBuffer`与启动层的设备判断：不能仅改YAML，需去掉NPU遇到的CUDA/A100硬限制。
4. `adaptive_rotary_fusion.py`：每个乘法先舍入到输入类型，再相加；不能直接FP32一口气融合。
5. `kerv_libero_goal_musa_optimized.yaml`：优化组合和分组验证方法可参考，行数白名单需在NPU重测。

MUSA Triton内核不能仅把设备字符串替换为NPU就宣称适配完成。应先迁移可复用的树、KV与
工作区部分，再逐项验证RoPE/Add-RMSNorm/GEMM的数值和实际收益。

## 5. 本次已落地与检查范围

- 原样补回 [C++树算子](../openvla/experiments/robot/csrc/kerv_tree_ops.cpp)，与归档内容哈希一致。
- 在当前Ascend容器补齐Ninja编译工具，完成CPU扩展实际编译。
- [15项检查](../tests/test_tree_operator_port.py)全部通过：4种树规模×3种路径上限，与现有
  Python规划器的节点、retrieval、mask、position和统计精确一致；检查双batch打包/填充和非法拓扑。
- 日志：[tree_operator_tests.log](../outputs/moore_comparison/tree_operator_tests.log)。
- 源码对照：[implementation_diff.patch](../outputs/moore_comparison/implementation_diff.patch)、
  [哈希记录](../outputs/moore_comparison/source_comparison.json)。

复查命令（Ninja在项目venv内，需加入PATH）：

```bash
docker exec -w /data/data/KERV -e KERV_DEVICE_TYPE=cpu -e MAX_JOBS=2 \
  kerv-ascend bash -c \
  'export PATH="/data/data/KERV/.venv/bin:$PATH"; exec python -m pytest tests/test_tree_operator_port.py -q'
```

以上为最初CPU检查范围。随后已完成下述隔离NPU移植和真实权重/闭环验证。

## 6. 已完成的 Ascend 隔离移植

工作代码：`ports/moore_ascend/`；启动：`scripts/run_kerv_moore_ascend.sh`。
经 FlagScale 原有 `run.py` 和 KERV adapter 启动，不另造推理引擎。

- 新增NPU设备/同步路径，解除CUDA/MUSA/A100限制；C++草稿树与验证树实际执行。
- 使用全部路径、稳定同分排序；批量Kalman、批量KV提交、持久drafter KV、解码workspace、
  pinned控制缓冲均有真实命中。线性/RoPE融合与图默认关闭。
- 首先保留摩尔原动作协议：原生与优化各220步均失败，但动作220/220相同，排除运行时差异。
- 明确另设 `KERV_ACTION_PROTOCOL=seven`，原生/优化同时修正root+6、KF偏移、SD终止条件。
  这不是摩尔原轨迹复现，而是在同一修正参考下验证运行时。
- 修正协议后，LIBERO Goal task0/init0/seed7，两组均115动作成功，115/115动作及token完全一致。
  原生排除首动作506.257ms、优化263.306ms，描述性闭环比1.923×。
- 原始F.rms_norm在32768个随机BF16值上与CPU结果精确一致；保持原函数和共享embedding。
  不再把原生NPU RMSNorm当作已证实错误，也不把手写RMS替代加入默认。
- 原摩尔协议的157观测回放曾测得模型2.011×、pipeline1.988×，动作精确一致；
  **该旧协议闭环失败，这组速度只能作诊断**。修正协议正式回放见主README的最新表。
- NPU RoPE新增片上gather实现，单算子正确，16观测模型动作32/32一致；整模型未显示进一步
  受益，因此保留实验开关，不加入默认配置。旧MUSA内核直接适配则明显变慢，已淘汰。

配对证据：[paired_episode.json](../outputs/moore_seven_checks/paired_episode.json)。
最终命令、完整回放和已知范围：[README_ASCEND.md](../README_ASCEND.md)。


## 7. 修正协议的正式完整回放

`outputs/moore_seven_runtime_full/benchmark.json`：seven协议，物理chip4，157连续观测，
完整预热1遍、测量2遍，314/314优化动作精确一致，最大差0。
模型514.554→261.260ms，**1.970×**；pipeline520.354→266.628ms，**1.952×**。
C++树/Kalman/KV提交/控制缓冲有实际命中，drafter KV分配一次且fallback为0。
图和融合候选关闭，采用原F.rms_norm；该比值仅相对相同数学配置的KERV native。
