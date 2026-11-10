# 摩尔节点 Softmax 随机错误定位与精度保护修复

日期：2026-10-27。设备仅使用本机物理 GPU0，MTT S5000 80 GiB。

## 1. 定位结论与边界

本轮已定位到 **当前 torch_musa/muDNN 原生 Softmax 执行路径的偶发概率错误**。同一输入可输出非零遮罩概率，整行概率和超过 1。错误可在移除模型、量化权重、树构建及仿真后独立复现。因此这项中间张量分歧不能解释成量化舍入误差，也不能归因于 INT8 解码。

定位到的是原生算子执行路径；尚未区分闭源库、编译器或驱动内部的具体实现原因。本地补丁绕过已复现的错误，不是厂商运行时升级。

此前 `IndexSelectVectorKernel` 非法地址仍是独立未关闭问题。本轮 Softmax 定位不能证明该非法地址与之同源，也不能单凭未再次崩溃宣布其已修复。历史最终动作分歧也没有全部逐次追溯到同一算子。

## 2. 环境与不变量

| 项目 | 值 |
| --- | --- |
| 驱动 | 3.3.5-server |
| torch | 2.9.1 |
| torch_musa | 2.9.1+a18d871 |
| Triton MUSA | 3.2.0 |
| Transformers | 4.40.1 |
| 容器 | kerv-musa-runtime |
| 权重 | 既有 OpenVLA LIBERO-Goal verifier、SpecVLA drafter |
| 量化策略 | 75 个 MLP 投影，原 GPTQ/INT8 + BF16 保护组；本轮未改变策略或重新校准 |
| 策略 SHA256 | `4261780057e3b2f937b8f528aea94de0ca8237e2553074689e6539b83410d61a` |
| Attention | 既有 eager Llama attention；未启用 Graph、SDPA 或新的 attention 算法 |

没有升级依赖、驱动或镜像。没有重置 GPU，也没有停止其他进程。新证据保存在独立目录，既有量化与链路优化实验目录保持原样。

## 3. 定位过程及证据

证据目录：`outputs/stability_diagnosis/20261027T052303Z/`，下文路径均相对此目录。

1. `targeted_async_v1/`：对历史发生最终动作分歧的第 61、122 帧各重复 20 次；保持原帧编号、观测、历史动作及 Kalman token 前缀。只在动作完成后复制诊断快照到 CPU。40 次预测中有 5 次首次 hidden state 分歧，变化集中于单行；本批最终动作未改变。
2. `targeted_inner_v2/`：进一步记录归一化、attention、MLP 输出。40 次预测捕获 3 次首次分歧，均位于 verifier `self_attn.output`，其归一化输入完全一致。Embedding 索引在动作结束后检查，没有提前插入 CPU 索引检查同步。
3. `targeted_attention_v3/`：拆分 QK 矩阵乘、Softmax、AV 矩阵乘。第 122 帧重复 20 次，捕获 3 次首次分歧，均为 **Softmax 输入逐位一致而输出不同**。第一处失败位于 verifier 第 18 层。
4. `softmax_trace_evidence.json`：用 CPU 独立复算 `.pt` 快照，验证输入一致性、异常列、概率和及参考误差。此复算不导入 MUSA 算子。

第一份有效证据为 `targeted_attention_v3/difference_r3_f122.pt`：

| 观测 | 值 |
| --- | --- |
| 输入 | BF16，`[1, 32, 254, 537]`，两次逐位一致 |
| 输出 | FP32 |
| 异常行 | batch 0 / head 18 / query 110 |
| 正常行概率和 | 0.9999999404 |
| 异常行概率和 | 6.8241701126 |
| 遮罩输入值 | 约 `-3.3895e38` |
| 正常遮罩概率 | 0 |
| 异常遮罩概率 | 约 0.48534748，共 12 个变化位置 |
| 正常输出相对 CPU 最大误差 | 约 `5.96e-8` |
| 异常输出相对 CPU 最大误差 | 约 0.48534748 |

另两份证据为 `difference_r8_f122.pt`、`difference_r16_f122.pt`。变化发生在不同层、不同 attention 行，不能把故障限制为某一固定层的权重问题。

上述诊断加入了快照与挂钩，不能用其耗时作为性能指标。

## 4. 脱离整模型的独立复现

脚本：[benchmark_musa_softmax.py](../../src/acceptance/benchmark_musa_softmax.py)。从实际失败快照提取输入，**先在 CPU 转成 FP32**，随后在 GPU 单独调用 Softmax。这也排除了推理时 BF16 → FP32 转换作为该次独立失败的必要条件。

| 实验 | 原生 Softmax | 完整 Triton 参考 | 保守修复 |
| --- | --- | --- | --- |
| `softmax_micro_v1`，各 2000 次 | 3 次概率异常 | 0 次概率异常 | 未运行 |
| `softmax_micro_v3`，各 2000 次，填充/释放同尺寸暂存区 | 2 次概率异常 | 0 次概率异常 | 0 次最终概率异常，实际修复 2 行 |
| v3 已检查输出相对 CPU 最大绝对误差 | 0.49028295 | `2.3842e-7` | `2.9802e-7` |

“概率异常”指遮罩概率非零或行概率和误差超过 `1e-5`。这里的出现次数是本次具体实验观察，不是跨机器故障率估计。

保守修复在 v3 中有 4 个相邻迭代间 FP32 非逐位一致事件，来自异常行进入/退出重算路径；其误差仍在上表范围。不能将“概率检查通过”写成“2000 次所有 FP32 输出逐位一致”。

初版保守内核出现 MUSA 编译器不支持 `llvm.is.fpclass.f32` 的链接错误；失败的 `guarded_softmax_micro_v2`、`safe_softmax_unit_v1` 均保留。改为有序区间比较后，`safe_softmax_unit_v2` 的 5 项针对性测试通过。失败目录不覆盖、不计为通过。

## 5. 为什么选择保守修复

完整 Triton Softmax 的计算精度达到微算子参考要求，但改变了正常行的数值路径。`safe_full_replay_v1/` 的三轮 141 帧自身完全一致，却每轮只有 120/141 帧与此前普通原生 Softmax 对照的 token 一致。故未将其选为精度保持部署方式。

当前补丁：[musa_safe_softmax.py](../../src/runtime_opt/KERVRuntimeOptimization/musa_safe_softmax.py)。保守路径先调用原生 Softmax，随后用设备端 Triton 内核逐行检查：

- 概率和位于 `[0.99999, 1.00001]`。
- 不含负概率。
- 强遮罩位置，即输入小于 `-1e30`，概率为零。

检查通过的行不写回，保留原生输出 bit；异常行用 FP32 `exp(x - max(x)) / sum(exp(x - max(x)))` 重算，并递增设备端修复计数。行归约参照本地 FlagGems 的 Softmax 算法，接入本地 `KERVRuntimeOptimization`，不代表该补丁已被上游 FlagGems 或 FlagScale 接收。

实现仅替换 32 个 eager Llama verifier attention 的私有 functional 命名空间，复用 Transformers 原 forward 代码。未全局注册或覆盖 `torch.nn.functional.softmax`；KV、RoPE、QK/AV GEMM、dropout 及输出投影保持原实现。单元测试验证了正常行 bit 保持、异常行修复、尾维、非连续输入、dtype、空输入、非有限输入及局部接入。

这是针对已复现概率异常的防护，不是对任意后端错误的完整检测器。修复仅覆盖推理，训练路径仍走原生实现。新的正常/异常检查增加了 GPU 工作，性能应以带修复入口的实测为准。

## 6. 完整回放结果

`guarded_full_replay_v1/` 使用完整 141 帧 capture，预热 1 轮后测量 3 轮，总计测量 423 次动作预测；预热不计入下表耗时。每帧重放固定观测、相同历史动作和 Kalman token 前缀，不推进仿真。

| 指标 | 实测 |
| --- | --- |
| 三轮相对第 0 轮 token 一致数 | 141 / 141 / 141 |
| 三轮相对第 0 轮 raw action 一致数 | 141 / 141 / 141 |
| 三轮相对历史普通 Softmax 对照第 0 轮 token/raw action 一致数 | 各 141 / 141 / 141，即 423/423 |
| 相同的剩余原生参数 | 全部 SHA256 一致 |
| 相同的量化解码权重 | 75/75 个逐位一致 |
| 实际 Softmax 覆盖 | verifier 32/32 层 |
| Softmax 调用总数，含预热 | 67,584 |
| 实际修复异常概率行数，含预热 | 74 |
| 模型耗时均值 / 中位数 / P95 | 306.941 / 308.069 / 474.073 ms/动作 |
| Pipeline 耗时均值 / 中位数 / P95 | 320.351 / 321.257 / 487.611 ms/动作 |
| 该次完整回放非法地址 dump | 0 |
| 全部回归 | 63 项：62 通过，1 项因当前驱动不支持 Graph 捕获跳过 |

复算文件为 `guarded_replay_analysis.json`，脚本为 [check_guarded_replay.py](../../src/acceptance/check_guarded_replay.py)。它独立验证观测 hash、三个完整轮次、参数和解码权重、32 层实际调用及修复计数、实现文件 SHA256 与启动前清单一致、退出码及没有本次 dump。实际调用证据为 `guarded_full_replay_v1/safe_softmax.jsonl`。

此前未加防护的树优化入口三轮模型均值为 295.854 ms，当前 306.941 ms，历史顺序对照约多 3.75%；pipeline 分别为 307.595 与 320.351 ms，约多 4.15%。**这不是重新配对计时的性能验收**，不能据此给修复开销做严格归因，也不能直接把此前 1.637× 作为本修复入口的倍率。

这里的 423/423 表示一个固定输入序列的重复及特定历史轮次对照通过，不表示 INT8 与 BF16 动作全部一致，更不表示 LIBERO 任务成功率达到 100%。

## 7. 复现与使用

在宿主机 `/root/KERV`，使用已就绪的容器、权重与原策略：

```bash
export KERV_CALIBRATED_POLICY=/workspace/outputs/calibrated_quantization/20261026T045030Z_v2/model_refined/policy.json
./src/scripts/run_musa_calibrated_safe.sh \
  --max-episode-steps 8 \
  --output-dir /workspace/outputs/client_guarded_smoke_unique
```

输出目录必须新建。此命令是操作示例，不是闭环质量通过声明。默认 BF16 入口保持原样；保守修复与原量化策略共同接入的是显式实验入口。

独立微算子复现：

```bash
docker exec \
  -e PYTHONPATH=/workspace/src/runtime_opt:/workspace/src:/workspace/src/openvla \
  kerv-musa-runtime /workspace/.venv/bin/python \
  /workspace/src/acceptance/benchmark_musa_softmax.py \
  --input /workspace/outputs/stability_diagnosis/20261027T052303Z/targeted_attention_v3/difference_r3_f122.pt \
  --output /workspace/outputs/client_softmax_micro_unique \
  --repeats 2000 --poison-reuse
```

运行前确认 GPU0 没有其他任务。微算子输出记录各后端失败迭代、误差及修复计数。`operations/` 保存已登记命令、日志、退出码；首个微算子执行为事后记录，明确标注于 `softmax_micro_execution_note.json`。

## 8. 验收边界

本次修复不等于量化精度验收通过。量化的闭环任务成功率、套件指标、混合量化精度损失、修复后显存峰值及正式配对性能仍需各自实测。以前的峰值显存和 1.64× 链路收益不能直接沿用为新修复入口的指标。

原 `IndexSelect` dump 位于上一轮封存目录 `outputs/quantized_pipeline/20261027T044701Z_prefetch/diagnostics/tree_same_input/`。其参数对应 32064×4096 的 BF16 embedding 表及 8 个索引；dump 没有保存可验证的完整索引值。同步索引检查后未复现，仅能说明检查样本合法，不能排除异步数据或原生索引内核问题。

追加 `draft_embedding_stress_v1/`：提取 20 份真实 drafter TopK 输入，独立重复原生 TopK → 索引 gather → 8-token BF16 embedding，共 10,000 次。在索引生成与 embedding 之间不加入 CPU 边界检查；每 100 次统一检查结果。全部索引和所用行编码 embedding 检查通过，未产生非法地址 dump。该实验没有复现完整模型的内存布局、全部算子交互和权重，故结论仍为 **未复现、未关闭**。

源码记录说明：`source_guarded/` 保存保守修复及相关测试/工具实体，`source_guarded_manifest.json` 保存哈希；正式修复回放的启动前哈希与实际实现文件哈希已核对。完整 Triton 旧版本根据补丁记录重建于 `source_pure_triton_reconstructed/`，与其启动前 SHA256 完全匹配，明确标注为事后重建。更早编译失败版本保留命令、错误日志及演进说明，不能将最终源码冒充所有中间版本。

本报告保留上述未关闭项，避免把算子修复、量化质量、动作一致性及任务成功率混为一个结论。
