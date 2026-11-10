# 摩尔 MTT S5000 验收自测执行说明

> **2026-10-28 甲方技术报告：** [适配、推理优化与保精度压缩技术报告](MOORE_TECHNICAL_REPORT_20261028.md)，配套[实验明细](MOORE_TASK_AWARE_QUANTIZATION_20261028.md)、[操作指南](../../src/docs/MUSA_TASK_AWARE_QUANTIZATION.md)及[文档修订清单](MOORE_TECHNICAL_REPORT_20261028.manifest.json)。七类有损候选均未通过抽屉试点；保留 96 投影无损 BF16 存储。本轮与 BF16 均 170 步成功，动作及观测完全一致，allocated 峰值减少 2.107 GB，模型均值延迟增加 19.37%。37 项最终回归通过，988 个实验证据文件已封存；本次文档修订不改动原证据。

下方按日期列出历史追加报告。**从“2026-10-19 验收自测执行记录”开始的流程和 campaign ID 属于该历史批次**；最新部署选择及其指标以以上 2026-10-28 报告为准，不合并不同参考下的倍率与成功率。

> 2026-10-27 精度优先：[实测报告](MOORE_PRECISION_REFINEMENT_20261027.md)、[操作指南](../../src/docs/MUSA_PRECISION_FIRST.md)。96个MLP投影无损BF16压缩，两对完整闭环262/262动作和观测一致，实际allocated峰值少约2.10GB；模型耗时增加20.47%，原INT8策略质量状态另列。

> 2026-10-27 最新：[算子筛选与共享历史报告](MOORE_QUANTIZED_OPERATORS_FOLLOWUP_20261027.md)、[操作说明](../../src/docs/MUSA_QUANTIZED_OPERATORS_FOLLOWUP.md)。同一量化策略三轮 423/423 动作一致，模型 1.1669×、pipeline 1.1600×；原生向量解码收益不足未接入，量化质量门禁仍未通过。

> 2026-10-27继续优化：[量化链路报告](MOORE_QUANTIZED_PIPELINE_REPORT_20261027.md)、[操作指南](../../src/docs/MUSA_QUANTIZED_PIPELINE.md)。基于完整动作剖析接入批量树缓冲，性能和策略精度分开验收。

> 2026-10-27追加：[混合量化算子报告](MOORE_QUANTIZED_KERNEL_REPORT_20261027.md)、[操作指南](../../src/docs/MUSA_QUANTIZED_KERNELS.md)。专用算子已实现，整模型未获稳定提速，默认配置保留。

> 2026-10-26实施追加：[激活校准量化报告](MOORE_CALIBRATED_QUANTIZATION_REPORT_20261026.md)、[操作指南](../../src/docs/MUSA_CALIBRATED_QUANTIZATION.md)。新算法已实测，校准改善和任务精度分别判断；原失败保留。

> 机制设计：[低精度损失量化V2](QUANTIZATION_DESIGN_V2_20261026.md)。激活校准、二阶补偿、BF16保护组及独立验证；目前为设计和静态预算，尚无新精度实测。

> 2026-10-26混合精度追加：[混合量化及保精度存储报告](MOORE_MIXED_QUANTIZATION_REPORT_20261026.md)、[操作指南](../../src/docs/MUSA_MIXED_QUANTIZATION.md)。INT8混合策略和无损存储分别记账，候选权重一致不等于完整动作精度通过。

2026-10-26允许量化后的独立试验见[量化补充报告](MOORE_QUANTIZATION_REPORT_20261026.md)和[操作指南](../../src/docs/MUSA_QUANTIZATION.md)，证据位于`outputs/quantization/20261026T021321Z_w8a16/`。量化候选显存下降，但本轮任务质量未通过；下文2026-10-19 campaign数据作为历史记录保留，不合并统计。

## 2026-10-19 验收自测执行记录

本目录落实 [TestOutline.md](../../TestOutline.md) 中摩尔节点可执行的部分。用户已明确本轮仅负责摩尔机器。A100、昇腾、第三款芯片、智源模型、其他认可平台及独立第三方测试不在本机完成范围，不能由本次自测推断通过。

本轮 campaign：`20261019T072621Z_2f7723f2`。原始证据位于 [outputs/acceptance/20261019T072621Z_2f7723f2](../../outputs/acceptance/20261019T072621Z_2f7723f2/)。最终结果从该目录的 `summary.md`、`reports/local_results.json` 和 `reports/evidence_checks.json` 读取；不要将运行前的大纲或历史 README 当成本轮结果。

本轮面向交付的完整说明见[摩尔节点阶段测试报告](MOORE_TEST_REPORT_20261019.md)。三对闭环仅seed 7一致；seed 17/27存在相同观测下token分歧，不可写成三轮无损加速。证据完整性PASS只检查数据是否齐全、能否复算，不代表优化质量或任务书指标PASS。

## 本轮范围与预登记

- 设备：物理 GPU 0，容器内 `musa:0`，MTT S5000 80 GiB。开跑前 8 张卡均空闲，本轮在同一物理卡串行测速，不干预其他容器。
- 模型：公开 OpenVLA LIBERO-Goal verifier，固定 revision；SpecVLA LIBERO-Goal drafter，完整文件 SHA256 校验。
- 平台：LIBERO 框架的 LIBERO-Goal 套件；只计一个平台候选。
- 主闭环：任务 0、官方初始状态 0；KERV native/optimized 各 seed 7、17、27，每组 3 次，B1/B2 顺序为 AB、BA、AB。
- 这些 seed 为框架随机数 seed。现用官方 `get_libero_env` 固定调用 `env.seed(0)`；不同框架 seed 不等于不同仿真初始状态，不把同一个状态的重复执行包装成跨任务成功率。
- B0：隔离的真实上游 SpecVLA MUSA 移植，固定宽松接受阈值 9；先做 8 步和 CPU/MUSA 小模型数值验证，再执行 seed 7 完整闭环。
- 冒烟及诊断均只执行 8 步；KERV 保留规划时域 220。它们从完整任务 SR 分母中排除。
- 每个完整闭环最多 220 个策略动作，初始稳定化 10 个仿真步骤另计。每个 case 的容器内超时为 1800 秒。

[cases.csv](cases.csv) 是最初不可缩减的预登记清单，其原样副本和哈希保存在 campaign 的 `inventory/`。采集器冒烟修复的唯一重试，以及在独立 SpecVLA 适配可用后加入的 B0 用例，以 `cases_addendum_01.csv`、`cases_addendum_02.csv` 单独冻结，不覆盖原清单。所有失败、未开始项和重试父 ID 保留。

`cases_addendum_03.csv`登记额外观测采集。实际成功回合为141帧，固定输入回放每配置先完整预热一次，再计时三次，共423个动作；每帧使用相同原始观测和采集时的历史动作/token。该实验不执行仿真，不能计算闭环成功率；多worker同时驻留但串行计算，其显存不用于单模型资源对比。协议、逐帧记录和差异保存在campaign的`replays/fixed_input_v1/`。

## 工具与实际路径

宿主机工程 `/root/KERV`，容器工程 `/workspace`，Python `/workspace/.venv/bin/python`。宿主机使用下面的 `PYTHONPATH=src python3 -m ...`；这些统计工具只依赖标准库，不要求宿主机安装 torch。

| 工具 | 作用 |
| --- | --- |
| [preflight.py](../../src/acceptance/preflight.py) | 只读检查路径、任务书哈希、权重、容器、设备与磁盘；不安装或下载 |
| [run_matrix.py](../../src/acceptance/run_matrix.py) | 按冻结 CSV 串行调度；每次新目录、开跑检查空卡、GNU timeout、保存命令/退出码/显存采样 |
| [collect_metrics.py](../../src/acceptance/collect_metrics.py) | 在真实加载/动作/验证点记录耗时、动作、接受事件、内存和诊断传输 |
| [aggregate.py](../../src/acceptance/aggregate.py) | 从所有清单及原始记录重算，包含失败分母、零接受和 nearest-rank P95 |
| [check_evidence.py](../../src/acceptance/check_evidence.py) | 交叉检查动作/运行/采集记录、缺事件、长度、录像及源代码/权重哈希 |
| [record_command.py](../../src/acceptance/record_command.py) | 保存辅助命令、工作目录、完整 stdout/stderr、时间和退出码 |
| [prepare_specvla_port.py](../../src/acceptance/prepare_specvla_port.py) | 从固定上游源码创建独立 B0 MUSA 目录，拒绝覆盖旧移植 |
| [run_specvla_case.py](../../src/acceptance/run_specvla_case.py) | 使用真实上游 B0 模块执行同一 LIBERO 动作协议 |
| [validate_specvla.py](../../src/acceptance/validate_specvla.py) | 核查 B0 核心文件原样保留，执行 CPU/MUSA 接受边界及小模型前向对照 |

CLI 参数检查：

```bash
cd /root/KERV
PYTHONPATH=src python3 -m acceptance.preflight --help
PYTHONPATH=src python3 -m acceptance.run_matrix --help
PYTHONPATH=src python3 -m acceptance.aggregate --help
PYTHONPATH=src python3 -m acceptance.check_evidence --help
```

只读预检及完整权重哈希（在新盘点目录中保存）：

```bash
PYTHONPATH=src python3 -m acceptance.preflight --hash-weights \
  --output outputs/acceptance/20261019T072621Z_2f7723f2/inventory/recheck/preflight.json
```

已有证据目录不得用于重复执行。以下单用例命令适用于该用例尚未运行时；本轮完成后只复算，重新测量须先登记新 ID 和清单。

```bash
PYTHONPATH=src python3 -m acceptance.run_matrix \
  --campaign outputs/acceptance/20261019T072621Z_2f7723f2 \
  --cases docs/acceptance/cases.csv --case-id full_native_s7
```

批量运行/续跑已冻结的 KERV 清单：

```bash
PYTHONPATH=src python3 -m acceptance.run_matrix \
  --campaign outputs/acceptance/20261019T072621Z_2f7723f2 \
  --cases docs/acceptance/cases.csv --resume
```

`--resume` 跳过已经结束的 ID，包括失败项，不把失败静默重跑；已有未结束目录会报错，避免向旧 JSONL 追加。修复后的重试必须在附加清单中登记新 ID 和 `parent_run_id`。runner 当前固定使用物理 GPU 0 的专用容器，不会自动调度到未暴露的其他卡。

结果重算与证据检查：

```bash
PYTHONPATH=src python3 -m acceptance.aggregate \
  --campaign outputs/acceptance/20261019T072621Z_2f7723f2
PYTHONPATH=src python3 -m acceptance.check_evidence \
  --campaign outputs/acceptance/20261019T072621Z_2f7723f2 \
  --check-source-hashes --check-weight-hashes
```

派生 JSON/CSV 可以从原始文件重新生成；不要删除或修改 `runs/`、原始命令、预登记清单或历史失败。汇总工具在正式口径未确认时始终保留 `formal_acceptance_status=BLOCKED`，不会因为某次平均值有利就自动判课题通过。

## 指标定义与采集开关

只有设置 `KERV_ACCEPTANCE_CONTEXT` 指向 run 的 `context.json` 才安装采集器。现有 `run_musa.sh` 默认不设置该变量，原部署入口继续可用。新增任务/初始状态选择也只在显式传参时生效。

| 记录 | 本轮定义 |
| --- | --- |
| `model_seconds` | 现有 predict_action 调用前至 MUSA 同步完成；不含此前 CPU processor；起点没有设备预同步 |
| `pipeline_seconds` | 输入观测哈希完成后，经图像提取/裁剪、processor、输入传输、预测、反馈/阈值控制和执行动作转换，直到 env.step 前 |
| `env_seconds` | 实际 env.step 起止墙钟 |
| `wall_seconds` | 每步观测哈希开始至 env.step 返回；不含随后采集器日志刷盘 |
| `closed_loop_wall_seconds` | 首个已稳定观测到终止 step 及逐步日志完成，含仿真和步间日志；不含模型加载、初始稳定化和 MP4 编码 |
| `launcher_wall_seconds` | 宿主机 docker exec 启动至子进程退出，包含加载、仿真和视频等开销 |
| 首步 | 全部保留；另列排除首步的分布，不能只展示较快口径 |
| P95 | 排序后第 `ceil(0.95*N)` 项，1 起始；不采用插值 |

观测哈希覆盖 incoming RGB、末端位置、末端四元数与夹爪状态，并纳入 shape/dtype。哈希用于追溯输入一致性，不表示这些 proprio 字段实际进入 OpenVLA 或 Kalman 计算。模型加载与 allocator 采集从加载前开始，加载后和配置后的基值分别保留。

每次验证独立记录候选接受前缀，根 token、临时 bonus、补偿及 padding 不计入 `accepted_prefix_length`，全拒绝为 0。它是 posterior 的候选接受值；到最终七维动作边界可能截断，所以另存 `committed_accepted_prefix_length`。`compensation_length` 是直接补全的尾部长度；KERV 完整运行另外记录实际 Kalman 预测尾部数与候选/重复回退数，避免把全部补全都算成模型真正接受。

普通计时轮不额外回传整条候选，`proposed_length=null` 并保留包含 padding 的容量及说明；诊断轮回传实际选中候选并排除负 padding 得到真实 proposed length。所有运行保留逐事件接受值和原有 `runtime_stats.accept_lengths`，可交叉核对。

## 显存、传输和诊断的解释

内存同时采集 `allocated/reserved` 当前值与峰值。加载前 reset 一次，在首动作结束保存冷启动峰值后，再 reset 建立后续窗口。全运行初始化峰值取两个窗口的最大值；稳态峰值只使用首步后窗口，不拿增量冒充总量。CPU RSS 以 `/proc/self/statm` resident pages 计。设备 CLI 的 100ms 目标采样另存原始输出，实际采样间隔由 monotonic 时间差重算；采样峰值不能代替 allocator 瞬时峰值。

设备总显存与单进程显存是不同口径，当前 CLI JSON 的设备值不自动归为模型进程占用。统计同时保留 GB/GiB；D06 未确认前不择优选择 allocated/reserved。

诊断轮启用 TorchDispatch 跟踪跨设备 `_to_copy`、`copy_` 和隐式标量读取，保存源/目标设备、张量形状、两侧 logical bytes、方向、同步完成耗时及采集器自身额外复制标记。为计量完成时间会同步，因此**诊断时延不进入正式性能均值**。首动作另尝试 CPU+MUSA profiler，保存 Chrome trace 和未支持项。

这些是张量层的 logical payload，不是经过 DMA/驱动计数器确认的物理总线 byte；自定义 C++/Triton/驱动内部复制可能未覆盖。暂不将它判成 `<100M/s` 正式通过证据。原始 copy 时间为诊断同步区间，也不等于无观测扰动的纯传输 kernel 时间。模型初次加载传输不计入该逐步诊断窗口。

## 三组身份与适配限制

B1 为现有 KERV native，保留历史动作估计、补偿与动态阈值；B2 仅开启已有通过筛选的算子/运行时优化。B0 来自固定 SpecVLA 上游 commit，自己的 `cnets.py`、HF `utils.py` 和 `modeling_prismatic.py` 与原件哈希一致，`eagenerate` 保留其原始草稿/验证/停止逻辑。

B0 MUSA 兼容改动、原件/移植 hash 和补丁见 `ports/specvla_musa/port_manifest.json`、`musa_compatibility.patch`。改动包括设备选择、BF16 eager、CPU 加载 checkpoint、共享 verifier embedding、离线 processor、MUSA 同步计时及采集钩子。无 KERV Kalman/动态阈值/投影融合替换。

完整 B0 是本机 MUSA 上的参考实现移植，不等于已复现任务书 A100 的 65 秒实验。B0 使用隔离 harness；B1/B2 使用既有 FlagScale harness，逐步 JSON 日志量等存在差别。报告同时给 policy、model、闭环及 launcher 耗时，并披露这一限制；不拼接历史比值形成正式总加速。

## 2026-10-27 稳定性补充

本机原生 Softmax 偶发概率错误已通过相同输入及独立微算子复现，新增只重算异常行的修复入口。三轮固定输入回放423/423动作及token与特定历史轮次一致，真实推理修复74行，回归62通过/1跳过。见[定位与修复报告](MOORE_SOFTMAX_DIAGNOSIS_20261027.md)。该项不关闭量化质量门禁或此前IndexSelect非法地址，也不替代本页原任务书验收结果。

## 当前功能映射及材料

详细实际分工、函数与反馈来源见 [placement.md](placement.md)。本次不为了凑任务书的位置描述改变补偿算法。D01–D08 文件均为 `UNCONFIRMED`，不填写专家签字或第三方身份。

原源码、未提交/未跟踪改动和依赖完整快照保存在 campaign `inventory/source_before.tar.gz`；原 Git diff、环境包清单、镜像与任务书原件另存。每个 run 有自己的 `source_manifest.sha256`，后续仅统计工具改动另记版本，不能覆盖成“实际运行始终使用最终源码”。

完整闭环期间推理数学代码未调整；统计、调度及报告工具曾演进。`inventory/source_objects/`补充内容寻址版本，`source_object_coverage.json`明确列出未保存实体的较早版本，主要为分析工具与早期冒烟版本；不能声称每个历史中间版本均可完整重建。初始快照与最终源码包分别保留。

回归加入零接受、严格阈值边界、失败分母、缺失/重复事件、NaN/Inf、拒绝位置、补偿前缀与七维停止、回合历史重置和阈值范围测试。初次新增测试文件的语法错误日志也保留，修正后的有效回归从 `operations/regression_mechanisms_corrected/` 读取。

尚缺的正式材料由 [gap_register.md](gap_register.md) 登记；完整课题不得据此自测签认通过。

## 最终结果索引

固定输入回放：观测哈希423/423相同，动作/token422/423相同，门禁FAIL；模型/pipeline均值比2.275119×/2.198777×仅为描述性统计。独立检查命令：

```bash
cd /root/KERV
PYTHONPATH=src python3 -m acceptance.check_replay \
  --campaign outputs/acceptance/20261019T072621Z_2f7723f2
```

最终回归29项：28通过、1因驱动不支持Graph跳过。最终证据检查见`operations/final_evidence_check_scoped/`；原检查失败日志与具体范围修正保留，详见阶段报告第10节。证据PASS不代表动作保持或正式指标PASS。
