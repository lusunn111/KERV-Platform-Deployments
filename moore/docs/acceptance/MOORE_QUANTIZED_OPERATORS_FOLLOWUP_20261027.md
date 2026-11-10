# 摩尔 S5000：量化算子筛选与共享历史优化实测

日期：2026-10-27；Campaign：`20261027T071216Z`。

## 完整回放结果

最终候选继续使用已有 row1024 混合权重解码和原生 BF16 GEMM，合并树分支中重复的 Kalman 历史估计。新增原生 MUSA 向量解码已实现并验证，但实测收益不足，未接入最终候选。

相对相同量化 policy、相同 host tree builder 和相同 guarded Softmax 的对照，三轮合计模型倍率 **1.1669×**，pipeline 倍率 **1.1600×**。动作 token 完全一致 **423/423**，raw 动作一致 **423/423**，夹爪一致 **423/423**。

| 实现 | 模型平均 ms/动作 | 模型 P50 ms | 模型 P95 ms | pipeline 平均 ms/动作 |
| --- | ---: | ---: | ---: | ---: |
| safe_rows 对照 | 307.0975 | 310.0410 | 486.6842 | 318.8484 |
| shared_history 候选 | 263.1793 | 281.1825 | 355.7811 | 274.8675 |

| 轮次 | 动作/token 一致 | raw 最大误差 | 模型倍率 | pipeline 倍率 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 141/141 | 0 | 1.1735× | 1.1658× |
| 2 | 141/141 | 0 | 1.1648× | 1.1586× |
| 3 | 141/141 | 0 | 1.1623× | 1.1556× |

合并倍率按全部帧总耗时之比计算。参考不是 BF16 native 或独立 SpecVLA；不能将本轮倍率与历史不同实验相乘后汇报。

**原量化策略相对 BF16 的质量验收仍为 NOT_PASSED。** 固定回放一致只检查本轮优化带来的额外动作差异，没有重新验证完整任务成功率。

## 实现与等价条件

同一次树扩展的所有分支使用相同历史、过程噪声、观测噪声和窗口。内置预测函数先计算完整七维 Kalman 估计，最后才按剩余长度切片；有效历史存在时，前缀不参与估计。候选只计算一次完整估计，再复用相同 tail。原标量递推、FP64 Python float 运算顺序、round 和 token 范围裁剪均沿用。

空历史仍按各分支自身末 token 回退；自定义 kalman_predict_tokens 钩子可能依赖前缀或带副作用，改为逐分支调用。超出 action_dim 的 tail 也回退。树节点、候选顺序和原有 mask 不裁剪、不重排。

实际候选命中 **1548 次**批量入口，共处理 **75852 条**分支，含预热。最终配置验证 tree_builder=true、static_runtime=false、kalman_batch=true。两个 worker 使用相同 75 个投影的 policy、原生参数哈希及解码权重 SHA。

## 算子候选筛选

| 候选 | 实测决定 |
| --- | --- |
| 二维分块解码、联合 gate/up 解码与 GEMM | 真实 decoded weights 和联合输出逐位一致，但耗时增加，未接入 |
| 原生 MUSA vector8 解码 | 原生 .mu kernel 已编译运行；下表为真实 layer15 热微基准，收益不足，未接入 |
| 单 BF16 工作区复用 | 75 投影轮换在 M279/303 仅约 1% 收益，未接入最终候选 |
| 共享内置 Kalman 历史估计 | 本轮完整回放候选；保留原有解码和 GEMM |

原生 kernel 每线程读取 8 个 INT8 值，共享 scale/保护组映射；保护组直接向量复制原 BF16；其他值按 FP32 缩放后舍入 BF16。使用当前 MUSA stream，编译不在计时内；非对齐形状回退 Triton。它仍属于 W8 存储加 BF16 GEMM，没有声称使用 INT8 Tensor Core。

| 投影 | Triton row ms | 最快原生 ms | 倍率 | 权重逐位一致 |
| --- | ---: | ---: | ---: | --- |
| gate_proj | 0.131837 | 0.130139 | 1.0130× | true |
| up_proj | 0.132648 | 0.129584 | 1.0236× | true |
| down_proj | 0.133040 | 0.133003 | 1.0003× | true |

| 75 投影输入行数 | 原 row 总 ms | Triton 单工作区 ms | 原生单工作区 ms | 原生工作区倍率 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 14.0861 | 13.5504 | 13.4779 | 1.0451× |
| 279 | 16.5328 | 16.4839 | 16.3913 | 1.0086× |
| 303 | 16.5642 | 16.5232 | 16.4520 | 1.0068× |

微基准为驻留权重、预热后同步批次的中位数；轮换基准使用固定 seed 合成 BF16 激活，不是完整 Transformer。所有 75 投影在三种输入行数下的新旧输出逐位一致。以上数据不能代替端到端结果。

## 环境、协议、显存

单张 MTT S5000 80GiB，物理 GPU0；驱动 3.3.5-server，torch/torch_musa 2.9.1、Triton MUSA 3.2、Transformers 4.40.1。沿用现有 FlagScale 启动与本地 FlagOS/KERV 适配。未升级已安装驱动、MUSA Runtime 或 muDNN，也未应用已撤销 PR 的框架补丁。

固定 group32 混合 policy：75 个 MLP 投影，attention、vision、head、drafter 及原定保护层保留 BF16。量化 payload SHA 及所有解码 SHA 核验一致。原静态权重节省 2,592,483,200 bytes，既有 group map 为 60,000 bytes；共享历史候选没有新增 GPU 解码池或常驻 BF16 权重缓存。本轮没有重新测单模型冷进程显存峰值，不将静态预算写成实际峰值降幅。

每个 worker 先完整预热 141 帧，再 AB/BA/AB 串行测三轮。每帧恢复同一捕获观测、raw action 历史和 Kalman token 前缀；不运行仿真环境。两个 worker 同时驻留，计算串行。warmup frame77 有 tensor 导出，measured 无此插桩。该捕获集已用于多次研发，不作为未见精度集。protocol 的 operator-only 旧标签按 intent_addendum_shared_history.json 解释为同 policy 的工程对照，本候选实际改变的是主机历史计算复用。

两端均沿用相同 guarded Softmax：原生计算后检查概率不变量并选择性修复异常行。该机制不是所有潜在底层错误的完备检测；实际检查次数和修复行数见 reports/local_results.json。

## 验证与失败记录

新增针对性测试 **8 项通过**：原生解码 4 项，共享历史与完整树 4 项。共享历史覆盖 141 帧历史前缀、三种窗口、各 tail 长度、空/缺失历史、自定义钩子以及完整树四类 buffer；原生解码覆盖保护组、缩放、尾部、侧流和重复复用。三轮动作证据由独立 CPU 检查器从逐帧 JSONL 重新计算，见 evidence_checks.json。

首次 shared_history_v1 在加载审计时过早读取尚未由 runtime 初始化的配置属性，启动失败；修正为 loaded 阶段允许未初始化、final 阶段核验实际配置后，以新目录 shared_history_v2 重跑。失败命令、退出码和日志均保留。未覆盖失败记录，未重置 GPU/驱动。

本轮未完成相对 BF16 的质量门禁、完整闭环成功率矩阵或跨芯片验收，默认 BF16 入口保留。

## 交付索引

- 操作说明：[MUSA_QUANTIZED_OPERATORS_FOLLOWUP.md](../../src/docs/MUSA_QUANTIZED_OPERATORS_FOLLOWUP.md)。
- 完整回放：`replays/shared_history_v2/`；失败回放：`replays/shared_history_v1/`。
- 算子试验：`paired_micro_v1/`、`paired_micro_rows_v2/`；原生与工作区的 v1/v2 结果全部保留，表格采用增加入口参数检查后的 `native_micro_v2/`、`workspace_rotation_v2/`。
- 机器可读结果：`reports/local_results.json`；命令和退出码：`operations/`。
- 最终源码与文档归档：`inventory/source_final.tar.gz`；文件校验：`checksums.sha256`。

证据校验通过表示本轮数据可复算，不表示甲方全部任务书指标或量化质量验收通过。
