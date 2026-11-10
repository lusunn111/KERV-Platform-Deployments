# 本机执行与证据

## 2026-10-27 技术大纲与综合测评交付

- [昇腾技术大纲](ASCEND_TECHNICAL_OUTLINE_20261027.md)（[Word](ASCEND_TECHNICAL_OUTLINE_20261027.docx)）。
- [综合测评报告](reports/ASCEND_EVALUATION_REPORT_20261027.md)（[Word](reports/ASCEND_EVALUATION_REPORT_20261027.docx)）。
- [复算数据、CSV与证据索引](../../outputs/acceptance/ascend_technical_evaluation_20261027/summary.md)。

从169个源证据文件复核12组运行、120个完整回合、6组时延；配置复用20个task/init/seed，不能按120个独立质量样本解释。
本次新增模型运行0次，统一nearest-rank P95并核验原封存哈希；原始campaign不覆盖。
当前只交付昇腾本机技术大纲与测评，正式指标状态和外部依赖见报告第9节。

```bash
python -m acceptance.evaluate_ascend_delivery --output outputs/acceptance/ascend_delivery_recomputed_new
python -m acceptance.render_ascend_evaluation --evidence outputs/acceptance/ascend_delivery_recomputed_new
```

执行依据：[TestOutline.md](../../TestOutline.md)。本轮只负责当前昇腾，不代表课题整体验收。

2026-10-19证据目录：[`ascend_20261019T0730Z`](../../outputs/acceptance/ascend_20261019T0730Z/summary.md)。
原始测试、失败/修复、注册表和口径记录均保存在该唯一campaign中；旧outputs未覆盖。

工具位于根目录`acceptance/`，主机Python执行工具，模型在已有`kerv-ascend`容器运行。

```bash
python -m acceptance.preflight --help
python -m acceptance.run_matrix --help
python -m acceptance.collect_metrics --describe
python -m acceptance.aggregate --help
python -m acceptance.check_evidence --help
```

实际执行过的单例/批量/复算命令（同campaign已有目录会保留或跳过，不覆盖）：

```bash
python -m acceptance.run_matrix --campaign ascend_20261019T0730Z \
  --cases outputs/acceptance/ascend_20261019T0730Z/cases_retry1.json
python -m acceptance.run_matrix --campaign ascend_20261019T0730Z \
  --cases outputs/acceptance/ascend_20261019T0730Z/cases.json \
  --only paired_B1 paired_B2 transfer_B1 transfer_B2
python -m acceptance.aggregate --campaign-dir outputs/acceptance/ascend_20261019T0730Z
python -m acceptance.check_evidence --campaign-dir outputs/acceptance/ascend_20261019T0730Z
```

新一轮：先用唯一campaign ID执行preflight并冻结，再建立cases；先烟测再完整回归。仅同步确认的
源码到运行副本，并逐文件比较哈希。用例或采集器变更采用新的run ID并保留修复关系。

采集器只在`KERV_ACCEPTANCE_DIR`非空时启用；`KERV_ACCEPTANCE_TRANSFERS=1`启用重型拷贝诊断，
不用于正式速度。原始posterior接受长度不是已审计排除所有补偿/padding后的正式ΔL。
`MEMCPY_ASYNC`没有方向/bytes时不能将其与另一轮逻辑字节相除求带宽。

## 2026-10-26量化试验

[INT8显存与质量验证](reports/ASCEND_INT8_MEMORY_VALIDATION_20261026.md)。
项目允许量化，CPU加载阶段转换以覆盖全程显存峰值；最新code与2026-10-19冻结版本不同，
复现旧实验须使用当时源码快照，不能用当前源码检查结果冒充旧版本。

```bash
python -m acceptance.aggregate_quantization --campaign-dir outputs/acceptance/quant_20261026T0215Z
python -m acceptance.report_quantization --campaign-dir outputs/acceptance/quant_20261026T0215Z
```
