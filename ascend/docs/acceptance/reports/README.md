# 专项报告

- [昇腾综合测评报告](ASCEND_EVALUATION_REPORT_20261027.md)（[Word](ASCEND_EVALUATION_REPORT_20261027.docx)），2026-10-27交付；从原始记录复核120回合与6组时延，统一P95、指标对照与证据索引，配套[技术大纲](../ASCEND_TECHNICAL_OUTLINE_20261027.md)。

- [昇腾QKV batch GEMM与视图缓存](ASCEND_QKV_BATCHED_DISPATCH_20261027.md)，2026-10-27；含形状数值筛查、60个完整回合、三遍同卡测速、逐步动作比对与失败路径。

- [昇腾QKV融合与CPU预打包](ASCEND_QKV_FUSION_20261027.md)，2026-10-27后续；比上一版快4.22%，修复约3GB新增设备驻留；15/20低于上一版16/20，未替换推荐，包含实际HBM口径和全部失败尝试。

- [昇腾混合精度与保留舍入的融合算子](ASCEND_MIXED_PRECISION_OPERATORS_20261027.md)，2026-10-27，新增60回合；W8A16/BF16混合省2.085GB，平均时延降1.03%，两个初态各8/10（BF16各7/10），含失败候选与复现命令。

- [KERV 昇腾显存优化分析报告：2 GB目标](ASCEND_2GB_MEMORY_ANALYSIS_20261019.md)，2026-10-19。

报告引用已冻结的本机实验。新增报告不修改原实验记录或其校验清单。

- [昇腾INT8显存优化验证报告](ASCEND_INT8_MEMORY_VALIDATION_20261026.md)，2026-10-26，含10任务BF16/INT8配对试验。
- [昇腾INT8变慢原因核查](ASCEND_INT8_SLOWDOWN_DIAGNOSIS_20261026.md)，2026-10-26，同卡交替回放、真实输入kernel与设备trace。
- [以任务精度为优先的量化设计](ASCEND_ACCURACY_AWARE_QUANTIZATION_DESIGN_20261026.md)，2026-10-26，设计记录；后续实施范围和结果见原型试验报告。
- [精度优先量化原型试验](ASCEND_ACCURACY_AWARE_QUANTIZATION_PILOT_20261026.md)，2026-10-26，70个闭环回合、独立初态验证和同卡时延。

- [昇腾W8A8省显存与时延优化](ASCEND_W8A8_MEMORY_LATENCY_20261026.md)，2026-10-26，80回合、完整157帧同卡测速、保守混合配置与精度退化记录。
