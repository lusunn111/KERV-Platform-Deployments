# 摩尔：以闭环任务成功为准的混合量化

本轮允许动作差异，以实际任务成功为首要依据。实验记录及当前结论见 [报告](../../docs/acceptance/MOORE_TASK_AWARE_QUANTIZATION_20261028.md)，机器可读结果见 `outputs/task_aware_quantization/20261028T022212Z/reports/local_results.json`。候选未通过任务测试前，不覆盖既有默认入口。

甲方交付的架构、量化原理、配对指标、验收状态与证据索引汇总于[正式技术报告](../../docs/acceptance/MOORE_TECHNICAL_REPORT_20261028.md)。该报告是封存后的说明文档；其修订不改动实验目录，也不重新计算或扩大任务成功率分母。

## 最终选择

保留原有96个MLP投影无损BF16压缩。七类有损候选均在task0/state0试点失败，新解码原型未显示稳定Linear收益，因此没有切换默认配置。

本轮新BF16/推荐配置各在170步成功，170条动作、raw、执行动作及观测完全一致；96个设备解码权重SHA均等于原始BF16。allocated峰值节省2,107,057,664bytes，reserved峰值节省2,092,957,696bytes，静态权重节省2,122,448,562bytes。排除首步模型均值261.860→312.572ms（+19.37%），没有新增速度收益。最终37项针对性回归通过。

`recommended_exact_mlp/deployment_policy.json` 与实际复测的上一批推荐policy逐字节一致；SHA256为 `c36da97d0e13c2d69c3926014f2962831ef7b5271982de22849866187b8bc475`。当前case是上一批task0的重复复测，不增加独立任务/状态覆盖，也不构成整套成功率验收。

## 机制

- 注意力、视觉、drafter、输出头和KV缓存保留BF16；选择部分verifier MLP投影压缩。
- `PackedIntLinear` 支持真正的10/12-bit整数位打包，每组32列使用FP32 scale；`PackedPow2Linear` 将scale限制为2的整数次幂，只存int8指数。解码恢复FP32 scale，再将整数乘积舍入为BF16。
- 激活风险较高的列组直接保存原始BF16，GPU解码时覆盖；其余列通过GPTQ补偿量化误差。误差反馈计算包含实际部署的BF16舍入。
- 解码算子将拆位、scale还原及保护组覆盖合在一个Triton内核，随后使用原生BF16 GEMM。没有INT10硬件计算，也没有整模型BF16展开缓存。
- 非压缩模块保持原生BF16；可选的 `exact_bf16_groups` 属于此前的无损存储方案。各模块的实际编码以policy的 `encoding` 为准。

CPU载入时逐项核验原始BF16权重SHA、payload SHA、解码权重SHA；迁移到MUSA后再次核验实际解码SHA。这里的一致指与**选定量化权重**一致，不代表与原始BF16权重一致。

## 近零量化候选

普通W10、真实树GPTQ W10及少量W12投影在抽屉试点仍失败，后续追加 `nearzero_mlp/policy.json`。它不使用整数权重量化，而是对96个MLP投影保留BF16符号、尾数与分组指数差；只把指数范围下方及两种符号的最低边界值映射为同符号零，其余BF16位保持原样。

设每组最大BF16指数为e，编码基值为max(e−15,0)。非零基值时，被舍弃的幅值不超过2^(e−15−127)，因此每个权重绝对误差≤该组原始最大绝对值/32768。基值为0时没有更低指数，保留BF16次正规值。构建时检查有限值、实际误差界和所有保留值的原始位，再记录实际零化数量、权重RMSE及最大误差。

`nearzero_bf16_kernels.py` 单次拆位还原权重并处理零码，取消原无损codec的稀疏异常值补写kernel。仍使用BF16 GEMM，attention、vision、drafter、head与KV保持原精度。数学误差界不能代替闭环成功率，是否部署以报告和selection.json为准。

此policy的 `modules=[]` 表示没有预制整数payload；`nearzero_bf16_groups=["verify_mlp"]` 在CPU加载时生成近零编码。它与 `exact_bf16_groups` 或预制整数payload互斥。

## 校准与闭环验证

`acceptance/calibrated_tree_model_entry.py` 从三个已冻结C轨迹各抽24帧，共72帧，恢复每帧观测、动作历史及Kalman token历史，实际运行KERV。分别采样提示词和推测验证树的MLP输入，避免只用固定动作前缀进行teacher forcing。

每个MLP独立使用BF16上游的真实激活，gate/up量化后再生成down的校准输入。36帧已见诊断轨迹用于检查MLP输出相对RMSE，超过1%的块回退原生BF16。这些局部检查不等于整模型任务成功，也没有覆盖逐层累积误差。校准耗时、显存和样本不计入部署指标。

试点为task0/state0和task1/state10、seed7；BF16筛选参考来自上一批已封存成功轨迹。只有试点通过后才运行预先登记的三个额外状态，并重新启动BF16对照：task0/state10、task1/state20、task2/state10。统一使用220策略动作上限、相同guarded Softmax和共享Kalman历史。任务成功从 `episodes.jsonl` 读取，不从进程退出码推断。

动作不同会改变后续观测，不能把不同轨迹的逐步动作差直接当作量化误差。闭环延迟变化同时包含树结构和任务步数变化，不能直接当作同工作量算子加速。少量成功任务不能验证TestOutline中套件成功率损失<5%的界限，也未证明更严格的1个百分点目标。量化/压缩显存收益不替代同精度异构部署P06验收。

## 复现与部署

宿主机启动，容器为现有 `kerv-musa-runtime`，物理GPU0。依赖和checkpoint保持原版本；使用显式policy，不安装额外量化框架。

```bash
cd /root/KERV/src
C=/root/KERV/outputs/task_aware_quantization/20261028T022212Z

# 查看选择结果；没有通过时，不以候选替换已有部署。
cat "$C/inventory/selection.json"

# 用记录中的明确policy复测。容器参数使用/workspace路径。
# 下例为本轮复测通过、继续保留的原无损配置。
KERV_PRECISION_POLICY=/workspace/outputs/task_aware_quantization/20261028T022212Z/recommended_exact_mlp/deployment_policy.json \
  bash scripts/run_musa_precision.sh --max-tasks 1 --trials 1 \
  --max-episode-steps 220 \
  --output-dir /workspace/outputs/manual_task_aware_exact
```

校准、测试和完整case应串行使用GPU。即使CPU权重构建程序也可能初始化MUSA并持有2MiB上下文，因此不与设备队列并行。`run_matrix` 的空闲检查会保留拒绝启动记录，不应通过关闭检查、重置GPU或终止无关任务绕过。

冻结批次包含唯一run ID；重跑使用新批次或新的ID，不追加到旧日志。以下只演示如何运行新的预登记清单：

```bash
python3 -m acceptance.run_matrix --campaign /path/to/new_campaign \
  --cases /path/to/new_campaign/inventory/cases.csv
```

针对性测试使用容器Python，支持CPU编码测试和真实摩尔解码测试。设备测试须确认GPU空闲。

```bash
docker exec -e MUSA_VISIBLE_DEVICES=0 \
  -e PYTHONPATH=/workspace/src/runtime_opt:/workspace/src:/workspace/src/tests \
  kerv-musa-runtime /workspace/.venv/bin/python -m unittest \
  test_packed_intn.PackedCpuTests
```

具体已执行的测试类名、命令和退出码以批次 `operations/` 为准。校验记录包括打包边界、保护组、偏置、CPU/MUSA解码与原生GEMM、GPTQ宽位整数导出一致性。

JSON和Markdown报告由标准库脚本从原始run数据生成：

```bash
python3 -m acceptance.task_aware_quantization_report --campaign "$C"
```

封存后在副本中复算。`checksums.sha256` 核验文件完整性；完整性通过、算子正确、局部任务质量和整套任务成功率分别判断。

## 未推荐的解码算子

保留CSR融合恢复、静态展开、两级异常字典和两字节拆位的实现、设备正确性检查与真实权重AB/BA微基准。CSR和字典版本更慢；两字节拆位在最佳配置下基本持平，M=279部分形状仍较慢，因此均不接入推荐运行。源码位于 `runtime_opt/KERVRuntimeOptimization/inline_exact_bf16*`、`hash_exact_bf16*`、`lossless_bf16_fast_kernels.py`；原始样本位于本轮批次对应的benchmark目录。它们没有完成完整模型闭环，不将算子正确性写成任务通过或模型加速。

完整性验证在批次封存后可用 `sha256sum -c checksums.sha256`；源码和本轮文档归档在 `inventory/source_final.tar.gz`。检查所有成功/失败数据和退出码时使用 `operations/`、`runs/` 与 `reports/`，不只读取成功视频。
