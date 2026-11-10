# MUSA 校准量化试验操作指南

配套报告：[2026-10-26校准量化实测](../../docs/acceptance/MOORE_CALIBRATED_QUANTIZATION_REPORT_20261026.md)。本文描述实验入口，是否通过精度门槛以报告和原始任务结果为准，不将校准成功写成任务成功。默认optimized BF16入口保留。

## 1. 实现范围

- 原生KERV运行时，FlagScale启动，torch_musa执行；不与原optimized配置的加速倍数拼接。
- verifier第2–27层MLP使用group32 INT8，FP32 group scale、FP32通道逆scale；约5%的敏感连续输入列组保留原BF16。
- 前2/后4层、attention、视觉、drafter、embedding、动作头、norm、激活和KV保持原配置。
- GPTQ式误差补偿计入实际BF16解码舍入；保护组预先冻结，补偿只更新未保护列。Hessian在CPU FP64分解，误差补偿在MUSA FP32执行。
- 单block消融中alpha=.5的AWQ组合未优于alpha=0，故本轮全模型采用alpha=0的GPTQ+BF16保护组，不声称AWQ已产生额外收益。
- 逐层推进原BF16教师和当前量化候选状态；后续层读取量化上游产生的输入。MLP内部down_proj也用已量化gate/up的实际输入。
- 运行时先恢复本层BF16权重，覆盖保护组，再做一次原生BF16 GEMM；不宣称原生INT8 GEMM加速。

本轮是小规模pilot：3个任务各一个初始状态的前48帧作为C，历史task0的36帧作为诊断选择V。完整设计的60/45/300回合协议没有完成，不能宣称成功率损失≤1个百分点。

## 2. 运行前检查

```bash
cd /root/KERV
mthreads-gmi -i 0
docker inspect --format '{{.State.Running}}' kerv-musa-runtime
```

仅使用已分配GPU0、已有镜像和虚拟环境。不安装CUDA-only量化后端、不升级驱动。确认GPU0空闲；不得终止其他用户作业来启动实验。

## 3. 显式加载冻结量化产物

```bash
cd /root/KERV
KERV_CALIBRATED_POLICY=/workspace/outputs/calibrated_quantization/20261026T045030Z_v2/model_quant/policy.json \
  src/scripts/run_musa_calibrated.sh \
  --max-tasks 1 --trials 1 --max-episode-steps 220 \
  --override ++inference.runtime.acceptance_task_id=0 \
  --override ++inference.runtime.acceptance_init_state_id=0 \
  --override ++inference.runtime.seed=7
```

路径必须使用容器内`/workspace`路径。policy不完整、源BF16权重SHA、payload文件SHA、CPU解码SHA、设备解码SHA或shape不匹配均退出。CPU先加载量化buffer，随后才上MUSA，不会先把全模型BF16上卡；部署时不保留校准阶段的BF16 GPU备份。

如果追加输出敏感性筛选生成了`model_refined/policy.json`，可显式替换路径复现该候选。两种policy是不同候选，不能合并成功率；原失败case保留。

policy有模块名、shape、文件哈希、原BF16权重哈希、解码权重哈希、保护组统计和静态字节账；它不是标准Hugging Face checkpoint。加载入口是`run_calibrated_libero.py`，复用FlagScale的w8a16字段进行分发，实际实现以manifest中的`musa_calibrated_gptq_int8_bf16_groups_native_bf16_gemm`为准。

## 4. 数据与判定文件

实验根目录：`outputs/calibrated_quantization/20261026T045030Z_v2`。

| 文件/目录 | 含义 |
| --- | --- |
| inventory/cases*.csv | 执行前冻结的采集、完整回合及追加候选ID |
| block_pilot_v2/reconstruction.json | RTN、GPTQ、保护组及AWQ组合的单MLP block对照 |
| block_pilot_v2/action_replay.json | 仅替换一个block的36帧动作比较，不能当作全模型结果 |
| model_quant/layers.json | 每层实际使用的量化输入、误差和是否回退 |
| model_quant/teacher_forcing.json | 固定动作前缀下252个预测位置的诊断，不能当作KERV实际动作或SR |
| model_quant/*.pt及policy.json | 可部署INT8/scale/BF16保护组产物；主候选78个投影 |
| model_refined/ | 如触发追加筛选，记录单block恢复原BF16、组合、排序及最终policy |
| runs/*/calibrated_quantization.jsonl | 实际设备buffer dtype/bytes、CPU/MUSA解码一致性及真实forward次数 |
| runs/*/steps.jsonl、episodes.jsonl | 每动作/每回合证据；执行PASS和任务成功分别记录 |
| runs/*/memory.jsonl、device_samples.jsonl | framework冷/稳态峰值及整卡采样，两种口径分开 |
| replays/*/ | 同输入、同历史回放及全部帧输出；双模型驻留显存不参与单模型峰值比较 |
| reports/baseline_drift.json | 本轮BF16与历史BF16在相同配置下出现分歧的记录，不是根因修复声明 |
| reports/calibrated_execution_proof.json | 冻结policy、实际buffer、解码和前向次数的独立核对 |

兼容的旧`quantization.jsonl`采集器只识别原Int8WeightOnlyLinear类，本轮使用新类，实际执行证明以`calibrated_quantization.jsonl`为准。两者的schema不能混用。

## 5. 复算

```bash
cd /root/KERV
export PYTHONPATH=src
C=outputs/calibrated_quantization/20261026T045030Z_v2
python3 -m acceptance.calibrated_report --campaign "$C"
python3 -m acceptance.check_evidence --campaign "$C" --check-source-hashes --check-weight-hashes
python3 -m acceptance.check_quant_replay --campaign "$C"

# 仅在追加回放已实际完成时运行：
python3 -m acceptance.check_quant_replay --campaign "$C" \
  --experiment-name calibrated_refined_fixed_input_v1
```

新增实验不要向已完成run ID或已存在回放目录追加数据。使用新campaign或先冻结新CSV与新ID。初始候选失败后按登记规则停止扩展；输出敏感性筛选使用已见V，只是研发调参，后续闭环也不能冒充最终T测试。

## 6. 校准过程与回退

单block采集及顺序全模型校准入口分别为`acceptance/calibrated_block_entry.py`和`acceptance/calibrated_model_entry.py`。它们需要完整的预登记输入及匹配元数据，不应只拿policy重新当作校准数据运行。实际命令、环境和版本在inventory/及operations/留档。

如果触发输出敏感性筛选，按实际执行夹爪变化、6维动作bin绝对/最大误差、top1变化及KL排序，尝试恢复0/1/2/4/6个block并保留静态2GB预算。它会改变候选，不保证实际峰值仍省2GB，也不保证任务成功。

已有BF16基线也可能在同配置下出现不同动作和任务结果，必须记录重复控制，不将所有失败归因量化。最终仍需多任务、多状态、独立数据的成功率验证。

回退：`src/scripts/run_musa.sh`使用原默认BF16 optimized；`KERV_MUSA_PROFILE=native src/scripts/run_musa.sh`使用匹配的native BF16。原模型文件全程只读，没有覆盖checkpoint。
