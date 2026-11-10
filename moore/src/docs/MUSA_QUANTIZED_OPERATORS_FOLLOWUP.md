# MUSA 混合量化算子与共享历史优化操作说明

本轮继续优化固定的 INT8/BF16 保护组方案。运行环境仍为物理 GPU0、MTT S5000、`kerv-musa-runtime` 容器；驱动及已安装的 MUSA/muDNN 不变。量化 policy 使用 `outputs/calibrated_quantization/20261026T045030Z_v2/model_refined/policy.json`，75 个投影及其保护组不变。

本轮报告见 [实测报告](../../docs/acceptance/MOORE_QUANTIZED_OPERATORS_FOLLOWUP_20261027.md)。量化策略相对 BF16 的质量验收仍未通过；本页说明同一策略内部的性能优化与复现方法。

## 实现及选择

`calibrated_paired_ops.py` 实现二维分块解码、gate/up 联合解码和联合原生 GEMM。真实权重及激活对照逐位一致，但本机更慢，保留为实验代码。

`csrc/calibrated_decode.mu` 与 `calibrated_native_decode.py` 实现原生 MUSA 向量化解码。每线程读取 8 个 INT8 权重，共用组 scale；BF16 保护组整块复制；非保护组按 FP32 顺序乘 scale/inverse scale，再舍入 BF16。保留原生 BF16 GEMM，不改为激活 INT8。非 8 对齐的宽度或组长回退到 Triton 行解码。显式调用 `load_library(build_directory)` 后才能使用，编译排除在推理计时外。编译产物按源码 SHA 命名，原生候选尚未作为推理默认后端。

单工作区复用在 279/303 行的 75 投影轮换测试中仅约 1% 收益，没有据此宣称整模型提速，也未接入本轮最终候选。

`run_calibrated_shared_libero.py` 是本轮完整回放候选。它沿用原有 row1024 解码、原生 BF16 GEMM、host tree builder 和 guarded Softmax，在树构建时共用一次内置 Kalman 历史估计，再按各分支所需长度切片。数值计算仍是原标量 Kalman 函数。没有历史时保留每个前缀的重复末 token 回退；自定义预测钩子、超长 tail 等场景继续逐分支调用。避免将任意自定义钩子视为与前缀无关。

## 启动显式候选

在宿主机执行，脚本会通过既有容器和 FlagScale 桥接启动：

```bash
cd /root/KERV/src
export KERV_CALIBRATED_POLICY=/workspace/outputs/calibrated_quantization/20261026T045030Z_v2/model_refined/policy.json
bash scripts/run_musa_calibrated_shared.sh \
  --max-tasks 1 --trials 1 --max-episode-steps 220 \
  --output-dir /workspace/outputs/manual_shared_history
```

对照入口为 `scripts/run_musa_calibrated_safe.sh`，两者都启用相同的 Softmax 检查及异常行修复。不要将历史上未带保护的对照结果直接作为本次新入口的性能基线。输出目录使用新路径，保留原始运行记录。

关键配置：

| 配置 | 对照 | 共享历史候选 |
| --- | --- | --- |
| `tree_builder.enabled` | true | true |
| `tree_builder.static_runtime` | false | false |
| `tree_builder.kalman_batch` | false | true |
| 量化 policy、row 解码、BF16 GEMM | 相同 | 相同 |
| guarded Softmax | 启用 | 启用 |

## 三轮完整回放

先确认 GPU0 空闲。以下命令必须使用新的 campaign/experiment 目录；已有原始结果不覆盖。

```bash
cd /root/KERV/src
python3 -m acceptance.quantized_kernel_replay \
  --campaign /root/KERV/outputs/quantized_operators/NEW_CAMPAIGN \
  --capture /root/KERV/outputs/calibrated_quantization/20261026T045030Z_v2/inputs/diagnostic_capture \
  --policy /root/KERV/outputs/calibrated_quantization/20261026T045030Z_v2/model_refined/policy.json \
  --reference-backend safe_rows --candidate-backend shared_history \
  --rounds 3 --experiment-name shared_history
python3 -m acceptance.check_quantized_kernel_replay \
  --experiment /root/KERV/outputs/quantized_operators/NEW_CAMPAIGN/replays/shared_history
```

每个 worker 先完整预热 141 帧，随后 AB/BA/AB 串行测量。每帧恢复相同捕获观测、raw action 历史和 Kalman token 历史。两个 worker 同时驻留但不会同时执行 GPU 计算；该进程布局不能用于单模型冷启动显存结论。该捕获集已反复用于研发，不能称为新的未见质量集。

核验 `identity_checks.json`、`evidence_checks.json` 以及两端 `calibrated_kernel.jsonl`、`safe_softmax.jsonl`。候选另有 `calibrated_shared_history.jsonl`，记录实际调用次数、分支数和最终运行配置。加载阶段 runtime 属性可能尚未初始化，以 final 记录的实际配置和计数为准。

## 单算子与边界测试

GPU 测试与完整模型回放串行执行：

```bash
docker exec \
  -e PYTHONPATH=/workspace/src/runtime_opt:/workspace/src/openvla:/workspace/src:/workspace/src/tests \
  -e KERV_NATIVE_TEST_BUILD=/workspace/outputs/manual_native_build \
  -e MUSA_VISIBLE_DEVICES=0 \
  kerv-musa-runtime /workspace/.venv/bin/python -m unittest \
  test_calibrated_native_decode test_kalman_shared_history -v
```

原生解码测试覆盖有/无/全 BF16 保护组、非单位 inverse scale、部分组、非对齐宽度、侧流以及重复复用工作区。共享历史测试逐一比较 141 帧历史、三种窗口的所有 tail 长度，并验证完整树 tokens、retrieve indices、mask、position ids，以及自定义钩子回退。

真实权重的原生解码微基准：

```bash
docker exec -e PYTHONPATH=/workspace/src/runtime_opt:/workspace/src \
  -e MUSA_VISIBLE_DEVICES=0 kerv-musa-runtime /workspace/.venv/bin/python \
  -m acceptance.benchmark_native_quant_decode \
  --policy /workspace/outputs/calibrated_quantization/20261026T045030Z_v2/model_refined/policy.json \
  --output /workspace/outputs/manual_native_micro
```

微基准只表示所列真实权重和形状下的局部性能。完整回放动作一致也仅表明本轮优化相对同一有损量化策略的额外差异，不能替代相对 BF16 的精度或完整任务成功率验收。
