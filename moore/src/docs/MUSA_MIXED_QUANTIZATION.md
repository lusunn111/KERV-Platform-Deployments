# 摩尔 MUSA 混合精度与保精度压缩操作指南

**当前均为实验入口：** 两轮INT8混合策略均未通过初始闭环；optimized运行时无损存储虽逐bit还原权重，回放仍为139/141动作相同，完整任务未完成。追加native控制在task0/state0/seed7的170条动作、token及全过程观测全部一致，任务成功；该单case通过不等于完整套件SR。

本文配套报告为工作区 `docs/acceptance/MOORE_MIXED_QUANTIZATION_REPORT_20261026.md`。默认 BF16 optimized 入口保留。所有路径、设备与命令以当前 `/root/KERV` 和 `kerv-musa-runtime` 容器为准。

## 1. 已实现的两类方案

| 方案 | 精度与存储 | 适用状态 |
| --- | --- | --- |
| 混合 INT8/BF16 | verifier MLP 按投影选择 group-wise INT8；未选投影保留原 BF16；激活/GEMM 为 BF16 | 实验入口；两轮策略均有动作偏差，未通过初始闭环，不替换默认配置 |
| BF16 无损存储压缩 | 部分投影用 12-bit 编码、分组指数基值和稀疏原 BF16 例外；计算前逐 bit 还原 BF16 | 保精度备选；最终实测数据与使用边界见配套报告 |

第二类不是 FP12，也不是“12-bit 数值量化”。其存储值能够恢复全部 16 个 BF16 原始位，计算仍采用原生 BF16 GEMM。每层临时解码，额外耗时须与节省显存一起评估。

## 2. 环境检查

```bash
cd /root/KERV
mthreads-gmi -i 0
docker inspect --format '{{.State.Running}}' kerv-musa-runtime
```

使用已验证的容器和公开 checkpoint，不修改驱动或依赖。GPU0 须空闲；脚本不会替使用者终止已有任务。容器挂载 `/root/KERV` 为 `/workspace`，仅映射本次验证卡。

## 3. 保精度压缩入口（已验证单case的native配置）

```bash
cd /root/KERV
src/scripts/run_musa_lossless_native.sh \
  --max-tasks 1 --trials 1 --max-episode-steps 220 \
  --override ++inference.runtime.acceptance_task_id=0 \
  --override ++inference.runtime.acceptance_init_state_id=0 \
  --override ++inference.runtime.seed=7
```

上述脚本使用native KERV运行时，通过 FlagScale 启动专用 `run_lossless_libero.py`。CPU 加载原始 BF16 权重后编码，再将压缩缓冲搬到 MUSA；不会先把完整 BF16 verifier 上卡，也不会保存原 BF16 GPU 全模型备份。编码对象为32层的96个MLP投影及96个k/v/o attention投影；q_proj 保持原模块，视觉、drafter、embedding、动作头、激活和 KV 保留既有 BF16 设置。Linear 融合关闭，其余 KERV 树/Kalman/阈值参数保持匹配配置。

兼容现有 CPU 加载阶段时使用 `w8a16.enabled` 字段进行入口分发和互斥控制；实际算法以 manifest 中的 `implementation=musa_lossless_bf16_exp_delta12_native_bf16_gemm` 和 `lossy_quantization=false` 为准。不能根据历史字段名将该路径写成 INT8 GEMM。

第一次加载包含CPU编码、CPU位一致检查、设备解码哈希检查和内核编译；这些计入加载/初始化阶段，不混入逐动作模型耗时。初始化显存仍计入峰值。

optimized运行时压缩入口（该配置首个闭环失败，保留用于研究，不作为已通过候选）：

```bash
cd /root/KERV
src/scripts/run_musa_lossless.sh \
  --max-tasks 1 --trials 1 --max-episode-steps 220 \
  --override ++inference.runtime.acceptance_task_id=0 \
  --override ++inference.runtime.acceptance_init_state_id=0 \
  --override ++inference.runtime.seed=7
```

native压缩入口的匹配BF16基线为 `KERV_MUSA_PROFILE=native src/scripts/run_musa.sh` 加相同task/state/seed覆盖。native路径关闭optimized运行时但保留KERV算法；不能与optimized配置合并成功率，也不能将显存收益与原optimized加速倍数拼接。

## 4. 混合 INT8 实验入口

下述策略保留用于复现失败和误差分析，不能直接用于已通过精度验收的交付声明。

```bash
cd /root/KERV
KERV_MUSA_PROFILE=mixed src/scripts/run_musa.sh \
  --max-tasks 1 --trials 1 --max-episode-steps 220 \
  --override inference.runtime.w8a16.policy_path=/workspace/outputs/mixed_quantization/20261026T025710Z_mixed/calibration/policy.json \
  --override inference.runtime.w8a16.group_size=128

# 第二轮：替换 policy_path 为 calibration_refined/policy.json，group_size 为32。
```

冻结 policy 包含完整模块名、shape、group_size、INT8 与保留 BF16 的投影清单。加载阶段验证策略与模型拓扑，不匹配即报错；不允许把静默 BF16 回退当作量化成功。没有配置策略时，mixed profile 不会自动选取一个已失败策略。

校准过程先记录BF16参考，再逐投影试量化；只有所选校准帧上的完整动作token向量与重复BF16参考一致才接受。第1轮8帧；第2轮16帧，包括观察第1轮结果后补充的高误差帧。第2轮数据属于自适应筛选数据，不是严格未见测试集。只在少量帧通过门槛，不能推出闭环任务不掉精度。

## 5. 复算与回归

```bash
cd /root/KERV
export PYTHONPATH=src
C=outputs/mixed_quantization/20261026T025710Z_mixed
python3 -m acceptance.mixed_report --campaign "$C"
python3 -m acceptance.check_evidence --campaign "$C" --check-source-hashes --check-weight-hashes
python3 -m acceptance.check_quant_replay --campaign "$C"

# GPU0 空闲时执行；与性能测试串行。
docker exec \
  -e PYTHONPATH=/workspace/src/runtime_opt:/workspace/src:/workspace/src/tests \
  -e KERV_DEVICE_BACKEND=musa kerv-musa-runtime \
  /workspace/.venv/bin/python -m unittest discover -s /workspace/src/tests -v
```

不要向已完成run ID追加运行。新增实验应创建新campaign，或先冻结新的 `cases_addendum_XX.csv` 后采用新ID。`run_matrix` 的执行PASS仅表示数据采集与程序执行完整，任务成功另看 `task_success`；8步冒烟不计成功率。报告工具会保留未执行项、错误、失败和各策略版本，不能将不同候选混合成一个成功率。

## 6. 核验文件与回退

- `compression.jsonl`：每个压缩投影的原始/设备解码SHA256、实际设备buffer dtype和字节数，以及退出时真实forward次数；192个投影全部核对。
- `manifests/w8a16.json`：兼容字段下的实际压缩算法与静态字节账，需读取 implementation，不能单看文件名。
- `quantization.jsonl`：混合INT8的真实INT8 buffer、保留BF16权重和forward证据；无损压缩路径不产生INT8权重。
- `steps.jsonl`、`episodes.jsonl`、`memory.jsonl`、`device_samples.jsonl`：动作、闭环、framework峰值及整卡采样；单模型冷进程峰值与回放双驻留显存分开。
- `replays/quantized_fixed_input_v1/results.json`：正式同输入、固定历史对照，具体profiles见protocol；目录名沿用工具，不能据此认定它是有损量化。

回退：直接使用 `src/scripts/run_musa.sh`，默认 `optimized` BF16。原 checkpoint 全程只读，压缩不重写权重文件。压缩模块是推理专用入口，不支持训练；需要在新进程按原checkpoint重新加载，不将内部压缩buffer当作 Hugging Face 标准权重导出。
