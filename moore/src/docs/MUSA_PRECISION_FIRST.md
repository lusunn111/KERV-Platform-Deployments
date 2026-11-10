# MUSA 精度优先配置：无损 BF16 存储与混合候选

本配置针对之前 INT8 方案相对原始 BF16 存在动作偏差的问题。推荐入口将 verifier 的 96 个 MLP 投影保存为无损 BF16 压缩数据，attention、视觉、drafter、动作头等保持原配置。计算仍使用原生 BF16 GEMM；每个压缩投影使用前还原原始 16 个 BF16 bit。

**12 bit 指主体存储编码，不是 W12 有损量化，也不是 INT8 计算。** 编码保存符号/尾数和组内指数差，超出指数范围的元素另存原始 BF16。该机制没有新增权重量化误差；模型本身的 BF16 精度与任务能力不因此提高。

本次完整结果与显存、延迟代价见[实测报告](../../docs/acceptance/MOORE_PRECISION_REFINEMENT_20261027.md)。此前 75 投影 INT8、8/4 投影 INT8 混合候选及全无损候选的原始结果全部保留。

## 配置和存储预算

本机 campaign：`/root/KERV/outputs/precision_refinement/20261027T084306Z`。

| 配置 | 存储方式 | 静态权重节省 | 用途 |
| --- | --- | ---: | --- |
| `exact_mlp/deployment_policy.json` | MLP 96 投影无损 BF16 压缩，其余原生 BF16 | 2.122 GB | 本轮推荐的精度优先候选 |
| `exact_bf16/deployment_policy.json` | MLP 和部分 attention 共 192 投影无损 BF16 压缩，q_proj 原生 | 2.912 GB | 更大的静态节省，解码开销也更大 |
| `conservative_up8/deployment_policy.json` | 8 个 up_proj 使用既有 INT8 payload，其余符合条件的投影无损压缩 | 约 3.012 GB | 精度筛选未通过，不推荐 |
| `conservative_up4/deployment_policy.json` | 4 个 up_proj 使用既有 INT8 payload，其余符合条件的投影无损压缩 | 约 2.962 GB | 精度筛选未通过，不推荐 |

GB=10⁹ bytes。静态预算不等于 framework 显存峰值；GPU 解码临时张量、allocator reserved、KV 与图像工作区另计。本轮推荐配置的实际冷进程显存结果以报告为准。

## 直接运行

在宿主机执行，路径参数传给容器，因此 policy 和 output 使用 `/workspace`：

```bash
cd /root/KERV/src
export KERV_PRECISION_POLICY=/workspace/outputs/precision_refinement/20261027T084306Z/exact_mlp/deployment_policy.json
bash scripts/run_musa_precision.sh \
  --max-tasks 1 --trials 1 --max-episode-steps 220 \
  --output-dir /workspace/outputs/manual_precision_mlp
```

沿用现有 `kerv-musa-runtime` 容器、FlagScale 启动桥接、MUSA 驱动和 checkpoint。没有升级驱动、Runtime、muDNN 或框架，也没有重新训练。新配置是显式入口，已有默认入口保留。

配置中的 `w8a16.enabled=true` 借用既有“CPU 权重预处理后再上卡”的加载钩子。实际存储由 `run_precision_libero.py` 和 policy 决定，不能凭该旧开关判定使用了 INT8。推荐 policy 的 `modules=[]`、`module_count=0` 表示不注入 INT8；`exact_bf16_groups=["verify_mlp"]` 表示只对 MLP 执行无损压缩。

Hydra 的逗号字符串必须保留内层引号。手工拼装配置时应使用：

```bash
--override 'inference.runtime.w8a16.groups="verify_mlp,verify_attention"'
```

入口同时启用 host tree builder、共享原标量 Kalman 历史估计、guarded Softmax；static tree runtime 为 false。比较时，BF16 对照也必须使用相同设置。上轮量化内部的 1.1669× 不能直接套到本轮相对 BF16 的结果。

## 加载、解码和审计

1. 原始 BF16 checkpoint 在 CPU 加载。
2. policy 可以指定少量 INT8 payload；每个 payload 验证文件 SHA、原始 BF16 权重 SHA 与解码 SHA。推荐配置的 INT8 列表为空。
3. 在选定组中构建无损 BF16 编码，CPU 逐 bit 验证；q_proj 保持原生，供原运行时直接访问设备和 dtype。
4. 压缩模块迁移到 MUSA，每个模块实际解码一次，核验解码 SHA。推荐配置共 96 个模块。
5. 推理时按投影临时还原 BF16，随后调用原生 GEMM；没有常驻整模型 BF16 展开缓存。

`precision_storage.jsonl` 的 loaded 记录包含每个模块的 storage、原始/解码/GPU SHA、存储字节和 policy SHA；final 记录实际 forward 次数、Kalman 命中与 Softmax 检查/修复次数。加载与哈希审计不混入动作推理计时。

关键源码：

- `runtime_opt/KERVRuntimeOptimization/lossless_bf16.py`：编码和分组选择。
- `runtime_opt/KERVRuntimeOptimization/lossless_bf16_kernels.py`：MUSA/Triton 逐 bit 解码和精确异常值覆盖。
- `openvla/experiments/robot/libero/run_precision_libero.py`：policy 加载、CPU 预处理、MUSA 审计与 Softmax 保护。
- `openvla/experiments/robot/libero/run_precision_reference_libero.py`：原始 BF16 对照。

## 核验已保存结果

宿主机使用标准库即可复算；下列命令会更新派生检查 JSON，封存目录中复算后应重建校验清单，或先复制目录：

```bash
cd /root/KERV/src
C=/root/KERV/outputs/precision_refinement/20261027T084306Z
python3 -m acceptance.check_precision_screening --campaign "$C"
python3 -m acceptance.check_precision_closedloop --campaign "$C" \
  --task-id 0 --state-id 0 \
  --candidate-run-id full_exact_mlp_task0_state0_s7_r1
python3 -m acceptance.check_precision_closedloop --campaign "$C" \
  --task-id 1 --state-id 10
```

`screening/` 是多份权重同卡驻留、串行执行的精度选择实验，不能用于冷显存结论。`runs/` 是独立冷启动、真实 LIBERO 闭环，包含动作、观测、内存、runtime 记录和视频。`inventory/cases*.csv` 保存原登记及唯一重试，原启动失败保留。

针对性测试：

```bash
docker exec \
  -e PYTHONPATH=/workspace/src/openvla:/workspace/src:/workspace/src/runtime_opt:/workspace/src/tests \
  kerv-musa-runtime /workspace/.venv/bin/python \
  -m unittest test_precision_storage -v
```

这组测试覆盖 exact-only、INT8/exact 混合、MLP-only、损坏 payload 和不完整 policy。完整性通过、权重无损和局部任务动作一致分别记录；它们不替代完整 LIBERO 成功率矩阵或跨芯片验收。
