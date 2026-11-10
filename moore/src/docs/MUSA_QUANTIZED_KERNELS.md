# MUSA 混合量化算子操作指南

此入口针对已经生成的 `kerv_calibrated_int8_bf16_v2` 权重包，优化 INT8/BF16 保护组的解码速度。它不重新校准、不量化激活、不更改保护组，也不代表此前量化候选的任务精度已经通过。

**2026-10-27实测：热Linear微基准有收益，但整模型三轮倍率约1.011/0.995/0.987，未通过性能门禁。入口保留用于实验和复现，不建议将其作为默认部署替换。** 详细数字以本轮报告为准。

## 运行方式

沿用现有 `kerv-musa-runtime` 容器、FlagScale 启动及 MUSA 适配。宿主机执行：

```bash
cd /root/KERV
export KERV_CALIBRATED_POLICY=/workspace/outputs/calibrated_quantization/20261026T045030Z_v2/model_refined/policy.json
bash src/scripts/run_musa_calibrated_kernel.sh
```

此命令显式启用实验后端。原来的 `run_musa_calibrated.sh` 保留旧解码器，用于对照；`run_musa.sh` 默认配置也保留。更多运行参数可追加在命令末尾，参照原校准量化指南。不要修改已封存的 policy 或 `.pt` 文件。

## 具体计算路径

旧路径每个投影依次执行 INT8 解码、BF16 保护列补丁、原生 BF16 GEMM。新路径使用 `calibrated_w8a16_ops._decode_rows`，一次 kernel 同时完成：

1. 读取 INT8 权重和 FP32 group scale，按需应用 inverse channel scale。
2. 按原有计算顺序舍入为 BF16。
3. 通过 group→slot 小表，把保护组还原为原始 BF16 值。
4. 写出临时 BF16 权重，交给原生 `F.linear`；计算完成后临时权重可释放。

网格按输出行和输入列块组织，每块 1024 个输入元素，避免对扁平元素索引反复计算非二次幂行宽的整除和取余。`down_proj` 的行宽为 11008，收益更明显。对当前 alpha=0 权重，初始化时确认 inverse scale 全为 1 后编译省略该乘法；每次前向没有 CPU 同步或动态调参。

75 个投影共新增 60000 字节 group 查询表，没有永久 BF16 权重缓存。原混合量化权重策略和保护组不变。当前仍需要一次临时 BF16 权重分配，不能写成“完全消除反量化中间权重”。

## 为什么没有默认使用融合 GEMM

源码也包含边解码边做矩阵乘的 `_gemm` 和小行数 `_gemv`，用于实验。它们在本机真实形状下未获得性能优势，不被正式实验入口调用。初版 N×K tile 转置后进入 dot 时出现数值错误；改为直接 K×N 布局后最小样例和实际形状测试通过。保留失败记录，不把编译成功等同于算子可用。

有 bias 的直接融合路径回退到新解码器加原生 `F.linear`，避免改变本机原生库的 bias 舍入顺序。当前 Llama MLP 本身没有 bias。通用直接 GEMM 的 FP32 累加顺序可能与原生库不同，因此不承诺所有输入逐位一致。最终选择的是权重逐位一致、继续使用相同原生 GEMM 的解码优化。

## 如何核验真实命中

加载后的 `calibrated_kernel.jsonl` 包含：

- `phase=loaded`：75 个投影的新解码权重 SHA256，必须逐个匹配冻结 policy 中的 `decoded_weight_sha256`。
- `persistent_decoded_weight_bytes=0`：没有常驻展开权重。
- `phase=final`：逐模块 `forward_calls` 和 `decode_calls`。加载校验本身也会增加一次 decode，不能把它误算成推理次数。

原有 `calibrated_quantization.jsonl` 仍记录 INT8 MUSA buffer 和原始解码审计。两个文件一起看，不能只检查配置开关。

## 复现微基准和固定输入回放

每次使用新的输出目录；调度器要求 GPU0 空闲，不会停止其他进程。下列 `new_campaign` 路径需要自行先创建，不能指向已经封存的实验目录。

```bash
cd /root/KERV
mkdir -p outputs/quantized_kernels/new_campaign
docker exec -e PYTHONPATH=/workspace/src/runtime_opt:/workspace/src \
  kerv-musa-runtime /workspace/.venv/bin/python -m acceptance.benchmark_calibrated_ops \
  --reference /workspace/outputs/calibrated_quantization/20261026T045030Z_v2 \
  --output /workspace/outputs/quantized_kernels/new_campaign/microbench \
  --rows-decode-only --rows 1 279 303 --projections gate_proj up_proj down_proj

PYTHONPATH=src python3 -m acceptance.quantized_kernel_replay \
  --campaign /root/KERV/outputs/quantized_kernels/new_campaign \
  --capture /root/KERV/outputs/calibrated_quantization/20261026T045030Z_v2/inputs/diagnostic_capture \
  --policy /root/KERV/outputs/calibrated_quantization/20261026T045030Z_v2/model_refined/policy.json \
  --rounds 3
```

微基准是热缓存、重复输入的独立 Linear，不等于模型加速。回放的两个模型同时驻留、串行计算，三轮顺序 AB/BA/AB，使用相同 141 帧观测以及相同历史动作和 Kalman token 前缀；每个 worker 先完整预热一次。逐帧模型/pipeline 时间、动作和 tokens 保存在 `replays/same_policy_rows_v1/`。这组比较是“同一量化策略，旧算子对新算子”，其倍率不能写成“相对 BF16 native”。

单元测试在空闲 GPU 上串行运行：

```bash
docker exec -e PYTHONPATH=/workspace/src/runtime_opt:/workspace/src:/workspace/src/tests \
  -e KERV_DEVICE_BACKEND=musa kerv-musa-runtime \
  /workspace/.venv/bin/python -m unittest test_calibrated_w8a16_ops -v
```

涵盖尾部维度、部分/全部/零保护组、非单位 inverse scale、bias、非连续输入、空行和错误输入。微基准的数值对照是同一解码权重的原生 GEMM；量化对 BF16 原模型的精度问题必须另行评估。

本轮实际结果和限制见[量化算子报告](../../docs/acceptance/MOORE_QUANTIZED_KERNEL_REPORT_20261027.md)。
