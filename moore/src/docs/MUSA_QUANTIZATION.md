# 摩尔 MUSA W8A16 权重压缩操作指南

本入口根据2026-10-26用户确认新增，属于独立实验配置。实测结论及质量失败见工作区 `docs/acceptance/MOORE_QUANTIZATION_REPORT_20261026.md`。显存下降不能代替任务成功率检查；默认 `optimized` 仍使用BF16。

## 实现与适用范围

验证环境为MTT S5000、torch/torch_musa 2.9.1、Transformers 4.40.1及Triton MUSA 3.2，使用现有 `kerv-musa-runtime` 容器；未安装TorchAO或bitsandbytes，也未升级驱动/运行时。

专用配置仅压缩OpenVLA verifier的32层MLP，每层gate/up/down共96个投影。attention、vision、drafter、embedding、动作头、激活及KV保持原精度；当前checkpoint的 `pretraining_tp=1`。其他模型、分组组合及并行训练方式不属于本轮验证范围。

对每个输出通道沿输入维分组：

```text
scale = max(abs(weight_in_group)) / 127     # FP32，零组用1
qweight = clamp(round(weight / scale), -127, 127).to(INT8)
weight_for_this_call = (qweight.float() * scale).to(BF16)
output = native_linear(BF16_input, weight_for_this_call, unchanged_bias)
```

这是有损的对称权重量化。INT8权重与FP32 scale在推理期间常驻MUSA；执行到某层时，Triton只为该层生成临时BF16矩阵，随后调用MUSA原生浮点GEMM，函数返回后释放临时权重。不缓存整份BF16模型，也不在推理期间向CPU卸载权重。不能将其描述成原生INT8 GEMM加速。

量化在CPU加载checkpoint并组装drafter之后、整个模型搬到MUSA之前完成。原BF16 checkpoint只读，每次启动重新准备INT8 buffers，CPU转换时间包含在模型加载时间中。本轮未实现独立量化checkpoint的导出/快速加载格式。

| 分组 | 原选中权重(bytes) | INT8+FP32 scale(bytes) | 静态节省(bytes) |
| --- | ---: | ---: | ---: |
| 128 | 8657043456 | 4463788032 | 4193255424 |
| 32 | 8657043456 | 4869586944 | 3787456512 |

这是96个投影的存储核算，实际GPU峰值还包括其他模型权重、KV、工作区及临时矩阵。量化配置关闭BF16投影融合，因此reserved变化还包含融合打包及allocator缓存行为；不能把reserved总差全部归因于INT8位宽。

## 启动

从宿主机工作区运行group128候选：

```bash
cd /root/KERV
KERV_MUSA_PROFILE=quantized src/scripts/run_musa.sh \
  --max-tasks 1 --trials 1 --max-episode-steps 220 \
  --output-dir /workspace/outputs/quantized_new_run_id
```

group32候选：

```bash
cd /root/KERV
KERV_MUSA_PROFILE=quantized src/scripts/run_musa.sh \
  --max-tasks 1 --trials 1 --max-episode-steps 220 \
  --override inference.runtime.w8a16.group_size=32 \
  --output-dir /workspace/outputs/quantized_g32_new_run_id
```

输出目录每次使用新ID，保留失败结果。只跑某个官方初始状态时，可额外传入：

```text
--override ++inference.runtime.acceptance_task_id=0
--override ++inference.runtime.acceptance_init_state_id=0
--override ++inference.runtime.seed=7
```

`--load_in_8bit` / `--load_in_4bit` 是另一套checkpoint加载接口，MUSA仍明确拒绝；不要用它们替代上述profile。专用配置使用 `w8a16.enabled=true, backend=musa, groups=verify_mlp, group_size=128, strict=true`。转换失败会退出，不能将BF16回退计为量化通过。

## 检查实际执行

普通启动在 `manifests/w8a16.json` 中记录实现、分组、每层名称/形状/字节和 `fallback=false`。验收采集还保存 `quantization.jsonl`：

1. `after_model_load`：96个INT8 buffer实际在 `musa:0`，scale为FP32，分组与候选吻合。
2. `process_exit`：96个模块均有真实前向调用计数。
3. `memory.jsonl`：加载前开始的allocated/reserved峰值，以及首步后的独立稳态窗口。
4. `steps.jsonl` / `episodes.jsonl`：完整七维有限动作、步数、任务成功与终止原因。

标准部署命令不会自动启用验收采集。需要完整证据时，使用 `acceptance.run_matrix` 和冻结清单；该工具将 `KERV_ACCEPTANCE_CONTEXT` 注入子进程。验收不能只看打印了“W8A16”，也不能只看模型文件大小。

## 代码与复算

| 文件 | 职责 |
| --- | --- |
| `runtime_opt/KERVRuntimeOptimization/weight_only_int8.py` | 分组CPU量化、INT8 buffers、逐层反量化及原生Linear |
| `runtime_opt/KERVRuntimeOptimization/weight_only_int8_kernels.py` | MUSA/CUDA Triton单次分配反量化 |
| `openvla/experiments/robot/openvla_utils.py` | 在模型上卡前准备量化权重 |
| `openvla/experiments/robot/flagos_ops.py` | 量化后端选择、manifest及严格错误处理 |
| `runtime_opt/FlagScale/configs/kerv_libero_goal_musa_w8a16.yaml` | 独立量化profile |
| `acceptance/quantization_report.py` | 从原始数据重算显存、完成率、执行证明 |
| `acceptance/quant_replay.py` | 同观测/历史的BF16与量化动作误差对照 |
| `tests/test_weight_only_int8.py` | 8项CPU/MUSA数值、边界、存储、缓存及模型前向测试 |

```bash
cd /root/KERV
PYTHONPATH=src python3 -m acceptance.quantization_report \
  --campaign outputs/quantization/20261026T021321Z_w8a16
```

正式统计需扩展任务、初始状态和seed；不能用单个成功或节省显存掩盖失败样本。原有BF16的跨进程一致性问题与量化质量分别记录，量化没有在本轮修复前者。
