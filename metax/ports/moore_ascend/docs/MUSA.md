# 在摩尔 MUSA 上运行 KERV

甲方交付的完整部署流程、技术大纲、计时边界、指标复核和证据索引见 [详细交付指南](../../README.md)。本文件为运行摘要。

本工作区使用 KERV 推理代码和 SpecVLA 的 LIBERO-Goal drafter 权重，
通过 FlagScale 启动。实际设备是 MTT S5000，采用 `torch_musa` BF16。
提供 `native` 基线和 `optimized` 优化配置。
这是摩尔芯片适配；当前环境没有验证昇腾硬件。

## 算子和推理优化

优化配置为 `runtime_opt/FlagScale/configs/kerv_libero_goal_musa_optimized.yaml`。
算子命中日志与运行统计用于核对实际执行，配置开关不等同于命中。

| 路径 | 优化配置 |
| --- | --- |
| Residual Add + RMSNorm | KERV adaptive Triton 内核，按行数选择 verifier 层，保留 BF16 舍入位置 |
| 全局 RMSNorm 替换 | 关闭；FlagGems 注册路径导致动作轨迹改变，未通过完整任务成功门禁 |
| QKV、Gate-Up 投影 | 合并权重，调用 MUSA 原生 GEMM；只启用本机微基准有收益的行数 |
| Q/K RoPE | KERV adaptive Triton 融合，drafter 和 verifier 均适配 MUSA |
| SwiGLU | 原生 SiLU 和乘法原位执行；实测更慢的融合内核保持关闭 |
| 草稿树、验证树 | 补齐发布包缺失的 C++ 树构建源文件；保留全部验证路径 |
| Kalman、KV 提交 | 批量构建和提交，沿用原生 MUSA 张量算子 |
| 解码工作区、提示词、草稿 KV | 复用工作区、缓存提示词 token、持久化 drafter KV |
| Verify-Accept 控制传输 | 复用 MUSA 设备缓冲区和 pinned CPU 缓冲区 |
| Graph、常驻 verifier KV、量化 | 关闭；当前驱动拒绝 Graph 捕获，常驻 verifier KV 组合有性能退化 |

FlagGems 使用兼容当前 Triton 3.2 的 v3.0，修复 MUSA `PrivateUse1`
dispatch key 和缺失的通用 heuristics；保留显式配置和测试入口。
默认优化配置采用 FlagOS/KERV Runtime 的算子，FlagGems 全局注册保持关闭。
普通 RMSNorm 微基准约有 1.8–2.2 倍收益，但启用全局替换的完整轨迹到 220 步
未完成任务，因此不能以单算子速度替代任务精度门禁。

草稿候选改用稳定排序：BF16 下子节点可能与父节点同分，原来的非稳定
Top-K 会丢掉父节点。相同分数优先保留较早的祖先，避免生成非法树。
新增测试覆盖同分候选、C++/Python 树一致性、融合算子数值、Kalman 批量与逐分支
预测一致性、缓存和控制传输。RoPE 在长短序列及 indexed/direct 布局下与
原生 BF16 结果逐元素一致。

`outputs/musa_operator_benchmark_final.json` 保存真实模型尺寸的微基准；
微基准加速比仅对应单个算子，不代表整个任务的加速比。

## 已完成的验收

最终优化配置完整运行成功，仍在第 170 步完成同一任务。
原生基线与优化运行的任务、初始状态、种子和 220 步规划上限一致，
两条轨迹的七维动作逐步比较结果保存在 `outputs/musa_optimization_comparison.json`。
170 条动作及其 token 全部一致，验证周期有 1 次差异。

| 完整 episode，同一张 MTT S5000 | 原生基线 | 优化配置 |
| --- | --- | --- |
| 任务成功 / 动作数 | 成功 / 170 | 成功 / 170 |
| 首步，含首次推理懒初始化/预热 | 10.208 秒 | 9.260 秒 |
| 排除首步的平均动作推理耗时 | 0.523 秒 | 0.230 秒 |
| 排除首步的中位数 / P95 | 0.519 / 0.926 秒 | 0.237 / 0.350 秒 |
| 验证周期 / Kalman 补偿动作 | 488 / 42 | 489 / 42 |

本次完整运行的平均延迟比为 **2.27 倍**，降低约 **56%**。
这是各配置一条轨迹的比较，不代表整套 LIBERO 的成功率或多次重复的统计结论。
日志确认解码工作区和提示词缓存各命中 170 次，控制缓冲传输 489 次，
批量 KV 提交 447 次，草稿 KV 只分配 1 次且没有回退；Graph 回放为 0。
最终回归共 18 项：17 项通过，Graph 捕获因当前驱动不支持而跳过 1 项；
日志为 `outputs/musa_optimization_tests.log`。

最终验收：`outputs/kerv_libero_goal_musa_optimized_final/acceptance.json`；
动作、运行计数、实际算子实现和录像保存在同目录。
数值不符合门禁的全局 RMSNorm 实验日志也保留在 `outputs`，未用作成功结果。

以下是最初原生基线的验收记录（2026-09-30，单张 MTT S5000）：

| 验收 | 结果 |
| --- | --- |
| MUSA FP32/BF16 运算及 LIBERO reset、初始化、渲染 | 通过 |
| SpecVLA drafter 参数键和形状检查 | 通过 |
| 真实权重 8 步 KERV 推理 | 通过，8 条动作和运行统计 |
| LIBERO-Goal 第 0 个任务、第 0 个初始状态 | **成功，第 170 步打开柜子中间抽屉** |
| 算子、KV 路径、drafter/verifier 前向、启动器回归 | 7 项通过 |
| 新构建镜像的 MUSA 和仿真检查 | 见 `outputs/preflight_image_musa.json` |

完整 episode 的 170 条动作均为有限的七维数值，记录了 488 个验证周期，
42 步采用 Kalman 补偿。首步耗时 10.208 秒；排除首步后，169 步的
平均动作推理耗时为 0.523 秒，中位数 0.519 秒，P95 为 0.926 秒。
计时从 `predict_action` 调用前开始，至动作推理结束后的 MUSA 同步完成；
不包含此前的 CPU 图像裁剪/processor 处理、模型权重加载、仿真步进或录像保存。
首步包含首次实际推理的懒初始化/编译预热等开销，不含循环前完成的权重加载。
计时起点没有设备预同步，先前排队的输入传输尾部也可能被终点同步覆盖；不是纯 kernel 时间。
这是最初原生基线的运行验收；优化配置对比见上表。

## 当前工作区直接运行

宿主机工作区为 `/root/KERV`；KERV 源码位于 `src`，容器挂载为 `/workspace`。
已配置容器名为 `kerv-musa-runtime`，只暴露物理 GPU 0。
`run_musa.sh` 默认选择已通过验收的优化配置并新建带时间戳的输出目录。
`run_kerv.py` 的通用入口仍默认选择原生配置，传入 `--musa-profile optimized` 即可切换。

```bash
cd /root/KERV

# 8 步真实权重检查；建议每次使用新输出目录
./src/scripts/run_musa.sh --musa-profile optimized --max-episode-steps 8 \
  --output-dir /workspace/outputs/smoke_musa_new

# 完整 episode，默认最多 220 个策略动作
./src/scripts/run_musa.sh --musa-profile optimized \
  --output-dir /workspace/outputs/libero_goal_new

# 在宿主机核对完整输出
python3 src/scripts/summarize_musa_run.py \
  outputs/libero_goal_new --max-steps 220

# 原生基线仍可单独运行
./src/scripts/run_musa.sh --musa-profile native \
  --output-dir /workspace/outputs/libero_goal_native_new
```

默认执行 LIBERO-Goal 第 0 个任务的一次 episode；`--max-tasks` 和 `--trials`
可扩大评测范围。动作日志以追加方式写入，复跑请使用新的输出目录。

## 文件位置

- `outputs/kerv_libero_goal_musa/acceptance.json`：本次完整 episode 验收摘要。
- `outputs/kerv_libero_goal_musa_optimized_final/`：优化后的完整成功 episode。
- `outputs/musa_optimization_comparison.json`：完整任务延迟对比及实际优化命中。
- `outputs/musa_operator_benchmark_final.json`：本机算子微基准及 RMSNorm 数值差异。
- `outputs/kerv_libero_goal_musa/actions.jsonl`：原始动作、执行动作、终止标记、阈值和耗时。
- `outputs/kerv_libero_goal_musa/runtime_stats.jsonl`：接受长度、验证周期、Kalman 和运行配置。
- `outputs/kerv_libero_goal_musa/rollouts/`：任务日志、逐步计时及 MP4。
- `outputs/kerv_libero_goal_musa/flagscale/inference_logs/`：FlagScale 启动脚本和完整输出。
- `outputs/smoke_musa_v5/`：通过的 8 步检查。
- `outputs/preflight_musa.json`、`outputs/preflight_image_musa.json`：设备和仿真检查。
- `outputs/reproducibility.json`、`outputs/requirements-musa.freeze.txt`：源码、环境和验证记录。
- `outputs/reproducibility-optimized.json`、`outputs/requirements-musa-optimized.freeze.txt`：优化配置的复现记录。

## 环境复现

保留当前 `src` 修改，准备下列上游源码版本于 `third_party`：

| 组件 | 上游 | 版本 |
| --- | --- | --- |
| KERV | https://github.com/zhengzihaoPKU/KERV | `4abb3dfdbe24783a738a23a893a6bd374beca0a3` |
| FlagScale | https://github.com/flagos-ai/FlagScale | `f8e9a62c499d526a6ffa47f533bcf342330b7326` |
| FlagGems（目录 FlagGems-v3） | https://github.com/flagos-ai/FlagGems | v3.0，`ed8c864062f7d037d3de6574df8895f45a60fd53` |
| SpecVLA（权重来源参考） | https://github.com/PineTreeWss/SpecVLA | `56ef2985e25c7903d2759254efb754327fb975f7` |
| LIBERO | https://github.com/Lifelong-Robot-Learning/LIBERO | `8f1084e3132a39270c3a13ebe37270a43ece2a01` |

本地 LIBERO 从固定版本源码归档解压，`UPSTREAM_REVISION` 记录版本。
只需要套件自带的 BDDL、资产和初始状态，无需下载训练数据集。

```bash
# 补充与当前 MUSA/Triton 环境兼容的 FlagGems 源码
git clone --branch v3.0 --depth 1 https://github.com/flagos-ai/FlagGems \
  third_party/FlagGems-v3

# 适用于已经准备源码的工作区；需要 Docker 的 mthreads runtime
./src/scripts/bootstrap_musa.sh

# 下载真实 verifier 和 SpecVLA drafter（已存在的文件会复用）
docker exec kerv-musa-runtime /workspace/.venv/bin/python \
  /workspace/src/scripts/prepare_checkpoints.py

docker exec kerv-musa-runtime /workspace/.venv/bin/python \
  /workspace/src/scripts/check_drafter.py /workspace/checkpoints/kerv-drafter

docker exec kerv-musa-runtime /workspace/.venv/bin/python \
  /workspace/src/scripts/preflight_musa.py --simulation

docker exec kerv-musa-runtime /workspace/.venv/bin/python \
  -m unittest discover -s /workspace/src/tests -v
```

`Dockerfile.musa` 基于摩尔的
`rc5.1.0-v2.9.1-S5000-py310` 镜像。`requirements-musa.txt` 继承镜像内的
PyTorch 2.9.1、torch_musa 2.9.1 和 torchvision 0.24.1，避免安装 CUDA 版 torch。
Transformers 为 4.40.1、timm 为 0.9.10、NumPy 为 1.26.4。
仿真使用 CPU OSMesa 渲染，库提取到独立目录，避免覆盖容器挂载的摩尔驱动库。

Verifier 使用公开的 `openvla/openvla-7b-finetuned-libero-goal`，固定 Hugging Face
revision 为 `fa5ae1e7509348889295bba8e08621d8b55e9baf`。
Drafter 来源为 SpecVLA README 中的 LIBERO-Goal Google Drive 链接；
`checkpoints/*/source.json` 记录来源。Drafter 的冗余 embedding 是 32064 行，
KERV 原有加载逻辑丢弃它并共享 verifier embedding；其余参数检查完全匹配。

## 适配范围

- 增加 `--backend musa`、设备选择、随机种子和 MUSA 同步计时。
- MUSA 使用 eager attention，显式应用 attention 配置，避免依赖 CUDA Flash Attention。
- 复用 KERV/FlagOS 具身算子注册；原生和优化配置分别保存实际算子命中记录。
- `native` 保留原始草稿树生成、并行验证、动态阈值和 Kalman 补偿。
  `optimized` 额外启用上述已适配路径；Graph、Triton 静态树 attention 和量化保持关闭。
- LIBERO 的初始化状态加载适配 PyTorch 2.9；补充包入口以支持 editable 安装。
- FlagScale 启动脚本保留推理子进程退出码；KERV 验收遇到异常立即失败。
- 修复 YAML CLI 对 `off` 和空字符串的转换，防止合法运行配置在解析时改变含义。

外部兼容修改由 `scripts/apply_dependency_fixes.py` 和
`patches/flagscale-inference-exit-status.patch`、`patches/flaggems-v3-musa.patch`
保存并可重复应用。
当前验收范围为 MUSA BF16 推理，CUDA/昇腾硬件和训练流程未在本机回归。
