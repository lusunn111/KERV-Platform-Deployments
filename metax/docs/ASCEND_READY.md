# Ascend 环境与恢复运行

当前环境和结果以 [README_ASCEND.md](../README_ASCEND.md) 为准。
模型、仿真、原始 TF 预处理、公共算子和 NPU 图执行均已准备，不需要在线下载才能推理。

推荐运行入口已更新为 `scripts/run_kerv_moore_ascend.sh`；优化代码隔离于
`ports/moore_ascend/`，默认seven协议、eager执行，已验证模型1.970×（对KERV native）。
本页下方保留旧图适配的环境检查命令；新版闭环与测速命令请使用主README最新章节。

## 目录

- 工作目录 `/root/KERV`。
- 主机运行副本 `/public-flash/zhiyuan/liyuzhuo/data/KERV`。
- 容器 `kerv-ascend`，挂载 `/data/data/KERV`，Python `.venv/bin/python`。
- verifier `checkpoints/openvla-libero-goal`、drafter `checkpoints/kerv-drafter`。
- LIBERO `third_party/LIBERO`，路径配置 `configs/libero/config.yaml`。

修改后同步项目代码到运行副本，不能只编辑 `/root/KERV` 就直接启动容器。
运行封装可用 `KERV_CONTAINER` / `KERV_CONTAINER_ROOT` 覆盖默认容器与目录。

## CPU readiness 检查

```bash
scripts/create_kerv_container.sh
scripts/run_kerv_container.sh --check-ready --load-model --check-env
```

检查权重长度和索引的 982 个 tensor、CPU 加载 verifier/drafter、processor 和软件渲染。
不进行 NPU 模型推理。报告在运行副本 `outputs/readiness.json`。
需要推理时先检查 `npu-smi info`，物理 chip ID 使用 `KERV_NPU_ID` 指定；图优化闭环
约占 31 GB HBM，建议至少预留 36 GB。容器存在并不代表 NPU 仍有空闲。

## 准备新环境

在兼容 Ascend 容器内执行：

```bash
apt-get update
apt-get install -y libosmesa6 libgl1 libegl1 libglib2.0-0
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements-ascend.txt
.venv/bin/python -m pip install --no-deps --no-build-isolation -e openvla
.venv/bin/python scripts/check_kerv_ready.py --load-model --check-env
```

保留镜像匹配的 torch / torch_npu，以及已装的 FlagGems / NPU Triton。
TensorFlow 2.18.0 是原图像预处理依赖，默认不得静默使用 PIL 近似替代。
LIBERO 通过 PYTHONPATH 导入，官方初始状态只使用本地已知 NumPy 文件。
当前运行离线加载，不依赖代理；旧文档中的 baai-proxy 包装命令目前不存在。

## 执行

```bash
KERV_NPU_ID=4 scripts/run_kerv_container.sh --dry-run
KERV_NPU_ID=4 scripts/run_kerv_container.sh \
  --output-dir outputs/npu_smoke_repro --max-tasks 1 --max-episode-steps 8
```

完整闭环、图执行和测速命令见 README。封装为前台运行，测试结束时模型进程退出并释放
NPU 内存；保留容器和本地权重，可在芯片空闲时直接恢复。
