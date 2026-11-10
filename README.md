# KERV 三平台部署代码

本仓库保存 2026 年 11 月 10 日从三台实验机读取的 KERV 源码快照，包括工作区中已提交和未提交的代码、平台配置、启动脚本、技术文档及第三方依赖源码。

| 平台 | SSH 别名 | 仓库目录 | 服务器源目录 | 已有启动入口 |
| --- | --- | --- | --- | --- |
| 摩尔 MTT S5000 | `baai-moore-1` | `moore/` | `/root/KERV`，核心源码在 `src/` | `src/scripts/run_musa.sh` |
| 昇腾 910C | `baai-new-1` | `ascend/` | `/root/KERV` | `scripts/run_kerv_moore_ascend.sh` |
| 沐曦 C550 | `baai-metax-1` | `metax/` | `/root/KERV` | `scripts/run_kerv_metax.sh` |

## 目录与部署关系

每个平台目录保持对应服务器 `/root/KERV` 的文件布局。使用既有启动脚本时，该平台目录应对应服务器的 KERV 工作区根目录。脚本包含现有容器和挂载路径，需要按目标机器的环境配置使用。

昇腾容器实际挂载 `/public-flash/zhiyuan/liyuzhuo/data/KERV` 到 `/data/data/KERV`。本次比较的 1748 个共有代码、配置和文档文件与 `/root/KERV` 一致，另补入实际运行目录中的 `configs/libero/config.yaml`。

摩尔使用容器 `kerv-musa-runtime`，宿主 KERV 目录挂载为 `/workspace`，Python 环境为 `.venv`。昇腾使用 `kerv-ascend` 和实际运行目录中的 `.venv`。沐曦使用 `kerv-metax`，宿主 KERV 目录挂载为 `/workspace`，Python 环境为 `.venv-metax`。各平台模型路径均使用 `checkpoints/openvla-libero-goal` 与 `checkpoints/kerv-drafter`。

## 快照状态

摩尔 `src` 原有 Git 基点为 `4abb3dfdbe24783a738a23a893a6bd374beca0a3`，本快照包含其后的工作区修改。昇腾源目录通过文件清单记录版本。

沐曦原有 Git 基点为 `cf41388bdeb16a80ea94bb20c18e45ff07e88e47`。已验证配置为 `numerical_fixed`。快照同时保存暂停前准备的 SpecVLA 对照、组合压缩配置与 GPU 等待脚本，这部分工作的状态见 `metax/docs/METAX_MODEL_PERFORMANCE.md`。模型性能调试与等待任务处于用户要求的暂停状态，使用时需单独决定是否恢复。

## 文件清单与依赖

每个平台的 `SOURCE_MANIFEST.json` 记录服务器源路径、各文件大小和 SHA-256（文件内容校验值）、已有 Git 版本及工作区状态。昇腾额外配置的实际来源也记录在清单内。上传前逐文件核验了下载内容。

快照范围为源码、配置、文档和仿真资源。权重、运行输出、日志、虚拟环境、缓存、聊天导入归档和编译二进制由清单中的 `excluded` 列表记录。部署时需要另行准备模型权重、Python 环境和对应厂商运行环境。

第三方源码和资源保留原目录中的许可证及声明。具体授权以各组件自身许可证为准。
