# muDNN 3.4.0 版本对照测试

日期：2026-10-27。证据：`outputs/softmax_latest/20261027T065058Z`。

## 结果

官方 APT 源当前提供的 muDNN 3.4.0.0 在本轮测试中未复现原先的 Softmax 概率错误：共 12,000 次调用，逐元素与 CPU double 参考比较，0 次失败。其中 9,000 次使用 Runtime 5.2.0，另外 3,000 次使用 Runtime 5.1.0。

旧 muDNN 3.3.0 换用 Runtime 5.2.0 后仍出现 3 / 3,000 次错误。因此，单独更新 Runtime 没有消除本次旧库故障。

| muDNN | Runtime | 行数 × 列数 | 尾部输入 | 错误调用数 / 调用数 |
| --- | --- | --- | --- | ---: |
| 3.4.0 | 5.1.0 | 8128 × 537 | -10000 | 0 / 3000 |
| 3.4.0 | 5.2.0 | 8128 × 537 | -10000 | 0 / 3000 |
| 3.4.0 | 5.2.0 | 8128 × 538 | -10000 | 0 / 3000 |
| 3.4.0 | 5.2.0 | 8128 × 539 | 0 | 0 / 3000 |
| 3.3.0，交叉对照 | 5.2.0 | 8128 × 537 | -10000 | **3 / 3000** |

先前 muDNN 3.3.0 / Runtime 5.1.0 的相同合成用例分别为：537 列 7/3000、538 列 3/3000、539 列有限输入 8/3000。原始记录保存在上一份报告及其封存证据中。

以上结论是“这些用例下未复现”，不等于证明新版所有 Softmax 形状或完整业务流程都无错。

## 实际测试的版本范围

官方当前公开的框架发行版为 [torch_musa 2.11.0.post1](https://github.com/MooreThreads/torch_musa/releases/tag/2.11.0.post1)；[SDK 5.2.0 组件表](https://docs.mthreads.com/musa-sdk/version-5.2.0/releasenote/)列出 muDNN 3.4.0、Toolkit 5.2.0 和 Linux Driver 5.2.0。

本轮只在隔离进程中加载下载的 muDNN 和 Runtime，没有升级宿主驱动或替换现有推理环境。硬件为物理 GPU0、MTT S5000；宿主驱动仍是 `3.3.5-server`，进程实际映射的用户态驱动为 `libmusa.so.4.3.5`。

因此，本轮验证了新版 muDNN/Runtime 在当前机器上的这些用例，没有完成“整套最新版软件栈”、torch_musa 2.11.0.post1 或模型闭环验证。不能把结果扩展为完整新版环境已经验收通过。

## 方法与版本核验

- 复用 `src/acceptance/repro_mudnn_softmax.cpp`，仅调用 muDNN 和 musart，无 PyTorch、模型或量化实现。
- 合成输入、CPU double 参考和容差与旧版复现保持一致。每次调用检查完整输出及行概率和。
- 从官方 HTTPS APT 软件源下载指定 DEB，按当次 Packages 索引中的 SHA-256 校验；仅解包到本轮证据目录，不执行安装脚本。
- muDNN 3.4.0.0 包 SHA-256：`06bce3268590d614fd278150a5c45941273d92a4636d3e1f64400b1a02add8ea`。
- 通过子进程 `/proc/<pid>/maps` 记录全部 muDNN、musart 和 libmusa 路径，确认每组确实使用指定库，没有混入旧 muDNN 组件。
- 使用 3.4.0 C 头文件重编译了同一份复现源码。新旧复现程序的 ELF `.text` SHA-256 完全相同：`f67a3da7dfcc1e2addefe29d2b2d0bad65753581edefd8305ddb86597a85111a`。
- GPU 测试按组串行执行，仅使用 GPU0。

抓取工具、隔离运行工具分别为 `src/acceptance/fetch_mudnn_candidate.py`、`src/acceptance/run_mudnn_candidate.py`。各组 `results.json` 保存调用参数、退出码、库选择检查和输出统计，`matrix_results.json` 汇总五组结果。

下载过程中将单连接传输切换为 Range 并行续传，旧下载进程是本次自行创建的进程，已明确终止并保留 `-15` 操作记录。完整文件仍经过最终 SHA-256 校验。构建阶段的头文件目录及链接名配置失败也保留了日志，不计作数值测试结果。

## 二进制对照

提取了两个版本同名 `mp_31` Softmax 特化，机器码确实不同：

| 项目 | muDNN 3.3.0 | muDNN 3.4.0 |
| --- | ---: | ---: |
| 函数长度 | 13528 字节 | 13264 字节 |
| 解码指令数 | 1119 | 1102 |
| 显式 `AF32.EXP2 BURST4` 指令数 | 5 | 0 |

旧函数 SHA-256：`90c98bb41522c20cf56e8af07bdd59fe0f83738f3bdf73a092118bff7ffdf62c`。

新函数 SHA-256：`bae4c16323438324900cc4915a0de92bfa6956f454a379c4ea2f4da7195be2ec`。

这些是指令差异的观测，没有做单点消融，不能据此断言 EXP2 的 burst 路径就是根因。也没有厂商针对该故障的修复说明可以确认。

## PR 处理

按用户要求，[MooreThreads/torch_musa #158](https://github.com/MooreThreads/torch_musa/pull/158) 已关闭，未合并；关闭时间为 2026-10-27 15:06:53（北京时间）。正文补充了上述新版测试及旧版交叉对照结果。

原补丁、旧版复现和新版验证记录保留在本地。关闭 PR 没有修改已安装框架、业务默认入口或驱动。
