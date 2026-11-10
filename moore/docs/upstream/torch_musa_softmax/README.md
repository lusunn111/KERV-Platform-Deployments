# torch_musa Softmax 上游提交包

> **状态：PR 已按用户要求关闭，未合并。** 新版 muDNN 3.4.0 本轮共 12,000 次未复现，旧版 3.3.0 配合新 Runtime 的交叉对照仍为 3/3000 次错误。详见[新版测试报告](../../acceptance/MOORE_SOFTMAX_LATEST_VERSION_20261027.md)。以下保留原提交和验证记录。

本包针对本机 muDNN 3.3.0 / MTT S5000 宽行 FP32 Softmax 的偶发错误，提供有限范围的上层规避补丁。内部精确故障机制尚未确认。

- [完整定位报告](../../acceptance/MOORE_SOFTMAX_ROOTCAUSE_20261027.md)
- [候选补丁](../../../src/patches/torch-musa-softmax-mudnn3300.patch)
- [英文 PR 正文](PR_BODY.md)
- [纯 C++ 复现程序](../../../src/acceptance/repro_mudnn_softmax.cpp)
- [实际派发及回归验证工具](../../../src/acceptance/check_softmax_upstream_patch.py)

补丁基于 `MooreThreads/torch_musa` main：`2838e77d774a8bcc2593126ad07a3314a399d517`。包含派发规避、19 项回归及纯 C++ 复现。只编译头文件版本为 muDNN 3.3.0、运行架构为 mp_31、FP32 最后一维大于 512 且不能被 4 整除时启用规避。

## 验证

- 本机编译修改后的完整派发代码到隔离 namespace：3,000 次逐元素 CPU 对照，0 次失败。
- 以工作进程内的候选注册运行新增上游测试：19 passed，16.94 秒。
- 补丁格式及反向应用检查通过。
- 完整新版 torch_musa 构建/CI 和其他芯片未测试；后续 muDNN 3.4.0 隔离测试见新版报告。

证据：`outputs/softmax_rootcause/20261027T055434Z/`。

## 应用与测试

在对应基线的完整 torch_musa 源码树执行，先检查再应用：

```bash
git apply --check /root/KERV/src/patches/torch-musa-softmax-mudnn3300.patch
git apply /root/KERV/src/patches/torch-musa-softmax-mudnn3300.patch
```

按该仓库的构建说明重编译后运行：

```bash
MUSA_VISIBLE_DEVICES=0 python -m pytest \
  tests/unittest/operator/test_softmax.py -q \
  -k 'wide_row_boundaries or unaligned_masked_rows_repeated'
```

本机验证用隔离 namespace，不要求先替换运行环境，具体命令见完整定位报告。

## 发布状态

已关闭：[MooreThreads/torch_musa #158 — Work around intermittent fp32 softmax errors on mp_31](https://github.com/MooreThreads/torch_musa/pull/158)，未合并。

- fork：`lusunn111/torch_musa`
- 分支：`fix/softmax-mp31-mudnn3300`
- 提交：`b2becfa`，仅包含派发修改、测试和纯 C++ 复现这 3 个文件。
- 发布前追加直接对照：同一新增回归在原始库上失败，12 个元素不符，最大绝对误差约 0.006055878；候选代码通过该回归的 3,000 次循环。
- PR 正文保留了内部精确根因、完整 main 构建及其他版本未验证的限制。提交成功不代表维护者已评审或合并。

用户完成 GitHub 设备授权后，通过独立 CLI 创建 fork 并提交。远端 PR 元数据、文件清单及提交哈希已复核，保存在证据目录 `upstream_pr.json`、`upstream_pr_files.json`。

提交后已退出本次 CLI 登录并删除临时凭据目录。

厂商二进制、反汇编、模型 tensor 和登录凭据不包含在公开提交包中。
