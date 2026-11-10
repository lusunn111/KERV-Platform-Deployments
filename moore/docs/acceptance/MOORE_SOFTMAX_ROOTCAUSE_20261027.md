# 摩尔 Softmax 独立复现、源码检索与二进制分析报告

> **后续状态：** 新版 muDNN 3.4.0 在本机共 12,000 次调用中未复现，旧版交叉对照仍出错；PR #158 已按要求关闭，未合并。见[新版验证报告](MOORE_SOFTMAX_LATEST_VERSION_20261027.md)。本报告保留旧版本的定位过程和原提交记录。

日期：2026-10-27。证据目录：`outputs/softmax_rootcause/20261027T055434Z`。

## 1. 结论与证据边界

已经用仅链接 **muDNN、musart 的纯 C++ 程序**复现概率错误。该程序不依赖 PyTorch、KERV、模型权重、量化实现或 Triton。因此，当前环境中 muDNN Softmax 执行路径本身存在可独立复现的数值正确性问题。

进一步定位到了 `libmudnn_ops.so` 中的 `softmax_accurate_column` 特化。通过单独解包官方编译器组件取得 `libmusaasm.so`，完整解码出目标函数的 **13,528 字节、1,119 条 GPU 指令**。另用自己编译的 `mp_31` 加一内核校验了解码链路，能得到对应的加载、加法、写回和结束指令。

**目前完成的是组件和内核特化级定位，尚未证明具体哪条内部语句或指令导致错误。** 对齐相关尾部处理、共享内存交互和指令依赖仍是调查方向。二进制中确有多处同步指令，不能把“缺少 barrier”写成已经确认的根因；也不能据此区分厂商源码、编译器生成代码或驱动/硬件交互中的缺陷。

已提交公开 `torch_musa` 层的有限范围规避补丁：[PR #158](https://github.com/MooreThreads/torch_musa/pull/158)，包含纯 C++ 复现和回归测试。修改后的实际派发代码通过 3,000 次全概率检查，新加入的 19 项回归全部通过。该补丁属于绕开故障内核的上层修复，不等于修改了 muDNN 内部实现。PR 发布状态单独记录在 `docs/upstream/torch_musa_softmax/README.md`。

## 2. 实测环境

| 项目 | 值 |
| --- | --- |
| 设备 | 物理 GPU0，MTT S5000，80 GiB，`mp_31` |
| 驱动 | `3.3.5-server` |
| MUSA Toolkit | `5.1.0` |
| muDNN | `3.3.0`，commit `bd105d42d4413d10f51cf91785cf4e06acee30ac` |
| muDNN 构建分支 | `release_musa_5.1.0` |
| PyTorch / torch_musa | `2.9.1` / `2.9.1+a18d871` |
| 容器 | `kerv-musa-runtime` |
| Python | `/workspace/.venv/bin/python` |
| 候选补丁基线 | 官方 main `2838e77d774a8bcc2593126ad07a3314a399d517` |

GPU 测试均在 GPU0 串行执行。未重置 GPU、卸载驱动或替换正在使用的库。工具包仅解压到证据目录。

## 3. 从业务现场到独立复现

### 3.1 业务现场输入

使用上一轮已封存记录中的实际 attention 输入：

```text
outputs/stability_diagnosis/20261027T052303Z/
  targeted_attention_v3/difference_r3_f122.pt
```

原始输入是 BF16 `[1, 32, 254, 537]`，Softmax 计算输入先在 CPU 转成 FP32，再拷到 GPU。原始两条业务路径的输入逐 bit 一致。异常表现包括掩码列出现接近行内最大概率的值、整行概率和明显超过 1；错误集中出现在列 514/515 起的部分尾部位置。

对有限掩码值 `-10000`，正常 FP32 Softmax 应在这些位置下溢为零。实际非零概率远超舍入误差，不能用正常浮点误差解释。不同输入下不应一概把所有错误值都描述为“恰好等于最大概率”。

### 3.2 排除输出缓存和极值掩码解释

`src/acceptance/softmax_rootcause_probe.py` 只使用 torch/torch_musa，每种设置重复 2,000 次：

| 设置 | 错误调用数 / 调用数 |
| --- | ---: |
| 原生 `torch.softmax` | 2 / 2,000 |
| 每次先把固定输出缓冲填为 NaN，再调用 `_softmax.out` | 2 / 2,000 |
| 每次先把固定输出缓冲填为 0.25，再调用 `_softmax.out` | 4 / 2,000 |
| 极小有限掩码改成 `-10000` | 1 / 2,000 |
| 掩码改成 `-inf` | 1 / 2,000 |
| 最后一维 537 补到 1024 | 0 / 2,000 |

每种设置结束后都验证输入未被修改。这些结果削弱了“仅输出未初始化”和“仅极小有限掩码溢出”两种解释。零错误表示本轮未观察到，并不是永久无错的证明。

记录：`wrapper_probe_actual_v1/results.json`，以及同目录失败 tensor 和 profiler trace。

### 3.3 直接调用 muDNN C API

`src/acceptance/mudnn_direct_softmax.py` 使用 `ctypes` 调用 `mudnnSoftmaxForwardEx`，PyTorch 仅提供内存和结果检查，不调用 GPU PyTorch Softmax。

- 相同实际输入：**7 / 3,000** 次调用出错。
- handle、tensor descriptor、算子 descriptor 和输出缓冲持久复用。
- 显式关联当前 MUSA stream。
- FP32、最后一维、ACCURATE 算法，工作区大小为 0。
- profiler 确认仍调用同一个厂商 GPU 内核特化。

记录：`direct_capi_actual_v1/results.json`、`trace.json`。

### 3.4 完全脱离 PyTorch 的纯 C++ 程序

源码：`src/acceptance/repro_mudnn_softmax.cpp`。所有输入均通过简单正弦函数生成，CPU 使用 double 计算期望概率。每次 GPU 调用后，把完整输出复制回 CPU，检查每个元素及行概率和。

| 合成输入设置 | 行数 | 错误调用数 / 调用数 |
| --- | ---: | ---: |
| 列数 537，尾部值 -10000 | 8,128 | **7 / 3,000** |
| 列数 538，尾部值 -10000 | 8,128 | **3 / 3,000** |
| 列数 539，尾部值为普通有限值 0 | 8,128 | **8 / 3,000** |
| 原始 537，实际存储/计算补到 540 | 8,128 | **0 / 5,000** |
| 列数 516 | 8,128 | **0 / 3,000** |
| 列数 537，小行数 | 16 | **0 / 10,000** |

539 列实验说明问题可以影响普通有限输入；516 列和 540 列实验说明不能简单归纳成“非 2 的幂都会错”。目前证据支持优先调查不能被 4 整除的宽行特化及尾部处理。小行数未复现不能单独证明竞争条件。

这些调用失败次数是特定机器、版本和负载下的观测，不是稳定的故障概率估计。矩阵中各设置改变了实际工作负载或内核选择，不能只根据补齐后无错就断言某个地址访问必然越界。

## 4. 二进制分析的具体进展

### 4.1 正确的目标库与设备镜像

实际内核位于 `/usr/local/musa/lib/libmudnn_ops.so`；`libmudnn.so` 提供接口层。之前检查的 `libmudnn_tensor_reduce.so` 不是本次目标内核的所在库。

从 ELF 的 `.mt_fatbin` 读取 Clang offload bundle，选择 `musa-mtgpu-mt-musa--mp_31` 设备镜像，再按动态符号表和节地址提取目标函数。

| 项目 | 值 |
| --- | --- |
| 库 SHA-256 | `2a917aa6096619e0dd4b74b6b842f821a1835c7d699aad03ed0545e2f59fa479` |
| 函数虚拟地址 | `0xd565400` |
| 函数长度 | `13528` 字节 |
| 函数机器码 SHA-256 | `90c98bb41522c20cf56e8af07bdd59fe0f83738f3bdf73a092118bff7ffdf62c` |
| 解码结果 | 3,382 个 32-bit word，1,119 条可变长度指令 |

实际 attention 出错特化为：

```text
softmax_accurate_column<float, float, SoftmaxMode(0),
  4, 64, 16, SoftmaxDir(0), 4, 16, SoftmaxFusedMode(0),
  SoftmaxFusedOp<...>, DeviceTensor<...>, false, false, true>
```

实际 attention launch：grid `[508,1,1]`，block `[1024,1,1]`，shared memory 34,380 字节。516 列 profiler 记录到同族但末尾模板参数为 `true,false,false` 的另一特化。

### 4.2 反汇编工具的获取与校验

自带 `llvm-objdump` 对目标机器码给出 `<unknown>`，指定 `mp_31` 后仍没有得到可读指令。随后从已安装驱动中找到其调用 `DisassembleBinary`、加载 `libmusaasm.so` 的线索。

依据[官方 APT 配置说明](https://docs.mthreads.com/musa-sdk/version-5.1.0/install_guide/)，只下载配置 DEB 并读取其中的软件源地址；再读取 Packages 索引，下载 `libmthreads-gpucomp_5.1.0_amd64.deb`，按索引 SHA-256 校验后单独解包。没有执行安装脚本、配置系统 APT 或替换驱动。

适配程序 `src/acceptance/disassemble_musa_binary.cpp` 通过 `dlopen` 调用该库。入口 ABI 是从驱动调用位置推断的内部接口，运行在独立 CPU 分析进程；不作为业务运行依赖。

为检查解码链路，另用本机 mcc 编译 `mp_31` 的已知 `y[i]=x[i]+1` 内核。提取后使用同一解码方式得到加载、加法、写回和结束序列，51 个 word 完整消耗。编译器直接 `-S` 输出 MUSA 汇编的尝试不受支持，随后采用正常 host object 内的设备镜像进行验证。

### 4.3 当前能从指令看到什么

可识别共享内存 `LMA.LMRD/LMWR`、128-bit 向量访问、尾部标量写入、`AF32.EXP2/RCP`、条件控制流和多处 `ASYNC.SYNC`。输出前也存在同步。

因此已有真实指令可供继续分析，但“有指令”并不自动等于“已经找到故障指令”。尚缺少充分的 ISA 内存/依赖语义验证、错误现场数值对应的数据流证据，或能够单点改变行为的受控实验证据。没有将任何猜测写成确定的厂商源代码错误。

提取的厂商设备镜像和反汇编保存在 `vendor_binary_local_only/`；下载工具在 `vendor_tools_local_only/`。公开补丁仅包含自行编写的源码、测试和复现程序。

## 5. 源码检索结果

| 检查范围 | 结果 |
| --- | --- |
| 本机和容器中的 SDK、头文件及源码目录 | 找到 muDNN 公共接口头及库，未找到目标内核源码 |
| `/home/torch_musa`、`/home/pytorch` | 找到框架和调用封装源码；不是厂商 Softmax 内核源码 |
| `/home/mccxadmin/tutor/utilities` 中旧 SDK/muDNN 归档 | 检查归档列表，未发现目标 Softmax 源码 |
| 官方公开 `MooreThreads/torch_musa` | 找到 `SoftMax.cpp` 调用与现有 ATen MUSA 内核复用路径 |
| 官方公开 `MooreThreads/mudnn-frontend` | 检查时为空仓库 |
| `flagos-ai/FlagDNN` | 是独立实现及厂商库参考接口，不是目标 muDNN 私有内核源码 |
| 内部仓库线索 | 发现构建 checkout 的内部 torch_musa remote；推测的 muDNN 地址未能访问，不能据此确认仓库存在或内容 |

“未找到”只覆盖本次实际检查范围，不表示证明所有渠道都没有源码。

## 6. 已实现的上游候选修复

补丁：`src/patches/torch-musa-softmax-mudnn3300.patch`。

命中条件同时满足：

1. 编译时 `MUDNN_VERSION == 3300`。
2. 当前设备 `getMUSAArch() == 310`。
3. FP32 普通 Softmax，最后一维、非空输入。
4. 最后一维长度大于 512 且不能被 4 整除。

命中时调用已经随框架提供的、musify 后的 ATen MUSA Softmax 内核。函数名保留 `structured_softmax_cuda_out`，执行的是本机 `libmusa_kernels.so` 中的 MUSA 代码。functional 和 `out=` 通过共同派发点覆盖，继续沿用现有自动微分及 backward 算子。

这是根据观测故障特化设置的保守规避条件，没有声称该范围所有形状都已分别复现。版本门禁取编译头文件；只替换动态库而不重编译不会自动改变门禁。muDNN 3.4.0、新版完整 torch_musa 构建及其他芯片尚未验证。

### 6.1 验证方式和结果

为验证修改的实际 C++ 派发代码，单独编译完整候选 `SoftMax.cpp` 到隔离 namespace。框架安装文件保持原样。新增 pytest 的执行进程临时注册候选实现，让 `torch.softmax`、`out=` 和 autograd 走正常框架接口；该注册只在该工作进程有效。

| 验证 | 结果 |
| --- | --- |
| 实际 attention 输入，单独测试既有 ATen MUSA 内核 | 3,000 次未观察到概率错误；抽查 CPU 最大绝对误差 `1.7881393e-7` |
| 修改后的实际派发代码，合成 `[8128,537]` | 3,000 次逐元素与 CPU 比较，0 次失败；抽查最大绝对误差 `1.8626451e-9` |
| 新增上游回归 | **19 passed**，66 deselected，16.94 秒 |
| 边界形状 | 511、512、513、516、537、538、539、540、1024 |
| 接口和语义 | 非连续输入、`out=` 输出地址、独立 stream、autograd 梯度均通过 |
| 非最后一维 muDNN 派发 | 与 CPU 对照通过 |
| 补丁格式 | `git diff --check`、反向 `git apply --check` 通过 |

发布前又运行了完全相同的新增压力回归，但使用未修改的已安装算子。测试明确失败：12 / 4,364,736 个元素不符，最大绝对误差为 0.006055878009647131，位置 `(3988,514)`，CPU 参考为 0。记录：`operations/upstream_regression_original_negative_v1/stdout_stderr.log`、`original_negative_junit.xml`。这与候选代码同一回归的 3,000 次通过结果构成直接对照。

isolated 编译验证经历了两次构建头文件配置失败，之后修正分析程序的 include 路径与 pybind 头后通过；失败日志保留在 `upstream_patch_full_dispatch_v1/v2` 操作记录中。

合成 `[8128,537]` 的一次诊断计时批次（20 次预热、500 次调用、批次边界同步）得到 muDNN 0.04211 ms/次、候选 0.03512 ms/次。该数字仅是单算子单批观测，没有重复统计，也不代表整模型加速。

### 6.2 与当前 KERV 入口的关系

本报告中的候选补丁尚未替换已安装 `torch_musa`。业务侧继续使用上一轮报告中的异常行检查及局部重算入口。两者是不同实现，应分别验证；候选原生 ATen 内核的舍入顺序可能改变正常行，不能沿用异常行修复入口此前的动作一致性结果。

本轮未新增闭环成功率评测，不关闭量化精度门禁，也不解释此前 IndexSelect 非法地址问题。

## 7. 复现操作

### 7.1 纯 C++ 复现与补齐对照

在本工作区宿主机执行：

```bash
docker exec kerv-musa-runtime g++ -O3 -std=c++17 \
  /workspace/src/acceptance/repro_mudnn_softmax.cpp \
  -I/usr/local/musa/include -L/usr/local/musa/lib \
  -Wl,-rpath,/usr/local/musa/lib -lmudnn -lmusart \
  -o /tmp/repro_mudnn_softmax

docker exec -e MUSA_VISIBLE_DEVICES=0 kerv-musa-runtime \
  /tmp/repro_mudnn_softmax --columns 537 --repeats 3000

docker exec -e MUSA_VISIBLE_DEVICES=0 kerv-musa-runtime \
  /tmp/repro_mudnn_softmax --columns 537 --pad 4 --repeats 5000
```

程序返回码：0 为本轮未观察到错误；1 为复现出错误概率；2 为参数或运行接口错误。先确认 GPU0 空闲，再顺序执行。出现错误的测试退出 1 是预期诊断证据，不能把它改成成功或丢弃日志。

### 7.2 验证候选实际派发及上游回归

输出目录每次必须为新目录：

```bash
docker exec -e MUSA_VISIBLE_DEVICES=0 -e PYTHONPATH=/workspace/src \
  -e MAX_JOBS=2 kerv-musa-runtime /workspace/.venv/bin/python \
  -m acceptance.check_softmax_upstream_patch \
  --output /workspace/outputs/softmax_patch_dispatch_repeat

docker exec -e MUSA_VISIBLE_DEVICES=0 -e PYTHONPATH=/workspace/src \
  -e MAX_JOBS=2 kerv-musa-runtime /workspace/.venv/bin/python \
  -m acceptance.check_softmax_upstream_patch --upstream-tests \
  --output /workspace/outputs/softmax_patch_regression_repeat
```

该验证工具依赖当前容器的构建 checkout 和 generated headers，适用于本机复核。移植到其他机器时优先在完整 torch_musa 构建环境应用补丁并运行新增测试。

### 7.3 二进制提取与分析工具

`src/acceptance/extract_musa_kernel.py` 从 ELF/offload bundle 读取指定符号，记录库与函数 SHA-256；`disassemble_musa_binary.cpp` 对已提取机器码调用本地解包工具。具体长符号和完整调用参数见 `operations/extract_vendor_kernel_verified_v1/command.json`、`operations/disassemble_vendor_verified_v1/command.json`。

## 8. 后续技术问题

内部精确机制仍需以下证据之一：对应版本的厂商内核源码和编译参数；充分的 ISA 语义及可验证的数据流分析；或受控、可归因的指令/实现修改前后对照。当前可向维护者提交纯 C++ 复现、版本、受影响特化、形状对照及上层规避补丁，供其在内部定位。

公开发布不包含模型数据、厂商镜像、厂商反汇编或登录凭据。报告中的零故障记录和单算子耗时均保持各自测试范围。
