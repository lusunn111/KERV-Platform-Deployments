# 昇腾复现差异排查（2026-10-15）

> 摩尔归档对照新增结论：我此前的归一化、embedding和KF统一7-token改变，与摩尔成功参考
> 的算法行为不同；运行时也未完整移植。此前实验保留为各自配置结果，后续须先建立兼容参考。
> 具体差异及已补回的C++树算子见 [MOORE_ASCEND_COMPARISON.md](MOORE_ASCEND_COMPARISON.md)。

用户确认摩尔与 NVIDIA 上的 KERV 准确率正常，因此本次检查先对齐输入和模型语义，
再评价优化性能。此前只跑一个失败 episode、将 1.23× AR 回放作为主要结果，
不足以判断昇腾准确率或对齐论文加速比。

## 确认和修正的差异

### 1. 图像预处理

原环境缺少 TensorFlow，部署时曾使用 PIL 近似 Lanczos resize 和中心裁剪。
两者的 fractional crop、像素中心、量化等处理不完全相同。相同 LIBERO 初始状态下：

| 检查 | 实测 |
|---|---:|
| resize 像素平均绝对差（0–255） | 1.104917 |
| crop 像素平均绝对差（0–255） | 3.712598 |
| crop 像素值不同比例 | 79.540% |
| processor tensor 最大绝对差 | 1.473003 |

恢复原 TensorFlow 预处理后，同一 AR 任务从 220 步失败变为 157 步成功。
因此该近似输入不能继续用作准确率等价的默认路径。
已安装 TensorFlow 2.18.0，并加入 `requirements-ascend.txt`；缺少 TF 时默认报错，
仅显式 `KERV_ALLOW_APPROX_PREPROCESS=1` 才允许诊断用近似路径。

证据：[输入对比](../outputs/preprocessing_comparison/report.json)、
[对比脚本](../scripts/compare_kerv_preprocessing.py)、
[恢复后 AR episode](../outputs/npu_ar_tensorflow/rollouts/libero_goal_spec_relaxed/episode_summary.json)。

### 2. drafter 数值语义和检查点

SpecVLA 原始 RMSNorm 在 FP32 中求平方均值/归一化，然后转换回输入类型再乘 gamma。
本地运行时优化版用 `F.rms_norm`；CANN 融合的中间舍入顺序可能不同。
NPU drafter 已恢复原 FP32/cast/gamma 顺序，CUDA 分支保留原有实现。
`KERV_NPU_DRAFT_RMS_REFERENCE=0` 仅用于对照旧 NPU 行为。

同时保留 SpecVLA 检查点中的 embedding，避免未经验证地覆盖它；缺少 embedding
的检查点仍可从 verifier 继承。实际权重对比，二者转换为 BF16 后只有
约 0.0225% 元素不同，最大差为 2.98e-8；这不是已经证明的主要失败原因。
不能把这两项修正单独声称为已完成的跨芯片数值等价验证。

### 3. 现成 FlagOS 能力

容器已有 FlagGems 5.3.1、Triton 3.5.0，其活动 target 是
`backend=npu, arch=Ascend910_9382`。之前统一禁用 Triton 的判断过于宽泛。
当前已实际调用 FlagGems Ascend RMSNorm，以及基于 FlagGems pointwise 工具的
SiLU/multiply 实现；16 个独立案例通过 tensor 容差检查。

但 RMSNorm/SwiGLU 的整模型回放仍出现 action 差异（旧参考下最大约 0.215），
因此 `flag_gems` 保持实验选项，未作为准确率通过的默认后端。
attention 接入也保留为实验项，不能仅根据 API 存在宣称推理已加速。

### 4. 原有阶段图复用 NPUGraph

StageGraphCache 增加 `torch_npu.NPUGraph` 后端，保留输入克隆、拷贝、结构签名和缓存上限。
NPU 图使用独立内存池，避免不同阶段的输出复用同一池造成意外覆盖。
prompt 和 drafter 图在完整 episode 中已有实际重放记录。

扩展 verifier 图时，发现原树 mask 布尔索引触发 `aclnnNonzeroV2` 和主机同步，
无法在 NPUGraph 中捕获。已替换为等价的固定形状 `masked_fill_`；对原始 CPU tree mask
显式转移到 NPU。verifier 图只接收由 tensor tuple 组成的不可变 past，
并将 tree mask 作为每次更新的输入，避免复用旧 mask；不支持的状态保留 eager。

修复后的小规模回放中，两种 KERV 的 32 个 action 完全一致、最大差 0；
实际捕获 20 个阶段图，prompt 重放 47 次、drafter 重放 101 次、verifier 重放 36 次，
捕获失败计数 0。该小规模测试不是全套 LIBERO 的准确率或速度复现。

证据：[图回放检查](../outputs/npu_verify_graph_parity/benchmark.json)。

## 闭环检查

### 动作 token 数量的两个边界问题

接入 action-only head 后，首个观测的前 7 个 token 与完整 head 完全一致，但旧结束条件
在恰好生成 7 个时没有终止，随后生成第 8 个。解码取最后 7 个，导致整组动作错位。
已修正 equality 边界，完成 7 个动作 token 就结束，不依赖额外的 EOS logits。

KF 补齐也曾把已由 verifier 生成的第一个动作 token 当成额外的起始 token，
又补齐 7 个 DoF，最终产生 8 个；历史取前 7 个、动作取后 7 个，二者错位。
已保留 verified root，只补剩余 6 个 DoF，并从正确的 KF 维度偏移取值。
推理入口现在检查 speculative action 的 token 数量，避免静默解码错位。

CPU 回归检查覆盖已接受 0、2、6 个 draft token 的情况，全部通过；
修复后的 NPU action-head/图优化回放 32 个动作与正确参考完全一致（最大差 0）。
QKV、Gate-Up 各自和组合的 packed GEMM 在 NPU 上仍出现动作差异，故仅保留显式实验选项；
即使局部更快，也不作为准确率通过的配置。FlagGems attention 使用显式展开的 HF mask，
适配其 head stride 与 bool mask 约定，仍需独立的动作检查。
权重打包融合仍在单独检查，不能把不通过一致性检查的版本作为默认。

证据：[修复前 token 追踪](../outputs/action_token_diagnosis.json)、
[CPU 回归检查](../tests/test_action_protocol.py)、
[action head 回放](../outputs/npu_action_head_parity/benchmark.json)。

原 TF 预处理、SpecVLA drafter、NPU 参考 RMSNorm、native 算子与 prompt/drafter 图：

| LIBERO Goal 任务 | 结果 | 策略步数 |
|---|---|---:|
| open the middle drawer of the cabinet | False | 220 |
| put the bowl on the stove | True | 85 |
| put the wine bottle on top of the cabinet | True | 104 |

上表是动作 token 边界修复前的历史闭环 smoke 检查，2/3 成功；不能用它断言整个
LIBERO 的准确率已与 NVIDIA/摩尔一致。第一项在关闭 KERV 树运行时优化的参考配置下
也失败，所以不能把所有失败都归因于图优化。

证据：[三任务摘要](../outputs/npu_kerv_reference_rms/rollouts/libero_goal_spec_relaxed/episode_summary.json)。

## 测速口径修正

额外发现 KERV `eagenerate` 原来无条件每 4 步强制 KF，即使 `use_kalman=False`。
这使旧 Naive SD / SpecVLA 配置和定义不符，也与 README 的 4 次 SD 后 1 次 KF 不符。
已将强制补齐条件改为先检查 Kalman enabled，并将周期显式配置化：默认 4，保留原码的
已运行算法设置；可用 `+inference.runtime.kalman_interval=5` 对齐 README 的文字描述。
不静默更改周期来改变准确率/加速比。测速脚本检查关闭 KF 的配置没有强制 KF。
旧错误开关下的 partial 结果保留作诊断，不用于结论。

[上游 README](https://github.com/zhengzihaoPKU/KERV#main-results) 明确：LIBERO Goal
2.34× 的基线是 Naive VLA+SD，AR 是另一种比较。新增 `naive-sd`：
验收阈值 0、关闭 Kalman、关闭树运行时优化；`specvla` 为固定阈值 9。

正式回放改用成功 AR episode 的全部 157 张连续图像，覆盖前、中、后期，保持历史和
阈值随 step index 更新。每个配置完整预热一遍，再测量两遍并检查确定性；使用同一
权重、输入、裁剪、BF16 与规划长度。统计 NPU 同步的 model/pipeline 时间，
不包含模型加载、预热、MuJoCo 和视频；原始延迟与 action 均保留。

最终完整回放输出目录：`outputs/npu_benchmark_final/`；`npu_benchmark_corrected` 中旧 partial 不作正式结果。
每个配置完成后先保存 `benchmark.partial.json`，全部通过后写 `benchmark.json`。
部分完成或动作一致性失败的结果不作为最终验证结果。


## 7-token 修复后的最终闭环（2026-10-15）

原 TF、BF16、SpecVLA Goal drafter、KF 周期 4、native 公共算子、NPU 图与 action head。
10 个 Goal 任务各第一个官方初始状态、各 1 次，最多 220 步：6/10 成功。
总计 1,502 次 action 均恰好 7 token，KF 历史与实际解码全部一致。
这不是论文多初始状态的完整准确率评测，无法据此宣称与 NVIDIA/摩尔一致。

证据：[完整摘要](../outputs/npu_goal_final/rollouts/libero_goal_spec_relaxed/episode_summary.json)、
[动作和调用检查](../outputs/npu_goal_final/validation.json)。

CANN fusion attention 在单个 BF16 张量上与 SDPA 相同，但 16 个整模型观测中有 4 个
动作不同、15 个 DoF 改变、最大差 0.169874；没有采用其 1.17× 诊断速度作为正式增益。
FlagGems attention 候选触发 AICore 错误，退出后原生 NPU 检查正常；未重置设备。


## 完整动作门限与最终测速

157 个连续观测、预热 1 遍、测量 2 遍。图+action head 候选仅第 48、56 帧的动作不同，
最大差 0.016087；短 16 帧门限没有覆盖到这个情况。保留图和公共算子、关闭 head 后
314 个 action 全部与参考完全一致。最终配置关闭 head，不把它作为可无损启用的 NPU 优化。

最终 pipeline 平均延迟：AR 263.343 ms、Naive SD 353.248 ms、SpecVLA 266.494 ms、
KERV native 365.500 ms、KERV+公共算子+图 362.245 ms。相对 Naive SD 为 0.975×，
相对 KERV native 为 1.009×，未达到 2.4×。4 组确定性基线按元数据/输入哈希严格核对后复用，
优化组在相邻串行窗口重测；原始报告记录复用来源。

原生 FlagScale `run_kerv_entrypoint` 配置已补齐 `inference=ascend`，
将直接使用其 bundled 算子运行关闭 head 的最终闭环，不经过 release bridge。


## FlagScale 原生入口最终闭环完成

`inference=ascend`、native 公共算子、三阶段 NPUGraph、head=false，
Goal 10 个任务各第一个官方状态、各 1 次：6/10 成功，1,451 次 action 全部恰好 7 token，
KF 历史/解码全部一致。实际记录 5 类 NPU 算子。完整 benchmark 314 个优化 action 与参考
精确一致，但相对 Naive SD 为 0.975×，尚未复现 2.4×。全部自己的测试模型进程正常退出，
容器与依赖保留 ready；未停止其他用户任务。

最终证据：[native adapter 闭环检查](../outputs/flagscale_ascend/validation.json)、
[正式测速](../outputs/npu_benchmark_final/benchmark.json)。此前 graph+head 的6/10仅保留作诊断。


## 摩尔运行时完整迁移后的修复与复测

以上是根目录旧适配的历史。新版本隔离于 `ports/moore_ascend/`，经同一FlagScale入口启动。
此前未获得收益的主要实现差异是：缺失C++树源文件、NPU设备条件未覆盖控制缓冲和持久KV、
批量KV提交关闭、验证路径被限制。移植这些运行时路径后已有真实命中与整体速度提升。

保留摩尔原动作协议时，原生/优化都220步失败且动作完全一致；强制KF共55步，均有历史/解码
错位。仅将两组共同切换到显式seven协议，修正root+6、KF偏移与SD边界后，二者都115次动作
完成Goal task0，且115/115动作与token相同、历史全部一致。这表明此环境的动作边界需要修正，
并不支持“优化导致准确率下降”的归因。

此前怀疑NPU RMSNorm不能作为结论：使用原F.rms_norm，对32768个随机BF16值与CPU参考
精确一致；手写cast顺序反而可改变结果。当前保留摩尔原F.rms_norm和共享embedding。
证据：`outputs/moore_npu_norm/check.log`。

闭环排除首动作506.257→263.306ms（1.923×），只覆盖单个配对成功episode。
固定157观测、完整预热一遍、测两遍的报告作为正式性能证据，见主README最新章节；
不与旧图版本或原摩尔8-token协议的诊断速度混用。

RoPE直接设备适配数值虽正确，模型速度却显著下降。新NPU内核将旋转索引改为片上gather，
32头S279约0.236ms，精确一致；16帧整模型动作一致但尚无额外加速，默认仍关闭。
`outputs/moore_seven_rope_vector_smoke/benchmark.json`保留实际命中与计时，不能把微基准加速
加到最终模型比值。完整迁移对照见`docs/MOORE_ASCEND_COMPARISON.md`。


修正协议正式完整回放已完成：157观测×2测量，314动作精确一致；模型514.554→261.260ms，
1.970×；pipeline520.354→266.628ms，1.952×。证据：
`outputs/moore_seven_runtime_full/benchmark.json`。这是KERV原生→运行时的加速，尚未达到2.4×。
