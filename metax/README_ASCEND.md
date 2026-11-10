# KERV 昇腾运行、正确性排查与加速比

## 技术大纲与测评交付（2026-10-27）

[技术大纲](docs/acceptance/ASCEND_TECHNICAL_OUTLINE_20261027.md)（[Word](docs/acceptance/ASCEND_TECHNICAL_OUTLINE_20261027.docx)）、
[综合测评](docs/acceptance/reports/ASCEND_EVALUATION_REPORT_20261027.md)（[Word](docs/acceptance/reports/ASCEND_EVALUATION_REPORT_20261027.docx)）、
[复算数据与证据索引](outputs/acceptance/ascend_technical_evaluation_20261027/summary.md)。

本机技术文档和测评材料已交付：复核169个源证据文件、120个完整回合（配置复用20个task/init/seed）及6组时延。
综合报告统一nearest-rank P95；默认BF16，省内存显式推荐W8A16/BF16混合。课题正式验收的未满足项单列，不由本机结果替代。

---

## 2026-10-27再优化：保持投影形状的QKV BMM与视图缓存

未通过全部预设门槛，保留上一版W8A16/BF16混合为推荐候选，BMM＋缓存视图仅供显式实验。
只在257..320行使用batch=3的BF16 QKV GEMM，其他形状保持原三个投影；权重在CPU预打包，缓存视图共享同一NPU存储。
现有MLP W8A16/BF16混合与Ascend Triton SwiGLU保持不变。

| 本轮最终缓存候选 | 结果 |
|---|---:|
| Goal10/init1、init2 | 8/10、8/10 |
| 与上一版逐步观测和动作一致 | 2649/2649步 |
| 同卡对上一版平均模型时延变化 | -0.3045% |
| 同卡对BF16 runtime平均模型时延变化 | +0.1063% |
| 完整allocated峰值GB | 14.9509～14.9512 |
| 相对历史BF16的allocated节省GB | 2.0887～2.0935 |

两组各157帧×3遍AB/BA。allocated、reserved和驱动HBM分别记录，口径不能互相替代；allocated节省尚未达到2GiB。
动作一致指相对上一版量化候选；小样本不证明相对BF16总体精度无损。
中间18层拼接版6/10、7/10和初版BMM略慢的结果全部保留。默认仍为BF16。

```bash
KERV_NPU_ID=14 KERV_QKV_BACKEND=bmm KERV_QKV_LAYERS=all KERV_LIBERO_INITIAL_STATE_OFFSET=1 \
  scripts/run_kerv_quant_qkv_ascend.sh \
  --output-dir outputs/quant_bmm_new --max-tasks 10 --trials 1 \
  --max-episode-steps 220
```

[完整过程、失败路径与限制](docs/acceptance/reports/ASCEND_QKV_BATCHED_DISPATCH_20261027.md)、
[最终完整结果](outputs/acceptance/qkv_bmm_cached_20261027/reports/qkv_results.json)。

---

## 2026-10-27后续：QKV速度选项与CPU预打包

已增加显式的verifier 32层BF16 QKV融合，复用现有FlagScale KERV接口。
CPU预打包避免在NPU上临时替换权重，消除了该融合新增的约3GB设备驻留；推理中没有CPU权重搬运。

| 最终CPU预打包版的新测对照 | 参考均值ms | 新版均值ms | 变化 |
|---|---:|---:|---:|
| 同chip14，上一版混合量化→新版 | 294.275 | 281.860 | −4.22% |
| 同chip10，BF16→新版 | 297.516 | 285.368 | −4.08% |

每组157帧×3遍、逐帧AB/BA交替。allocated全程峰值约14.95GB，比BF16省 **2.088～2.091GB**。
但最终Goal10/init1、init2为 **6/10、9/10**，上一版为8/10、8/10；合计15/20低于16/20。
**未通过逐初态质量门槛，上一版W8A16/BF16混合仍是推荐候选。QKV仅作为实验性速度选项。**

reserved与驱动HBM也独立记录：init1同chip0的驱动10秒采样峰值为22,284→20,487MB，
不能把allocated省2GB等同于所有口径均严格省2GB。相关差异与失败尝试完整保留。

显式测试QKV版本时，先检查目标chip空闲并使用新输出目录：

```bash
KERV_NPU_ID=14 KERV_LIBERO_INITIAL_STATE_OFFSET=1 \
  scripts/run_kerv_quant_qkv_ascend.sh \
  --output-dir outputs/qkv_quant_new --max-tasks 10 --trials 1 \
  --max-episode-steps 220
```

[完整过程、缓存诊断与限制](docs/acceptance/reports/ASCEND_QKV_FUSION_20261027.md)、
[最终结果与逐步数值复现](outputs/acceptance/qkv_cpu_pack_20261027/reports/qkv_results.json)。
本轮共完成60个回合，后两轮为同一20条轨迹的加载修复复现，不能视为60个独立质量样本。

---

## 2026-10-27补充：W8A16/BF16混合与SwiGLU融合

最新完成60个完整闭环回合，推荐显式试用：**verifier中间18层MLP W8A16/group32，
15/16/19/20层down保留BF16，gate/up预打包融合，Ascend Triton SwiGLU保留BF16中间舍入**。
通过现有FlagScale adapter启动，新增算子位于本地NPU适配层。

| 本轮最终候选 | BF16 | W8A16/BF16混合 |
|---|---:|---:|
| 全程allocated峰值GB，含加载 | 17.040 | 14.954 |
| 同卡157帧×3遍平均模型时延ms | 293.300 | 290.277 |
| 模型时延P95 ms | 504.565 | 489.535 |
| Goal10/init1成功数 | 7/10 | 8/10 |
| Goal10/init2成功数 | 7/10 | 8/10 |

**省2.085GB（1.942GiB），平均时延降低1.03%，P95降低2.98%。**
三遍平均时延均降低；本轮两初态共14/20→16/20。
满足2GB十进制目标，尚不满足2GiB；小样本与重复使用的任务不能证明总体精度无损。
默认仍为BF16。全W8A16和W8A8/W8A16混合的失败候选也保留在报告中。

```bash
KERV_NPU_ID=14 KERV_NPU_QUANT=mlp_dynamic_int8 \
  KERV_NPU_QUANT_POLICY=/data/data/KERV/outputs/acceptance/weight_only_hybrid4_20261027/policy.pt \
  KERV_LIBERO_INITIAL_STATE_OFFSET=1 \
  scripts/run_kerv_moore_ascend.sh \
  --output-dir outputs/w8a16_hybrid4_new --max-tasks 10 --trials 1 \
  --max-episode-steps 220
```

运行前检查物理chip空闲并使用新输出目录。入口沿用历史名称`mlp_dynamic_int8`，
该policy实际为W8A16/BF16，未使用INT8激活。
[完整过程与复现](docs/acceptance/reports/ASCEND_MIXED_PRECISION_OPERATORS_20261027.md)、
[最终质量、内存和计时结果](outputs/acceptance/weight_only_hybrid4_20261027/reports/hybrid4_results.json)。

---

## 2026-10-27：量化融合算子与保守混合配置

已完成80个闭环回合。通过现有FlagScale KERV adapter，使用昇腾原生INT8矩阵乘，
合并gate/up与SwiGLU/动态量化；第15、16、19、20层down保留BF16。

| 本轮同卡实测 | BF16 | 混合配置 |
|---|---:|---:|
| 全程allocated峰值GB，含加载 | 17.040 | 14.802 |
| 完整157帧平均模型时延ms | 296.218 | 289.865 |
| 模型时延P95 ms | 514.555 | 481.520 |
| Goal10/init2成功数 | 7/10 | 7/10 |
| Goal10/init1成功数 | 7/10 | 6/10 |

**省显存2.237GB（2.084GiB），平均模型时延降低2.14%，P95降低6.42%。**
三遍交替测量均未变慢。两初态合计14/20 → 13/20，仍有质量退化，不能承诺不降精度；
原BF16默认保持不变。结果限于本机和本轮回放，小样本不证明总体质量边界。

```bash
KERV_NPU_ID=14 KERV_NPU_QUANT=mlp_dynamic_int8 \
  KERV_NPU_QUANT_POLICY=/data/data/KERV/outputs/acceptance/quant_noslow_20261026T054934Z/calibrated_hybrid4/policy.pt \
  KERV_LIBERO_INITIAL_STATE_OFFSET=2 \
  scripts/run_kerv_moore_ascend.sh \
  --output-dir outputs/w8a8_hybrid4_new --max-tasks 10 --trials 1 \
  --max-episode-steps 220
```

运行前检查目标chip空闲和输出目录未使用；`KERV_NPU_QUANT=none`使用BF16参考。
[过程、实现、所有候选和限制](docs/acceptance/reports/ASCEND_W8A8_MEMORY_LATENCY_20261026.md)、
[完整混合配置结果](outputs/acceptance/quant_noslow_20261026T054934Z/reports/hybrid4_results.json)。

---

## 2026-10-26：精度优先量化试验结果

本轮推荐候选为 **中间18层MLP W8A16/group32**。完成70个完整闭环回合：

| 配置 | 开发初态0成功数 | 独立初态1成功数 |
|---|---:|---:|
| BF16参考 | 7/10 | 7/10 |
| 原group128 | 7/10 | 7/10 |
| **group32** | **9/10** | **9/10** |
| 激活校准混合group | 未跑该配置闭环 | 8/10 |

group32独立验证的allocated全程峰值为14.826GB，比BF16节省 **2.218GB（2.066GiB）**。
另行同卡逐帧交替测速：245.002 → 250.610ms，时延增加 **2.29%**。
小样本不证明正式成功率损失边界；本轮实现激活校准与逐投影policy，完整GPTQ仍待接入。

```bash
KERV_NPU_ID=14 KERV_NPU_QUANT=mlp_int8 \
  KERV_NPU_QUANT_LAYERS=7:25 KERV_NPU_QUANT_GROUP_SIZE=32 \
  KERV_LIBERO_INITIAL_STATE_OFFSET=1 \
  scripts/run_kerv_moore_ascend.sh \
  --output-dir outputs/quant_g32_new_run --max-tasks 10 --trials 1 \
  --max-episode-steps 220
```

运行前检查目标chip空闲，使用新输出目录。
[本轮完整报告](docs/acceptance/reports/ASCEND_ACCURACY_AWARE_QUANTIZATION_PILOT_20261026.md)、
[原始任务与计时数据](outputs/acceptance/quant_accuracy_20261026T045119Z/reports/results.json)。

---

## 2026-10-26：已验证的INT8显存配置

项目已明确允许量化。新增NPU原生W8A16路径：CPU加载阶段量化verifier中间18层MLP，
权重INT8、激活BF16、group128，仍经FlagScale adapter执行原KERV运行时。

| 本轮Goal10试验 | BF16优化版 | INT8中间18层 |
|---|---:|---:|
| allocated全程峰值GB，含加载 | 17.044 | 14.705 |
| 成功回合/总数 | 7/10 | 7/10 |
| 观察到的单步模型均值ms | 286.898 | 328.289 |

**显存减少2.339GB（2.179GiB）**。这不是量化额外加速：不同芯片和轨迹的单步时延仅描述成本。
4个任务的成败发生互换；小样本不能证明正式多状态评测下损失严格小于5%。
默认仍可使用BF16参考，显存优先时显式选择量化配置：

```bash
KERV_NPU_ID=4 KERV_NPU_QUANT=mlp_int8 KERV_NPU_QUANT_LAYERS=7:25 \
  KERV_NPU_QUANT_GROUP_SIZE=128 scripts/run_kerv_moore_ascend.sh \
  --output-dir outputs/quant_middle18_reproduce --max-tasks 1 --trials 1 \
  --max-episode-steps 220
```

需先确认物理chip空闲、输出目录未使用。`KERV_NPU_QUANT=none`恢复BF16。
[完整量化报告](docs/acceptance/reports/ASCEND_INT8_MEMORY_VALIDATION_20261026.md)、
[本轮原始结果](outputs/acceptance/quant_20261026T0215Z/reports/quantization_results.json)。

### INT8变慢原因核查

同卡、固定64观测、逐帧AB/BA交替运行，每组预热64次再测128次：BF16 **245.478ms**，INT8 **254.301ms**，时延增加 **3.59%**。
验证轮数316 → 314，原始接受长度略升。真实输入的量化矩阵乘比BF16慢约17%～34%，并增加host包装操作。
原先跨卡、不同轨迹的14.4%不能全部归为量化的固定代价；串行测试顺序也存在明显波动。
[原因核查报告](docs/acceptance/reports/ASCEND_INT8_SLOWDOWN_DIAGNOSIS_20261026.md)、
[同卡对照原始结果](outputs/acceptance/quant_diagnosis_20261026T032238Z/diagnosis.json)。

精度优先的下一版采用真实VLA数据校准、逐投影BF16/INT8选择及敏感投影的小group：
[量化机制设计](docs/acceptance/reports/ASCEND_ACCURACY_AWARE_QUANTIZATION_DESIGN_20261026.md)。
该方案尚待实施与闭环验证；不能把局部tensor误差降低当作任务精度已经提升。

---


## 2026-10-19：按 TestOutline 的本机实验

本轮仅当前昇腾。资源检查后使用空闲物理chip4做完整配对、chip6做三轮回放、chip8做独立诊断。
**全部为新实测，没有把历史速度填入本轮结果。**

| 本轮检查 | 结果 |
|---|---|
| 完整Goal task0/init0/seed7，B1/B2配对 | 两组均115动作成功；观测/动作/token/done精确一致 |
| 固定157观测，AB/BA/AB三轮 | 每配置471动作全部精确一致，跨轮确定 |
| 回放模型 / pipeline加速 | **2.102× / 2.080×**，对KERV B1 |
| 单配对闭环含首步模型 / 策略 / 含仿真墙钟 | 1.872× / 1.831× / 1.450× |
| 框架allocated峰值B1→B2 | 16,886,204,928→17,031,623,168 bytes，增加145,418,240，未省2GB |
| 8步诊断Tensor.item计数B1→B2 | 3535→8；不是独立消融，不能单独归因全部收益 |
| 定向NPU / 统计边界测试 | 4项 / 5项通过；首次失败和修复保留 |

正式B0 Spec-VLA尚未做本后端等价性核对；传输采集仍缺后端全覆盖和DMA方向/bytes关联；
当前KF用历史输出token，阈值按步数衰减，不能称为实测机器人运动反馈。**未宣称课题验收通过或达到2.4×。**

[本轮完整总结](outputs/acceptance/ascend_20261019T0730Z/summary.md)、
[三轮回放原始报告](outputs/acceptance/ascend_20261019T0730Z/reports/replay_results.json)、
[逐回合与资源报告](outputs/acceptance/ascend_20261019T0730Z/reports/results.json)、
[执行工具与命令](docs/acceptance/README.md)。

---

以下为2026-10-15的已验证版本及历史记录，范围和计时口径以各节为准。

测试日期：2026-10-15（北京时间）。使用 OpenVLA verifier、SpecVLA Goal drafter，
经 FlagScale 在 Ascend 上运行 LIBERO Goal。FlagScale 管理 worker，模型前向由 KERV 执行。

## 当前可用版本：摩尔运行时迁移到 Ascend

**推荐入口是 `scripts/run_kerv_moore_ascend.sh`，代码位于 `ports/moore_ascend/`。**
该版本与之前图优化实验隔离，补齐了摩尔成功版的 C++ 草稿/验证树、批量 Kalman/KV 提交、
持久化 drafter KV、解码工作区和 pinned 控制缓冲。保留全部验证路径和稳定同分选择，
保留原 `F.rms_norm` 与共享 embedding；图、packed linear 和 RoPE 融合默认关闭。

### 正确性与计时范围

- LIBERO Goal task0、官方初始状态0、seed7、220步上限：原生版和优化版均 **115次动作后成功**。
  两组 **115/115动作和token完全一致**，每次恰好7 token，Kalman历史与解码一致。
- 同一成功闭环排除首动作：原生506.257ms，优化263.306ms，**1.923×**。
  这是一次配对闭环的描述性计时；正式固定观测速度见下表。
- 配对证据：[paired_episode.json](outputs/moore_seven_checks/paired_episode.json)，
  [原生验收](outputs/moore_seven_native_baseline/acceptance.json)、
  [优化验收](outputs/moore_seven_native_rms/acceptance.json)。
  `native_rms` 是历史诊断目录名，这次成功运行使用原始 `F.rms_norm`，没有启用手写 RMS。
- 这只验证一个配对 episode，不能外推完整 LIBERO 成功率；下方旧版的6/10也不能代替新版评测。

### 正式固定观测测速：模型1.970×，pipeline 1.952×

157个连续观测、每组全序列预热一遍再测两遍，物理chip4串行测量。
**314/314个优化动作与参考精确一致，最大差0，两次重复各自也完全确定。**

| 配置 | 模型均值 ms | pipeline均值 ms |
|---|---:|---:|
| KERV native，seven协议 | 514.554 | 520.354 |
| KERV runtime，seven协议 | 261.260 | 266.628 |
| 加速比 native / runtime | **1.970×** | **1.952×** |

模型计时不含CPU预处理/processor；pipeline包含这些准备。两者均不含仿真、模型加载和预热。
实际执行：C++验证树构建1413次、批量Kalman1416次、批量KV提交1299次、
解码workspace471次、控制缓冲传输1416次，drafter KV只分配1次、fallback为0。
这些计数包含预热，不能除以314当作每动作计数。

证据：[完整报告](outputs/moore_seven_runtime_full/benchmark.json)、
[模型日志](outputs/moore_seven_runtime_full/launch.log)、
[测试脚本快照](outputs/moore_seven_runtime_full/benchmark_script_snapshot.py)。
图、RoPE、packed linear、全局FlagGems RMSNorm都关闭；不将整体收益归于某个独立kernel。
相对Naive SD的新版本比值尚未重测，不能拼接旧版Naive SD时间。

**比较基线是 KERV native → KERV runtime，尚不能称为相对 Naive SD 的2.4×。**
摩尔归档的2.2713×也是前者；论文相对Naive SD的比较属于另一种基线。

### 重现已通过的 FlagScale 闭环

```bash
cd /root/KERV
# 物理chip 4；原生与优化串行运行。先用npu-smi确认空闲。
KERV_NPU_ID=4 KERV_MOORE_PROFILE=native scripts/run_kerv_moore_ascend.sh \
  --output-dir outputs/reproduce_moore_native --max-tasks 1 --max-episode-steps 220
KERV_NPU_ID=4 KERV_MOORE_PROFILE=optimized scripts/run_kerv_moore_ascend.sh \
  --output-dir outputs/reproduce_moore_runtime --max-tasks 1 --max-episode-steps 220
```

启动器使用容器 `kerv-ascend`、路径 `/data/data/KERV`，通过
`third_party/FlagScale/run.py` 加载隔离目录的 NPU 配置。模型、LIBERO、TensorFlow 和
Ninja已在该运行环境准备好。工作副本和运行副本位置见下方环境表，修改源码后需先同步。

### 重现固定观测测速

```bash
KERV_NPU_ID=4 scripts/benchmark_kerv_moore_ascend.sh \
  --output /data/data/KERV/outputs/moore_seven_reproduce
```

使用已有成功AR轨迹的157个连续观测（`outputs/npu_ar_tensorflow/observations/`），
每组完整预热一遍，再测两遍；原生/优化串行，同片、同权重、BF16、seed7、动态阈值15→3、
规划长度220、KF开启、seven协议。processor token cache两组都关闭，模型计时与pipeline计时
均同步NPU；不含加载、预热、仿真和视频。保留每帧延迟、动作、输入SHA256和实际命中。

### 本次解决的问题

1. 摩尔代码保留的 CUDA/MUSA/A100 判断阻断 NPU 工作区和控制缓冲。已接入 NPU 设备、
   同步与 pinned 传输，C++树实际编译执行；两种树构建、批量Kalman和KV提交都有命中计数。
2. 原摩尔动作协议在此NPU上原生/优化两组均220步失败，但220个动作完全一致。
   每4步的KF补齐生成root+7，历史取前7、解码取后7；这不是运行时优化引入的差异。
   将两组共同切换为root+6，并修正补齐偏移和SD终止边界后，两组均115步成功。
   启动器默认 `KERV_ACTION_PROTOCOL=seven`，`moore` 只保留复查原行为。
3. 原 `F.rms_norm` 在32768个随机BF16值上与CPU参考完全一致；没有证据将失败归咎于
   NPU RMSNorm。新版保留它，手写FP32替代仅作为显式诊断，不作为默认配置。
4. MUSA RoPE直接迁移虽数值正确，整模型却变慢。另写了NPU片上gather版本，32头S279
   单算子约0.236ms、输出精确一致；16观测整模型动作也一致，但尚无整体收益，因此默认关闭。
   单算子速度不能替代完整模型收益。

实现对照：[MOORE_ASCEND_COMPARISON.md](docs/MOORE_ASCEND_COMPARISON.md)。
排查记录：[ASCEND_INVESTIGATION.md](docs/ASCEND_INVESTIGATION.md)。

---

## 历史：迁移摩尔运行时之前的验证

以下结果属于根目录旧NPU适配及其图配置，保留用于对照，不是上述推荐入口的成绩。

### 旧版验证结果

**FlagScale 原生入口闭环：6/10 成功，共 1,451 次策略动作。**
最终配置：native 公共算子 + NPU 图，关闭 action head。全部动作恰好 7 token，
KF 历史与实际解码完全一致；模型进程正常退出，测试 NPU 已释放，容器保留 ready。
证据：[episode 摘要](outputs/flagscale_ascend/rollouts/libero_goal_spec_relaxed/episode_summary.json)、
[验证记录](outputs/flagscale_ascend/validation.json)、[日志](outputs/flagscale_ascend/launch.log)。
10 段视频复制到 `outputs/flagscale_ascend/videos/`；原始视频仍在运行副本的 `rollouts/`。

| Goal 任务 | 成功 | 策略步数 |
|---|---|---:|
| open the middle drawer of the cabinet | 是 | 128 |
| put the bowl on the stove | 是 | 80 |
| put the wine bottle on top of the cabinet | 是 | 95 |
| open the top drawer and put the bowl inside | 否 | 220 |
| put the bowl on top of the cabinet | 否 | 220 |
| push the plate to the front of the stove | 是 | 120 |
| put the cream cheese in the bowl | 否 | 220 |
| turn on the stove | 是 | 71 |
| put the bowl on the plate | 是 | 77 |
| put the wine bottle on the rack | 否 | 220 |

**正式动作检查通过：314/314 个 KERV 优化动作与参考完全一致，最大差 0。**
最终配置：native 公共算子 + prompt/drafter/verifier NPUGraph，**关闭 action head**。

| 配置 | 模型均值 ms | pipeline 均值 ms | 基线 / KERV+算子+图 |
|---|---:|---:|---:|
| AR | 257.656 | 263.343 | 0.727× |
| Naive VLA+SD，严格阈值 0 | 347.045 | 353.248 | 0.975× |
| SpecVLA，固定阈值 9 | 259.559 | 266.494 | 0.736× |
| KERV native，公共算子/图关闭 | 359.092 | 365.500 | 1.009× |
| KERV+公共算子+NPU 图 | 354.951 | 362.245 | 1.000× |

**该旧配置没有复现 2.4×。** 相对 Naive SD 是 0.975×，本条轨迹上略慢；相对 AR 为 0.727×。
公共算子与图相对 KERV native 为 1.009×，约 0.9% 延迟差，在共享服务器上基本持平，
不将这个小差值宣传为可靠的融合收益。该比较同时包含公共算子和图，并非某一个算子的独立贡献。

实际捕获 25 个图；prompt/draft/verifier 重放分别 470/832/437 次，597 次按有界 cache
策略回退 eager。图确实执行，但并非每个动态形状都有图缓存。当前已验证路径使用 native
算子实现；FlagGems/CANN/packed GEMM 融合候选未通过完整动作门限，不能把其局部速度加到这里。

证据：[完整测速报告](outputs/npu_benchmark_final/benchmark.json)。4 组基线使用相同源码、
输入与参数在前一个串行测量窗口完成，最后优化组重测；脚本核对输入 SHA256、权重路径、
物理设备、BF16、seed、规划长度、KF 周期、预热与重复数，并检查缓存动作确定性。
复用来源和 profile 列表记录在报告 `reused_baseline_report` / `reused_baseline_profiles`。
源码或基线配置改变后必须重新测量；该复用只用于此次隔离 action-head 差异。

此前 graph+action-head 候选在完整 157 帧中第 48、56 帧有差异，最大 0.016087；
去掉 head 后完整回放精确一致，所以 action head 已移出 NPU 默认 profile。
[失败候选报告](outputs/npu_benchmark_graph_head_diagnostic/benchmark.json) 保留为诊断，不能作为通过结果。

这次每个任务仅测第一个官方初始状态，属于 10 个 episode 的闭环检查，不是论文的完整
多初始状态评测。方法之间动作可以不同；KERV 优化前后另行检查动作完全一致。
上游 [KERV README](https://github.com/zhengzihaoPKU/KERV#main-results) 的 Goal 2.34×
相对 **Naive VLA+SD**，本记录同时报告 Naive SD 和 AR 比较，不能混用基线。

## 用到的 FlagScale / FlagOS 接口

| 接口 | 运行中的用途 | 状态 |
|---|---|---|
| FlagScale `run.py`、Hydra runtime 配置 | 配置、worker 启动、日志与产物目录 | 已实际启动全部验证 |
| `torch.ops.flagos_embodied.*` | RMSNorm、SwiGLU、树打包、embedding 与验收控制 | NPU native 实现已实际调用 |
| KERV 静态树缓存、控制传输接口 | 复用树结构、合并控制结果传输 | 已启用 |
| KERV `StageGraphCache` + `torch.npu.NPUGraph` | prompt、drafter、verifier 捕获与重放 | 显式启用、记录实际重放次数 |
| action head | verifier 输出从 32000 vocab 收缩到 256 个动作 token | 短回放通过；完整回放 2 帧有差异，默认关闭 |
| FlagGems 5.3.1 / NPU Triton | RMSNorm、SwiGLU 和 attention 候选融合 | 可调用；旧诊断有动作差异，修复后的完整门限尚未通过，保留实验开关 |
| CANN `npu_fusion_attention` | attention 候选融合 | 小张量检查通过，整模型动作有差异，未采用 |
| QKV / Gate-Up packed linear | 复用已有自适应融合接口 | NPU 整模型动作有差异，未采用 |

所以算子和图接口确实运行起来了，但 **native 接口调用不代表已获得融合内核的性能收益**。
19 个公共算子接口中，默认选择 15 个；microbenchmark 覆盖 16 个案例，其中部分接口
并未在此模型配置中触发。不能把所有注册算子都称为已在闭环执行。

FlagScale 的 [Ascend guide](third_party/FlagScale/docs/ascend.md) 还提供 `vllm-plugin-FL`
服务路线；本次使用的是已经集成的 KERV/OpenVLA PyTorch adapter，模型没有交给 vLLM engine。
因此单独设置 vLLM 的插件变量不会替换这条自定义树验证路径里的 kernels，仍需逐项接入并验证。

主要入口：

- [启动器](run_kerv.py) 与 [容器封装](scripts/run_kerv_container.sh)。
- [FlagScale bridge](runtime_opt/FlagScale/inference_kerv.py) 与 [配置](runtime_opt/FlagScale/configs/kerv_libero_goal.yaml)。
- [NPU 模型 hooks](openvla/experiments/robot/ascend_ops.py)、[设备抽象](openvla/experiments/robot/device_utils.py)。
- [公共算子实现](runtime_opt/KERVRuntimeOptimization/embodied_ops/kerv_ops.py)。
- [回放测速脚本](scripts/benchmark_kerv.py)。

## 环境与本地资源

| 项目 | 配置 |
|---|---|
| 芯片 | Ascend910，物理 chip 4，64 GiB HBM |
| 容器 / 架构 | `kerv-ascend` / ARM64，镜像 `tk-qwen35-env:20260920` |
| Python / torch / torch_npu | 3.12.13 / 2.7.1+cpu / 2.7.1.post8 |
| CANN / npu-smi | 容器 `cann-9.0.1` / 25.5.0 |
| FlagGems / Triton | 5.3.1 / 3.5.0；Triton target `npu`, arch `Ascend910_9382` |
| Transformers / TensorFlow / NumPy | 4.40.1 / 2.18.0 / 1.26.4 |
| MuJoCo / robosuite | 3.2.7 / 1.4.1 |
| 推理 / 仿真渲染 | BF16 / CPU OSMesa |

`torch` 的 `+cpu` 标签不妨碍匹配的 `torch_npu` 扩展执行推理。项目 venv 继承系统包，
保留容器匹配的 torch / torch_npu，避免原 CUDA requirements 覆盖它们。

```text
工作目录：    /root/KERV
运行副本：    /public-flash/zhiyuan/liyuzhuo/data/KERV
容器路径：    /data/data/KERV
```

容器读取运行副本，因此修改工作目录后必须同步代码；产物也要复制回来。
原始 git clone 遇到代理问题，源码使用 GitHub 归档，工作目录没有原始 git 元数据。

| 资源 | 本地路径 / 来源 |
|---|---|
| verifier | `checkpoints/openvla-libero-goal`；[OpenVLA Goal](https://huggingface.co/openvla/openvla-7b-finetuned-libero-goal)，4 个分片、982 个 tensor |
| drafter | `checkpoints/kerv-drafter`；[SpecVLA](https://github.com/PineTreeWss/SpecVLA) 的 [Goal 权重](https://drive.google.com/drive/folders/1W7nBHM9-bf9tq4NQDUfles583OUfrtLv)，734,546,940 字节 |
| LIBERO | `third_party/LIBERO`，commit `8f1084e3132a39270c3a13ebe37270a43ece2a01`；包含 assets/init states |
| FlagScale | `third_party/FlagScale`；[官方项目](https://github.com/flagos-ai/FlagScale/) |
| 仿真路径 | `configs/libero/config.yaml`，readiness 脚本生成 |

所有模型从本地离线加载。初期网络代理返回 503，后续恢复访问补齐模型和依赖；运行不再
依赖代理。当前主机没有旧记录中的 `baai-proxy` 包装命令，不要照旧文档调用它。

## 正确性问题与修复过程

1. **恢复原 TensorFlow 图像预处理。** PIL 近似替代导致同一观测 crop 的 79.54% 像素不同，
   processor 最大差 1.473。恢复 TF 后，同一 AR 抽屉任务从失败变为 157 步成功。
   默认缺 TF 会报错；只有显式 `KERV_ALLOW_APPROX_PREPROCESS=1` 才允许诊断用近似路径。
2. **修复 KF 开关。** 旧代码关闭 Kalman 仍每 4 步强制补齐，污染 Naive SD / SpecVLA 基线。
   现在先检查 enabled，测速也检查没有强制 KF。默认周期 4 保留源代码行为；周期 5 为显式实验。
3. **修复 7 维动作边界。** 普通 SD 满 7 个动作 token 未及时结束；KF 又把已生成的 root
   当成额外 token，产生 8 个。历史取前 7 个、解码取后 7 个，所有 DoF 错位。
   现在 root+剩余 6 维、正确 KF 偏移，并检查 speculative 输出恰好 7 个 token。
4. **旧适配的 drafter 数值选择（不用于新版）。** 使用原 FP32 norm→BF16 cast→gamma 的顺序，保留
   checkpoint 自身 embedding；新版隔离移植改为摩尔的原 `F.rms_norm` 和共享 embedding。
5. **适配 NPU 图捕获。** 复用有界 stage graph cache，独立 graph pool；树 mask 改为固定
   形状 `masked_fill`，避免 boolean advanced indexing 的 Nonzero/主机同步阻止捕获。
6. **逐项检查已有融合接口。** packed GEMM、FlagGems 和 CANN attention 未通过整模型动作
   一致性检查，未放入最终测速配置，诊断加速比不算正式收益。

CPU 回归 [test_action_protocol.py](tests/test_action_protocol.py) 覆盖接受 0、2、6 个 draft
动作的 7 维补齐，3 项通过。详细历史记录及失败实验见
[ASCEND_INVESTIGATION.md](docs/ASCEND_INVESTIGATION.md)。旧 PIL / KF 错误基线下的 1.23×
和失败 episode 已撤下主结果；它们仅是历史诊断，不能证明最终速度或准确率。

## 复现命令

### 检查与准备

```bash
cd /root/KERV
npu-smi info
scripts/create_kerv_container.sh
scripts/run_kerv_container.sh --check-ready --load-model --check-env
KERV_NPU_ID=4 scripts/run_kerv_container.sh --dry-run
```

`KERV_NPU_ID` 为物理 chip ID，并非 npu-smi 左侧板卡编号。完整图优化运行峰值约 31 GB
HBM，建议空闲芯片预留至少 36 GB。readiness 在 CPU 检查权重、processor 和仿真。

新容器环境：

```bash
apt-get update
apt-get install -y libosmesa6 libgl1 libegl1 libglib2.0-0
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements-ascend.txt
.venv/bin/python -m pip install --no-deps --no-build-isolation -e openvla
.venv/bin/python scripts/check_kerv_ready.py --load-model --check-env
```

LIBERO 直接加入 PYTHONPATH，已修复 Python 3.12 的 collections.abc 和 PyTorch 2.7
官方 NumPy 初始状态的加载兼容性。更多准备说明见 [ASCEND_READY.md](docs/ASCEND_READY.md)。

### 10 个 Goal 任务各 1 次

```bash
KERV_NPU_ID=4 scripts/run_kerv_container.sh \
  --output-dir outputs/npu_goal_repro --max-tasks 10 --trials 1 \
  --max-episode-steps 220 \
  --override inference.runtime.embodied_ops.record=true \
  --override inference.runtime.runtime_stats_path=/data/data/KERV/outputs/npu_goal_repro/runtime.jsonl \
  --override +inference.runtime.ascend_graphs=true \
  --override +inference.runtime.ascend_action_head=false
```

默认 profile 是 `kerv-ops`；`ar` 自回归，`naive-sd` 阈值 0 / KF off，`specvla` 固定
阈值 9 / KF off，`kerv-native` 保留 KERV 静态树但关闭公共算子和上述显式优化。

### 同输入测速

成功 AR episode 的全部 **157 张连续观测** 已保存在 `outputs/npu_ar_tensorflow/observations`。
先结束自己的闭环进程，再执行：

```bash
KERV_NPU_ID=4 scripts/run_kerv_container.sh --benchmark \
  --output-dir outputs/npu_benchmark_repro --max-episode-steps 220 \
  --override '+inference.runtime.benchmark_profiles="ar,naive-sd,specvla,kerv-native,kerv-ops"' \
  --override +inference.runtime.benchmark_samples=157 \
  --override +inference.runtime.benchmark_repeats=2 \
  --override +inference.runtime.benchmark_warmup=1 \
  --override +inference.runtime.benchmark_observations=/data/data/KERV/outputs/npu_ar_tensorflow/observations \
  --override +inference.runtime.ascend_graphs=true \
  --override +inference.runtime.ascend_action_head=false
```

每种配置完整预热 1 遍，再测 2 遍，314 个 action；每遍重置 history 和 rollout。
KERV 阈值按 step 和 220 步规划长度从 15 向 3 更新，KF 周期 4；Naive SD 与 SpecVLA
实际没有强制 KF。相同图像、task、权重、BF16、裁剪和规划长度，检查重复动作确定性，
并要求 KERV 优化前后动作完全一致。

`model_mean_ms` 排除 processor；`pipeline_mean_ms` 包含 processor 和模型；均使用 NPU
同步的 `perf_counter`，排除加载、预热、仿真和视频写盘。算子磁盘记录仅在预热开启。
加速比 = 基线平均 pipeline 延迟 / KERV+算子平均延迟。图捕获开销在预热，单次冷启动
和闭环包含捕获的耗时不等于这里的稳态回放延迟。报告保留输入 SHA256、全部动作与原始延迟。

`benchmark.partial.json` 是中途产物；`benchmark.json` 也需检查动作一致性标志。
实验候选未通过检查时会写诊断 JSON 并返回非零，不能把这个文件当成通过结果。
本机其他用户仍在别的芯片运行，因此这些数据代表共享服务器环境。

### FlagScale 算子 microbenchmark

```bash
docker exec -w /data/data/KERV/third_party/FlagScale \
  -e ASCEND_RT_VISIBLE_DEVICES=4 -e OMP_NUM_THREADS=16 -e PYTHONPATH=. \
  kerv-ascend /data/data/KERV/.venv/bin/python examples/kerv/benchmark_ops.py \
  --device npu --backend native --warmup 50 --iterations 100 --repeats 5 \
  --output /data/data/KERV/outputs/npu_ops_native.json
```

16/16 案例通过：浮点 `rtol=atol=0.02`，整数精确一致；整模型动作使用更严格的完全一致检查。
局部算子容差通过不能代替模型动作或 LIBERO 准确率验证。

## 直接使用 FlagScale 已集成的 KERV 入口

补充 [ascend.yaml](third_party/FlagScale/examples/kerv/conf/inference/ascend.yaml)，
复用 `flagscale.models.kerv.entrypoint.run_kerv_entrypoint` 在 worker 内直接执行，
算子从 FlagScale bundled runtime 导入。该入口不经过 release bridge。

主机封装：[run_kerv_flagscale_ascend.sh](scripts/run_kerv_flagscale_ascend.sh)。

```bash
KERV_NPU_ID=4 scripts/run_kerv_flagscale_ascend.sh action=test \
  inference.kerv.arguments.max_tasks=1 \
  inference.kerv.arguments.max_episode_steps=8 \
  inference.kerv.arguments.max_planning_steps=220
```

改为 `max_tasks=10`、`max_episode_steps=220` 可用同一入口运行闭环。
`KERV_FLAGSCALE_OUTPUT` 指定容器内绝对输出路径，默认 `/data/data/KERV/outputs/flagscale_ascend`。
最终测速使用前文 release bridge；关闭 head 的最终 10 任务闭环直接通过本原生入口执行。

## 实验开关

默认 NPU backend 为 `native`；以下候选保留源码但未列入通过配置：
`+inference.runtime.ascend_backend=flag_gems`、`+inference.runtime.ascend_attention_backend=cann`、
`+inference.runtime.ascend_attention_backend=flag_gems`、`+inference.runtime.ascend_linear_fusion=true`。
测速切换公共算子后端使用 `+inference.runtime.benchmark_ops_backend=flag_gems`。
已安装 NPU Triton 不意味着原 CUDA kernels 可直接编译；后端和 kernel 要分别验证。
