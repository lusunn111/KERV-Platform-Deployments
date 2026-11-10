# 沐曦 C550 进度

<!-- metax-model-performance-live -->
## 模型性能任务已按用户要求暂停

CPU适配和检查结果已保存，GPU队列及Codex自动重试已停止。本轮GPU实验尚未开始，恢复时需重新检查资源并显式启动。
<!-- metax-model-performance-live -->















## 上一阶段验收状态（2026-11-05 03:23 UTC，历史记录）

- 推荐数值稳定入口：`scripts/run_kerv_metax.sh --profile numerical_fixed`，保留原生FP32 pow/mean，尾融合保留BF16乘法类型；通用支持条件外走原生，无动作/测试集/逐层结果判定。原生RMS回退入口`numerical_native_rms`。旧optimized_v2保留为数值不同实验，不覆盖任何旧结果。
- 两次独立进程的3任务×2初态均 **5/6、696步、1928次验证**。观测/image hash/token/raw/executed/done/success/阈值/终止原因逐字段与bf179f6第一轮原生/optimized一致，验证候选、接受前缀、KF补偿与边界也一致。非仅成功数相同。
- 已采样65个验证器RMS层、28种真实形状、19个早晚步骤、4745次RMS调用，包含6验证循环长路径及周期KF，修复输出位差0。未声称所有可能输入普遍位精确；完整两轮轨迹一致结论限定同模型、权重、任务/初态、seed与环境。
- 根因实测：首次BF16差step0/layer0/[1,279,4096]38处由1/sqrt替换可复现；该次mean虽155行FP32位差，单换mean未变BF16，不将其混称同一贡献。后续层rsqrt试验仍有归约差，原生mean替换消除本批非零数值差。额外FP32末乘在本后端丢-0，保留BF16乘恢复；FP32乘lower为pk.fma(...,+0)的LLVM和转换边界实验已保存。
- 四组同输入220步AB/BA：native353.472020ms、第一轮219.126675ms、旧v2实验170.828971ms、修复181.583823ms。修复/native **1.946605×**，修复/第一轮 **1.206752×**，策略/native **1.925705×**；不为维持2×调整精度、阈值、树、图像、权重、seed、220时域或停止边界。
- 首轮修复完整模型/策略/闭环130.109817/136.983491/241.248891s；独立复跑130.309811/137.411998/242.473709s，含首步及成功终止步。每回合去首步的均值/中位/P95及全部配置原始统计在episode-stats-final.json和报告；修复去首步690步模型均值177.916838ms、中位186.722557ms、P95284.998138ms（首轮）。
- allocated17,174,868,992B，reserved19,549,650,944B，两次一致；较旧v2 allocated -331,776B、reserved +301,989,888B，未包装为显存优化。设备/进程1秒采样另存。各正式profile加载前/释放后allocated均0，计入完整加载与图池。
- 可复跑报告：docs/METAX_NUMERICAL_FIX.md；原始记录/源码包/核命中/根因实验：outputs/metax_numerical_20261105/。`python3 -m ports.metax.validate_numerical`（含独立复跑）passed；`render_numerical_report`可重建全部表格。
- 无正在执行GPU命令，物理卡0已释放；kerv-metax独立容器保留。仅此主机/工程/容器操作，未改全局驱动/其他服务器/其他任务；本地阶段提交，不推送。bf179f6、5b7de3c及历轮所有原始结果完整保留。
- 本轮没有扩展压缩或B0。无外部权限/资源阻塞。未来扩大Goal holdout仍需独立质量评估，未执行也未假称普遍保证。

以下为阶段历史，历史“执行中”状态按上方当前状态解读。

## 2026-11-04：P00/P03 环境与源码盘点（执行中）

- 起始源码快照：`79466c7bd7f68feb795f0d17a60e68340caf12ea`，原目录没有任何提交，已完整保存迁入代码（不含权重）。不推送远端。
- 主机 `bm-zksg-wx-zone1-d-mc550-64g-2-111`，8×MetaX C550，驱动3.8.1、宿主MACA3.7.1.5。初查无GPU进程，每卡858MiB基础占用。
- 仅选物理卡0/PCI 0000:2b:00.0 → 容器逻辑cuda:0。厂商PyTorch使用CUDA兼容API，实际为 `2.10.0+metax3.8.1.0`，`torch.version.maca=3.8.1.0`；BF16矩阵乘及同步、allocated/reserved接口已真实执行。
- 独立容器 `kerv-metax`，现有镜像 `1a10a6cc5674`（maca-pytorch:3.8.1.2-torch2.10-py312-ubuntu24.04-amd64）；独立venv `.venv-metax` 继承厂商torch。Docker默认bridge失效，使用host网络绕过，无宿主网络修改。
- 原始盘点：`outputs/metax_inventory/`。根盘余55GB，优先只下载推理权重；不清理他人数据。
- 已读任务大纲、昇腾工作状态/技术大纲、实际port launcher和FlagScale桥。推荐算法起点为 `ports/moore_ascend`，保留旧实现。沐曦适配作为独立目录及配置。
- seven协议：root为首动作token，接受前缀排除root；KF补偿从1+accepted位置开始，尾部6维；原生图像TF JPEG/Lanczos/中心裁剪与动作反归一化保留。
- 发现旧launcher将max_episode_steps同步用于max_planning_steps；沐曦入口将规划时域固定220并独立暴露，短流程只限动作。
- 真实B0还需追溯SpecVLA上游实现；不将关闭KERV开关当作B0。

正在执行：依赖准备、真实verifier/drafter获取、LIBERO无显示渲染及FlagScale入口适配。
下一步：加载/短流程 → 一个完整同卡串行B1/B2回合 → 真实输入诊断/优化 → 扩展Goal任务/初态。
真正阻塞：目前无已确认需要用户处理的阻塞；权重尚未获取，四项指标尚未测得，历史结果不计入。

### P03 更新（04:41 UTC）
- LIBERO task0 OSMesa reset/render已通过，原始记录 `outputs/metax_inventory/render02.log`。
- seven KF根/补偿边界与C++树拓扑对照18项通过；FlagScale新旧runner退出码0/23测试通过。
- drafter公开真实权重下载完成；verifier revision `fa5ae1e7509348889295bba8e08621d8b55e9baf` 正在下载（只包含仓库推理内容）。
- SpecVLA上游 `56ef2985e25c7903d2759254efb754327fb975f7` 已只读取回至 `outputs/SpecVLA-reference`。发现其动作边界 `>6` 与七维解码尾切片和当前修正版不同，待独立适配/审计，B0仍未测。
- 测量器已适配厂商CUDA兼容API；逐验证保留包含零的posterior原始前缀、七维实际提交前缀、root、补偿和padding审计；新增实际drafter/verifier forward设备观察。

### P03/P04 更新（04:44 UTC）
- 真权重native 8动作短流程通过，FlagScale全链路退出码0；输出 `outputs/metax_native_smoke_01/`。GPU实际forward：verifier24次、drafter80次。
- 16个verify（包括2个零接受），posterior原始前缀均值2.5，提交前缀均值2.125，root16、KF补偿6、padding0。allocated峰值16,970,799,104B。首动作冷启动单列，不作为预热后性能结果。
- 正在执行完整task0/init0/seed7：`scripts/run_kerv_metax.sh --profile native --output-dir outputs/metax_native_pair_01`；外层日志 `outputs/metax_inventory/native-pair01.log`。之后串行执行优化候选，未同时占用第二张卡。
- B0边界修正选择已通过文本问题请求用户确认；这一项待答复，B1/B2独立继续。

### P04/P05 更新（04:49 UTC）
- 完整B1 task0/init0/seed7正常结束220动作，策略结果失败0/1（不排除失败轨迹）。643次验证包含269个零，原始接受前缀均值1.4199066874，实际提交前缀0.9673405910，KF补偿275，padding0。
- 本回合模型83.898s、策略86.146s、完整闭环117.880s；首动作保留，非正式预热后速度结论。allocated峰值17,051,619,328B；reserved17,244,880,896B，厂商逐秒观测另存 `vendor-memory.log`。
- B2诊断短流程8动作与B1观测/token/raw/executed/done全部精确一致；真实C++草稿/验证树与批量KF/KV、控制合并均命中。该轮保存真实未中心裁剪图像至 `outputs/metax_optimized_smoke_01/observations/`，其耗时不用于正式速度。
- 完整B2正在执行：`outputs/metax_optimized_pair_01`，外层日志 `outputs/metax_inventory/optimized-pair01.log`。全树、不剪枝、不启用图执行与融合浮点内核。
- 根据真实逐阶段记录（排除首动作的181步窗口），初始prompt/草稿约111ms/动作、verify约68ms、posterior约65ms、提交/后续草稿约108ms；CPU/GPU enqueue含混合边界，非纯kernel时间。
- 已准备独立W8A16/group32实验实现及真实矩阵正确性检查，尚未运行/尚无节省或质量结论。原生BF16保持独立配置。

### 完整配对通过（04:51 UTC）
- `outputs/metax_inventory/paired-result.json` 从原始记录复算：220步观测哈希、动作token、raw/executed动作、done、阈值均精确一致。B1/B2同为0/1；接受前缀均值同为1.4199066874（运行时优化不改变接受策略）。
- 模型 B1→B2：83.898456s→54.146411s，1.549474×，耗时下降35.46197%；策略86.145887s→56.564486s，1.522968×；闭环117.880083s→88.906524s，1.325888×，下降24.57884%。均保留首动作，限此单回合。
- B2 allocated峰值17,030,045,696B，reserved19,079,888,896B。原生allocated17,051,619,328B、reserved17,244,880,896B；reserved反而升高，不能声称BF16优化省2GB。
- 厂商1秒观测峰值：B1设备18,785,508KB/进程17,486MiB，B2设备20,573,408KB/进程19,232MiB；原始日志保留，采样峰值不等于瞬时真实峰值。
- MetaX全词表Triton验证归约合成真实形状49×8×32064、零/整数/小数阈值及首同分路径测试通过。尚未接入默认配置，未把这个内核的收益写入B2。
- W8A16真实checkpoint矩阵11008×4096：Triton解压与native重构BF16逐元素完全一致，GEMM逐元素完全一致；相对原BF16输出L2差约0.00613，属于有损量化，仍待闭环质量。
- 串行资源锁 `/tmp/kerv-metax-card0.lock` 已加入本任务入口。正在先执行8真实观测、完整序列预热后3遍AB/BA回放：`outputs/metax_replay_AB_01` / `outputs/metax_replay_BA_01`，日志 `outputs/metax_inventory/replay.log`；W8A16 8动作短流程已排队于同一锁，日志 `outputs/metax_inventory/w8a16-smoke01.log`。
- 后续运行自动保存逐文件源码SHA256与源码压缩包（不含权重/环境/日志），避免后续修改影响复现。首对回合只有起始Git快照、阶段Git和解析配置，尚无每次启动源码包；这一证据限制保留。

### 回放与启动修复（04:55 UTC）
- 固定8真实观测、每轮完整预热后测3遍：AB模型1.449864×、pipeline1.443188×；BA模型1.576801×、pipeline1.562599×，两轮动作全部精确一致。原始时序保留，AB/BA变化如实呈现。
- 首次W8短流程与Goal3×init0/1启动在模型执行前的源码快照失败：厂商镜像缺git可执行文件。已只在独立容器安装git，并同步bootstrap依赖；失败输出保留，不计任务质量分母，使用新目录重试。
- 实际FlagScale故障探针通过：真实子进程23 → FlagScale非零1，记录 `outputs/metax_failure_probe_01/result.json`。

### 计量审计与量化短流程（04:58 UTC）
- seven守恒审计逐步通过：B1/B2各220动作，root643＋实际提交候选622＋KF补偿275＝1540＝220×7。原始posterior总前缀913中有291个超出当前动作停止边界。
- 主接受长度确定采用实际提交连续候选前缀：622/643＝0.9673405910；保留原始posterior均值1.4199066874作诊断，避免边界外候选虚增成绩。原始事件不改写，复算v2见 `outputs/metax_inventory/paired-result-v2.json` 和 `event-audit.json`。
- W8A16短流程02正常完成8动作，actual W8 forward1200次，8步中6步token与BF16完全相同（有损差异已发生）。allocated峰值15,012,854,272B，同长度B2峰值16,979,894,784B，省1,967,040,512B，尚未到2GB。
- 已定位量化模型loaded分配比理论存储节省少52,428,800B，约每个量化矩阵1MiB分配粒度开销。准备只合并INT8/scale存储、保持每个量化值与kernel不变的packed版本；尚未实测。
- 当前GPU执行 `outputs/metax_native_goal3_init01_02`（3任务×2初态），B2 `outputs/metax_optimized_goal3_init01_02` 在同锁排队。失败的首轮启动不计有效回合。

### 多任务评测执行中（05:03 UTC）
- B1已完成前4回合：task0/init0失败，task0/init1成功，task1/init0和init1成功；当前task2，尚不以未完成集合计算最终成功率。
- 上游SpecVLA原函数的CPU协议反例已复现：正确接受6个候选加root后未停止，最终生成8个token并切末7个丢root；单处`>6`→`>=6`提案可保留root。仅协议单元，不是B0模型运行。具体JSON：`outputs/metax_inventory/specvla-boundary-audit-v3.json`。
- packed量化GPU等价检查 → 8动作 → 3任务×2初态已排入同卡串行锁，日志 `outputs/metax_inventory/packed-pipeline.log`。须前一步成功才继续，未把待跑工作写成已通过。

### B1多任务组完成（05:05 UTC）
- `metax_native_goal3_init01_02` 退出0，Goal任务0/1/2各init0/1共6回合，成功5/6；逐动作raw共696步，1928次验证，主接受前缀2089/1928＝1.0835062241。
- raw模型总232.317148s、完整闭环349.335163s；allocated峰值17,051,619,328B、reserved17,345,544,192B。输出 `outputs/metax_inventory/native-goal3-summary.json` / `native-goal3-event-audit.json`。
- 发现旧控制台/旧timing导出在`done`处先break，再append时间，漏5个成功终止动作而打印691；raw Recorder已在break前保存，696步均守恒通过。已将旧导出append移至break前，算法/动作/主raw计量不变；旧日志不改写。
- B2多任务组已开始，日志 `outputs/metax_inventory/optimized-goal3-init01-02.log`；packed流水线继续等锁。

### B1/B2多任务配对通过（05:11 UTC）
- B2退出0，同样5/6成功；全部696步观测、token、动作、done、阈值精确一致，1928次验证、实际接受2089，seven守恒全部通过。
- 模型232.317148→153.676999s（1.511724×，下降33.850%）；策略240.099410→160.620708s（1.494822×）；闭环349.335163→264.449049s（1.320992×，下降24.299%）。接受长度差0，质量差0个百分点/相对0%，限6回合。
- B2 allocated峰值17,032,327,680B，reserved19,390,267,392B；B1为17,051,619,328B /17,345,544,192B。框架与厂商口径分别保留于 `goal3-paired-result.json`。
- 后续packed量化pipeline → 独立GPU kernel诊断 → 同输入B2/Q1预热AB/BA对照，均同卡锁串行执行，相关日志分别 `packed-pipeline.log`、`device-diagnostic.log`、`quant-replay.log`（位于outputs/metax_inventory）。

### root/bonus来源细分（05:14 UTC）
- 逐验证事件的`root_length=1`一直表示每棵树的root；第二轮及之后的tree root来自前一次verifier bonus。为避免把两种来源混称，汇总现分为首动作root与后续verifier bonus，原字段/原始数据不改写。
- 六回合每配置：初始root696、后续bonus1232、实际接受候选2089、KF补偿855，总4872＝696×7；1928是tree-root总数。逐事件root/接受token切片也已与最终动作token完全核对，见 `goal3-token-origin-audit.json`。


### 压缩候选对照与无损路线（05:29 UTC）
- W8A16 packed六回合791动作/2229次验证，实际前缀2325/2229＝1.0430686406；初始root791、bonus1438、补偿983，seven守恒通过。质量4/6低于B2的5/6，新增失败task0/init1。完整对照 `outputs/metax_inventory/goal3-with-w8a16-results.json`。
- 新增独立 `--profile bf16_packed`：low字节原样保存，high字节按每矩阵16项字典编码为4bit，字典外值以int32/int16例外表精确恢复；全程恢复BF16位模式后走厂商BF16 GEMM。96个MLP矩阵加8个attention输出投影，共104矩阵；GPU只保留一份共享临时BF16工作区。
- 两个真实矩阵的GPU权重位模式、GEMM均精确一致；8真实动作与B2观测/token/raw/executed逐步精确一致。allocated峰值14,842,354,688B，同长度B2省2,137,540,096B（2.13754GB、1.99074GiB），仅短流程结果，仍待完整组峰值。
- 无损还原增加实际耗时。单独测其完整闭环/预热后速度，绝不将B2速度与无损配置的省显存拼成一组结果。
- 初次device诊断和量化回放因脚本名`profile.py`遮蔽stdlib而在模型运行前导入失败；已改名`device_profile.py`，以新目录重试。失败日志不计任务分母。


### 独立实验收尾（05:49 UTC）
- 无损BF16完整组退出0，696步/1928次验证与B2逐项精确一致，5/6成功；主接受前缀1.0835062241，完整计量见goal3-all-results。
- Q2模型307.489422s、策略314.606274s、闭环419.354503s；峰值allocated14,894,787,584B、reserved17,255,366,656B。速度回退如实保留，候选只作为省显存选项。
- 预热回放：B1/B2合并模型1.513358×、pipeline1.502916×；B2/W8A16为162.389→180.697ms（慢11.274%）；B2/无损为169.458→342.899ms（慢102.350%），无损两轮动作精确、每轮allocated均省2,137,540,096B。
- 修复后的实际device profiler记录MACA kernel，四个scope成功；matrix active近似不变，标准步copy4073→66、KF步2153→38，CPU API时间与GPU DMA不混淆。
- 本机源码包与逐文件SHA、四个部署的退出码/设备映射、696步B1/B2/Q2精确对照、6轮预热回放和4个真实kernel trace校验通过。依赖本地附加包已锁定constraints，厂商镜像digest和实际MACA Version.txt已记录；GPU进程最终为空。

### 第二轮候选筛选（09:09 UTC）
- 第一轮每回合排除首动作，690步加权模型均值 B1 328.327511ms、B2 214.976217ms；每回合均值/中位数/nearest-rank P95、含首步/闭环时间均存 `outputs/metax_v2_20261104/round1-episode-stats.json`。不采用各回合倍率的算术平均。
- 真实 verifier 输入行数208–301（49是树路径数，不是唯一节点数）；QKV微基准约1.11–1.42×，部分形状BF16输出改变；Gate-Up约1.02–1.035×，两形状反而变慢。暂不选择投影合并。
- compact接受归约32次真实验证审计通过，路径/前缀/bonus及选中logits行精确；8观测正式回放169.418→170.088ms，无净收益，隔离为实验开关。compact+vision_graph 169.750→163.387ms，动作一致；非最终配置。
- RoPE微基准5.1–5.6×，所有采集形状输出位模式及stride精确。RMS约3.8–4.3×，279行输入有38个BF16元素差异，属于FP32归约顺序差，保留BF16舍入边界；必须进一步评估动作及闭环，不宣称隐层完全一致。
- 首次组合回放在草稿CPU position_ids传入Triton时失败（pointwise_vision01），模型质量分母不计该次；已增加同设备索引转换，pointwise_vision02重试，日志同目录 `pointwise-vision02.log`。
- 捕获一条完整220步真实轨迹 `representative_capture/observations` 用于后续完整历史回放，包含晚期阈值及周期KF；捕获耗时不用于速度成绩。
- 下一步：点算子组合及单项消融→完整220步AB/BA→3任务×2初态→串行无损解码改进。当前无外部权限阻塞。B0仅边界定义独立待决：原SpecVLA的>6判停可产生8token并尾切丢root，暂不改变定义、不阻塞B1/B2。

### 第二轮实测推进（09:16 UTC）
- 组合rms,rope,vision_graph：8冻结真实观测/1轮完整预热/3轮测量，模型168.853921→124.994860ms（1.350887×相对旧optimized），策略173.849277→129.832585ms；第2/3步动作变化，`actions_exact=false`，不能称无损运行时优化。文件pointwise_vision02/benchmark.json。
- 单项rope：168.612→154.485ms，动作精确；rms：170.297→146.646ms，动作变化。vision单项在锁队列，所有试验保留真实命中计数。
- `pointwise_goal3_init01` 正经FlagScale执行六回合，目前前三回合2/3成功，但task0/init1变为131步（旧103），696步完全一致结论明确不适用；必须等待完整组单独报质量/时间。
- 新增trace_phases.py依据CPU launch关联设备事件，分别计算scope墙钟与设备区间并集，绝不相加CPU/GPU重叠。输出round1-phase-attribution.json。普通B2步骤prompt+initial drafter设备活跃56.61ms、verify82.26ms、posterior0.22ms、KV+next drafter7.68ms；它们不是关键路径可直接相加的收益预算。
- 压缩v2单pass块内例外高字节恢复已实现为独立实验文件，尚未接入默认配置/尚无收益结论；GPU评估等待速度主线完成。

### 六回合数值候选完成（09:19 UTC）
- rms,rope,vision_graph经FlagScale完整跑完6回合，5/6成功，与B1/B2成功数相同（0个百分点、相对0%），但722动作/2000次验证（旧696/1928），实际前缀2174/2000=1.087（旧1.083506）。不能称轨迹完全一致，逐回合首次token分歧与步数见pointwise-trajectory-audit.json。
- 模型130.261169s、策略138.153574s、闭环249.624491s；allocated17,175,200,768B、reserved19,247,661,056B（含加载/图捕获）。排除6个首动作后模型171.328031ms、中位179.403137ms、P95273.221130ms；因新旧轨迹不同，不将两者均值比当作同输入加速。
- 独立视觉图回放有约5.6%模型收益且动作精确；归约rsqrt变体和QKV+点算子组合在锁内继续筛选，正式全220步AB/BA尚未执行。
- 220步冻结fixture与旧B2的观测/token/raw/executed/done/阈值全部一致，representative-fixture-audit.json，可用于长短验证路径和晚期KF回放。没有修改旧服务器或B0。

### 正式测速输入补强（09:28 UTC）
- rsqrt/禁止FMA变体在首层9个实际形状位模式一致，但完整8步仍在第4步改变动作（max差0.034185），不能推广为全模型位精确。该组合125.119691ms，旧optimized168.668385ms，单轮1.348056×；正在单独跑6回合质量final_goal3_init01。
- 新增 `--frozen-history`：从fixture原始steps读取每步raw_action/action_tokens前缀，计时前给所有profile相同KF历史，并验证native/runtime生成token与fixture一致。数值候选的普通图像回放可能反馈不同KF历史，旧诊断结果只作为同观测诊断，不冒称全部输入相同。
- 当前单卡锁队列：final_goal3_init01六回合 → frozen8_AB / frozen8_BA（三profile native/旧runtime/v2，每个1预热+3测量）→ replay220_AB / replay220_BA（每profile完整220预热+220测量，冻结历史）。日志均在第二轮目录。若质量筛选失败，隔离候选后重新指定正式配置；不凑2×。
- 原replay8_AB是未冻结历史的诊断且精确动作断言失败，数据保留不改写；取消的只有本任务尚未取得锁的等待进程，未终止任何模型/他人任务。

### 质量淘汰与正式配置（09:30 UTC）
- rsqrt候选task0/init1新增失败，质量最多4/6，已明确淘汰。六回合继续收集完整负结果（目录final_goal3_init01名称只是历史任务名，不代表最终交付）；取消其尚未开始的测速等待进程，不中断正在执行的模型。
- 当前选择 `rms,rope,vision_graph`（pointwise_goal3_init01已完成5/6）。保留BF16权重/激活及归一化两处BF16舍入，FP32归约与native存在差异，动作/轨迹不同且明确单列。rms_rsqrt仅留实验开关。
- 默认optimized_v2配置改为该已通过pilot的组合，旧optimized/native完全独立，CLI可逐项关闭回退。正式冻结历史selected8_AB/BA → selected220_AB/BA在原锁排队，全部包含同场native/第一轮runtime/v2三组；不再用历史253.071除本轮新时长宣称达标。
- QKV叠加123.512ms，但只比124.995ms微幅改善且动作差异更多，不推广。compact无净收益、Gate-Up形状回退、rsqrt质量失败均独立保留，未无差别开启。

### 冻结回放阶段（09:41 UTC）
- selected8_AB/BA已完成，相同图像+参考token/raw_action历史：合并模型native242.074890ms、旧runtime168.059083ms、v2 124.823718ms。native/v2=1.939334×、runtime/v2=1.346371×；策略246.608242/172.349445/129.037495ms，native/v2=1.911136×。没有使用旧253.071ms基线凑2×。
- selected220_AB已完成native和runtime各220步预热+220测量，v2正在测量；均保持完整历史、220规划时域。尚无最终长序列AB/BA结论。
- 从实际分段进一步选取固定prefill子区域图执行补充筛选，prefill_screen01在锁后排队；只捕获无past、tree_mask=None的原verifier前向，检查所有输出tensor（logits/hidden/KV）后才复用。未启用默认配置，未宣称收益。旧selected220_BA等待该筛选结果后再决定具体正式配置，取消的仅本任务等待锁进程。
- rsqrt失败组最终3/6、937步，新增task0/init1及task2/init0失败，退出0且模型正常完成，属于质量失败；原始结果完整保留。
- 本轮中文技术说明已落盘docs/METAX_V2_TECHNICAL_REPORT.md，结果表待正式测速和压缩实测后生成。

### 长序列正序与图候选（09:45 UTC）
- selected220_AB完整220预热+220测量全部完成，native357.6479ms、旧runtime223.4609ms、v2 176.5608ms，native/v2=2.025636×、策略=1.994134×（精确数值见benchmark.json）。仅单轮，不作为最终合并达标结论；selected220_BA正在执行。
- prefill_screen01：实测prefill捕获1次/replay32次、全部输出tensor首帧bit-exact、8步动作相对v2_base精确；126.290942→125.100384ms，仅1.009517×。allocated17,119,780,352→17,580,325,888B，多460,545,536B。不纳入默认，独立prefill_graph开关保留。
- 加速主线正式配置不再无差别添加小收益项。下一步在反序完成后转压缩：lossless-screen.log评估真实矩阵的解压/例外/临时写入/GEMM；lossless-matmul-screen.log进一步筛选块内精确解码+BF16 GEMM（无持久BF16工作区）。两者已排队于同一锁，绝不与正式测速重叠。
- docs/METAX_V2_PERFORMANCE.md与render_v2_report.py已落盘，可从raw和summary重新生成任务集/预热回放性能表；最终仍需加入反序及压缩实测。

### 压缩筛选准备（09:49 UTC）
- 速度反序最后native正在执行（完整预热220步后测220步），压缩GPU进程仍未取得锁。
- 压缩微基准已改用冻结真实layer0 MLP输入：gate输入直接取probe01捕获，down输入用原始BF16 gate/up/SiLU/multiply重建。attention o_proj仅复用代表性真实归一化张量作形状筛选，明确不是实际o_proj输入，完整链路另验证。
- 拟筛选原两核恢复、新单pass恢复（全局/寄存器字典）、块128内解码+BF16 GEMM融合。融合kernel无权重BF16工作区，但必须分别验证恢复位模式及GEMM输出/累加差异，尚未选择或启用默认。

### 速度主线完成，压缩开始（09:51 UTC）
- selected220_BA完成：v2 171.264424ms、runtime220.960944ms、native356.784677ms，native/v2=2.083239×。正序2.025636×，合并按总样本均值，结果selected220-summary.json；共每profile440测量动作，各轮先完整220步预热，图像/参考KF历史/阈值/220时域固定。
- 短8步合并仍为1.939334×，不挑选长序列结论遮盖短片段。完整6回合质量5/6，轨迹改变，模型130.261169s、策略138.153574s、闭环249.624491s，仍原样保留。
- 独立prefill图没有足够收益，默认保持rms,rope,vision_graph，不启用QKV/GateUp/compact/prefill_graph。第一轮optimized/native不变，bf179f6及其原始数据保留。
- lossless-screen GPU微基准已取得单卡锁；完成后lossless-matmul-screen串行评估。优先选能保留BF16位模式、减少真实延迟、真实峰值仍省≥2GB的方案；不拼配置。

### 压缩诊断与计量修正（10:08 UTC）
- 三真实矩阵解码约0.57ms/90MB，原生BF16 GEMM仅0.14–0.17ms；单pass和寄存器字典无实质改善。fused matmul27项中9项共享内存超64KiB，其余22–98ms且多数GEMM输出不同，失败隔离，不推广。
- 双BF16工作区预取prefetch8_AB：动作与v2基线精确，旧codec289.922975ms→预取287.999413ms，约0.66%收益，不默认推广。工作区180,355,072B全部注册并计入峰值。
- 发现同进程多profile视觉graph stream导致BLAS工作区缓存每次遗留128MiB，早期回放的后续profile allocated峰值含前profile缓存，不能直接视作独立部署峰值。raw不改写；完整六回合各独立进程的峰值有效。
- 已在benchmark两profile之间、完成模型释放/同步后调用厂商torch._C._cuda_clearCublasWorkspaces，再下一组从加载前reset峰值；每组记录pre_load/post_release分配。不是在测量中清峰值，所有加载、图池、当前stream工作区及解码缓存仍计入。
- 现转向四权重向量访存：low32/high16输入、pair高字节预展开表、uint64恢复写入，原例外写回与厂商GEMM不变。vector-screen.log在单锁运行，逐矩阵验证权重位模式和GEMM输出精确后测吞吐。
- 速度最终合并：220步native357.216481ms、旧runtime222.211205ms、v2 173.912696ms；native/v2=2.053999×、旧runtime/v2=1.277717×；策略362.711253/227.414519/179.073020ms，native/v2=2.025494×。8步仍1.939334×，质量5/6并注明轨迹变化。

### 无损压缩取舍收敛（10:14 UTC）
- 向量2/4权重恢复在三真实矩阵权重位模式、厂商GEMM输出均精确，但90MB解码仍约0.57ms，未改善；不推广向量表额外存储。
- 双缓冲原codec289.923→287.999ms仅约0.66%，额外90MB，当前不值得推广；fused dot严重退化且多数非位精确，单pass/向量均无实质吞吐收益。没有把这些失败方案和原配置节省量拼接。
- 当前评估更少压缩覆盖的Pareto点：仅94个MLP矩阵（attention和layer0 gate/up原生BF16），复用原已验证codec及厂商BF16 GEMM，不改任何权重值。预测峰值可能仍省≥2GB，必须以实测确认；不提前宣称达标。
- compression8_AB正在执行三组：同一v2 BF16、原104矩阵无损、94矩阵无损；全部冻结同一原始KF历史。进程间BLAS工作区释放已加入，pre_load/post_release原始字节均记录。下一步按实测筛选后做反序、完整220及6回合闭环。

### 选择压缩短回放通过（10:22 UTC）
- compression8_AB/BA三profile完整预热+每轮3测量均完成，原生v2 125.163460ms、104矩阵292.038022ms、94矩阵288.399237ms。94相对104仅1.012617×（耗时降1.246%），相对v2 BF16仍慢2.304181×，如实保留明显开销。
- 两轮94矩阵allocated均实省2,026,985,984B；各组pre_load/post_release分配均0，加载/图捕获/工作区全部计入。实际动作精确于v2 BF16，无权重/归一化/GEMM精度改变。
- 94矩阵配置bf16_packed_v2已指向独立选择策略；原bf16_packed和第一轮数据不变。compression_goal3_init01经FlagScale正在执行6回合，当前回合仍在Recorder缓冲，待回合flush后逐字段校验；完整结果未出。
- 队列：6回合 → plain-matmul-control（独立短对照，确认融合严重退化来源）→ compression220_AB/BA。均同卡互斥，未与正式测速重叠。

### 压缩闭环推进（10:28 UTC）
- compression_goal3_init01已完成5回合4/5成功；已flush的628步观测/token/raw/executed/done/阈值与v2 BF16精确一致（compression-prefix-audit.json），第6回合仍在执行。Recorder每回合flush，未flush内容不提前宣称一致。
- 已增加独立validate_v2.py整合审计：6回合/逐动作/zero接受/原始prefix/root/bonus/KF守恒、8与220冻结输入、实际Maca设备/FlagScale、所有加载峰值及BLAS清理、≥2GB目标。待全部正式输出完成后执行，不把缺失数据当通过。

### 压缩闭环完成与最后修复筛选（10:35 UTC）
- 94矩阵完整组退出0：5/6、722动作、2000验证、实际提交2174/2000=1.087；与v2 BF16观测/token/raw/executed/done/阈值逐字段精确，seven守恒通过，compression-quality-pair.json及compression-event-audit.json可复算。
- 模型277.453014s、策略285.130729s、闭环394.881736s；allocated15,148,214,784B，较v2 BF16实省2,026,985,984B；reserved17,221,812,224B。相对v2 BF16仍显著更慢，没有解决接近BF16延迟的问题。
- plain-matmul-control：90MB设备copy0.159ms（读写总1.132TB/s），普通Triton BF16 GEMM最好0.379ms，vendor0.171ms。因此先前fused22–98ms并非单纯硬件不可用；定位prefix scan/布局路径，修正为每128块的短稀疏异常列表，去掉逐权重cumsum。原BF16权重bits与GEMM累加分别验证。
- compression220_AB正在运行，不中断。之后仅排一个direct-screen短筛选（3矩阵×3真实行数、固定plain控制较优tile）；原94矩阵BA暂缓，视筛选结果选择，未把待运行算作完成。

### 最后测量收尾（10:50 UTC）
- compression220_AB完整三组通过且动作精确：v2 171.130723ms、104矩阵398.131133ms、94矩阵385.340512ms；最后BA正在运行，仍不提前填最终合并结果。
- direct-screen原始权重位模式检查全部通过，但短异常列表融合仍13–37ms，虽较早期部分改善，远慢于原恢复+vendor GEMM，淘汰不部署。plain对照结果与负实验均保留，不据此宣称普遍硬件不可能。
- 当前唯一GPU命令是compression220_BA，日志第二轮目录compression220-BA.log；完成后运行summarize_replays / validate_v2 / render_v2_report，刷新技术报告和性能表，再记录最终Git提交。
- 第一轮bf179f6、原始结果、native/optimized/bf16_packed保持；新默认配置仅BF16三项和94矩阵lossless，所有失败内核独立实验且未默认打开。

### 第二轮最终验收（2026-11-04 10:57 UTC）
- 全部8/220步AB/BA、两组六回合闭环完成，delivery-validation.json为passed；无后台GPU测量遗留。
- 原94配置闭环与v2 BF16的722步动作/观测及2000次验证完全一致；相对第一轮696步的差异来自v2数值变化，未将两者混称。
- failed候选compact、投影合并、rsqrt、prefill图、单pass/向量/预取/inline dot均保留原始失败或无收益证据，不默认启用；GPU资源全程单锁串行，不改其他服务器/全局驱动/他人任务。

## 数值修复起始快照

# 沐曦 C550 进度

## 当前状态（2026-11-04 10:57 UTC，第二轮实测与交付完成）

- 新增独立 `optimized_v2`：rms/rope/vision_graph；`bf16_packed_v2`：同一v2运行时＋94矩阵无损存储。旧native/optimized/bf16_packed、`bf179f6` 和第一轮所有原始结果保留，tag `metax-round1-bf179f6`；本轮只本地提交，不推送。
- 速度正式结果：220步冻结图像＋参考KF历史、完整预热、正反顺序合并，native357.216481ms → 第一轮runtime222.211205ms → v2 173.912696ms；native/v2 **2.053999×**、runtime/v2 **1.277717×**，策略native/v2 **2.025494×**。8步短回放native/v2 **1.939334×**，分列，不挑口径。
- 质量：现有3任务×2初态，native/第一轮/v2均5/6（差0个百分点、相对0%）；v2是722步，与原696步轨迹不同。主接受前缀1.087，原1.083506，增量仅0.003494，未达+0.5；零接受/root/bonus/KF均独立保留。
- 无损94矩阵配置：5/6，722步/2000次验证及全部动作/观测/阈值与v2 BF16精确一致；allocated17,175,200,768→15,148,214,784B，实省 **2,026,985,984B = 2.026985984GB = 1.887777805GiB**。reserved及设备观测原始记录另存，加载峰值/图池/工作区全部计入。2GB目标参照同一v2 BF16；相对旧native节省1,903,404,544B，未到2GB，明确区分。
- 无损开销未消除：同输入220步v2 BF16 171.689656ms、原104矩阵396.627005ms、94矩阵384.000737ms；94相对104 **1.032881×**（耗时降3.183%），仍为BF16的 **2.236598倍耗时**。8步仅1.012617×改进。交付实测Pareto，不将BF16速度与压缩显存拼成同一配置。
- 完整6回合BF16模型/策略/闭环130.261169/138.153574/249.624491s；压缩277.453014/285.130729/394.881736s，含冷首动作及失败回合，不以回放替代闭环。
- `python3 -m ports.metax.validate_v2` 已通过；原始数据、源码包、命中、事件、显存、正反回放在 `outputs/metax_v2_20261104/`。技术说明：`docs/METAX_V2_TECHNICAL_REPORT.md`；性能表：`docs/METAX_V2_PERFORMANCE.md`；复跑入口：`ports/metax/README.md`。
- 当前无正在运行的GPU实验；物理卡0已释放，mx-smi无GPU进程，独立容器保留可复跑。无外部权限/资源阻塞。下一轮可独立扩大Goal holdout并开发适合MACA的高效无损GEMM，均尚未执行，不伪称已完成。
- B0保持独立待决：原SpecVLA的>6判停可生成8token并尾切丢root，未改变其定义、未填B0倍率、未阻塞本轮工作。6回合pilot不能证明大规模质量稳定。

### 数值修复全层诊断（02:38 UTC）
- 首差：step0、model.layers.0.input_layernorm、[1,279,4096]、flat90192，native0.000217437744140625，旧v2 0.0002193450927734375；该次38 BF16输出不同，FP32 mean155/279行不同。
- rsqrt试验第一处非零数值差在step0的model.layers.2.input_layernorm，[1,279,4096]有21输出不同，首层一致不能推广。全层旧v2共19179处输出位差，rsqrt_actual共14529处；本批FMA开/关所有采集阶段差0。
- 独立native_mean_inverse对照：原生mean作为输入时，epsilon/sqrt/reciprocal/rsqrt全部0位差；说明本批Triton原生归约替换是关键来源之一，旧1/sqrt与rsqrt另作因子实验证明，不靠推测。
- 原型尾融合曾有373处+0/-0差：x=+0且weight为负时，强转FP32乘得到+0，原生BF16乘为-0；保留BF16乘法类型后，native_inv_tail和native_mean_tail在2210个真实调用全部0位差。未宣称所有可能输入均位精确。
- 视觉13帧、verifier RoPE1088次、draft RoPE105次、draft原生RMS105次位差均0；global restore identity通过。单次draft T=1的singleton stride不同但数据一致，完整696轨迹恢复，未归为数值原因。
- 修复release仅恢复global/vision、不恢复实例RMS的生命周期缺口；历史对照每次新加载模型，诊断native入口断言全部原生，因此不把该缺口错误归为历史轨迹分歧根因。新增restore_audit记录。
- 微基准真实两层输入：原生约0.091ms，native_inv尾融合约0.055ms，native_mean尾融合约0.043ms。选无分层/动作判定的统一native_mean尾融合进入正式验证。

### 正式回放执行中（02:48 UTC）
- fixed8_AB四组通过，修复版137.3081ms对native241.0341ms为1.755425×，对第一轮167.7610ms约1.22×；原始动作精确。old_v2已还原65实例RMS与2个global函数，restore_audit全部true。
- fixed220_AB → fixed220_BA → fixed_goal3_init01在同一卡锁串行链中；正确性断言失败则停止后续依赖。新增修复profile token参考断言，与native/runtime一样在计时后逐步核对；旧v2保留数值差异实验。
- root-factor受控替换和长路径all_layer_long_diag03已排队锁内，不与正式测量重叠。后者选参考数据6验证循环step88/90/96/97/109及late step218，避免早晚期采样只覆盖KF短路径。没有把这些step用于运行时回退。

### 正反测速通过、最终闭环执行（03:08 UTC）
- fixed220_AB/BA均通过，native/第一轮/旧v2/修复同图像/参考历史、各完整220预热后220测量。合并模型353.472020/219.126675/170.828971/181.583823ms；修复/native1.946605×，修复/第一轮1.206752×；策略修复/native1.925705×。旧v2动作仍不同，单独作为实验，不保住2×而补偿偏差。
- 长路径补充完成：step88/90/96/97/109（6验证循环）及218，2535次RMS调用、65层、17形状；修复输出0位差。合计两组4745次全层真实调用，root/all-layer-summary-final.json记录全部形状与阶段差。
- 首次BF16差异受控替换：只换1/sqrt为旧公式即复现38处；只换该次8warp归约，BF16输出差0（虽FP32 mean155行差）。不把最早mean位差混称为首次BF16差异的唯一原因；rsqrt全层归约差异由native_mean替换消除。
- signed-zero-boundary实测FP32乘法中间就已变+0，后续BF16cast为+0；构造-0单独cast可保留。Triton LLVM中FP32提升末乘被lower为llvm.mxc.pk.fma.f32(...,+0)，即使enable_fp_fusion=False；BF16类型末乘lower为scalar fmul再cvt。类型保留是修复，不是靠开FMA开关推断。
- numerical_fixed六回合正在经FlagScale执行；尚未完成不提前宣称最终修复。validate_numerical.py已备好，要求所有696步字段、1928验证事件、全层检查、token和还原状态均通过。

### 数值修复最终验收（2026-11-05 03:23 UTC）
- 修复与回退、两次独立修复完整回合、四profile正反测速、65层真实shape检查、所有696动作及1928验证事件守恒均通过，numerical-validation.json为passed。
- README推荐numerical_fixed，并明确旧v2实验身份；不保住原不一致配置的2倍口径。数值修复报告由真实JSON生成，不把首层一致或随机张量推广成全模型结论。
- 所有GPU任务已结束，资源释放；未增加压缩/B0矩阵。第一轮原始轨迹和第二轮不一致实验均保留用于审阅。

## 2026-11-06：模型性能阶段恢复（CPU准备与waiting_gpu）

- 已读取20261105-model-performance/task.md；起点cf41388，上一轮容量错误退出后无新增GPU结果。当前GPU0被他人PID1063812占用约55154MiB，未加载GPU模型。
- 独立真实SpecVLA原版/单条件边界修复版已实现；原始上游文件字节/SHA保留，CPU协议16案例与命名空间隔离、计量无干扰通过。真实上游权重CPU严格加载通过；两真实MLP共90177536元素CPU codec还原位差0。首次CUDA序列化CPU加载失败日志保留，现两版共同显式map_location=cpu，不改数学行为。
- 数值修复+原lossless94矩阵组合独立配置numerical_fixed_packed已准备；native/numerical_fixed/旧v2均保留。CPU检查不代表GPU质量或速度收益。
- 已登记开发init0/1、未使用init2..6作Goal10×5冻结验证；当前只准备五配置短流程/六回合，共10个GPU job。尚不在holdout调参。
- tmux窗口model-gpu-queue正在每60秒等待；状态见outputs/metax_model_performance_20261105/queue-state.json。两次无进程+基础显存、取得锁后复查、Git/配置/矩阵冻结再运行。CPU模拟真实监督循环/文件锁测试通过，模拟信息不进入生产GPU判断。
- 报告docs/METAX_MODEL_PERFORMANCE.md及原始outputs/metax_model_performance_20261105/。真正阻塞仅GPU0尚未释放；下一步等待器自动执行已冻结开发实验，再据真实B0差距决定C/holdout。未宣称模型性能已改善。
