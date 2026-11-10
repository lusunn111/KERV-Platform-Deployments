# 技术指标测试大纲：昇腾分册

版本：1.0，2026-10-27。工程：KERV / OpenVLA / SpecVLA drafter / FlagScale。
课题：面向多模态智能的端侧异构计算系统高性能协同优化，编号 Z251100008125044。
测试性质：项目组本机自测及已有实测证据复核。范围为当前昇腾服务器。

## 1. 编制依据与范围

依据工程根目录 `TestOutline.md`、任务书原件及用户确认的量化授权D09。
任务书原件：`docs/acceptance/sources/2025-1215-课题任务书.pdf`，SHA256：
`94c0d5812d9bf43b48cd5689083d408f41753010aa0bfac537b109fdca0f3f4c`。

本分册覆盖可在本机核验的部署、功能、执行位置、算子、系统性能、量化内存和LIBERO Goal任务质量。
A100正式性能环境、其他国产型号、智源自研VLA、其他仿真平台和第三方独立测试由总大纲对应工作单元承接。
这些单元的缺项保留在指标对照表，不由本机成绩替代。

本次文档日期是复核与交付日期；实验日期分别见原始run.json，不能将历史实测改写为当日新运行。

## 2. 被测系统与版本管理

| 项目 | 本机配置 |
|---|---|
| 芯片 | npu-smi显示Ascend910；当前软件查询Ascend910_9382；资产级具体SKU未确认 |
| 资源 | 8块板卡、16个物理chip，每chip报告65536MB；每个模型单chip运行 |
| 容器 | kerv-ascend |
| 解释器 | /data/data/KERV/.venv/bin/python；Python3.12、Torch2.7.1+cpu、torch_npu2.7.1.post8 |
| 编辑目录 | /root/KERV |
| 运行副本 | 宿主机/public-flash/zhiyuan/liyuzhuo/data/KERV，容器/data/data/KERV |
| 当前代码 | ports/moore_ascend；根目录旧实现保留，实验按各自源码快照复现 |
| verifier / drafter | 本地openvla-libero-goal / kerv-drafter；身份以权重清单和SHA256核验 |
| 平台 | LIBERO框架的Goal套件，10个任务；不能将不同套件自动计为独立平台 |
| 输入与动作 | 原TensorFlow中心裁剪；图像和任务文本输入；7维动作、seven token协议 |
| 软件栈补充 | 容器CANN路径cann-9.0.1，驱动npu-smi 25.5.0；Triton实际模块3.2.0、包元数据3.5.0，版本与加载路径在交付environment.json分列 |
| 执行参数 | batch1、seed7、规划长度220、每回合上限220，算法与接受阈值按冻结配置 |

当前目录缺原始Git元数据，版本由源码归档、逐文件哈希、配置和依赖版本共同标识。
历史运行使用当时冻结代码，不能用当前源码哈希相同与否重新判定历史实验是否成功。
每次实际运行保存物理chip到逻辑npu:0的映射、退出码、起止时间和配置。

## 3. 实现路线与实际设备分工

| 环节 | 实现与位置 | 测试重点 |
|---|---|---|
| 仿真与输入准备 | LIBERO/MuJoCo、TF裁剪、processor在CPU | 任务/初态一致，输入哈希、图像裁剪、动作单位和夹爪变换 |
| 草稿与验证 | SpecVLA drafter、OpenVLA verifier在NPU | 真实kernel与模块命中、7 token停止边界 |
| 树与缓存 | C++构建拓扑；NPU打包/embedding/KV；批量提交与复用 | 稳定同分选择、候选不丢失、缓存重置与新输入 |
| Kalman补齐 | CPU对历史模型动作token估计，候选写回含NPU张量操作 | 全拒绝/部分接受、位置不偏移、补齐不算真实接受前缀 |
| 动态阈值 | 按步数和规划长度衰减 | 冻结曲线与边界，不描述为实测运动反馈自适应 |
| 混合量化 | CPU加载阶段转换；NPU原生W8A16 GEMM与Ascend Triton SwiGLU | 不在NPU临时保留原BF16量化权重；保留SiLU中间BF16舍入 |
| 实验QKV BMM | CPU预打包、NPU batch=3 GEMM，仅257..320行；其余原三个投影 | 原矩阵输出宽度不变、视图共享存储、形状回退和动作一致性 |

现有Kalman输入为历史输出token，并非真实关节/末端传感器反馈。proprio虽被观测，未作为当前OpenVLA/KF输入。
任务书不同章节对补偿位置的表述差异由总大纲D08处理；本机报告按真实路径记录，不虚构已完成运动学反馈链。

FlagScale实际链路：运行脚本 → ports/moore_ascend/run_kerv.py → third_party/FlagScale/run.py → worker →
本地FlagScale KERV bridge → run_kerv_libero.py → PyTorch/torch_npu。
本地FlagScale仓库已有examples/kerv与flagscale/models/kerv/ops；当前测得的NPU量化/融合实现来自port。
配置中的backend=vllm不代表由vLLM模型引擎执行。公开版本、上游合并及独立第三方复现另行核验。

## 4. 比较配置与归因

| 标识 | 配置 | 比较用途 |
|---|---|---|
| B0 | 真正SpecVLA参考实现 | 未完成本后端等价性核对；不能用关闭Kalman的KERV代替 |
| B1 | KERV native、BF16 | 系统优化参考；保留KERV算法机制 |
| B2 | KERV optimized、BF16 | 与B1隔离运行时收益；默认部署精度 |
| Q1 | B2＋W8A16/BF16混合、融合gate/up与保留舍入的SwiGLU | 经D09授权的省内存候选；与B2比较精度/内存/速度 |
| Q2 | Q1＋形状限制QKV BMM和权重视图缓存 | 实验项；与Q1比较派发开销和行为，与B2另测总体开销 |

Q1固定verifier层7:25，50个投影W8A16/group32；15/16/19/20层down保留BF16，其余不新增量化。
policy SHA256：`363e245a3b8265d4165cf18e78b1491fecffa1e8e6019fe2a2506661db3c62d7`。
`mlp_dynamic_int8`是历史入口名，该policy实际激活仍为BF16，不是W8A8。

系统同精度比较要求输入、权重、动作和算法机制一致；量化比较允许行为改变，但必须测完整任务质量。
禁止跨日期/芯片/轨迹拼接比值；B1→B2的加速不能改称B0→B2，也不能与另一轮量化比值相乘。

## 5. 样本与执行规则

1. 冒烟为1任务×1初态×8动作，只判运行与字段，不能纳入完整任务成功率。
2. 完整回归按平台终止或220步结束，保留所有策略失败与超步数。
3. 现有质量探索为Goal10×初态1/2×seed7，每配置20回合；初态反复用于开发，非未见验证集。
4. 固定输入测速为157观测，每模型完整预热1序列，再测3序列。量化同卡每帧AB/BA交替；旧B1/B2按AB/BA/AB分轮。
5. 总大纲的正式样本建议为每组10任务×10官方初态×3种子=300回合，B0/B1/B2共900回合；当前数据不冒充该矩阵完成。
6. 同一物理chip不并发模型测速；双模型交替测速的驻留仅用于时延诊断，不能作为单模型部署内存。
7. 基础设施失败、重试和配置修复保留独立ID；策略失败保留分母；原始输出不追加重写。

## 6. 指标、统计与判据

| 指标 | 计算/口径 | 本机与正式结论的边界 |
|---|---|---|
| 模型时延 | 同步边界内模型时间；均值、中位数、P95和原始序列 | 不包含全部processor/仿真，不能代替65秒任务耗时 |
| 策略链路 | processor、搬运、模型、解码等pipeline时间 | 与纯模型分列 |
| 完整闭环 | 首个策略观测就绪到终止env.step与记录；含首动作 | 加载、reset/settling、视频另列；按原采集边界报告 |
| P95 | 排序后第ceil(0.95N)项，nearest-rank | 历史插值P95保留；本报告从raw重新计算 |
| 成功率 | 成功数/全部预登记有效完整回合 | 报百分点差、相对损失、失败交换与样本数；小样本不证明总体损失<5% |
| 接受长度 | sum(原始接受前缀)/全部验证次数，保留0 | root/bonus/补偿/padding须审计；未审计不判ΔL>0.5 |
| 显存 | 基线同口径峰值减测试峰值 | allocated/reserved/驱动分列；量化精度明确披露 |
| 内存单位 | GB=10^9B，GiB=2^30B | 2GB与2GiB分别判断，厂商MB原始标签保留 |
| 传输 | 双向bytes/观察窗口；另列bytes/实际copy累计时间 | ATen逻辑字节不代表全部DMA，缺方向/时长不能判<100M/s |

内存峰值包含加载初始化和完整任务；采集器100ms读取框架峰值，驱动工具10秒采样可能漏瞬时峰值。
接受长度>0.5、成功率损失<5%和传输<100M/s保留严格不等式。D01-D08正式解释待确认，不能在结果后选择有利解释。
Wilson区间仅作当前成功比例的描述性补充；重复任务和初态相关性不因一个区间消失。

Q1/Q2工程观察门槛：完整动作有限且7维；逐初态成功数不低于参考；allocated节省≥2e9B；
新测同卡平均模型时延不增加。Q2额外检查相对Q1的完整观测/动作与验证工作量一致。
这些是工程门槛，不等于任务书所有指标的正式验收判据。

## 7. 测试项目与证据

| 用例 | 操作 | 判断标准 | 证据 |
|---|---|---|---|
| ASC-01 环境/版本 | 核对芯片、容器、权重、源码、运行副本 | 身份/映射可追溯，SHA一致，已有材料不覆盖 | inventory、source/weight manifest |
| ASC-02 机制与算子 | 定向NPU测试；检查树/补齐/阈值、量化和缓存 | 边界正确、无陈旧输入、量化视图共享唯一storage | 机制4项、动态量化9项日志及命中 |
| ASC-03 系统等价 | 同task0/init0/seed7完整B1/B2 | 数组等长，观测/token/raw/executed/done逐步一致 | paired_B1/B2原始JSONL |
| ASC-04 同输入时延 | 固定157帧、3遍、完整预热与AB/BA | 样本无缺失，序列确定，按真实基线报告 | replay/timing原始数组 |
| ASC-05 任务质量 | 对齐Goal10/init1/2完整回合 | 保留失败与7维有限动作；逐任务成败与总率均报告 | episodes.csv、quality.csv |
| ASC-06 部署资源 | 单模型加载至10回合结束采峰值 | 三口径分列，量化授权与精度披露；判断2GB及2GiB | memory、vendor_memory、quality.csv |
| ASC-07 执行位置 | 查NPU kernel和CPU范围、算子真实命中 | 注册不代替执行；历史token反馈如实描述 | CANN诊断、placement、operator命中 |
| ASC-08 接受/传输 | 审计逐事件与双向搬运范围 | 0事件保留；缺完整DMA或B0时不判正式阈值 | verify、transfers及缺口表 |
| ASC-09 集成/退出码 | 审核FlagScale实际worker、配置、错误返回 | 运行链路正确，异常可见；未实测模型不填支持 | inference.yaml、worker、stdout、run.json |
| ASC-10 证据/复算 | 只读原日志，统一统计，核对哈希 | 能从raw复算，所有结论有source索引 | evaluation.json、evidence_index.csv、validation.json |

## 8. 复现和交付

当前推荐入口、离线权重与初始化参数见README_ASCEND.md。先核实目标chip空闲、使用新输出目录。

```bash
KERV_NPU_ID=14 KERV_NPU_QUANT=mlp_dynamic_int8 \
  KERV_NPU_QUANT_POLICY=/data/data/KERV/outputs/acceptance/weight_only_hybrid4_20261027/policy.pt \
  KERV_LIBERO_INITIAL_STATE_OFFSET=1 \
  scripts/run_kerv_moore_ascend.sh \
  --output-dir outputs/mixed_reproduce_new --max-tasks 10 --trials 1 \
  --max-episode-steps 220
```

量化关闭时用`KERV_NPU_QUANT=none`。实验Q2另用`KERV_QKV_BACKEND=bmm KERV_QKV_LAYERS=all`和QKV脚本。
复核旧实验使用其源码快照；上述当前命令不是恢复旧源码版本的替代品。

综合测评复算命令，输出目录必须新建：

```bash
python -m acceptance.evaluate_ascend_delivery \
  --output outputs/acceptance/ascend_delivery_recomputed_new
python -m unittest acceptance.test_statistics -v
```

本机交付包括本大纲、综合测评报告、指标对照、原始证据索引、统计CSV/JSON、工具快照、哈希和复现命令。
本机文档与证据交付完成后，课题其余工作单元仍按总大纲和缺口表处理；不预填第三方签署结论。
