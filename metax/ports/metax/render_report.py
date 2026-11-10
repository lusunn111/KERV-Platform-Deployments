"""Render the Chinese MetaX report from retained, recomputable results."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
INV=ROOT/'outputs/metax_inventory'
read=lambda name:json.loads((INV/name).read_text())
results=read('goal3-all-results.json');runs=results['runs'];comparisons=results['comparisons']
replays=[read(name) for name in ['replay-summary-v2.json','quant-replay-summary.json','lossless-replay-summary.json']]
labels={'B1':'KERV native BF16','B2':'KERV optimized BF16','Q1':'W8A16/BF16（质量失败）','Q2':'无损BF16存储'}
lines=['# 沐曦 C550 部署、优化与阶段评测','',
'日期：2026-11-04。本报告仅使用本机新运行。已完成可复跑的真实权重LIBERO闭环；B0动作边界定义待用户答复，尚不声称完成全部验收指标。','',
'## 1. 实测结论','',
'- B1/B2在Goal三个任务、两个初态、seed7下均5/6成功，696步的观测、动作token、raw/executed动作、阈值和done全部精确一致。',
'- B1→B2模型1.5117×、策略1.4948×、完整闭环1.3210×；闭环耗时下降24.30%。',
'- W8A16虽省显存，但4/6成功，低于BF16的5/6，不推荐。',
'- 无损BF16存储与B2逐步精确一致，同样5/6成功，allocated省2,137,540,096B；模型耗时约为B2的2倍，闭环增加58.58%。这是单独的省显存配置，不与B2速度拼为同一结果。',
'- B0真实SpecVLA尚未运行，T(B0)/T(B1)、T(B0)/T(B2)、相对B0接受长度提升>0.5和成功率损失<5%均未测。六回合不能证明总体质量边界。','',
'## 2. 环境、模型和实际执行引擎','',
'主机 `bm-zksg-wx-zone1-d-mc550-64g-2-111`；8×MetaX C550中只选物理卡0（PCI `0000:2b:00.0`），容器内逻辑 `cuda:0`。初查无GPU进程；全程本任务作业由互斥锁串行执行，不终止他人任务。',
'独立容器 `kerv-metax`，本地厂商镜像ID `1a10a6cc5674`（maca-pytorch:3.8.1.2-torch2.10-py312-ubuntu24.04-amd64）；驱动3.8.1，宿主报告MACA3.7.1.5。容器 `/opt/maca/Version.txt`为3.8.1.3；实际Torch `2.10.0+metax3.8.1.0`、`torch.version.maca=3.8.1.0`，Torchvision0.25.0和Triton3.6.0均为厂商版本。BF16矩阵、同步、事件和显存API已实际执行。未覆盖厂商Torch、未升级驱动。',
'`.venv-metax`继承厂商包，单独安装Transformers4.40.1、Timm0.9.10、TF CPU2.16.2、Mujoco3.2.7、Robosuite1.4.1等。Python3.12不使用旧TF2.15环境；原JPEG/Lanczos/中心裁剪代码保留。未使用的镜像flash-linear-attention声明Transformers>=4.45，pip check保留该冲突记录；本链路未导入它。',
'Verifier：`openvla/openvla-7b-finetuned-libero-goal`，revision `fa5ae1e7509348889295bba8e08621d8b55e9baf`；drafter：SpecVLA公开Goal权重，严格核对非共享embedding参数。tokenizer/processor/statistics配套本地模型。逐文件哈希见 `outputs/metax_inventory/weights.sha256`。只下载推理内容，网络操作均通过baai-proxy，无永久代理修改。','',
'```text','scripts/run_kerv_metax.sh → ports/metax/run.py → third_party/FlagScale/run.py',
' → 本地worker脚本 → ports/moore_ascend/runtime_opt/FlagScale/inference_kerv.py',
' → run_kerv_libero.py → 厂商PyTorch/MACA + LIBERO/OSMesa','```','',
'FlagScale配置中的backend=vllm是现有inference runner注册名；实际模型引擎为PyTorch/MACA，不是vLLM。真实子进程退出23时，adapter/FlagScale返回非零1；新旧runner的0/23单测也已通过。','',
'## 3. 算法边界与适配','',
'推荐源码起点为ports/moore_ascend；根目录旧实现保留。草稿MMModel构树，OpenVLA并行验证，连续候选前缀在动作边界内提交。seven首动作root、后续verifier bonus、KF补偿、padding均不计为接受候选。KF保留root和已接受前缀，从正确的剩余维度补齐；每4个动作的周期补偿及原动态阈值函数保持不变。KF实际输入是历史模型token，proprio虽被观测但未作为当前OpenVLA/KF输入，不宣称完成真实传感器运动反馈。',
'图像旋转、TF JPEG编解码/Lanczos3/中心裁剪、processor、动作反归一化、夹爪变换保留。所有完整回合220步上限、220规划时域；8步冒烟只限制执行动作数。回合按原start_rollout重置历史和缓存。',
'B2启用C550上验证过的精确C++草稿/验证树、批量KF/KV提交、控制合并和工作区/提示token/草稿KV复用。全树、不剪枝；不启用图执行、旧设备融合内核或旧形状阈值。18项动作/树机制测试通过。','',
'## 4. 四项指标：六回合配对试跑','',
'条件：Goal task0/1/2 × init0/1 × seed7，同卡串行。每配置6回合；早期重复task0开发回合不并入这个分母。','',
'| 配置 | 成功/回合 | 实际动作数 | 验证次数 | 主接受前缀均值 | 模型秒 | 策略秒 | 闭环秒 |',
'|---|---:|---:|---:|---:|---:|---:|---:|']
for key,r in runs.items():
    lines.append(f"| {key} {labels[key]} | {r['successful_episodes']}/{r['complete_episodes']} | {r['policy_steps']} | {r['verify_calls']} | {r['mean_accepted_prefix']:.6f} | {r['model']['sum_seconds']:.3f} | {r['policy_pipeline']['sum_seconds']:.3f} | {r['closed_loop_seconds']:.3f} |")
lines+=['',
'B1/B2成功率83.333%，差0个百分点/相对0%；Q1为66.667%，对B2下降16.667个百分点/相对20%；Q2对B2差0个百分点/相对0%。B1/B2/Q2唯一失败均为task0/init0；Q1新增失败task0/init1。',
'主接受长度为动作边界内真正提交的连续候选前缀，包含零。B1/B2/Q2共1928次验证、提交2089，均值1.0835062241；原始posterior总2768、均值1.4356846473另列。六回合每组初始root696、后续bonus1232、候选2089、补偿855，总4872=696×7；逐事件token切片与最终动作核对通过，无接受padding。',
'原始事件root_length指每棵树的root，后续tree root来自上一轮bonus；汇总从action_tokens_before区分来源，避免混入接受长度。不能将AFEP或补齐后的7维当作接受长度。B1→B2接受长度差为0，未拿它冒充相对SpecVLA提升>0.5。',
'模型边界是predict_action调用，含内部树/接受/KF调度与动作解码，不等同纯矩阵kernel活动时间；策略再含图像处理/processor/搬运；闭环从首个策略观测就绪到终止env.step及动作记录，含仿真，排除加载/reset/settling/视频。首动作冷启动保留，模型加载和首动作在原始collector/steps中单列。',
'B1/B2模型/策略/闭环加速比分别1.511724×/1.494822×/1.320992×，耗时下降33.850%/33.102%/24.299%。Q2相对B2模型耗时增加100.09%、闭环增加58.58%。Q1轨迹长度变为791步，时间差还包含质量/工作量变化，应另看同输入回放。','',
'## 5. 显存：包括模型加载和全部任务','',
'| 配置 | allocated原始B | GB / GiB | reserved原始B | GB / GiB | 设备观测峰值KB | 进程观测峰值MiB |',
'|---|---:|---:|---:|---:|---:|---:|']
for key,r in runs.items():
    A=r['allocated_peak'];R=r['reserved_peak']
    lines.append(f"| {key} | {A['bytes']:,} | {A['GB']:.6f} / {A['GiB']:.6f} | {R['bytes']:,} | {R['GB']:.6f} / {R['GiB']:.6f} | {r['vendor_observed_device_peak_KB']:,} | {r['vendor_observed_process_peak_MiB']:,} |")
lines+=['','厂商原始单位按其67108864KB=64GiB总量解释为KB×1024B，进程MiB×2²⁰B；以下是换算值，不冒充原始字节API：','',
'| 配置 | 设备观测B / GB / GiB | 进程观测B / GB / GiB |','|---|---:|---:|']
for key,r in runs.items():
    d=r['vendor_observed_device_peak_bytes'];p=r['vendor_observed_process_peak_bytes']
    lines.append(f"| {key} | {d:,} / {d/1e9:.6f} / {d/2**30:.6f} | {p:,} / {p/1e9:.6f} / {p/2**30:.6f} |")
lines+=['',
'B2相对B1 allocated仅少19,291,648B，reserved反而多2,044,723,200B，不能声称BF16运行时优化省2GB。Q2相对B2 allocated少2,137,540,096B（2.137540GB、1.990739GiB），reserved少2,134,900,736B；达到2,000,000,000B门槛，未到2GiB。厂商数据保留原KB/MiB标签，为1秒观测峰值，可能漏瞬时峰值，不能替代框架allocated/reserved。',
'Q1在CPU加载时量化为group32 INT8+BF16 scale，Triton还原后BF16 GEMM，有损。packed存储只减少分配粒度浪费，不改善量化质量；首版unpacked仅省1.967GB，packed后省约2.026GB但质量失败。',
'Q2不近似权重：104矩阵（96个MLP及第7..14层8个attention输出投影）按低8位原存、高字节16项字典4bit编码、int32/int16例外表保存。Triton逐位还原至共享90,177,536B BF16工作区，再执行厂商GEMM。原checkpoint不改写，没有完整BF16设备影子副本；存储理论节省减工作区2,140,168,050B，实际峰值节省2,137,540,096B。反复还原及kernel/launch成本造成速度回退，不将文件压缩率当作显存或速度。','',
'## 6. 完整预热后的固定真实输入回放','',
'每轮8个真实仿真观测，先完整预热一序列，再测3序列；AB/BA反转顺序，每配置48个测量。无仿真，不作为成功率。','',
'| 对照 | 平均模型ms（参考→测试） | 模型比值 | pipeline比值 | 两轮动作精确 |',
'|---|---:|---:|---:|---|']
for r in replays:
    ref,test={'native_runtime':('native','runtime'),'runtime_w8a16':('runtime','quantized'),'runtime_lossless':('runtime','lossless')}[r['comparison']]
    A=r['profiles'][ref]['model']['mean_ms'];B=r['profiles'][test]['model']['mean_ms']
    lines.append(f"| {r['comparison']} | {A:.3f}→{B:.3f} | {r['model_speedup']:.6f}× | {r['pipeline_speedup']:.6f}× | {r['all_actions_exact']} |")
lines+=['',
'原生/优化AB模型1.449864×、BA 1.576801×，次序波动原样保留。压缩的同输入时延和每轮加载到推理峰值另见replay summary，不与其他配置拼接。原始时序、动作、输入SHA和nearest-rank P95均保留。','',
'## 7. 独立设备诊断与瓶颈','',
'真正MACA kernel已捕获，包括mcblas__Mck_bf16gemm。标准推测步/周期KF步各采一帧，B1/B2共四个trace，诊断不进入正式测速。标准步矩阵kernel总约83.013→82.985ms，GPU kernel合计150.977→145.903ms；mcMemcpyAsync数量4073→66、其CPU API总时间69.385→7.940ms。周期KF步拷贝2153→38，矩阵kernel约58.417→58.447ms。主要收益来自减少小拷贝、CPU调度和同步，而非假称矩阵乘吞吐大幅提升。API时间不是DMA时间；不据此宣称完整传输带宽指标通过。',
'Chrome trace及cost_breakdown在outputs/metax_device_diagnostic_02，包含真实kernel、copy、CPU调用和形状。常规测量不启用该重型profiler。','',
'## 8. B0阻塞与保留的失败记录','',
'真实SpecVLA上游已固定至56ef2985e25c7903d2759254efb754327fb975f7。原函数在root+6个候选时使用>6判断，可能继续生成第8个token，再取末7个而丢root。CPU协议反例及仅将>6改为>=6的提案都已复现（合成协议单元，不是模型部署）。此前已请求用户选择是否修正B0 seven停止边界并保留原版审计；答复前不更改该基线定义。没有把关闭KERV几个开关当作B0。',
'保留基础设施失败：初次依赖索引不可用后改用代理下的官方HTTPS索引；Docker默认bridge失效后独立容器用host网络，未改宿主网络；源码快照发现镜像缺git后只在容器补装；初次诊断脚本profile.py遮蔽stdlib，已改名device_profile.py并用新目录重试。模型前失败不计策略成功率，W8A16真实策略失败计入分母。','',
'## 9. 复跑、复算与证据','',
'```bash','KERV_METAX_GPU=0 ports/metax/bootstrap.sh',
'scripts/run_kerv_metax.sh --profile native --max-tasks 3 --trials 2 --output-dir outputs/native_new',
'scripts/run_kerv_metax.sh --profile optimized --max-tasks 3 --trials 2 --output-dir outputs/optimized_new',
'scripts/run_kerv_metax.sh --profile bf16_packed --max-tasks 3 --trials 2 --output-dir outputs/lossless_new',
'python3 ports/metax/evaluate.py --native outputs/native_new --optimized outputs/optimized_new \\',
'  --lossless outputs/lossless_new --output outputs/results_new.json','```','',
'先mx-smi确认目标卡可用，容器创建时冻结物理映射；输出目录必须不存在。真实权重获取、依赖和回放方法见[沐曦入口说明](../ports/metax/README.md)。',
'主要原始目录：outputs/metax_native_goal3_init01_02、metax_optimized_goal3_init01_02、metax_w8a16_packed_goal3_init01、metax_bf16_lossless_goal3_init01。每组含解析配置、launch/exit、源码包/逐文件哈希、actions、runtime_stats、raw steps/verify/memory/episodes/placement和vendor-memory.log。复算总表outputs/metax_inventory/goal3-all-results.json；回放总表replay-summary-v2.json、quant-replay-summary.json、lossless-replay-summary.json。',
'起始快照79466c7，后续按阶段本地提交，未推送远端；权重、凭据、巨大日志均不提交。首个开发配对回合没有逐次源码包，六回合主要结果已保存逐次源码包；所有限制保留。',
'证据索引为outputs/metax_inventory/evidence_index.json和checksums.sha256；源码快照/原始动作/回放/设备trace校验结果为validation.json。',
'当前完成范围为单款C550、OpenVLA、LIBERO Goal三任务双初态试跑。完整Goal10/更多初态/种子、B0正式对照、A100/其他模型或平台/第三方验收均不在已完成成绩内。后续扩大矩阵应先冻结B0协议，不能用旧机器或论文结果补齐。','']
(ROOT/'docs/METAX_TECHNICAL_REPORT.md').write_text('\n'.join(lines))
