"""Render the Ascend technical evaluation from the consolidated raw-data audit."""
import argparse
import csv
import json
from pathlib import Path
from acceptance.preflight import ROOT

LABELS={'bf16':'BF16 runtime B2','mixed':'混合量化Q1（推荐显式候选）','concat32':'32层拼接QKV','concat18':'中间18层拼接QKV','bmm':'初版BMM','bmm_cached':'缓存BMM Q2（实验项）'}
REQUIREMENTS=[
 ('1.1-F','运动反馈纠错补偿','5','BLOCKED','BLOCKED','模型推测验证/Kalman补齐实际运行；KF使用历史token','需真实运动反馈接口及D08确认'),
 ('1.1-C','两类VLA、至少三平台','5,7,8','BLOCKED','BLOCKED','仅OpenVLA＋LIBERO Goal','智源VLA、其他认可平台与接口；超出当前单机交付'),
 ('1.1-T','65 s/task参考下提升20%','5,7,16','BLOCKED','BLOCKED','本机B1→B2固定输入2.102倍；闭环墙钟1.450倍','缺真正B0后端等价性、A100条件及正式计时口径'),
 ('1.1-SR','各平台SR损失严格<5%','5,8,16','PASS','BLOCKED','本机BF16 14/20、Q1 16/20；有1失3得','不是300回合正式矩阵；不能证明总体损失界'),
 ('1.1-L','平均接受长度改善严格>0.5','5,8,16','BLOCKED','BLOCKED','有逐验证事件、保留0接受','缺B0与padding/root/bonus/补偿来源审计'),
 ('1.2-F','CPU反馈与加速器推理/补偿','6,8,16','BLOCKED','BLOCKED','CPU树/历史token KF、NPU模型与张量更新有证据','真实运动学反馈与任务书分工口径未完成确认'),
 ('1.2-B','CPU-加速器传输开销<100M/s','6,9,16','BLOCKED','BLOCKED','有ATen逻辑字节与独立CANN诊断','缺完整DMA方向/bytes/时长关联与D05定义'),
 ('1.2-M','显存节省2GB以上及优化推理开销','6,9,16','PASS','BLOCKED','D09授权Q1：allocated省约2.085GB、同卡均值-1.03%','本机工程观察通过；正式D06、模型/平台覆盖尚缺'),
 ('1.3-I','按FlagScale设计集成与开源','6,9','BLOCKED','BLOCKED','本地FlagScale KERV代码/入口存在且实际执行','本次未核验公开标签、上游接收及最终NPU适配是否随正式版本交付'),
 ('1.3-H','至少3款国产芯片','6','PASS','BLOCKED','当前只实测本台Ascend；16个chip是同型号资源','其他型号由对应节点分册承接；精确资产SKU待补'),
 ('1.3-R','第三方复现及智源VLA部署','6,9','NOT_STARTED','BLOCKED','本次为项目组本机自测','独立第三方环境、操作者与智源模型部署未在本机核验'),
 ('P10-materials','活动/专利/投稿/科技报告等材料','6,9,10,15,16,22','N/A','NOT_STARTED','本次仅交付技术大纲、测评和证据','非运行成果材料由项目材料清单核对，不推断其是否已具备'),
]


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join('---' for _ in headers)+'|']+['| '+' | '.join(map(str,row))+' |' for row in rows])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evidence',type=Path,required=True);a=p.parse_args()
    d=json.loads((a.evidence/'evaluation.json').read_text());v=json.loads((a.evidence/'validation.json').read_text())
    q=d['quality'];t=d['timings'];runs=d['runs'];legacy=d['legacy_runtime']
    quality=table(['配置','init1','init2','成功/总数','成功率','Wilson95%描述区间'],[
        [LABELS[k],str(r[0]['success'])+'/10',str(r[1]['success'])+'/10',f"{q[k]['success']}/{q[k]['total']}",f"{q[k]['rate']:.1%}",f"{q[k]['wilson95_descriptive'][0]:.1%}～{q[k]['wilson95_descriptive'][1]:.1%}"] for k,r in runs.items()])
    comparisons=[('mixed_vs_bf16','B2 BF16 → Q1混合'),('cached_vs_mixed','Q1混合 → Q2缓存BMM'),('cached_vs_bf16','B2 BF16 → Q2缓存BMM')]
    timing=table(['独立同卡比较','物理chip','参考均值ms','候选均值ms','时延变化','样本/profile'],[
        [label,t[k]['physical_device'],f"{t[k]['profiles']['bf16']['model_ms']['mean']:.6f}",f"{t[k]['profiles']['int8']['model_ms']['mean']:.6f}",f"{t[k]['change_percent']:+.4f}%",471] for k,label in comparisons])
    p95=table(['比较','参考P95 ms','候选P95 ms','三遍时延变化'],[
        [label,f"{t[k]['profiles']['bf16']['model_ms']['p95_nearest_rank']:.6f}",f"{t[k]['profiles']['int8']['model_ms']['p95_nearest_rank']:.6f}",' / '.join(f"{r['change_percent']:+.4f}%" for r in t[k]['per_repeat'])] for k,label in comparisons])
    memory=[]
    for label in ['bf16','mixed','bmm_cached']:
        for r in runs[label]:
            memory.append([LABELS[label].split('（')[0]+f" init{r['initial_state']} / chip{r['physical_device']}",
                f"{r['allocated_peak_bytes']/1e9:.6f}",f"{r['allocated_saving_vs_bf16_bytes']/1e9:.6f} / {r['allocated_saving_vs_bf16_bytes']/2**30:.6f}",
                f"{r['reserved_peak_bytes']/1e9:.6f}",r['driver_hbm_sampled_peak_mb'],'是' if r['driver_same_chip_as_bf16'] else '否'])
    memory=table(['单模型完整运行','allocated GB','allocated节省GB / GiB','reserved GB','driver采样峰值MB','driver与BF16同chip'],memory)
    current_legacy=legacy['profiles'];closed=legacy['closed_loop'];b1=closed['paired_B1'];b2=closed['paired_B2']
    req=table(['指标','本机观察','正式状态/缺项'],[[r[0]+' '+r[1],r[5],r[4]+'：'+r[6]] for r in REQUIREMENTS])
    accepts=table(['配置/初态','验证次数','0接受次数','原始posterior均值'],[
        [LABELS[k].split('（')[0]+f" init{r['initial_state']}",r['raw_acceptance']['count'],r['zero_acceptance_events'],f"{r['raw_acceptance']['mean']:.6f}"] for k in ['bf16','mixed','bmm_cached'] for r in runs[k]])
    pair=d['paired_quality']['mixed'];exact=sum(r['steps'] for r in d['behavior_parity'].values())
    rel=str(a.evidence.resolve().relative_to(ROOT));docs=ROOT/'docs/acceptance/reports/ASCEND_EVALUATION_REPORT_20261027.md'
    report=f'''# KERV 昇腾综合测评报告

版本：1.0；整理与复核日期：2026-10-27。性质：项目组本机技术自测。
实验日期：2026-10-19的运行时基线，以及2026-10-26～27的量化与算子实验，分别以原run.json为准。
范围：当前Ascend服务器、OpenVLA＋SpecVLA drafter、LIBERO Goal。本次交付没有新运行模型回合。
技术依据：[昇腾技术指标测试大纲](../ASCEND_TECHNICAL_OUTLINE_20261027.md)、根目录TestOutline.md及任务书原件。

## 1. 结论摘要

本机部署、运行时正确性检查、显存优化候选和可复算测评材料已经具备。
默认保持BF16，显式省内存优先采用Q1 W8A16/BF16混合；Q2缓存BMM为实验项。
正式课题验收状态仍为BLOCKED，具体原因按第9节逐项列出。

| 项目 | 有证据支持的本机结论 |
|---|---|
| 系统收益 | 2026-10-19，同输入B1→B2模型{legacy['model_speedup']:.3f}×、pipeline{legacy['pipeline_speedup']:.3f}×；这是KERV原生到运行时优化的比较 |
| Q1省内存与速度 | allocated约省2.085GB（约1.942GiB）；新测同卡模型均值{t['mixed_vs_bf16']['change_percent']:+.4f}% |
| 任务质量 | B2 BF16 14/20；Q1 16/20，有1个回合由成功变失败、3个由失败变成功 |
| Q2数值/动作 | 与Q1的{exact}个闭环步骤观测、token和动作逐步一致；同输入471个action及验证工作量也一致 |
| Q2时延 | 对Q1 {t['cached_vs_mixed']['change_percent']:+.4f}%，对新测BF16 {t['cached_vs_bf16']['change_percent']:+.4f}%；未过均值零回退门槛，不替换推荐 |
| 证据复核 | {v['source_files']}个源证据文件；{v['quality_runs']}组完整质量运行、{v['quality_episodes']}回合；6组独立时延比较；{d['original_frozen_hash_matches']}项与原冻结哈希一致 |

120回合来自6个配置复用相同20个task/init/seed组合，不是120个独立质量样本。
没有用2.102×乘另一轮量化比值，没有宣称达到2.4×或满足单卡A100的65秒参考目标。

## 2. 环境、配置与版本

宿主工作目录/root/KERV；容器kerv-ascend；实际运行副本/public-flash/zhiyuan/liyuzhuo/data/KERV，容器路径/data/data/KERV。
每模型单chip；npu-smi报告Ascend910、每chip65536MB。主机有16个物理chip，不能计成16种芯片。
Python3.12、Torch2.7.1+cpu配合torch_npu2.7.1.post8；容器工具链路径cann-9.0.1，驱动工具25.5.0。当前软件设备名为Ascend910_9382，不等同于资产SKU。实际导入Triton模块为3.2.0，环境分发包元数据为3.5.0；交付记录两者及加载路径，不按包标签替代运行模块版本。
推荐源码为ports/moore_ascend；没有原始Git元数据，使用每阶段源码归档、逐文件哈希和解析配置追溯。

Q1冻结policy：`{d['policy_sha256']}`。
verifier中间18层7:25，50个投影W8A16/group32，15/16/19/20层down保留BF16；其余不新增量化。
gate/up融合并CPU预打包，NPU执行原生weight-quant GEMM；Ascend Triton SwiGLU保留SiLU和乘法的BF16舍入。
Q2仅在257..320行使用QKV batch=3 GEMM，其他形状走原三个F.linear；缓存视图共享唯一权重storage。

量化由用户确认D09授权。量化的显存节省单列，不描述为同精度CPU/NPU异构部署本身的节省。
模型身份以本地checkpoint逐文件哈希清单核验，本报告不推测未记录的Hugging Face revision。

## 3. 测评方法与完整性

使用acceptance/evaluate_ascend_delivery.py只读解析旧campaign，向新交付目录写统计和索引。
检查了完整任务数、task/init对齐、唯一episode/step/verify ID、逐回合连续步数、失败保留、7维有限动作、seven token历史、量化CPU转换、原policy身份、算子命中和内存原值。
每份输入按SHA256登记，对有旧封存清单的已引用文件另核对原哈希；不重写旧campaign和旧结论。

P95统一采用nearest-rank：排序后第ceil(0.95N)项；均值、中位数及全部原始样本保留。
部分历史量化报告使用线性插值P95，本报告据raw重新计算，因此数值略有不同，原报告保持原样。
显存GB=1e9B，GiB=2^30B；npu-smi的MB沿用工具原始标签。

模型、pipeline、env.step、完整闭环分层处理。性能诊断与轻量计时分离；双模型驻留仅用于同卡AB/BA时延，不用于部署峰值。
机器并非严格独占，历史回放存在共享CPU干扰和轮间波动；三轮和复用任务的样本不足以证明总体时延或成功率不回退。

## 4. KERV原生到运行时优化

证据：2026-10-19 campaign，物理chip6的157帧固定观测，AB/BA/AB三轮，各profile471个动作。
动作逐值一致。该比较保持BF16、算法与动作协议；不以SpecVLA B0为基线。

| 模式 | 模型均值ms | 模型P95 ms | pipeline均值ms |
|---|---:|---:|---:|
| B1 native | {current_legacy['native']['model']['mean']:.6f} | {current_legacy['native']['model']['p95_nearest_rank']:.6f} | {current_legacy['native']['pipeline']['mean']:.6f} |
| B2 optimized | {current_legacy['runtime']['model']['mean']:.6f} | {current_legacy['runtime']['model']['p95_nearest_rank']:.6f} | {current_legacy['runtime']['pipeline']['mean']:.6f} |

单个Goal task0/init0/seed7完整闭环，两组均115动作成功且逐步一致。
含首动作的模型加速{b1['model_ms']['mean']/b2['model_ms']['mean']:.3f}×，pipeline {b1['pipeline_ms']['mean']/b2['pipeline_ms']['mean']:.3f}×，
包含仿真和记录的闭环墙钟{b1['wall_seconds']/b2['wall_seconds']:.3f}×。
单任务闭环与固定观测回放的分母和边界不同，不能相互替代。
原始轮次B1模型均值532.242/653.365/520.282ms，B2为260.020/290.678/260.870ms，波动不隐藏。

## 5. 量化与算子方案的完整任务结果

固定Goal10、初态1/2、seed7、TF裁剪、seven协议、规划/上限220；每配置20个完整回合。

{quality}

Q1相对BF16：两个配置都成功{pair['both_success']}个，丢失{pair['lost']}个、获得{pair['gained']}个，都失败{pair['both_failure']}个。
观察成功率损失为-10个百分点，相对损失-14.286%；负值只表示本小样本中提高，不是总体改善证明。
Wilson区间是比例的描述性补充，不处理任务/初态相关性，不能据此证明总体损失严格<5%。
初态已反复用于优化选择，不是未见测试集。总大纲建议的300回合/配置及B0/B1/B2共900回合矩阵未完成。

32层拼接版init1=6/10、init2=9/10；中间18层拼接版6/10、7/10。
微小的局部相对L2误差不能代替闭环质量验证；这两种融合均未替换推荐。
BMM初版虽保持任务结果与动作，但相对Q1平均时延+0.2535%，相对BF16+0.6790%，也未采用。

Q2与Q1的{exact}步观测/图像哈希、action token、raw/executed action、done、阈值及episode/step ID全部一致。
同卡固定输入471个action、verify记录和verifier形状也相同。该一致性相对Q1量化模型，不意味着与BF16相同。

## 6. 新测同卡模型时延

每组参考与候选同chip同时驻留，157帧完整预热1遍，再测3遍，每帧AB/BA交替；每profile471次。
各比较都有自己的新测参考，不能跨行比较绝对值或相乘。

{timing}

{p95}

Q1三遍均低于该组BF16参考均值，满足本次工程观察门槛。
Q2相对Q1三遍均值均降低，相对BF16两遍略高、一遍略低，合并+0.1063%；该小差不能单独证明显著变慢，也不能写成严格无回退。
预设零回退门槛未在看到结果后放宽，Q2仍是实验项。pipeline完整统计保存在evaluation.json和CSV源索引。
BF16与量化模型的动作/验证工作量允许不同，B2→Q1/Q2收益不是纯单核收益。

## 7. 单模型部署内存

独立单模型进程从加载初始化至10个完整回合结束，框架记录全程峰值；驱动10秒采样。

{memory}

Q1同初态allocated节省约2.085GB，满足2GB十进制观察目标，未到2GiB。
Q2 allocated节省2.0887～2.0935GB；reserved也单独列入quality.csv，不能用更有利的口径替换allocated。
Q2 init1在同chip0的驱动采样峰值22284→19901MB，下降2383MB；init2不是同chip的驱动对照，仅作参考。
框架allocated、reserved、驱动HBM不是同一口径，10秒采样可能漏瞬时峰值。比较基线为BF16 KERV runtime，不是真正SpecVLA B0。

CPU预打包避免先将原分立QKV搬入NPU再临时替换，修复了上轮融合新增的约3GB设备驻留。
设备内cat/copy与empty_cache尝试及其失败证据保留；不据此推断未观测的驱动内部泄漏归属。
推理中没有CPU权重卸载，视图缓存测试检查了storage指针和新输入后缓存清理。

## 8. 核心机制、算子与执行位置

当前量化运行的record_once算子日志实际记录了：{', '.join(runs['mixed'][0]['observed_operators'])}。
requested_operators列表只表示登记，不当作所有算子实际执行证明；record_once条数也不是完整调用数。
Q1两个初态的原生量化GEMM累计forward_calls分别为{runs['mixed'][0]['quantization']['forward_calls']}、{runs['mixed'][1]['quantization']['forward_calls']}。
SwiGLU的配置、实际forward路径、真实输入微基准和9项NPU定向测试另有记录，不能将局部kernel收益直接换算成端到端收益。
Q2 manifest确认32个verifier组CPU预打包、BMM与视图缓存，不触及drafter；公开配置没有使用JIT派发探针。

历史独立CANN诊断记录68204个NPU kernel、CPU Kalman范围、树验证与KF补齐范围。
主模型在NPU，CPU做仿真、预处理、树拓扑与历史token估计。Kalman基于历史输出token，不是已测得的关节/末端反馈；
动态阈值按步数衰减，不称为传感器运动状态自适应。

{accepts}

表中为原始posterior前缀统计，保留全拒绝0。尚未完成root/bonus/padding/补偿来源审计，也缺B0，不能判任务书ΔL>0.5。
历史8步ATen传输诊断记录双向逻辑字节，并另有CANN时间线；后端DMA方向、字节和时长未完整关联，不判<100M/s通过。
这些缺口不会被量化的显存改善覆盖。

## 9. 指标对应与完成边界

{req}

任务书第6页“不少于3款国产芯片”保留，服务器/卡/chip数量不替代型号数。
本机只承担Ascend分册；A100、智源模型、其他平台/芯片、独立第三方与非运行成果材料由总项目相应单元承接。
本报告完成本机技术测评材料交付，结论不是整个课题已通过验收。

## 10. 证据与复算

本次交付索引：`{rel}`。

- evaluation.json：原始记录复算结果、完整回合、六组计时、基线/量化身份和原哈希核验。
- quality.csv、episodes.csv：逐配置/逐回合结果，保留全部失败。
- timing.csv：逐对照的均值、中位数、nearest-rank P95与样本数。
- requirements_results.csv：任务书对应、已观察结果与未完成条件。
- evidence_index.csv、inventory/input_manifest.json：原始文件位置、角色、byte数与SHA256。
- inventory：复算工具、统计边界测试、环境、初稿大纲与本次文档源文件快照。
- 文件校验与交付包：见同目录delivery_checksums.sha256及delivery_manifest.json。

```bash
python -m acceptance.evaluate_ascend_delivery \\
  --output outputs/acceptance/ascend_delivery_recomputed_new
python -m acceptance.render_ascend_evaluation \\
  --evidence outputs/acceptance/ascend_delivery_recomputed_new
python -m unittest acceptance.test_statistics -v
```

原campaign保持不变。模型复现使用对应源码快照和冻结配置；当前Q1/Q2运行命令见技术大纲与README_ASCEND。
测试失败、参数解析修复、拒绝候选、插值/nearest-rank差异均保留，不只交付有利的摘要。
'''
    docs.write_text(report)
    with (a.evidence/'requirements_results.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['requirement_id','requirement','taskbook_pages','local_status','formal_status','observed_evidence','remaining_condition']);w.writerows(REQUIREMENTS)
    gaps=['# 昇腾分册缺口与总项目衔接','', '本机技术文档与证据交付完成；以下是对应正式指标的剩余条件，不是其他机器已经失败的结论。','',req]
    (a.evidence/'gap_register.md').write_text('\n'.join(gaps)+'\n')
    (a.evidence/'summary.md').write_text(f'''# 昇腾技术大纲与测评交付

复核日期：2026-10-27。新模型运行0次；从既有原始记录复算，旧campaign未修改。

- 源文件{v['source_files']}个，{v['quality_runs']}组完整运行/120回合；独立task/init/seed组合20个。
- B1→B2同输入模型{legacy['model_speedup']:.3f}×，不是SpecVLA基线。
- Q1 16/20、BF16 14/20；同卡均值{t['mixed_vs_bf16']['change_percent']:+.4f}%，allocated约省2.085GB。
- Q2与Q1 {exact}步一致；对BF16均值+0.1063%，未替换推荐。
- 默认BF16，省内存显式用Q1；总体正式验收状态BLOCKED，条件见requirements_results.csv。

主文档：docs/acceptance/ASCEND_TECHNICAL_OUTLINE_20261027.md 与 docs/acceptance/reports/ASCEND_EVALUATION_REPORT_20261027.md。
原始索引及所有CSV/JSON均在本目录。P95统一nearest-rank，旧报告不覆盖。
''')
    print(docs)

if __name__=='__main__':main()
