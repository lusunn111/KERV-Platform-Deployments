# 沐曦 C550 第二轮优化说明

本轮在 baai-metax-1 的原工程和 kerv-metax 容器内执行，第一轮 `bf179f6` 另存 tag `metax-round1-bf179f6`。第一轮所有原始输出保持原路径；本轮数据独立位于 `outputs/metax_v2_20261104/`。物理 GPU0 映射逻辑 cuda:0，所有 GPU 实验使用 `/tmp/kerv-metax-card0.lock` 串行运行。厂商 PyTorch、驱动和 MACA 未升级，没有网络/宿主共享环境修改。

## 选择依据与实现

首批捕获的真实 verifier 矩阵行数208–301，49对应路径数而非唯一树节点。第一轮普通步的矩阵 kernel 活跃时间约83ms，优化前后基本相同。进一步按 CPU launch 关联 trace（`trace_phases.py`）计算各 scope 的设备时间并集：第一轮 optimized 的 prompt+initial draft 56.61ms、verify82.26ms、posterior0.22ms、KV+next draft7.68ms。CPU scope 与设备工作重叠，不把二者相加，也不把这些累计区间当作完整关键路径。

第二轮独立事件诊断（`phase_diag01/phases.json`）对已预热普通/周期KF步骤采样，进一步区分视觉、verifier、drafter、posterior、KV copy。嵌套模块区间不相加，诊断耗时不进入正式速度统计。posterior较长CPU耗时含等待队列中的验证计算，实际设备归约很短，单纯重写该段不一定加速。

本轮主要组合由 `ports/metax/v2.py` 安装，逐项计数，原生和第一轮optimized不安装：

- `rope`：Q/K与位置索引处理合并为一个Triton核，显式将每个乘积先舍入BF16，再相加舍入BF16；不合并改变舍入的FMA。保持原输出stride。实际采集形状微基准输出/stride一致，5.1–5.6倍算子收益，单项8步模型168.612→154.485ms且动作一致。
- `rms`：只替换verifier的LlamaRMSNorm，原FP32归约、归一化后转BF16、乘BF16权重的边界保留。并行归约/逆平方根实现与原实现存在数值差异；首层279行输入有38个BF16元素不同，完整模型动作会变化。它是质量须单独评估的新配置，不能称全模型位精确或696步轨迹一致。
- `vision_graph`：只捕获固定形状的原始双视觉编码器前向。每步复制新pixel，复用相同数学运算。首次捕获逐元素核对特征，动态形状不静默复用。单项回放约5.6%模型收益，动作一致；图池内存、捕获时间计入部署峰值和冷启动。

未改变图像预处理、动作反归一化、seven根节点/接受前缀/补偿/停止边界、树规模、动态阈值、220规划时域、KF周期或BF16权重精度。运行时数值差异单独标注；不通过更换任务或修改基线争取2倍。

## 消融与淘汰

| 实验 | 实测 | 处理 |
|---|---|---|
| unique-node compact接受融合 | 169.418→170.088ms；32次真实验证的控制/选中logits行审计一致 | 无净收益，独立实验开关，不默认启用 |
| Gate-Up合并 | 大行数约1.02–1.035倍；208/255行更慢且部分BF16值改变 | 未推广 |
| QKV合并 | 孤立GEMM约1.11–1.42倍；叠加组合124.995→123.512ms但更多动作差异 | 收益有限，未推广 |
| RMS/RoPE/视觉图 | 六回合5/6，722步；旧配置5/6、696步 | 作为v2 pilot候选，正式冻结历史测速单列 |
| verifier prefill图 | 126.291→125.100ms，命中32次且动作/首帧全部输出一致；allocated多460.5MB | 仅1%收益，独立保留不默认 |
| RMS改rsqrt并禁止FMA | 首层9个形状全部BF16一致，但完整六回合3/6、937步 | 质量淘汰；`final_goal3_init01`只是历史目录名，绝非最终选择 |

完整原始动作、事件、接受前缀、时间与显存分别位于对应 `raw/` 和 `runtime_stats.jsonl`。失败实验保留，不拼接不同配置的速度、显存或质量成绩。

## 计量方式

`episode_stats.py`按每个完整回合分别排除第0步，再计算均值、中位数、nearest-rank P95；合并速度采用总时间/总动作数。第一轮B1/B2的690步模型均值328.327511/214.976217ms，合并1.527273倍；完整含首步结果仍为232.317148/153.676999s，闭环349.335163/264.449049s。

220步完整冻结序列正反回放合并：native357.216481ms、第一轮runtime222.211205ms、v2 173.912696ms，native/v2=2.053999倍、runtime/v2=1.277717倍；策略native/v2=2.025494倍。8步短片段合并只有1.939334倍，分别保留。v2对220步native模型耗时下降约51.31%，相对第一轮runtime下降约21.74%。

正式回放使用相同图像、原始动作历史、token历史、阈值、seed与220时域。`--frozen-history`在计时前从fixture的raw/steps.jsonl恢复参考前缀，并断言native/第一轮runtime产生原参考token；候选预测作为结果保留，不反馈到下一步冻结KF输入。真实闭环独立运行，使用自身实际动作反馈。图像回放但反馈各自历史的早期诊断明确不作为全部输入相同的证据。

每profile完整序列预热后测量；三profile顺序native/runtime/v2与反序。8步各测3轮，220步各测1轮，两次正反顺序合并。重型profiler独立运行，正式测量关闭逐阶段日志，三组收集开销相同。脚本保留源码SHA及源码包，原始计时数组可重新汇总。每回合首动作和冷启动不从闭环结果删除。

六回合候选rms/rope/vision_graph实际5/6，对第一轮0个百分点、相对0%；722步/2000次验证，真实提交连续前缀2174/2000=1.087（第一轮2089/1928=1.083506，增量约0.003494，未达到+0.5）。初始root722、后续bonus1278、补偿880、padding0，seven守恒通过。六条轨迹均发生token变化，因此原696步完全一致结论不适用于v2。该pilot不能证明大规模质量稳定。

## 无损存储候选

第一轮104矩阵采用low8+high4字典及int32/int16例外，恢复核和例外写回各一次launch，再写共享BF16工作区并调用厂商GEMM。文件小不等于显存峰值降低；共享工作区、图池及加载过程均计入峰值。

`lossless_v2.py`为独立单pass候选：15项高字节字典，code15表示例外；每1024权重记录例外前缀，核内scan定位例外高字节，低字节保持原样。恢复后仍调用厂商BF16 GEMM，不引入舍入、低比特GEMM或永久BF16影子。寄存器字典与全局字典两种解码路径先做真实矩阵微基准再选择。CPU加载模型先压缩再放GPU，峰值计数从加载前开始，任何缓存均包含在峰值中。

本轮默认压缩配置改用94个MLP矩阵，attention及layer0的gate/up保持原生BF16，codec与厂商GEMM不变。8步AB/BA三组对照：v2 BF16 125.163460ms、原104矩阵292.038022ms、94矩阵288.399237ms；94相对104仅1.012617倍，仍是v2 BF16的2.304181倍耗时。两轮allocated均省2,026,985,984B（2.026986GB、约1.887GiB）。完整闭环和220步结果另列，不按短回放推算。

多profile同进程比较已增加模型释放后的进程内BLAS工作区清理：视觉图新stream会遗留约128MiB缓存。清理发生在下一profile加载前，每组从加载前重新计峰值；当前加载、图池、工作区、缓存全部计入，pre_load/post_release分配原始字节保留。早期未清理的回放内存只作为原始过程记录，不作为独立部署峰值；各独立进程完整六回合峰值有效。

单pass/寄存器字典/向量2或4权重恢复都未改善0.57ms/90MB的实际恢复延迟；双缓冲预取289.923→287.999ms，新增90MB且收益不足；内联解码+dot有9项共享内存越界，其余22–98ms，多数GEMM结果不同。这些都隔离为实验，不将其峰值或速度拼成默认成绩。正式性能表见 [性能改进表](METAX_V2_PERFORMANCE.md)。

## 复跑入口

```bash
# 第一轮保持独立；原配置与代码行为不安装v2开关
scripts/run_kerv_metax.sh --profile native --max-tasks 3 --trials 2 --output-dir outputs/your_native
scripts/run_kerv_metax.sh --profile optimized --max-tasks 3 --trials 2 --output-dir outputs/your_v1
# 已通过pilot但存在数值/轨迹差异的v2
scripts/run_kerv_metax.sh --profile optimized_v2 --max-tasks 3 --trials 2 --output-dir outputs/your_v2
# 94矩阵无损存储 + 同一v2运行时；相对v2仍有明显恢复开销
scripts/run_kerv_metax.sh --profile bf16_packed_v2 --max-tasks 3 --trials 2 --output-dir outputs/your_compressed_v2
# 单项消融，可回退到空特性；仍保留第一轮runtime优化
scripts/run_kerv_metax.sh --profile optimized_v2 --v2-features rope,vision_graph --output-dir outputs/your_ablation
# 全历史冻结三profile回放（替换唯一输出目录，反序改BA）
scripts/benchmark_kerv_metax.sh --comparison native_runtime_v2 --v2-features rms,rope,vision_graph --frozen-history --allow-action-change --samples 220 --repeats 1 --profile-order AB --input-dir outputs/metax_v2_20261104/representative_capture/observations --output outputs/your_replay
```

真实FlagScale路径仍为run_kerv_metax.sh → ports/metax/run.py → third_party/FlagScale/run.py → 既有worker → 厂商PyTorch/MACA。配置backend=vllm只是FlagScale注册桥，不代表用vLLM执行模型。B0的上游SpecVLA动作边界定义保持独立待决，未修改、未计算B0倍率，也未阻塞B1/B2优化。


## 最终验收与限制

全部正式8/220步AB/BA、两组六回合闭环已完成，`python3 -m ports.metax.validate_v2` 返回passed，细节为 `outputs/metax_v2_20261104/delivery-validation.json`。94矩阵压缩与v2 BF16的722动作、2000次验证、2174实际候选前缀、722初始root、1278 bonus、880补偿精确一致；padding0。相对第一轮696动作的变化明确属于v2数值变化。

220步压缩三组正反合并：BF16 171.689656ms、104矩阵396.627005ms、94矩阵384.000737ms。94相对104速度1.032881倍、耗时下降3.183%，仍需BF16的2.236598倍模型时间，未消除无损还原的主要开销。8步只改善1.012617倍。短异常列表修复后的inline dot仍约13–37ms，不推广；不能将失败内核的理论收益填入最终结果。

完整6回合压缩模型277.453014s、策略285.130729s、闭环394.881736s；v2 BF16分别130.261169/138.153574/249.624491s。原始allocated 17,175,200,768→15,148,214,784B；reserved19,247,661,056→17,221,812,224B；设备1秒采样峰值21,191,213,056→19,165,343,744B，进程采样20,289,945,600→18,264,096,768B。采样峰值不是瞬时真峰值，字节换算依据原始vendor KB/MiB，PyTorch精确峰值单列。

省2GB的参照为同一v2 BF16配置，2,026,985,984B（2.026985984GB、1.887777805GiB），不是2GiB；相对第一轮native只省1,903,404,544B，未到2GB。相对第一轮原始无损配置，当前选择较快但显存略高，表中保持各配置自身的速度、显存和质量，未拼接。

六回合5/6仅是配对pilot，未覆盖全部Goal任务/初态，不能证明总体质量损失小于5%。接受前缀目标+0.5未达到；B0定义仍独立待决。工作区清理仅在模型之间进行，下一模型从加载前计峰值；没有为达标排除加载/图池/缓存。所有GPU实验已结束，物理卡0释放；无需修改其他服务器或全局共享环境。
