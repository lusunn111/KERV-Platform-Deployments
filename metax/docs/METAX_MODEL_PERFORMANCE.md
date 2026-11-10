# 沐曦模型性能调试：真实 SpecVLA 与稳定压缩组合

截至 2026-11-06 04:42 UTC，本阶段是 **CPU准备完成、等待GPU**。稳定参考为 `cf41388 / numerical_fixed`。当前没有本阶段新的GPU成功率、速度、接受长度或峰值显存成绩；下方待测项不能用历史结果填充。

2026-11-05 的代理因模型容量限制退出，代码未改变。恢复时重新读取授权任务与当前Git，2026-11-06 04:28 UTC实查物理卡0为他人PID1063812，约55154MiB进程显存；后续每分钟检查仍有该进程。没有杀他人进程、重启容器/驱动或改用其他机器。设备状态以实时 `outputs/metax_model_performance_20261105/queue-state.json` 为准，不能以历史空闲或利用率低判断。

## 已实际完成的代码及CPU检查

- 独立SpecVLA目录 `ports/metax/specvla/`，固定上游提交 `56ef2985e25c7903d2759254efb754327fb975f7`。`upstream_original/` 保存原始文件字节、许可证与SHA256；隔离命名空间加载原始verifier、drafter、tree/posterior/KV/update及动作解码，没有通过关闭KERV开关制造B0。
- 两入口 `upstream_original` 与 `boundary_fixed`。两者共同适配仅有绝对import路由及draft checkpoint `torch.load(..., map_location='cpu')`，后者避免CUDA序列化权重在CPU准备期偷偷回GPU。两版之间唯一算法条件差是 `utils.update_inference_inputs` 的 `>6 → >=6`。不改权重、树、阈值、seed或预处理来掩盖边界错误。
- 16个CPU协议案例覆盖首轮接受0..7以及后续root/bonus，验证单条件AST差异、缓存提交长度、原版8token后末七维丢首root的反例、修复版七维停止、计量包装前后输出相同。原始generated token、解码切片、丢弃root/candidate均单独保存，不把丢弃/bonus/KF计为接受。
- **真实权重CPU加载已通过**：原上游32层verifier、1层drafter、4096接口及七维动作、严格draft state_dict加载和共享embedding/KV reset均通过。模型全部参数仍在CPU，GPU context未初始化。两个真实MLP矩阵各45088768元素，原codec CPU还原位差0且SHA256一致；这不是GPU解码、速度或质量结论。首次CPU加载因checkpoint原CUDA存储标记失败，失败日志保留，显式CPU加载适配后重试通过。
- `numerical_fixed_packed` 独立配置：继承稳定修复版的 `rms_native_mean,rope,vision_graph`，仅将QUANT从none改为已验证的 `bf16_lossless_selective`（94个MLP）。旧 `bf16_packed_v2` 和旧RMS不改写。CPU配置检查确认三个数值特性完全一致。CPU压缩后才迁到GPU，原共享workspace、图池/解码缓存及所有加载峰值均进入collector；未重新尝试历史无收益融合解码。
- `run.py --prepare-only` 只生成FlagScale解析配置和源码快照，不导入Torch、不查询GPU、不启动worker。五种profile都准备同一3任务×2初态参数。独立SpecVLA入口复用既有图像JPEG/Lanczos、crop、processor与反归一化，独立执行上游生成；共享FlagScale桥及错误退出传播。
- 等待机制CPU测试实际执行其循环与文件锁，用模拟忙卡/两次空闲/锁内变忙/源码不一致状态验证：忙卡不运行；锁内变忙重新累计；源码冻结不一致拒绝启动。测试payload是临时目录里的无害CPU写文件，非GPU实验。

证据目录：`outputs/metax_model_performance_20261105/` 中的 `cpu-checks.json`、`queue-cpu-checks.json`、`cpu-weight-checks.json`、`cpu-weight-checks-attempt01.log`、`cpu-processor-check.log`、`specvla-entry-help.log`、`prepared_*/`。原始上游文件逐字节验SHA，不提交权重、日志或大文件。

## B0身份与接受计量

本批准备采用真实上游 **SpecVLA Relaxed、固定token距离阈值9**，对应已取回的 `run_libero_goal_Spec_Relaxed.py` 默认值；原版及边界修复版各自单列。上游strict Spec入口默认0，参数 `--specvla-threshold 0` 可显式复跑，不能把两个定义混用。正式任务书B0身份须在最终表格明确写出，目前不是将边界错误原版当公平倍率分母。KERV使用已有动态15..3与KF，这是已声明的算法差异。

主接受长度为每次验证实际提交到动作的连续草稿候选前缀，包含零。分母为各自所有真实验证事件；初始root、后续verifier bonus、KF、padding及尾切丢弃分别计数。SpecVLA原版需要从实际生成序列和末七维切片回溯保留区间，不能套用修复版七维裁剪假设计数。如果保留部分不是候选前缀，汇总标记，不能隐藏口径问题。逐动作检查 `root+bonus+接受+补偿+padding=7`。

+0.5目标的比较对象是边界修复的真实SpecVLA，不是KERV native与稳定优化版之差。SR、模型/策略/闭环时间、去每回合首步的均值/中位/nearest-rank P95及冷首步均从raw重新汇总。成功终止步保留。任务异常计失败，未执行初态明确标缺失，并给计划矩阵SR上下界，不从分母静默删除。

## 开发、冻结验证及执行次序

`evaluation-matrix.json` 已登记历史episode初态元数据。历史开发使用init0/1；从未用过的init2..6选作Goal10任务×5初态共50回合，seed7、动作上限及规划时域均220。本轮先只排3任务×init0/1开发六回合，50回合 **尚未启动**。最终候选要在开发集依据质量/接受/端到端联合选择并冻结，之后再单独解锁holdout；不在holdout调阈值或树。

准备队列依次为五配置各8动作短流程，再五配置各完整六回合：真实B0原版、修复版、KERV native、numerical_fixed、numerical_fixed_packed。所有GPU任务单卡串行、各自新进程。失败原版保留，其失败不妨碍独立配置；完整回合只依赖自身短流程通过。单个阶段完成后CPU汇总更新本文件，原始结果独立、不拼旧数据。根据开发测得的真实B0差距，再决定任务C的候选排序/缓存/位置拒绝率/树预算/阈值调试，当前没有盲调参数。

压缩组合必须与本阶段同一未压缩修复版六回合逐字段比较，再报告对SpecVLA、native、同版BF16三个不同参考的真实峰值差。目标≥2000000000B只能使用完整加载峰值证明；省显存与速度不拼不同配置。CPU驻留预算充足（恢复时可用约756GiB），模型CPU构造及编码峰值原始RSS另保留。磁盘恢复时约18GiB空闲，队列每次启动要求至少8000000000B可用，不清他人数据。

## 持久等待与复跑

等待器在本机 `tmux kerv-metax:model-gpu-queue` 内持续运行。每60秒仅查询GPU0进程/显存并写简短JSON，不打印全卡表。连续两次无计算进程且≤950MiB（历史基础858..872MiB），随后取得 `/tmp/kerv-metax-card0.lock` 并再查一次。检查Git HEAD、冻结源码/配置、工作树、矩阵SHA和磁盘后，锁持有期间直接启动独立容器命令，避免嵌套flock死锁。只有动态进度/报告允许变化。等待时间不进入实验耗时。

`waiting_gpu/running/completed/failed` 状态和每次检查保留在 `queue-state.json/queue-events.jsonl`，每个job独立log、命令、配置、源码快照与退出码。若监督进程中断时job仍running，重启拒绝隐式重跑，需先审计子进程与输出。测试脚本不会使用模拟空闲信息来启动实际GPU任务。

```bash
# 查看实时状态，不启动模型
cat outputs/metax_model_performance_20261105/queue-state.json
# CPU原始结果重汇总
python3 -m ports.metax.model_performance_summary
# 五配置入口均通过同一FlagScale桥；GPU空闲检查后才可手动执行
scripts/run_kerv_metax.sh --profile upstream_original --max-tasks 3 --trials 2 --output-dir outputs/spec_original_new
scripts/run_kerv_metax.sh --profile boundary_fixed --max-tasks 3 --trials 2 --output-dir outputs/spec_fixed_new
scripts/run_kerv_metax.sh --profile numerical_fixed_packed --max-tasks 3 --trials 2 --output-dir outputs/numerical_packed_new
```

<!-- measured-results -->

本阶段开发集原始结果（不拼接历轮数值）：

| 配置 | 状态 | SR成功/已执行/计划 | 模型/策略/闭环 s | allocated / reserved B | 主接受长度 |
|---|---|---|---|---|---|
| upstream_original | pending，等待GPU | — / — / 6 | — | — | — |
| boundary_fixed | pending，等待GPU | — / — / 6 | — | — | — |
| native | pending，等待GPU | — / — / 6 | — | — | — |
| numerical_fixed | pending，等待GPU | — / — / 6 | — | — | — |
| numerical_fixed_packed | pending，等待GPU | — / — / 6 | — | — | — |

原版仅单列，不用其边界错误放大倍率。配对差/区间、每回合去首步均值/中位/P95、含首步及设备采样见 `model-performance-results.json`。未完整的配置不计算正式配对倍率。

<!-- measured-results -->

当前未解决项：本阶段所有GPU模型指标待测；+0.5、≥2GB与质量目标未宣称达成；50回合仅阶段配对评测，需报告配对差与不确定性，不自动证明总体SR损失<5%。本机同条件对照不能替代任务书A100 65s/task条件。
