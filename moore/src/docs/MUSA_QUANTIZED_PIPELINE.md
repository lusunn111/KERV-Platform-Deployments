# MUSA 混合量化推理链路优化

本轮在原75投影校准量化策略上继续优化运行时，权重、保护组及原生BF16 GEMM均保留。此前量化策略的任务精度没有通过；本轮性能结果不能代替精度验收。最终实测见[链路优化报告](../../docs/acceptance/MOORE_QUANTIZED_PIPELINE_REPORT_20261027.md)。

## 显式启动入口

宿主机执行，模型及policy路径按已有工作区布局：

```bash
cd /root/KERV
export KERV_CALIBRATED_POLICY=/workspace/outputs/calibrated_quantization/20261026T045030Z_v2/model_refined/policy.json
bash src/scripts/run_musa_calibrated_tree.sh
```

该脚本沿用`run_musa_calibrated.sh`、FlagScale和既有MUSA适配，使用native配置并显式覆盖：

```text
entrypoint=run_calibrated_tree_libero.py
tree_builder.enabled=true
tree_builder.static_runtime=false
tree_builder.kalman_batch=false
```

量化部分仍使用group32、原BF16保护组和`row_decode_1024`算子。不启用树剪枝、静态树重排、RoPE/attention替换或Linear合并。原生树路径原本逐分支调用标量Kalman预测；新入口显式关闭Kalman批量数值路径，继续逐分支使用相同预测函数，只改变缓冲构建及传输方式。

旧`run_musa_calibrated_kernel.sh`保留为row解码对照；原默认BF16入口保留。选择实验量化入口不会使原量化策略自动获得质量通过状态。

## 优化依据与实现

上一轮完整模型没有提速，因此先在已预热真实动作上同时记录cProfile和MUSA事件。单帧插桩诊断显示Kalman候选树扩展的耗时明显高于量化解码，主要开销包括逐节点GPU标量读写、反复拼接tokens及祖先mask构造。

本轮复用仓库已经集成的FlagOS/KERV批量树路径，将其接入校准量化入口：

1. 一次将原始tokens和深度复制到CPU。
2. 按同样的节点顺序构建分支前缀，逐分支调用原Kalman预测。
3. 汇总附加tokens后一次拼接。
4. 在CPU构建祖先mask、位置编号及候选路径，再批量送回设备。

`_augment_tree_with_kf_branches_batched`与`_rebuild_tree_buffers_batched`是之前已存在的适配，本轮新增了显式量化运行入口、实际命中审计及同输入验证。没有声称这一实现已提交到上游FlagScale/FlagGems。

## 实际命中与数值检查

加载过程检查原payload哈希及75个解码权重SHA256；`calibrated_kernel.jsonl`记录实际MUSA模块和解码次数。新增`calibrated_tree_runtime.jsonl`在退出时记录：

- `tree_builder_enabled=true`及非零`tree_builder_hits`。
- `static_tree_runtime=false`，不走静态重排/裁剪。
- `kalman_batch_enabled=false`，保留标量预测数值路径。
- 实际加载的`policy_sha256`。

同输入诊断比较旧、新两条树构建路径的四个返回tensor：tokens、retrieve indices、mask、position ids。28次真实树扩展全部逐位一致；另有浅树、深树、分叉和无Kalman补充分支的回归测试。独立worker的完整动作一致数仍需查看回放，不由这些局部检查推断。

## 双槽提前解码试验

另实现`calibrated_prefetch.py`：两块BF16临时权重槽交替使用，通过MUSA事件保证“解码完成后才GEMM、GEMM结束后才能复用槽位”。75投影的实际槽位容量共180355072 bytes，相对逐次解码通常多一块90177536 bytes临时权重。它没有永久展开整份模型。

事件复用压力测试和75个实际权重数值检查通过，但本机轮换微基准没有性能收益，因此未选入本轮主性能候选。`prefetch`相关代码保留为实验，需要完整有序的verifier前向；拒绝重入、顺序变化和计算流切换，尚未完成完整模型部署验收。主启动脚本不会调用它。

## 复现对照实验

只使用物理GPU0。运行前确保空卡，所有测试串行执行；不要写入已封存campaign。新建目录后执行：

```bash
cd /root/KERV
mkdir -p outputs/quantized_pipeline/new_campaign
PYTHONPATH=src python3 -m acceptance.quantized_kernel_replay \
  --campaign /root/KERV/outputs/quantized_pipeline/new_campaign \
  --capture /root/KERV/outputs/calibrated_quantization/20261026T045030Z_v2/inputs/diagnostic_capture \
  --policy /root/KERV/outputs/calibrated_quantization/20261026T045030Z_v2/model_refined/policy.json \
  --reference-backend rows --candidate-backend tree_rows \
  --experiment-name same_policy_tree_v1 --rounds 3

PYTHONPATH=src python3 -m acceptance.check_quantized_kernel_replay \
  --experiment /root/KERV/outputs/quantized_pipeline/new_campaign/replays/same_policy_tree_v1
```

每个worker先完整预热141帧，然后按AB/BA/AB顺序串行计算。每帧恢复同一观测、历史raw动作及Kalman token前缀。回放不开启profiler或运行时逐步日志；两个模型同时驻留，回放显存不能当作单进程冷启动峰值。倍率的参考对象是上一轮row量化后端，不是原BF16模型。

原始逐帧时间和动作位于各worker的`measure_0/1/2.jsonl`，`results.json`为汇总，`evidence_checks.json`独立复算。额外的冷启动冒烟仅8步，用于检查执行、显存及实际命中，不计完整任务成功率。

本轮封存目录同时备份`inputs/replay_capture/`和`inputs/diagnostic_first8/`；原始命令和protocol保留当时实际使用的路径。搬迁复现时可将`--capture`改为备份的实际路径，映射和SHA见`inventory/input_manifest.json`。大体积checkpoint及量化payload仍需按依赖清单另行携带。

## 故障与验证边界

首次真实树诊断的挂钩未覆盖utils的实际入口，前两次不能算树路径验证通过；第三次修正后才得到有效覆盖。首个无覆盖诊断在原row路径上触发MUSA `IndexSelect`非法地址，已保留dump并只终止自有进程，未重置设备或停止其他任务。其根因未定位；后续带Embedding索引边界检查的诊断通过，不能据此宣称故障已修复。边界检查不在性能计时路径中。

原量化策略相对BF16的误差、既有运行重复性问题和未完成的完整成功率矩阵均需单独处理。报告中的证据完整性PASS只说明可复算。
