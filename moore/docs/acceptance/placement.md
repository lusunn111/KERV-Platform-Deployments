# P03：摩尔节点实际反馈与执行位置

本表依据当前执行源码，配合本轮逐验证事件、参数设备和诊断轨迹使用。它不把函数命名或配置声明直接当作设备执行证据。

| 阶段 | 文件/函数 | 输入与输出 | 实际位置/当前边界 |
| --- | --- | --- | --- |
| 仿真及反馈获取 | `run_kerv_libero.py:eval_libero`、`libero_utils.py:get_libero_image` | RGB、末端/夹爪状态；JPEG/resize后图像 | CPU OSMesa、TensorFlow 图像处理 |
| OpenVLA 输入 | `openvla_utils.py:get_vla_action` | 图像、任务文字→processor输入 | CPU processor 后搬到 MUSA；OpenVLA未使用 observation.state 作为模型输入 |
| 阈值调整 | `tool_utils.py:compute_dynamic_threshold` | step、220步时域、start=15、lower=3→阈值 | CPU Python；当前取决于步数和预定曲线，不直接读取实测关节状态 |
| 视觉/语言模型 | `modeling_speculation.py:SpecVLAforActionPrediction`、verifier模型 | 图像/token/隐藏状态 | MUSA BF16；真实 parameter devices 与前向钩子/trace 验证 |
| 草稿生成 | `cnets.py:MMModel` | multimodal hidden/input embedding→候选树token | MUSA；B0/B1/B2分别核对真实模块路径 |
| 运动估计 | `modeling_speculation.py:kalman_predict_tokens`、`kelman_predict_tokens`、`tool_utils.py:kalman_predict_from_history` | 历史模型动作/token序列→预测token | CPU Python 数值与历史整理；不是实测关节反馈滤波 |
| 候选树拓扑 | `flagos_tree_ops.py`、`csrc/kerv_tree_ops.cpp` | parent/节点/路径→retrieval/mask等 | 优化组合的拓扑构建为CPU C++；设备张量后续搬到MUSA |
| 并行验证 | `utils.py:tree_decoding/evaluate_posterior` | 候选与 logits→路径和接受前缀 | MUSA；接受标量回到CPU控制 |
| 补全决策及输出 | `modeling_speculation.py:_finalize_action_with_kf` | 接受前缀及历史估计→7个动作token | CPU读候选/计算尾部；构建及拼接张量在input_ids所在MUSA设备；候选或重复回退另记 |
| KV提交及下一轮 | `utils.py:update_inference_inputs` | 接受索引/KV/隐藏状态→缓存更新 | MUSA张量算子，CPU控制调度 |
| 动作解码/执行 | `predict_action`、`normalize_gripper_action/invert_gripper_action`、`env.step` | token→七维动作→仿真 | token/输出回到CPU，反归一化及仿真执行CPU |

LIBERO 七维动作通常包含平移、旋转及夹爪控制，本项目沿用 checkpoint 的 `libero_goal` 归一化统计和原有相机预处理。没有在本轮另行标定世界坐标的物理单位或真实机械臂执行周期，不把 token 差值阈值直接说成米/弧度误差。

当前补全还包括上游KERV已有的周期性路径：`(step_idx + 1) % 4 == 0` 时直接进行Kalman动作补全。因此不能把所有补偿描述为“模型拒绝后才触发”。本轮没有修改这项算法规则；直接补全、候选接受及回退分别统计。

`observation.state` 确实从实测仿真状态构造，但“构造了状态”不等于“补偿用了状态”。当前链路以历史模型动作/token估计运动；任务书若要求实际关节/末端测量反馈进入纠错补偿，需要另做接口定义和质量评测。本轮标记为口径/功能缺口，保留有效实现。

任务书第6/8/16页将推理及纠错补偿放在GPU；第15页将Compensation/Adjustment放在CPU。本机实际为上表的混合链路，MUSA是GPU后端。D08未确认前不声称完全符合任一种文字解释；也不盲目迁移CPU逻辑来修改已经验证的动作。

固定输入验证覆盖全接受、首次拒绝、中途拒绝、全拒绝、补偿保留前缀、补全到七维、回合历史清空和阈值上下界；完整闭环另检查实际补偿触发/未触发及B1/B2逐动作/token一致。工作区/缓存完整失效与真实状态闭环反馈仍需额外专项，不能用局部reset测试替代全部缓存场景。
