# 最新执行状态（2026-10-19）

只负责当前昇腾；按TestOutline完成本轮分阶段自测。最新总结：
[ascend_20261019T0730Z](../outputs/acceptance/ascend_20261019T0730Z/summary.md)。

- 新配对B1/B2均115动作成功，全部观测、动作、token精确一致。
- AB/BA/AB三轮157观测：每配置471测量动作精确一致。模型2.102×、pipeline2.080×，相对KERV native。
- 不同芯片分离正式低干扰回放与重型诊断；每芯片串行。
- 已补采逐验证、分层时间、100ms框架显存、10秒vendor快照和独立CANN轨迹。
- 未达到本轮B1→B2省2GB目标；没有声明2.4×、Spec-VLA总加速或正式验收完成。
- 原B0对照、M2/其他平台、DMA全覆盖/方向关联、D01-D08正式确认仍缺。
- 当前反馈为输出token历史，不等于实测运动反馈；详见本轮T03报告。
- 失败与修复全部保留，代码/环境ready；实时资源用npu-smi重新检查。

---

以下保留上一轮交接状态：

# 当前状态（2026-10-15）

## 推荐版本已跑通

- 推荐入口：`scripts/run_kerv_moore_ascend.sh`，代码隔离于`ports/moore_ascend/`。
- 基于用户提供的摩尔归档迁移，通过现有FlagScale launcher和KERV adapter运行。
- NPU执行C++树、批量Kalman/KV提交、持久drafter KV、解码workspace、pinned控制缓冲。
- BF16、原TensorFlow预处理、原F.rms_norm、共享embedding、稳定同分排序、全部验证路径。
- 默认seven动作协议（两组共同修正root+6/KF偏移/SD边界）；原moore协议保留为诊断开关。
- graph、packed linear、RoPE、全局FlagGems RMSNorm关闭。

## 已完成验证

- 配对LIBERO Goal task0/init0/seed7：原生和优化均115次动作成功，动作/token115/115精确一致。
- 每次恰好7 token，KF历史和实际解码一致；闭环排除首动作约1.923×。
- 正式157连续观测，全序列预热1遍、测2遍：314/314优化动作与参考精确一致，最大差0。
- 模型514.554→261.260ms（1.970×），pipeline520.354→266.628ms（1.952×）。
- 运行时实际命中：C++树1413、批量Kalman1416、KV提交1299、解码workspace471、控制缓冲1416；
  drafter KV分配1次、无fallback。计数包含预热。
- CPU C++树15项检查通过；seven协议3例（接受0/2/6）通过；两个新入口bash语法检查通过。
- 仅一个配对任务验证新版闭环；没有宣称新版完整LIBERO准确率或达到2.4×。
- 新版正式比值基线为KERV native，不是Naive SD；不能复用旧版Naive SD时间。

## 证据与复现

- `README_ASCEND.md`：复现命令、过程和范围。
- `outputs/moore_seven_checks/paired_episode.json`：闭环配对。
- `outputs/moore_seven_native_baseline/`、`outputs/moore_seven_native_rms/`：原生/优化动作、运行统计与视频。
  后者目录名是历史命名，实际使用原F.rms_norm，未启用手写替代。
- `outputs/moore_seven_runtime_full/benchmark.json`：正式完整回放。
- `scripts/benchmark_kerv_moore_ascend.sh`：重现同观测测速。
- `docs/MOORE_ASCEND_COMPARISON.md`：实现差异与归档来源。

## 实验与历史

- 原moore动作协议：两组220步均失败但动作完全一致；其2.011×速度仅作诊断。
- NPU原F.rms_norm对32768个随机BF16值与CPU精确一致，无证据归因于其数值错误。
- 新RoPE片上gather内核单算子精确一致，32头S279约0.236ms；16帧整模型32动作一致，
  但未完成完整回放及额外收益验证，默认关闭。直接迁移MUSA kernel曾明显拖慢模型。
- 根目录旧图版本6/10、原相对Naive SD 0.975×属于不同配置，保留历史，不当作新版结果。

## 环境

工作目录`/root/KERV`；运行副本`/public-flash/zhiyuan/liyuzhuo/data/KERV`；
容器`kerv-ascend`，容器目录`/data/data/KERV`，Python为`.venv/bin/python`。
Python3.12、torch2.7.1+cpu、torch_npu2.7.1.post8、TF2.18、FlagGems5.3.1、Triton3.5 NPU。
模型和LIBERO离线就绪。代码修改需同步到运行副本；芯片实时状态用`npu-smi info`确认。

测试收尾：本次模型进程均已正常退出，物理chip4/5释放，容器仍running，环境保留ready。
这是收尾时快照，当前实时占用需再查。见`outputs/npu_smi_moore_after.txt`；
汇总验收为`outputs/moore_ascend_validation.json`。
