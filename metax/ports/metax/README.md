# MetaX C550 实测适配


2026-11-05 数值修复后的推荐入口为 `numerical_fixed`；完整原生RMS回退为 `numerical_native_rms`。旧 `optimized_v2` 仍保留作数值有差异的实验配置，不能因为6回合成功数相同就当作轨迹一致。

```bash
scripts/run_kerv_metax.sh --profile numerical_fixed --max-tasks 3 --trials 2 \
  --output-dir outputs/numerical_fixed_your_run
scripts/run_kerv_metax.sh --profile numerical_native_rms --max-tasks 3 --trials 2 \
  --output-dir outputs/native_rms_your_run
```

修复保留原生FP32平方/归约，尾融合维持原生BF16乘法类型。65层/28种真实形状/4745次采样输出位差0，两个独立进程的完整六回合均5/6、696步，与第一轮观测/token/raw/executed/done及验证事件一致。此为已测范围结论，不保证所有可能输入普遍位精确。完整220步四组同输入AB/BA：native353.472ms、第一轮219.127ms、旧v2实验170.829ms、修复181.584ms；修复对native1.946605×，对第一轮1.206752×，不为保住2×改变算法或基线。

根因、符号零/LLVM证据、原始数据索引、每回合去首步均值/中位/P95、完整冷启动/闭环及显存见 [数值修复报告](../../docs/METAX_NUMERICAL_FIX.md)。离线复核：`python3 -m ports.metax.validate_numerical`；从原始结果重建报告：`python3 -m ports.metax.render_numerical_report`。

源码保留原迁入实现，以 `ports/moore_ascend` 的 seven 动作修正版为基础，共用算法；`ports/metax` 提供独立环境和 FlagScale 配置。实际模型执行引擎为厂商 PyTorch/MACA，张量设备名为 `cuda:0`；FlagScale 的 `backend: vllm` 是本地 runner 注册名，并非使用 vLLM 模型引擎。

已核验镜像 `cr.metax-tech.com/public-library/maca-pytorch:3.8.1.2-torch2.10-py312-ubuntu24.04-amd64`，本机ID `1a10a6cc5674`。厂商Torch2.10与原Transformers4.40.1搭配；Python3.12使用TF CPU2.16.2，仍执行原JPEG/Lanczos和中心裁剪，不替换为其他图像算法。

```bash
# 先mx-smi确认选卡无人占用；容器创建时固定物理卡，默认0。
KERV_METAX_GPU=0 ports/metax/bootstrap.sh
baai-proxy docker exec -e HTTP_PROXY -e HTTPS_PROXY -e ALL_PROXY -e NO_PROXY \
  -e HF_HUB_DISABLE_XET=1 kerv-metax /workspace/.venv-metax/bin/python \
  ports/moore_ascend/scripts/prepare_checkpoints.py --root /workspace/checkpoints
scripts/run_kerv_metax.sh --profile native --max-episode-steps 8 \
  --planning-horizon 220 --output-dir outputs/metax_native_smoke_new
scripts/run_kerv_metax.sh --profile native --max-tasks 1 --trials 1 \
  --initial-state 0 --output-dir outputs/metax_native_pair_new
scripts/run_kerv_metax.sh --profile optimized --max-tasks 1 --trials 1 \
  --initial-state 0 --output-dir outputs/metax_optimized_pair_new
```

输出目录必须不存在，避免覆盖原始数据。默认Goal、seed7、初态0、220动作上限及220规划时域；冒烟可只限制动作上限。沐曦入口 → `third_party/FlagScale/run.py` → worker生成脚本 → `ports/moore_ascend/runtime_opt/FlagScale/inference_kerv.py` → 真模型LIBERO闭环。每次保存解析后配置、实际命令、退出码、动作、验证事件、耗时、显存及任务结果。

第一轮 `optimized.yaml` 已通过6回合5/6，696步与native逐字段一致。第二轮 `optimized_v2.yaml` 已完成6回合5/6，但归一化FP32归约差异会改变动作：722步，不能称与第一轮轨迹一致。B0必须来自真实上游SpecVLA；上游原始代码的动作边界与修正seven协议有差异，不能把KERV开关关闭配置改名为B0。最新实测和未完成项见 `docs/METAX_PROGRESS.md`。

量化使用独立配置，当前为W8A16/BF16混合，不能视作无损BF16：

```bash
scripts/run_kerv_metax.sh --profile w8a16 --max-tasks 3 --trials 2 \
  --output-dir outputs/metax_quantized_new
python3 ports/metax/audit_events.py outputs/metax_quantized_new \
  --output outputs/metax_quantized_audit_new.json
python3 ports/metax/evaluate.py --native outputs/metax_native_goal3_init01_02 \
  --optimized outputs/metax_optimized_goal3_init01_02 \
  --quantized outputs/metax_quantized_new --output outputs/metax_results_new.json
```

主接受长度使用实际提交的连续候选前缀，包含零；原始posterior前缀独立保留。事件里的`root_length`表示每棵树的root；后续树root来自前一轮verifier bonus。汇总将初始动作root和后续bonus分别列出，均排除在接受长度之外，且与补偿一起验证每个动作恰好7维。

回放汇总工具：`ports/metax/summarize_replays.py`，接收含`benchmark.json`的AB/BA目录并重新计算。`ports/metax/device_profile.py`为重型诊断，只采真实观测下的标准推测步和周期KF步，不进入测速结果。后续源码快照包含代码及许可证，不含权重或凭据。

B0参考已固定到上游提交 `56ef2985e25c7903d2759254efb754327fb975f7`，协议反例见 `ports/metax/audit_specvla_boundary.py`。上游>6判停会形成8token后尾切丢root，边界定义独立待决，B0未运行；不能把现有KERV native当作SpecVLA，也不以该待决项阻塞B1/B2。


无损BF16存储方案使用独立入口，数学权重与原BF16逐位相同，但还原有时间成本：

```bash
scripts/run_kerv_metax.sh --profile bf16_packed --max-tasks 3 --trials 2   --output-dir outputs/metax_bf16_lossless_new
```

它对96个MLP和第7..14层的8个attention输出投影使用low8/high字节字典4bit与完整例外表，按需还原至一个共享BF16工作区再执行厂商GEMM。checkpoint文件不改写，不能把该配置的显存结果与普通optimized配置的速度合并。W8A16的6回合质量试验已降为4/6，不推荐；无损配置完整质量结果以最新技术报告为准。


推理附加依赖锁定在`constraints.txt`，厂商栈镜像的不可变digest记录于`vendor-image.json`；bootstrap不隐式拉取镜像或安装普通CUDA Torch。容器实际MACA Version.txt为3.8.1.3，Torch构建标记为metax3.8.1.0，两者分列保留。

```bash
scripts/benchmark_kerv_metax.sh --comparison native_runtime --samples 8 --repeats 3 \
  --profile-order AB --input-dir outputs/metax_optimized_smoke_01/observations --output outputs/replay_new_AB
# 再以新目录、--profile-order BA反转顺序。
# 压缩对照将--comparison设为runtime_w8a16或runtime_lossless。
scripts/profile_kerv_metax.sh outputs/device_diagnostic_new
python3 ports/metax/summarize_replays.py outputs/replay_new_AB outputs/replay_new_BA \
  --output outputs/replay_new_summary.json
python3 ports/metax/render_report.py
python3 ports/metax/validate_delivery.py
```

上述回放/诊断入口使用相同单卡互斥锁；device trace为独立重型诊断，不用于速度结果。最后两个脚本针对本次交付的原始目录重建报告和证据索引。更换原始目录时先使用通用summarize/compare/evaluate脚本。


## 第二轮入口与实测

第一轮源码 `bf179f6` 保存在tag `metax-round1-bf179f6`，原始outputs不覆盖。第二轮数据独立位于 `outputs/metax_v2_20261104/`，技术与性能说明分别为 [技术说明](../../docs/METAX_V2_TECHNICAL_REPORT.md)、[性能改进表](../../docs/METAX_V2_PERFORMANCE.md)。

```bash
scripts/run_kerv_metax.sh --profile optimized_v2 --max-tasks 3 --trials 2 \
  --output-dir outputs/v2_your_run
scripts/run_kerv_metax.sh --profile bf16_packed_v2 --max-tasks 3 --trials 2 \
  --output-dir outputs/v2_lossless_your_run
# 逐项消融，空字符串可回退第一轮runtime；均不是B0
scripts/run_kerv_metax.sh --profile optimized_v2 --v2-features rope,vision_graph \
  --output-dir outputs/v2_ablation_your_run
```

v2默认仅启用 `rms,rope,vision_graph`，QKV/Gate-Up/compact/prefill图等筛选未获足够收益，不默认启用。RMS保留BF16归一化/乘权重舍入边界，但FP32并行归约及逆平方根实现会造成数值变化，必须单独评估质量。rsqrt变体6回合3/6，已淘汰。220步完整冻结回放约2.05倍于同场native，8步短回放约1.94倍，两者分列。

`bf16_packed_v2` 使用94个MLP矩阵的原lossless codec与厂商BF16 GEMM，attention及layer0 gate/up原生；相对同一v2 BF16完整722步轨迹精确，实省2,026,985,984B，恢复仍明显变慢。它与第一轮104矩阵 `bf16_packed` 独立，不能混合两者的速度和显存。

```bash
# 完整真实历史回放，换新输出目录，反序改BA
scripts/benchmark_kerv_metax.sh --comparison native_runtime_v2 \
  --v2-features rms,rope,vision_graph --frozen-history --allow-action-change \
  --samples 220 --repeats 1 --profile-order AB \
  --input-dir outputs/metax_v2_20261104/representative_capture/observations \
  --output outputs/v2_replay_your_AB
# 压缩三组对照：同一v2 BF16、原104矩阵、选择94矩阵
scripts/benchmark_kerv_metax.sh --comparison v2_compression \
  --compression-codec bf16_lossless_selective --v2-features rms,rope,vision_graph \
  --frozen-history --samples 220 --repeats 1 --profile-order AB \
  --input-dir outputs/metax_v2_20261104/representative_capture/observations \
  --output outputs/v2_compression_your_AB
python3 -m ports.metax.render_v2_report
python3 -m ports.metax.validate_v2
```

冻结历史会在计时前恢复同一参考raw_action及七token前缀，不把候选动作反馈到下步KF；真实闭环使用自身动作反馈。正式预热与测量分开，重型profiler独立。多模型对照在两组之间释放进程内BLAS工作区缓存，每组仍从加载前统计完整峰值，并保存pre_load/post_release字节；第一轮report/validate脚本只适用于原轮数据，第二轮使用上述独立入口。
