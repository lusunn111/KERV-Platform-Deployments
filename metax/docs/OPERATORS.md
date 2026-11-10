# KERV Runtime Operators and Integration

## Operator surface

KERV contributes 19 interfaces under `torch.ops.flagos_embodied`.

| Interface | Role | Release status |
|---|---|---|
| `kerv_rms_norm` | standalone verifier RMSNorm; Ascend model hook | native default; NPU fusion experimental |
| `kerv_silu_mul` | SiLU and elementwise multiply fusion | mainline |
| `kerv_add_rms_norm` | residual addition and RMSNorm fusion | mainline |
| `kerv_verify_accept_control` | token match, accepted length and path selection | mainline |
| `kerv_static_tree_pack` | static-tree template and candidate packing | mainline |
| `kerv_static_tree_attention` | prefix/ancestor-only tree attention | mainline |
| `kerv_action_projection_select` | action-vocabulary projection and selection | mainline |
| `kerv_value_cache_store` | resident V-cache write | available, disabled by default |
| `kerv_kv_commit` | accepted-path K/V commit | mainline |
| `kerv_tree_embed_pack` | token gather, embedding and tree-buffer packing | mainline |
| `kerv_rope_kv_store` | RoPE and resident K/V write | mainline |
| `kerv_action_verify_accept` | action projection and Verify-Accept fusion | mainline |
| `kerv_draft_action_topk` | drafter action projection and Top-K | mainline |
| `kerv_kv_accept_commit` | accepted-path gather and commit | mainline |
| `kerv_vision_add_layer_norm` | visual residual and LayerNorm fusion | mainline |
| `kerv_vision_bias_gelu` | visual bias and GELU fusion | mainline |
| `kerv_o_proj_residual_rms_norm` | attention epilogue fusion | experimental |
| `kerv_down_proj_residual_rms_norm` | MLP epilogue fusion | experimental |
| `kerv_logical_kv_commit` | logical-address-only K/V commit | experimental |

On CUDA, `backend=auto` keeps native fallbacks and selects specialized
implementations for supported batch-one BF16/FP16 layouts. On Ascend the
validated default is `backend=native`. Experimental NPU `auto` RMSNorm uses
CANN and passes the microbenchmark's tensor tolerance, but changed generated
action tokens in replay tests; it is excluded from the default deployment.

## Ascend execution

The FlagScale bridge selects native operators, eager attention, CPU wall-clock
timing with NPU synchronization, and disables CUDA-specific graph, event,
kernel and persistent-runtime paths. The installed FlagGems/NPU Triton backend
is available through explicit experimental adapters. Optional NPUGraph replay
uses the shared stage-cache interface and passes the 314-action exact check;
the restricted action head is disabled after two full-sequence differences.
Verifier RMSNorm and verifier/drafter SwiGLU
call the public operator namespace through NPU model hooks. The drafter's
RMSNorm remains its original implementation to preserve BF16 rounding.
The NPU acceptance control uses int32 prefix/count arithmetic and converts its
public count back to int64. Tree packing and acceptance operators are also
called by the speculative generation path.

Registration, microbenchmark correctness, actual model invocation, and speedup
are separate checks. The deployment manifest lists enabled interfaces, while
the operator hit log records which interfaces actually ran. The measured
results and reproduction commands are in [README_ASCEND.md](../README_ASCEND.md).

## Transformer fusion

Two model-level fusion routes are enabled by measured row-count allowlists:

- **QKV fusion:** Q, K and V share the same input, so their weights are packed
  and the three projections are executed as one GEMM.
- **Gate-Up-SwiGLU fusion:** Gate and Up share the same input; one packed GEMM
  produces both branches, followed by an in-place SiLU/multiply epilogue.

The allowlists in
`KERV-RuntimeOptimization/FlagScale/configs/kerv_libero_goal.yaml` are part of the A100
reference profile. Other dimensions fall back to the native CUDA path instead
of assuming that a Triton kernel is always faster.

## System execution

The final runtime also contains non-kernel optimizations:

- fixed-width, operatorized verification-tree templates;
- persistent prefix/tree workspaces and resident K/V cache;
- verifier, prompt and drafter CUDA Graph replay;
- shared graph memory pool and process-level warm-up;
- persistent decode/control buffers and fused CPU-GPU control transfer;
- pinned image input and asynchronous H2D copy;
- two-stream commit/next-draft overlap;
- cached prompt tokenization and asynchronous runtime logging.

Quantized W8A16, compact trees and the three experimental epilogues are not
enabled by the BF16 safe configuration.
