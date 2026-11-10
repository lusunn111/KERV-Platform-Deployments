#!/usr/bin/env bash
# Explicit opt-in: existing mixed-precision MLP policy plus verifier-only QKV.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export KERV_NPU_QUANT="${KERV_NPU_QUANT:-mlp_dynamic_int8}"
export KERV_NPU_QUANT_POLICY="${KERV_NPU_QUANT_POLICY:-${KERV_CONTAINER_ROOT:-/data/data/KERV}/outputs/acceptance/weight_only_hybrid4_20261027/policy.pt}"
if [[ "${KERV_QKV_BACKEND:-concat}" == bmm* ]]; then
    QKV_FUSION_ROWS="$(seq -s, 257 320)"
else
    QKV_FUSION_ROWS="$(seq -s, 1 512)"
fi
exec "$PROJECT_ROOT/scripts/run_kerv_moore_ascend.sh" \
    --override inference.runtime.linear_fusion.enabled=true \
    --override "inference.runtime.linear_fusion.qkv_backend=${KERV_QKV_BACKEND:-concat}" \
    --override "inference.runtime.linear_fusion.qkv_layers=\"${KERV_QKV_LAYERS:-all}\"" \
    --override "inference.runtime.linear_fusion.qkv_rows=\"${QKV_FUSION_ROWS}\"" \
    --override 'inference.runtime.linear_fusion.qkv_input_sizes="4096"' \
    --override inference.runtime.linear_fusion.gate_up_rows=null \
    --override inference.runtime.linear_fusion.swiglu_rows=null \
    --override inference.runtime.linear_fusion.add_rms_norm_rows=null \
    "$@"
