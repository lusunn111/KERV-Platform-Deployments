#!/usr/bin/env bash
set -euo pipefail
: "${KERV_PRECISION_POLICY:?Set KERV_PRECISION_POLICY to the precision-first policy inside /workspace}"
KERV_MUSA_PROFILE=native exec "$(dirname "$0")/run_musa.sh" \
  --override inference.runtime.entrypoint=/workspace/src/openvla/experiments/robot/libero/run_precision_libero.py \
  --override inference.runtime.w8a16.enabled=true \
  --override inference.runtime.w8a16.backend=musa \
  --override inference.runtime.w8a16.strict=true \
  --override 'inference.runtime.w8a16.groups="verify_mlp,verify_attention"' \
  --override "inference.runtime.w8a16.policy_path=$KERV_PRECISION_POLICY" \
  --override inference.runtime.linear_fusion.enabled=false \
  --override inference.runtime.tree_builder.enabled=true \
  --override inference.runtime.tree_builder.static_runtime=false \
  --override inference.runtime.tree_builder.kalman_batch=true "$@"
