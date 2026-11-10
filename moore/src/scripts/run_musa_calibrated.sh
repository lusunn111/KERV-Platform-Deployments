#!/usr/bin/env bash
set -euo pipefail
: "${KERV_CALIBRATED_POLICY:?Set KERV_CALIBRATED_POLICY to the frozen policy.json path inside /workspace}"
KERV_MUSA_PROFILE=native exec "$(dirname "$0")/run_musa.sh" \
  --override inference.runtime.entrypoint=/workspace/src/openvla/experiments/robot/libero/run_calibrated_libero.py \
  --override inference.runtime.w8a16.enabled=true \
  --override inference.runtime.w8a16.backend=musa \
  --override inference.runtime.w8a16.strict=true \
  --override inference.runtime.w8a16.groups=verify_mlp \
  --override inference.runtime.w8a16.group_size=32 \
  --override "inference.runtime.w8a16.policy_path=$KERV_CALIBRATED_POLICY" \
  --override inference.runtime.linear_fusion.enabled=false "$@"
