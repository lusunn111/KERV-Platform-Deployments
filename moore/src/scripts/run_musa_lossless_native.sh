#!/usr/bin/env bash
set -euo pipefail
KERV_MUSA_PROFILE=native exec "$(dirname "$0")/run_musa.sh" \
  --override inference.runtime.entrypoint=/workspace/src/openvla/experiments/robot/libero/run_lossless_libero.py \
  --override inference.runtime.w8a16.enabled=true \
  --override inference.runtime.w8a16.backend=musa \
  --override inference.runtime.w8a16.strict=true \
  --override 'inference.runtime.w8a16.groups="verify_mlp,verify_attention"' \
  --override 'inference.runtime.w8a16.policy_path=""' \
  --override inference.runtime.w8a16.group_size=128 \
  --override inference.runtime.linear_fusion.enabled=false "$@"
