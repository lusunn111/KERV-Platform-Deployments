#!/usr/bin/env bash
set -euo pipefail
KERV_MUSA_PROFILE=quantized exec "$(dirname "$0")/run_musa.sh" \
  --override inference.runtime.entrypoint=/workspace/src/openvla/experiments/robot/libero/run_lossless_libero.py \
  --override 'inference.runtime.w8a16.groups="verify_mlp,verify_attention"' \
  --override 'inference.runtime.w8a16.policy_path=""' \
  --override inference.runtime.w8a16.group_size=128 "$@"
