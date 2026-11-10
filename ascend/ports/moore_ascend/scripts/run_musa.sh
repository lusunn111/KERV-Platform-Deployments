#!/usr/bin/env bash
set -euo pipefail
musa_profile="${KERV_MUSA_PROFILE:-optimized}"
run_stamp=$(date +%Y%m%d_%H%M%S)
exec docker exec \
  -e PATH=/workspace/.venv/bin:/usr/local/musa/bin:/usr/local/bin:/usr/bin:/bin \
  -e KERV_DEVICE_BACKEND=musa \
  -e LIBERO_CONFIG_PATH=/workspace/.libero \
  -e WANDB_MODE=disabled \
  -e KERV_DEBUG_TRACE_SECONDS="${KERV_DEBUG_TRACE_SECONDS:-0}" \
  kerv-musa-runtime /workspace/.venv/bin/python /workspace/src/run_kerv.py \
    --backend musa \
    --musa-profile "$musa_profile" \
    --flagscale-root /workspace/third_party/FlagScale \
    --base-checkpoint /workspace/checkpoints/openvla-libero-goal \
    --draft-checkpoint /workspace/checkpoints/kerv-drafter \
    --libero-config /workspace/.libero \
    --output-dir "/workspace/outputs/kerv_musa_${musa_profile}_${run_stamp}" "$@"
