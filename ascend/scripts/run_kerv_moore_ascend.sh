#!/usr/bin/env bash
# Run the isolated Moore-algorithm NPU port through FlagScale.
set -euo pipefail
CONTAINER="${KERV_CONTAINER:-kerv-ascend}"
CONTAINER_ROOT="${KERV_CONTAINER_ROOT:-/data/data/KERV}"
exec docker exec -w "$CONTAINER_ROOT" \
    -e ASCEND_RT_VISIBLE_DEVICES="${KERV_NPU_ID:-4}" \
    -e KERV_DEVICE_BACKEND=npu \
    -e KERV_MOORE_PROFILE="${KERV_MOORE_PROFILE:-optimized}" \
    -e KERV_ACTION_PROTOCOL="${KERV_ACTION_PROTOCOL:-seven}" \
    -e KERV_NPU_DRAFT_RMS_REFERENCE="${KERV_NPU_DRAFT_RMS_REFERENCE:-0}" \
    -e KERV_NPU_QUANT="${KERV_NPU_QUANT:-none}" \
    -e KERV_NPU_QUANT_GROUP_SIZE="${KERV_NPU_QUANT_GROUP_SIZE:-128}" \
    -e KERV_NPU_QUANT_LAYERS="${KERV_NPU_QUANT_LAYERS:-all}" \
    -e KERV_NPU_QUANT_POLICY="${KERV_NPU_QUANT_POLICY:-}" \
    -e KERV_QUANT_CALIBRATION_DIR="${KERV_QUANT_CALIBRATION_DIR:-}" \
    -e KERV_LIBERO_INITIAL_STATE_OFFSET="${KERV_LIBERO_INITIAL_STATE_OFFSET:-0}" \
    -e KERV_ACCEPTANCE_DIR="${KERV_ACCEPTANCE_DIR:-}" \
    -e KERV_ACCEPTANCE_RUN_ID="${KERV_ACCEPTANCE_RUN_ID:-}" \
    -e KERV_ACCEPTANCE_TRANSFERS="${KERV_ACCEPTANCE_TRANSFERS:-0}" \
    -e KERV_LIBERO_ROOT="$CONTAINER_ROOT/third_party/LIBERO" \
    -e OMP_NUM_THREADS=16 -e TF_NUM_INTRAOP_THREADS=1 -e TF_NUM_INTEROP_THREADS=1 \
    -e PYOPENGL_PLATFORM=osmesa -e MUJOCO_GL=osmesa \
    -e PYTHONUNBUFFERED=1 -e HF_HUB_OFFLINE=1 -e MAX_JOBS=2 \
    "$CONTAINER" bash -c \
    'export PATH="$PWD/.venv/bin:$PATH"; exec python ports/moore_ascend/run_kerv.py --backend npu --device "$ASCEND_RT_VISIBLE_DEVICES" --runtime-profile "$KERV_MOORE_PROFILE" --flagscale-root third_party/FlagScale --base-checkpoint checkpoints/openvla-libero-goal --draft-checkpoint checkpoints/kerv-drafter --libero-config configs/libero "$@"' \
    kerv-moore-npu "$@"
