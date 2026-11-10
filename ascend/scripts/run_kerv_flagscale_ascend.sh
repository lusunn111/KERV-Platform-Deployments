#!/usr/bin/env bash
# Use FlagScale's bundled in-process KERV adapter, without the release bridge.
set -euo pipefail
CONTAINER="${KERV_CONTAINER:-kerv-ascend}"
CONTAINER_ROOT="${KERV_CONTAINER_ROOT:-/data/data/KERV}"
OUTPUT_ROOT="${KERV_FLAGSCALE_OUTPUT:-$CONTAINER_ROOT/outputs/flagscale_ascend}"
exec docker exec -w "$CONTAINER_ROOT/third_party/FlagScale" \
    -e KERV_SOURCE_ROOT="$CONTAINER_ROOT" \
    -e KERV_BASE_CHECKPOINT="$CONTAINER_ROOT/checkpoints/openvla-libero-goal" \
    -e KERV_DRAFT_CHECKPOINT="$CONTAINER_ROOT/checkpoints/kerv-drafter" \
    -e LIBERO_CONFIG_PATH="$CONTAINER_ROOT/configs/libero" \
    -e ASCEND_RT_VISIBLE_DEVICES="${KERV_NPU_ID:-4}" \
    -e OMP_NUM_THREADS=16 -e TF_NUM_INTRAOP_THREADS=1 -e TF_NUM_INTEROP_THREADS=1 \
    -e TRITON_ALL_BLOCKS_PARALLEL=1 -e PYTHONUNBUFFERED=1 \
    -e PYOPENGL_PLATFORM=osmesa -e MUJOCO_GL=osmesa -e HF_HUB_OFFLINE=1 \
    "$CONTAINER" bash -c \
    'export PATH="$KERV_SOURCE_ROOT/.venv/bin:$PATH"; exec python run.py --config-path examples/kerv/conf --config-name inference inference=ascend "$@"' \
    kerv-ascend "experiment.exp_dir=$OUTPUT_ROOT" "$@"
