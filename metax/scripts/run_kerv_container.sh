#!/usr/bin/env bash
set -euo pipefail
CONTAINER="${KERV_CONTAINER:-kerv-ascend}"
CONTAINER_ROOT="${KERV_CONTAINER_ROOT:-/data/data/KERV}"
if [[ "${1:-}" == "--check-ready" ]]; then
    shift
    exec docker exec -w "$CONTAINER_ROOT" -e OMP_NUM_THREADS=16 \
        -e TF_NUM_INTRAOP_THREADS=1 -e TF_NUM_INTEROP_THREADS=1 "$CONTAINER" \
        .venv/bin/python scripts/check_kerv_ready.py "$@"
fi
exec docker exec -w "$CONTAINER_ROOT" -e KERV_PYTHON=.venv/bin/python -e OMP_NUM_THREADS=16 \
    -e TF_NUM_INTRAOP_THREADS=1 -e TF_NUM_INTEROP_THREADS=1 \
    -e TRITON_ALL_BLOCKS_PARALLEL=1 \
    -e HF_HUB_OFFLINE=1 -e PYTHONUNBUFFERED=1 \
    -e LIBERO_CONFIG_PATH="$CONTAINER_ROOT/configs/libero" "$CONTAINER" \
    bash scripts/run_kerv_ascend.sh "${KERV_NPU_ID:-4}" \
    --flagscale-root third_party/FlagScale --libero-config configs/libero "$@"
