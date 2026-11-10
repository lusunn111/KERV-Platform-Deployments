#!/usr/bin/env bash
# Same-chip quantization diagnosis; caller must supply a fresh output directory.
set -euo pipefail
CONTAINER="${KERV_CONTAINER:-kerv-ascend}"
CONTAINER_ROOT="${KERV_CONTAINER_ROOT:-/data/data/KERV}"
exec docker exec -w "$CONTAINER_ROOT/ports/moore_ascend" \
  -e ASCEND_RT_VISIBLE_DEVICES="${KERV_NPU_ID:-14}" \
  -e KERV_DEVICE_BACKEND=npu -e KERV_ACTION_PROTOCOL=seven \
  -e KERV_NPU_DRAFT_RMS_REFERENCE=0 \
  -e OMP_NUM_THREADS=16 -e TF_NUM_INTRAOP_THREADS=1 -e TF_NUM_INTEROP_THREADS=1 \
  -e HF_HUB_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e MAX_JOBS=2 \
  -e PYTHONPATH="$CONTAINER_ROOT/ports/moore_ascend/runtime_opt:$CONTAINER_ROOT/ports/moore_ascend:$CONTAINER_ROOT/ports/moore_ascend/openvla" \
  "$CONTAINER" bash -c \
  'export PATH="/data/data/KERV/.venv/bin:$PATH"; exec python /data/data/KERV/acceptance/diagnose_quantization.py "$@"' \
  quantization-diagnosis --input-dir "$CONTAINER_ROOT/outputs/npu_ar_tensorflow/observations" "$@"
