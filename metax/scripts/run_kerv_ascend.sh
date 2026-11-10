#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASCEND_ENV="${ASCEND_TOOLKIT_HOME:-${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-8.5.0}}/set_env.sh"
if [[ -f "${ASCEND_ENV}" ]]; then
    # shellcheck disable=SC1090
    source "${ASCEND_ENV}"
fi

PYTHON_BIN="${KERV_PYTHON:-python}"
export KERV_DEVICE_TYPE=npu
if [[ $# -gt 0 && "$1" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    export ASCEND_RT_VISIBLE_DEVICES="$1"
    shift
else
    export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
fi
unset CUDA_VISIBLE_DEVICES
"${PYTHON_BIN}" - <<'PY'
import torch

try:
    import torch_npu  # noqa: F401
except ImportError as exc:
    raise SystemExit("torch_npu is required; run this script in an Ascend-enabled environment") from exc

if not torch.npu.is_available():
    raise SystemExit("No Ascend NPU is available")
PY

cd "${ROOT_DIR}"
exec "${PYTHON_BIN}" run_kerv.py --device-type npu --device "${ASCEND_RT_VISIBLE_DEVICES}" "$@"
