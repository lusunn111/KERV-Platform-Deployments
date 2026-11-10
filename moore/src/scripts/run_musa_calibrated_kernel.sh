#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "$0")/run_musa_calibrated.sh" \
  --override inference.runtime.entrypoint=/workspace/src/openvla/experiments/robot/libero/run_calibrated_kernel_libero.py "$@"
