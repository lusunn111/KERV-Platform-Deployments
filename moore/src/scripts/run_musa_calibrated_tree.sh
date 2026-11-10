#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "$0")/run_musa_calibrated.sh" \
  --override inference.runtime.entrypoint=/workspace/src/openvla/experiments/robot/libero/run_calibrated_tree_libero.py \
  --override inference.runtime.tree_builder.enabled=true \
  --override inference.runtime.tree_builder.static_runtime=false \
  --override inference.runtime.tree_builder.kalman_batch=false "$@"
