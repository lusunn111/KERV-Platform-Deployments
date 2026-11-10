#!/usr/bin/env bash
set -euo pipefail
exec flock /tmp/kerv-metax-card0.lock docker exec -w /workspace \
  -e PYTHONPATH=/workspace/ports/moore_ascend/runtime_opt:/workspace/ports/moore_ascend:/workspace/ports/moore_ascend/openvla:/workspace \
  -e KERV_DEVICE_BACKEND=metax -e KERV_ACTION_PROTOCOL=seven \
  -e OMP_NUM_THREADS=16 -e TF_NUM_INTRAOP_THREADS=1 -e TF_NUM_INTEROP_THREADS=1 \
  -e PYTHONUNBUFFERED=1 -e HF_HUB_OFFLINE=1 kerv-metax \
  /workspace/.venv-metax/bin/python ports/metax/device_profile.py "$@"
