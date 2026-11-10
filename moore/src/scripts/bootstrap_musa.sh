#!/usr/bin/env bash
set -euo pipefail
workspace_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
docker build --build-arg HTTP_PROXY --build-arg HTTPS_PROXY \
  -f "$workspace_root/src/docker/Dockerfile.musa" -t kerv-musa:local "$workspace_root/src/docker"
if ! docker container inspect kerv-musa-runtime >/dev/null 2>&1; then
  docker run -d --name kerv-musa-runtime --runtime=mthreads --shm-size=8g \
    --env MTHREADS_VISIBLE_DEVICES=0 --env MTHREADS_DRIVER_CAPABILITIES=all \
    --env HTTP_PROXY --env HTTPS_PROXY --env NO_PROXY \
    --mount "type=bind,source=$workspace_root,target=/workspace" kerv-musa:local
else
  docker start kerv-musa-runtime >/dev/null
fi
docker exec kerv-musa-runtime python -m virtualenv --system-site-packages /workspace/.venv
docker exec kerv-musa-runtime /workspace/.venv/bin/python \
  /workspace/src/scripts/apply_dependency_fixes.py
docker exec kerv-musa-runtime /workspace/.venv/bin/python -m pip install \
  --index-url https://pypi.org/simple cmake==3.31.6
docker exec -e PATH=/workspace/.venv/bin:/usr/local/musa/bin:/usr/local/bin:/usr/bin:/bin \
  kerv-musa-runtime /workspace/.venv/bin/python -m pip install \
  --index-url https://pypi.org/simple --no-build-isolation -r /workspace/src/requirements-musa.txt
docker exec kerv-musa-runtime /workspace/.venv/bin/python -m pip install \
  --index-url https://pypi.org/simple --no-deps -e /workspace/src/openvla \
  -e /workspace/third_party/FlagScale -e /workspace/third_party/LIBERO
docker exec -e PATH=/workspace/.venv/bin:/usr/local/musa/bin:/usr/local/bin:/usr/bin:/bin \
  kerv-musa-runtime /workspace/.venv/bin/python -m pip install \
  --index-url https://pypi.org/simple --no-deps -e /workspace/third_party/FlagGems-v3
docker exec kerv-musa-runtime /workspace/.venv/bin/python /workspace/src/scripts/configure_libero.py
