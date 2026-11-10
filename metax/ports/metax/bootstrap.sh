#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
# Reuse this locally verified vendor image; never install stock torch over it.
image=${KERV_METAX_IMAGE:-cr.metax-tech.com/public-library/maca-pytorch@sha256:1a10a6cc56741f5e84da5dd6f0aa9dc4cfd420d66188e43983c9a3efd9d85ab0}
physical=${KERV_METAX_GPU:-0}
if ! docker container inspect kerv-metax >/dev/null 2>&1; then
  mx-smi -i "$physical" --show-process
  docker run --pull=never -d --network host --name kerv-metax --device /dev/dri --device /dev/mxcd \
    --shm-size 8g -v "$PWD:/workspace" -w /workspace \
    -e CUDA_VISIBLE_DEVICES="$physical" -e MACA_VISIBLE_DEVICES="$physical" "$image" sleep infinity
fi
docker exec kerv-metax python -m venv --system-site-packages /workspace/.venv-metax
# Each networked invocation uses the approved temporary proxy wrapper.
baai-proxy docker exec -e HTTP_PROXY -e HTTPS_PROXY -e ALL_PROXY -e NO_PROXY kerv-metax \
  /workspace/.venv-metax/bin/python -m pip install --index-url https://pypi.org/simple \
  --no-cache-dir -c /workspace/ports/metax/constraints.txt -r /workspace/ports/metax/requirements.txt
baai-proxy docker exec -e HTTP_PROXY -e HTTPS_PROXY -e ALL_PROXY -e NO_PROXY kerv-metax \
  bash -c 'apt-get update && apt-get install -y --no-install-recommends libosmesa6 libgl1 libglib2.0-0 libegl1 git'
docker exec kerv-metax /workspace/.venv-metax/bin/python ports/moore_ascend/scripts/configure_libero.py
docker exec kerv-metax /workspace/.venv-metax/bin/python ports/metax/preflight.py
