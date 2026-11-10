#!/usr/bin/env bash
set -euo pipefail
CONTAINER="${KERV_CONTAINER:-kerv-ascend}"
IMAGE="${KERV_IMAGE:-tk-qwen35-env:20260920}"
SHARED_ROOT="${KERV_SHARED_ROOT:-/public-flash/zhiyuan/liyuzhuo/data/KERV}"
CONTAINER_ROOT="${KERV_CONTAINER_ROOT:-/data/data/KERV}"
if docker container inspect "$CONTAINER" >/dev/null 2>&1; then
    docker start "$CONTAINER"
    exit 0
fi
test -d "$SHARED_ROOT/checkpoints"
# This host's Ascend driver requires the same host PID/IPC setup as the
# existing working Ascend containers. ASCEND_RT_VISIBLE_DEVICES selects NPU 4.
docker run -d --name "$CONTAINER" --privileged --network host --ipc host --pid host \
    -e ASCEND_RT_VISIBLE_DEVICES="${KERV_NPU_ID:-4}" \
    -v "$SHARED_ROOT:$CONTAINER_ROOT" \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro \
    -v /usr/local/Ascend/add-ons:/usr/local/Ascend/add-ons:ro \
    -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
    -v /usr/local/dcmi:/usr/local/dcmi:ro \
    -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi:ro \
    -w "$CONTAINER_ROOT" "$IMAGE" sleep infinity
