#!/usr/bin/env bash
set -euo pipefail
# Physical selection is frozen when the isolated container is created.
exec flock /tmp/kerv-metax-card0.lock docker exec -w /workspace -e PYTHONUNBUFFERED=1 kerv-metax \
  /workspace/.venv-metax/bin/python ports/metax/run.py "$@"
