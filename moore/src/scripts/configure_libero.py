"""Create an explicit, noninteractive LIBERO configuration for this workspace."""

from pathlib import Path

import yaml

root = Path("/workspace/third_party/LIBERO/libero/libero")
datasets = Path("/workspace/data/libero")
datasets.mkdir(parents=True, exist_ok=True)
config = Path("/workspace/.libero")
config.mkdir(exist_ok=True)
(config / "config.yaml").write_text(yaml.safe_dump({
    "benchmark_root": str(root), "bddl_files": str(root / "bddl_files"),
    "init_states": str(root / "init_files"), "assets": str(root / "assets"),
    "datasets": str(datasets),
}))
