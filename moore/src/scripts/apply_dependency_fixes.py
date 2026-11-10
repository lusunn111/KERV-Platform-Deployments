"""Apply the narrowly scoped compatibility fixes to the pinned dependencies."""

import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    root = parser.parse_args().workspace
    libero = root / "third_party/LIBERO/libero"
    init = libero / "__init__.py"
    if not init.exists():
        init.write_text('"""LIBERO package root for editable inference installs."""\n')
    benchmark = libero / "libero/benchmark/__init__.py"
    original = "torch.load(init_states_path)"
    fixed = 'torch.load(init_states_path, map_location="cpu", weights_only=False)'
    content = benchmark.read_text()
    if original in content:
        benchmark.write_text(content.replace(original, fixed))
    elif fixed not in content:
        raise RuntimeError("Unrecognised LIBERO init-state loader; review the source revision")

    for checkout, patch_name in (
        ("FlagScale", "flagscale-inference-exit-status.patch"),
        ("FlagGems-v3", "flaggems-v3-musa.patch"),
    ):
        repository = root / "third_party" / checkout
        patch = root / "src/patches" / patch_name
        already_applied = subprocess.run(
            ["git", "apply", "--reverse", "--check", str(patch)],
            cwd=repository, capture_output=True,
        ).returncode == 0
        if not already_applied:
            subprocess.run(["git", "apply", "--check", str(patch)], cwd=repository, check=True)
            subprocess.run(["git", "apply", str(patch)], cwd=repository, check=True)
    print("LIBERO, FlagScale and FlagGems compatibility fixes are ready")


if __name__ == "__main__":
    main()
