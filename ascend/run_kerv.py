#!/usr/bin/env python3
"""Single public launcher for KERV and its runtime optimizations."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _runtime_root(root: Path) -> Path:
    preferred = root / "KERV-RuntimeOptimization"
    legacy = root / "runtime_opt"
    if preferred.is_dir():
        return preferred
    if legacy.is_dir():
        return legacy
    raise SystemExit(
        "KERV runtime package not found. Expected "
        f"{preferred} or {legacy}."
    )


def main() -> None:
    root = Path(__file__).resolve().parent
    runtime_root = _runtime_root(root)
    parser = argparse.ArgumentParser(description="Run KERV on LIBERO through FlagScale")
    parser.add_argument(
        "--flagscale-root",
        default=os.environ.get("FLAGSCALE_ROOT", "../FlagScale"),
        help="FlagScale checkout containing run.py",
    )
    parser.add_argument(
        "--base-checkpoint",
        default=os.environ.get("KERV_BASE_CHECKPOINT", "checkpoints/openvla-libero-goal"),
    )
    parser.add_argument(
        "--draft-checkpoint",
        default=os.environ.get("KERV_DRAFT_CHECKPOINT", "checkpoints/kerv-drafter"),
    )
    parser.add_argument(
        "--libero-config",
        default=os.environ.get("LIBERO_CONFIG_PATH", "~/.libero"),
    )
    parser.add_argument("--output-dir", default="outputs/kerv_libero_goal")
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--device-type",
        choices=("auto", "cuda", "npu", "cpu"),
        default=os.environ.get("KERV_DEVICE_TYPE", "auto"),
        help="execution backend; auto prefers CUDA, then Ascend NPU",
    )
    parser.add_argument("--max-tasks", type=int, default=1)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--max-episode-steps", type=int, default=220)
    parser.add_argument("--profile", choices=("ar", "naive-sd", "specvla", "kerv-reference", "kerv-native", "kerv-ops"), default="kerv-ops")
    parser.add_argument("--override", action="append", default=[], help="FlagScale Hydra override (repeatable)")
    parser.add_argument("--benchmark", action="store_true", help="Run identical-observation latency replay through FlagScale")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compose the FlagScale launch script without loading checkpoints",
    )
    parser.add_argument(
        "--background",
        action="store_true",
        help="Run through FlagScale in the background instead of streaming logs",
    )
    args = parser.parse_args()

    flagscale_root = _path(args.flagscale_root)
    flagscale_run = flagscale_root / "run.py"
    base_checkpoint = _path(args.base_checkpoint)
    draft_checkpoint = _path(args.draft_checkpoint)
    libero_config = _path(args.libero_config)
    output_root = _path(args.output_dir)

    if not flagscale_run.is_file():
        raise SystemExit(f"FlagScale launcher not found: {flagscale_run}")
    if not args.dry_run:
        missing = [
            path
            for path in (base_checkpoint, draft_checkpoint, libero_config)
            if not path.exists()
        ]
        if missing:
            details = "\n".join(f"  - {path}" for path in missing)
            raise SystemExit(f"Required model/environment paths are missing:\n{details}")

    output_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    # FlagScale invokes its bridge with `python`; use the same interpreter
    # that launched KERV, including the isolated environment's dependencies.
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    device_type = args.device_type
    if device_type == "auto":
        device_type = "npu" if os.environ.get("FS_PLATFORM", "").lower() == "npu" else "cuda"
    env.update(
        {
            "FLAGSCALE_ROOT": str(flagscale_root),
            "KERV_RELEASE_ROOT": str(root),
            "KERV_RUNTIME_OPT_ROOT": str(runtime_root),
            "KERV_PYTHON": sys.executable,
            "KERV_BASE_CHECKPOINT": str(base_checkpoint),
            "KERV_DRAFT_CHECKPOINT": str(draft_checkpoint),
            "KERV_OUTPUT_ROOT": str(output_root),
            "LIBERO_CONFIG_PATH": str(libero_config),
            "KERV_DEVICE_TYPE": device_type,
            "PYOPENGL_PLATFORM": env.get("PYOPENGL_PLATFORM", "osmesa"),
            "MUJOCO_GL": env.get("MUJOCO_GL", "osmesa"),
        }
    )
    if device_type == "npu":
        env["ASCEND_RT_VISIBLE_DEVICES"] = args.device
        env.pop("CUDA_VISIBLE_DEVICES", None)
    elif device_type == "cuda":
        env["CUDA_VISIBLE_DEVICES"] = args.device
    env["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (
            str(runtime_root),
            str(root),
            str(root / "openvla"),
            str(root / "third_party" / "LIBERO"),
            str(flagscale_root),
            env.get("PYTHONPATH", ""),
        )
        if value
    )

    action = "dryrun" if args.dry_run else ("run" if args.background else "test")
    command = [
        sys.executable,
        str(flagscale_run),
        "--config-path",
        str(runtime_root / "FlagScale" / "configs"),
        "--config-name",
        "kerv_libero_goal",
        f"action={action}",
        f"inference.runtime.max_tasks={args.max_tasks}",
        f"inference.runtime.num_trials_per_task={args.trials}",
        f"inference.runtime.max_episode_steps={args.max_episode_steps}",
        f"inference.runtime.max_planning_steps={args.max_episode_steps}",
    ]
    command.extend([
        f"inference.runtime.use_spec={str(args.profile != 'ar').lower()}",
        f"inference.runtime.embodied_ops.enabled={str(args.profile == 'kerv-ops').lower()}",
        f"inference.runtime.tree_builder.enabled={str(args.profile in {'kerv-native', 'kerv-ops'}).lower()}",
    ])
    if args.profile in {"naive-sd", "specvla"}:
        threshold = 0 if args.profile == "naive-sd" else 9
        command.extend(["inference.runtime.use_kalman_fallback=false", "inference.runtime.use_kalman_tree=false",
                        f"+inference.runtime.accept_threshold_start={threshold}",
                        f"+inference.runtime.accept_threshold_min={threshold}"])
    command.extend(args.override)
    if args.benchmark:
        command.append(f"experiment.task.entrypoint={root / 'scripts/benchmark_kerv.py'}")
    print("[KERV]", " ".join(command), flush=True)
    subprocess.run(command, cwd=flagscale_root, env=env, check=True)


if __name__ == "__main__":
    main()
