"""Exercise MUSA and optional LIBERO rendering without downloading model weights."""

import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import torch

from openvla.experiments.robot.device_runtime import get_device, synchronize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulation", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("/workspace/outputs/preflight_musa.json"))
    args = parser.parse_args()
    device = get_device()
    if device.type != "musa":
        raise SystemExit(f"Expected MUSA, got {device}")
    x = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    for dtype in (torch.float32, torch.bfloat16):
        actual = x.to(device=device, dtype=dtype) @ x.to(device=device, dtype=dtype)
        torch.testing.assert_close(actual.float().cpu(), x @ x, rtol=0.02, atol=1)
    synchronize(device)
    # Import the real inference entrypoint, including local HF model classes.
    from openvla.experiments.robot.libero.run_kerv_libero import GenerateConfig

    report = {"device": str(device), "name": torch.musa.get_device_name(0),
              "visible_devices": torch.musa.device_count(), "inference_import": GenerateConfig.__name__,
              "versions": {name: importlib.metadata.version(name)
                           for name in ("torch", "torch_musa", "torchvision", "transformers", "timm", "numpy")}}
    if args.simulation:
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        suite = benchmark.get_benchmark_dict()["libero_goal"]()
        task = suite.get_task(0)
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
        try:
            env.seed(7)
            obs = env.reset()
            obs = env.set_init_state(suite.get_task_init_states(0)[0])
            for _ in range(2):
                obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
            frame = obs["agentview_image"]
            assert frame.shape == (256, 256, 3) and np.isfinite(frame).all()
            report["simulation"] = {"task": task.name, "image_shape": list(frame.shape), "steps": 2}
        finally:
            env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
