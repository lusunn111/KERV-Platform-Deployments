#!/usr/bin/env python3
"""Check local KERV resources on CPU, without starting accelerator inference."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "openvla"), str(ROOT / "runtime_opt"), str(ROOT / "third_party/LIBERO")]
os.environ["KERV_DEVICE_TYPE"] = "cpu"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
os.environ.setdefault("MUJOCO_GL", "osmesa")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load-model", action="store_true", help="Load real verifier and drafter on CPU; do not infer")
    parser.add_argument("--check-env", action="store_true", help="Reset and render a LIBERO task using software rendering")
    args = parser.parse_args()
    import yaml
    from types import SimpleNamespace

    config_dir = ROOT / "configs/libero"
    config_dir.mkdir(parents=True, exist_ok=True)
    libero_root = ROOT / "third_party/LIBERO/libero/libero"
    paths = {"benchmark_root": str(libero_root), "bddl_files": str(libero_root / "bddl_files"),
             "init_states": str(libero_root / "init_files"), "assets": str(libero_root / "assets"),
             "datasets": str(libero_root.parent / "datasets")}
    Path(paths["datasets"]).mkdir(parents=True, exist_ok=True)
    (config_dir / "config.yaml").write_text(yaml.safe_dump(paths))
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)

    versions = {name: importlib.metadata.version(name) for name in
                ["torch", "torch-npu", "transformers", "tokenizers", "timm", "numpy", "draccus",
                 "accelerate", "robosuite", "mujoco", "bddl", "imageio", "imageio-ffmpeg",
                 "tensorflow", "flag-gems", "triton"]}
    base = ROOT / "checkpoints/openvla-libero-goal"
    draft = ROOT / "checkpoints/kerv-drafter"
    index = json.loads((base / "model.safetensors.index.json").read_text())
    keys = set()
    total_bytes = 0
    for name in sorted(set(index["weight_map"].values())):
        file = base / name
        with file.open("rb") as stream:
            length = struct.unpack("<Q", stream.read(8))[0]
            if length > 10_000_000:
                raise ValueError(f"Invalid safetensors header: {file}")
            header = json.loads(stream.read(length))
        entries = {k: v for k, v in header.items() if k != "__metadata__"}
        expected = 8 + length + max(v["data_offsets"][1] for v in entries.values())
        if file.stat().st_size != expected:
            raise ValueError(f"Incomplete checkpoint: {file}")
        keys.update(entries)
        total_bytes += file.stat().st_size
    if keys != set(index["weight_map"]):
        raise ValueError("Verifier index and tensor keys do not match")
    if not (draft / "config.json").is_file() or not (draft / "pytorch_model.bin").is_file():
        raise FileNotFoundError("Drafter config or weights missing")

    from experiments.robot.openvla_utils import get_processor, get_vla
    from transformers import AutoProcessor
    from openvla.prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from openvla.prismatic.extern.hf.processing_prismatic import PrismaticProcessor
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    cfg = SimpleNamespace(pretrained_checkpoint=str(base), spec_checkpoint=str(draft),
                          use_spec=True, parallel_draft=False, accept_threshold=9,
                          load_in_8bit=False, load_in_4bit=False)
    processor = get_processor(cfg)
    report = {"versions": versions, "verifier_bytes": total_bytes, "verifier_tensors": len(keys),
              "processor": type(processor).__name__, "model_loaded_on_cpu": False,
              "simulation_checked": False, "accelerator_inference": "not run"}
    if args.load_model:
        model = get_vla(cfg)
        report["model_loaded_on_cpu"] = True
        report["model_class"] = type(model).__name__
        del model
    if args.check_env:
        from libero.libero import benchmark
        from experiments.robot.libero.libero_utils import get_libero_env, get_libero_image
        suite = benchmark.get_benchmark_dict()["libero_goal"]()
        states = suite.get_task_init_states(0)
        env, description = get_libero_env(suite.get_task(0), "openvla", 224)
        try:
            env.reset()
            obs = env.set_init_state(states[0])
            image = get_libero_image(obs, 224)
            if image.shape != (224, 224, 3):
                raise ValueError(f"Unexpected rendered image: {image.shape}")
            report.update(simulation_checked=True, task=description, image_shape=list(image.shape))
        finally:
            env.close()
    output = ROOT / "outputs/readiness.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
