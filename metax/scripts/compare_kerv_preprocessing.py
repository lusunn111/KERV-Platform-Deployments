#!/usr/bin/env python3
"""Compare the former PIL approximation against original TensorFlow inputs."""
import os
from pathlib import Path
import sys
import json
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "openvla"), str(ROOT / "third_party/LIBERO")]
os.environ["KERV_DEVICE_TYPE"] = "cpu"
os.environ["KERV_ALLOW_APPROX_PREPROCESS"] = "1"
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import numpy as np
from PIL import Image
import tensorflow as tf
from libero.libero import benchmark
from experiments.robot.libero import libero_utils as lu
from experiments.robot.openvla_utils import crop_and_resize, get_processor


def main():
    output = ROOT / "outputs/preprocessing_comparison"
    output.mkdir(parents=True, exist_ok=True)
    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    env, task = lu.get_libero_env(suite.get_task(0), "openvla", 256)
    try:
        env.reset()
        obs = env.set_init_state(suite.get_task_init_states(0)[0])
        for _ in range(10):
            obs, _, _, _ = env.step(lu.get_libero_dummy_action("openvla"))
        exact = lu.get_libero_image(obs, 224)
        lu.tf = None
        approx = lu.get_libero_image(obs, 224)
    finally:
        lu.tf = tf
        env.close()
    image = tf.image.convert_image_dtype(exact, tf.float32)
    image = crop_and_resize(image, 0.9, 1)
    image = tf.image.convert_image_dtype(tf.clip_by_value(image, 0, 1), tf.uint8, saturate=True)
    exact_crop = image.numpy()
    width = int(round(224 * 0.9**0.5))
    left = (224 - width) // 2
    approx_crop = np.asarray(Image.fromarray(approx).crop((left, left, left+width, left+width)).resize((224, 224), Image.Resampling.BILINEAR))
    processor = get_processor(SimpleNamespace(pretrained_checkpoint=str(ROOT / "checkpoints/openvla-libero-goal")))
    x = processor.image_processor(Image.fromarray(exact_crop), return_tensors="pt")["pixel_values"].numpy()
    y = processor.image_processor(Image.fromarray(approx_crop), return_tensors="pt")["pixel_values"].numpy()
    report = {"tensorflow": tf.__version__, "task": task,
              "resize_pixel_mae": float(np.abs(exact.astype(float)-approx.astype(float)).mean()),
              "crop_pixel_mae": float(np.abs(exact_crop.astype(float)-approx_crop.astype(float)).mean()),
              "crop_pixels_different_fraction": float(np.mean(exact_crop != approx_crop)),
              "processor_max_abs_difference": float(np.abs(x-y).max()),
              "processor_mean_abs_difference": float(np.abs(x-y).mean())}
    for name, data in (("tensorflow", exact_crop), ("pil", approx_crop)):
        Image.fromarray(data).save(output / f"{name}.png")
    np.savez_compressed(output / "inputs.npz", exact_image=exact, approximate_image=approx,
                        exact_pixels=x, approximate_pixels=y, task=task)
    (output / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
