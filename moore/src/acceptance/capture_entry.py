"""Explicit observation-capture entry; leaves the normal evaluator untouched."""
from pathlib import Path
import runpy
import numpy as np
from acceptance.collect_metrics import Collector

original_begin = Collector.begin_step
original_finish = Collector.finish_episode
keys = ("agentview_image", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")


def begin(self, task_id, episode, step_id, observation):
    if not hasattr(self, "capture_frames"):
        self.capture_frames = {key: [] for key in keys}
    for key in keys:
        self.capture_frames[key].append(np.array(observation[key], copy=True))
    return original_begin(self, task_id, episode, step_id, observation)


def finish(self, done, limit):
    original_finish(self, done, limit)
    if hasattr(self, "capture_frames"):
        # Outside policy/episode timing; preserve lossless incoming observations.
        np.savez_compressed(self.root / "observations.npz",
                            **{key: np.stack(values) for key, values in self.capture_frames.items()})


if __name__ == "__main__":
    Collector.begin_step = begin
    Collector.finish_episode = finish
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "openvla/experiments/robot/libero/run_kerv_libero.py"), run_name="__main__")
