"""Trace generated action tokens and the slice used for detokenization."""
import os
import sys
from pathlib import Path
from types import SimpleNamespace
import json
import gc

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "openvla"), str(ROOT / "runtime_opt")]
os.environ.setdefault("KERV_DEVICE_TYPE", "npu")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
import numpy as np
import torch
from experiments.robot.openvla_utils import get_vla, get_processor, get_vla_action
from experiments.robot.flagos_ops import enable_tree_builder, enable_action_head


def main():
    source = ROOT / "outputs/npu_ar_tensorflow/observations/task0_episode0_step0000.npz"
    with np.load(source) as data:
        obs, task = {"full_image": data["image"]}, str(data["task"])
    traces = []
    for restricted in (False, True):
        cfg = SimpleNamespace(pretrained_checkpoint=str(ROOT / "checkpoints/openvla-libero-goal"),
            spec_checkpoint=str(ROOT / "checkpoints/kerv-drafter"), use_spec=True, parallel_draft=False,
            accept_threshold=15, accept_threshold_start=15, load_in_8bit=False, load_in_4bit=False)
        model = get_vla(cfg).eval()
        processor = get_processor(cfg)
        enable_tree_builder(model, True, None, static_runtime=True, verify_max_paths=48)
        enable_action_head(model, restricted, 256, False, True, None, None)
        original = model.eagenerate
        trace = {"restricted_action_head": restricted}

        def traced(*args, **kwargs):
            output = original(*args, **kwargs)
            tokens = output.sequences if hasattr(output, "sequences") else output
            trace["generated_ids"] = tokens[0].tolist()
            trace["generated_count"] = tokens.shape[1]
            trace["last_seven"] = tokens[0, -7:].tolist()
            trace["history_first_seven"] = model._token_history[-1]
            return output

        model.eagenerate = traced
        model.start_rollout(max_steps=220, kalman_cfg={"enabled":True,"tree_enabled":True}, store_history=True)
        action = get_vla_action(model, processor, cfg.pretrained_checkpoint, obs, task, "libero_goal",
                    generate_mode="speculative", center_crop=True, step_idx=0, history=[], max_steps=220,
                    dynamic_threshold=15, use_kalman=True, kalman_tree_enabled=True)
        trace["action"] = action.tolist()
        traces.append(trace)
        print(json.dumps(trace), flush=True)
        del model, processor, original, traced
        gc.collect()
        torch.npu.empty_cache()
    output = ROOT / "outputs/action_token_diagnosis.json"
    output.write_text(json.dumps(traces, indent=2)+"\n")


if __name__ == "__main__":
    main()
