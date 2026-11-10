"""Validate the action/runtime logs and write a compact acceptance report."""

import argparse
import json
import math
import statistics
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--max-steps", type=int, required=True)
    args = parser.parse_args()
    actions = read_jsonl(args.run / "actions.jsonl")
    runtime = read_jsonl(args.run / "runtime_stats.jsonl")
    assert actions and len(actions) == len(runtime), "Missing actions or runtime records"
    for step, (action, stats) in enumerate(zip(actions, runtime)):
        assert action["step"] == step == stats["step_index"], "Non-contiguous action steps"
        # A speculative cycle can accept an extra token before the decoder
        # selects the final seven action tokens.
        assert stats["generated_token_count"] >= 7
        assert len(stats["action_token_ids"]) == 7
        for key in ("raw_action", "executed_action"):
            assert len(action[key]) == 7 and all(math.isfinite(x) for x in action[key])
        assert math.isfinite(action["latency_seconds"]) and action["latency_seconds"] > 0
    success = actions[-1]["done"]
    assert success or len(actions) == args.max_steps, "Episode ended before its limit without success"
    latencies = [row["latency_seconds"] for row in actions]
    steady = latencies[1:] or latencies
    ordered = sorted(steady)
    videos = list(args.run.glob("rollouts/**/*.mp4"))
    assert videos, "Missing episode replay"
    report = {
        "execution_passed": True,
        "task_success": success,
        "termination": "task_success" if success else "step_limit",
        "steps": len(actions),
        "step_limit": args.max_steps,
        "actions_finite": True,
        "runtime_records": len(runtime),
        "first_action_seconds": latencies[0],
        "all_actions_mean_seconds": statistics.mean(latencies),
        "excluding_first_action": {
            "count": len(steady),
            "mean_seconds": statistics.mean(steady),
            "median_seconds": statistics.median(steady),
            "p95_seconds": ordered[math.ceil(0.95 * len(ordered)) - 1],
        },
        "kalman_completion_steps": sum(bool(row["forced_kalman"]) for row in runtime),
        "verification_cycles": sum(row["cycle_count"] for row in runtime),
        "video": str(videos[0].relative_to(args.run)),
        "timing_note": "Measured immediately before predict_action through device synchronization on return. Excludes prior CPU image/processor preparation, model loading, environment stepping and video saving; no device pre-synchronization at the start. First action includes inference-time lazy initialization/warmup. Compare only runs with matching task, initial state and step limit.",
    }
    target = args.run / "acceptance.json"
    target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
