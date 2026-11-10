"""Compare accepted single-episode MUSA runs and audit actual optimization hits."""

import argparse
from collections import Counter
import json
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("optimized", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline, optimized = [json.loads((run / "acceptance.json").read_text())
                           for run in (args.baseline, args.optimized)]
    for acceptance in (baseline, optimized):
        assert acceptance["execution_passed"] and acceptance["task_success"], "Task success gate failed"
    assert baseline["step_limit"] == optimized["step_limit"], "Planning horizons differ"
    native_actions, optimized_actions = [read_jsonl(run / "actions.jsonl")
                                        for run in (args.baseline, args.optimized)]
    assert {(x["task_id"], x["episode"]) for x in native_actions} == {
        (x["task_id"], x["episode"]) for x in optimized_actions}
    runtime = read_jsonl(args.optimized / "runtime_stats.jsonl")[-1]
    manifests = args.optimized / "manifests"
    actual_implementations = {}
    for name in ("linear_fusion_hits.jsonl", "rope_fusion_hits.jsonl"):
        rows = read_jsonl(manifests / name)
        assert rows, f"No actual operator hits: {name}"
        actual_implementations[name] = dict(Counter(row["implementation"] for row in rows))
    if (manifests / "flag_gems_hits.log").exists():
        rows = read_jsonl(manifests / "flag_gems_hits.log")
        actual_implementations["flag_gems_hits.log"] = dict(Counter(row["implementation"] for row in rows))
    flag_gems = json.loads((manifests / "flag_gems.json").read_text())
    tree = runtime["tree_builder"]
    system = runtime["system_optimization"]
    counters = {key: tree[key] for key in (
        "operator_build_hits", "draft_tree_operator_hits", "kalman_batch_hits",
        "kv_commit_batched_hits", "cache_hits", "cuda_graph_replay_hits")}
    counters.update({key: system[key] for key in (
        "decode_workspace_hits", "workspace_append_hits", "prompt_token_cache_hits",
        "persistent_draft_kv_allocations", "persistent_draft_kv_fallbacks", "control_buffer_transfers")})
    for key in ("operator_build_hits", "draft_tree_operator_hits", "kalman_batch_hits",
                "kv_commit_batched_hits", "decode_workspace_hits", "control_buffer_transfers"):
        assert counters[key] > 0, f"Configured optimization did not execute: {key}"
    assert tree["verify_max_paths"] == 0, "Verification paths were pruned"
    native_latency, optimized_latency = [a["excluding_first_action"] for a in (baseline, optimized)]
    report = {
        "baseline": {"path": str(args.baseline), **baseline},
        "optimized": {"path": str(args.optimized), **optimized},
        "mean_latency_speedup": native_latency["mean_seconds"] / optimized_latency["mean_seconds"],
        "mean_latency_reduction_percent": 100 * (1 - optimized_latency["mean_seconds"] / native_latency["mean_seconds"]),
        "p95_latency_speedup": native_latency["p95_seconds"] / optimized_latency["p95_seconds"],
        "actual_implementation_distinct_hit_records": actual_implementations,
        "flag_gems_enabled": flag_gems["enabled"],
        "actual_cumulative_runtime_counters": counters,
        "equal_actions_at_matching_step": sum(a["raw_action"] == b["raw_action"]
                                              for a, b in zip(native_actions, optimized_actions)),
        "limitations": ["One episode per profile; not suite-wide accuracy or statistically stable speedup",
                        "Different action trajectories may produce different verification workloads",
                        "Operator hit logging records distinct shapes, not every invocation",
                        "Graph and resident verifier KV cache remain disabled"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in (
        "mean_latency_speedup", "mean_latency_reduction_percent", "p95_latency_speedup",
        "actual_cumulative_runtime_counters")}, indent=2))


if __name__ == "__main__":
    main()
