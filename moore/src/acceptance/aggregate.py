"""Recompute local results, retaining failed cases and all strict boundaries."""
import argparse
import csv
import math
from pathlib import Path
from acceptance.common import distribution, dump, load, rows, committed_candidate_prefix


def aggregate(campaign):
    campaign = Path(campaign)
    cases = []
    for path in sorted((campaign / "inventory").glob("cases*.csv")):
        with path.open() as stream:
            cases.extend(csv.DictReader(stream))
    registered_ids = [x["run_id"] for x in cases]
    if len(registered_ids) != len(set(registered_ids)):
        raise ValueError("Duplicate registered run IDs")
    result = []
    for case in cases:
        folder = campaign / "runs" / case["run_id"]
        run = load(folder / "run.json") if (folder / "run.json").exists() else {}
        entry = {**case, "status": run.get("status", "NOT_STARTED"), "task_success": run.get("task_success"),
                 "formal_threshold_status": "BLOCKED", "formal_reason": "H1 local self-test; B0/A100/coverage/decisions not complete"}
        steps = rows(folder / "steps.jsonl"); verify = rows(folder / "verify.jsonl"); memory = rows(folder / "memory.jsonl")
        episodes = rows(folder / "episodes.jsonl")
        if steps:
            ids = [(x["episode_id"], x["step_id"]) for x in steps]
            if len(ids) != len(set(ids)):
                raise ValueError("Duplicate steps in " + case["run_id"])
            if [x["step_id"] for x in steps] != list(range(len(steps))):
                raise ValueError("Missing/non-contiguous step IDs in " + case["run_id"])
            for key in ("model_seconds", "pipeline_seconds", "env_seconds", "wall_seconds", "cpu_seconds"):
                entry[key] = distribution(x[key] for x in steps)
            entry["model_seconds_excluding_first"] = distribution(x["model_seconds"] for x in steps[1:])
            entry["first_model_seconds"] = steps[0]["model_seconds"]
            entry["steps"] = len(steps)
            entry["observation_hashes"] = [x["observation_hash"] for x in steps]
            for x in steps:
                for key in ("raw_action", "executed_action"):
                    if len(x[key]) != 7 or not all(math.isfinite(v) for v in x[key]):
                        raise ValueError("Invalid action " + case["run_id"])
        if sum(x.get("verify_events", 0) for x in steps) != len(verify):
            raise ValueError("Missing verification events in " + case["run_id"])
        if verify:
            ids = [(x["episode_id"], x["step_id"], x["verify_id"]) for x in verify]
            if len(ids) != len(set(ids)):
                raise ValueError("Duplicate verification IDs")
            for step in steps:
                events = [x for x in verify if x["step_id"] == step["step_id"]]
                if len(events) != step["verify_events"] or [x["verify_id"] for x in events] != list(range(len(events))):
                    raise ValueError("Missing verification event")
            entry["accepted_prefix_length"] = distribution(x["accepted_prefix_length"] for x in verify)
            entry["committed_candidate_prefix_length"] = distribution(committed_candidate_prefix(x) for x in verify)
            entry["acceptance_definition_note"] = "posterior acceptance and action-boundary-clamped committed candidate prefix are both reported; formal D04 unsigned"
            entry["verification_events"] = len(verify)
            entry["zero_acceptance_events"] = sum(x["accepted_prefix_length"] == 0 for x in verify)
            entry["compensation_events"] = sum(x["compensation_triggered"] for x in verify)
            entry["completion_tail_tokens"] = sum(x["compensation_length"] for x in verify)
        if episodes:
            entry["closed_loop_wall_seconds"] = episodes[0]["closed_loop_wall_seconds"]
        for key in ("allocated_bytes", "reserved_bytes", "peak_allocated_bytes", "peak_reserved_bytes", "cpu_rss_bytes"):
            values = [x[key] for x in memory if x.get(key) is not None]
            entry["maximum_" + key] = max(values) if values else None
        entry["after_load_memory"] = next((x for x in memory if x["phase"] == "after_model_load"), None)
        entry["first_action_memory"] = next((x for x in memory if x["phase"] == "first_action_end"), None)
        steady = [x for x in memory if x["phase"] == "action_end"]
        entry["steady_peak_allocated_bytes"] = max((x["peak_allocated_bytes"] for x in steady if x.get("peak_allocated_bytes") is not None), default=None)
        entry["steady_peak_reserved_bytes"] = max((x["peak_reserved_bytes"] for x in steady if x.get("peak_reserved_bytes") is not None), default=None)
        transfers = [x for x in rows(folder / "transfers.jsonl") if not x.get("observer_generated")]
        if transfers:
            entry["diagnostic_logical_payload_bytes_by_direction"] = {d: sum(x["payload_upper_of_logical_bytes"] for x in transfers if x["direction"] == d) for d in ("H2D", "D2H", "D2D")}
            entry["diagnostic_copy_sync_wall_seconds"] = sum(x["copy_sync_wall_seconds"] for x in transfers)
            entry["transfer_threshold_status"] = "BLOCKED: incomplete physical-DMA/custom-kernel coverage; diagnostic synchronizes copies"
        samples = rows(folder / "device_samples.jsonl")
        used = []
        intervals = []
        previous = None
        for sample in samples:
            import json
            import re
            try:
                devices = json.loads(sample["raw"])["GPU"]
                value = next(x for x in devices if x["Index"] == case["physical_device_id"])["FB Memory Usage"]["Used"]
                match = re.fullmatch(r"([0-9]+)MiB", value)
                if match:
                    used.append(int(match.group(1)) * 1024**2)
            except (KeyError, ValueError, StopIteration):
                pass
            if previous is not None:
                intervals.append(sample["elapsed_seconds"]-previous)
            previous = sample["elapsed_seconds"]
        entry["vendor_sampled_device_peak_bytes"] = max(used) if used else None
        entry["vendor_actual_sample_interval_seconds"] = distribution(intervals)
        entry["vendor_sampling_note"] = "whole device sampled usage, not framework/process peak; initial resource check idle; not every instantaneous peak guaranteed"
        result.append(entry)
    pairs = []
    full = [x for x in result if x["kind"] == "full"]
    for base in [x for x in full if x["profile"] == "native"]:
        matched = [x for x in full if x["profile"] == "optimized" and all(x[k] == base[k] for k in ("model_id", "platform_id", "chip_id", "task_id", "init_state_id", "seed", "repeat_id", "physical_device_id"))]
        if len(matched) != 1:
            continue
        opt = matched[0]
        pair = {"native": base["run_id"], "optimized": opt["run_id"], "status": "INCOMPLETE"}
        if base["status"] == opt["status"] == "PASS":
            left, right = [rows(campaign / "runs" / x["run_id"] / "steps.jsonl") for x in (base, opt)]
            pair["same_step_count"] = len(left) == len(right)
            for key in ("raw_action", "executed_action", "action_tokens", "observation_hash"):
                pair[key + "_equal_records"] = sum(a[key] == b[key] for a,b in zip(left,right))
            pair["all_actions_and_tokens_identical"] = len(left) == len(right) and all(a[k] == b[k] for a,b in zip(left,right) for k in ("raw_action", "executed_action", "action_tokens"))
            pair["same_success"] = base["task_success"] == opt["task_success"]
            pair["status"] = "PASS" if pair["all_actions_and_tokens_identical"] and pair["same_success"] else "FAIL"
            for key in ("model_seconds", "model_seconds_excluding_first", "pipeline_seconds"):
                pair[key + "_mean_speedup"] = base[key]["mean"] / opt[key]["mean"]
            pair["closed_loop_wall_speedup"] = base["closed_loop_wall_seconds"] / opt["closed_loop_wall_seconds"]
            pair["initialization_peak_allocated_saving_bytes"] = base["maximum_peak_allocated_bytes"] - opt["maximum_peak_allocated_bytes"] if base["maximum_peak_allocated_bytes"] is not None and opt["maximum_peak_allocated_bytes"] is not None else None
            pair["accepted_prefix_mean_delta_B2_minus_B1"] = opt["accepted_prefix_length"]["mean"] - base["accepted_prefix_length"]["mean"]
            pair["committed_candidate_prefix_mean_delta_B2_minus_B1"] = opt["committed_candidate_prefix_length"]["mean"] - base["committed_candidate_prefix_length"]["mean"]
            pair["not_B0_comparison"] = True
        pairs.append(pair)
    groups = []
    for profile in ("specvla", "native", "optimized", "optimized_unfused", "quantized", "quantized_g32", "mixed", "lossless_bf16_12", "lossless_bf16_12_native", "calibrated"):
        valid = [x for x in full if x["profile"] == profile]
        executed = [x for x in valid if x["status"] in ("PASS", "FAIL")]
        groups.append({"profile": profile, "registered_full_cases": len(valid), "executed_cases_including_errors": len(executed),
                       "task_successes": sum(x["task_success"] is True for x in executed),
                       "execution_failures": sum(x["status"] == "FAIL" for x in executed),
                       "success_rate_over_executed_cases": sum(x["task_success"] is True for x in executed) / len(executed) if executed else None,
                       "not_started_or_blocked": len(valid)-len(executed), "scope": "self-test registered subset only, not official suite SR"})
    return {"campaign_id": campaign.name, "runs": result, "pairs": pairs, "groups": groups, "formal_acceptance_status": "BLOCKED"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign", type=Path, required=True)
    args = p.parse_args()
    report = aggregate(args.campaign)
    dump(args.campaign / "reports/local_results.json", report)
    columns = ["run_id", "kind", "profile", "task_id", "init_state_id", "seed", "status", "task_success", "steps", "closed_loop_wall_seconds", "maximum_peak_allocated_bytes", "maximum_peak_reserved_bytes"]
    with (args.campaign / "reports/local_results.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader(); writer.writerows(report["runs"])
    print(args.campaign / "reports/local_results.json")


if __name__ == "__main__":
    main()
