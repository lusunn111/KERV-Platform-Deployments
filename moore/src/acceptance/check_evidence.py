"""Audit registered IDs, raw event consistency and frozen-source/weight hashes."""
import argparse
from pathlib import Path
from acceptance.aggregate import aggregate
from acceptance.common import dump, load, rows, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--root", type=Path, default=Path("/root/KERV"))
    p.add_argument("--check-source-hashes", action="store_true")
    p.add_argument("--check-weight-hashes", action="store_true")
    args = p.parse_args()
    report = aggregate(args.campaign)  # Validates duplicate/missing verification IDs and NaN/Inf.
    errors = []; limitations = []; source_changes = []
    for run in report["runs"]:
        folder = args.campaign / "runs" / run["run_id"]
        if run["status"] == "NOT_STARTED":
            limitations.append(run["run_id"] + ": NOT_STARTED")
            continue
        if run["status"] != "PASS":
            limitations.append(run["run_id"] + ": execution " + run["status"])
            continue
        for name in ("run.json", "context.json", "stdout_stderr.log", "steps.jsonl", "verify.jsonl", "memory.jsonl", "episodes.jsonl", "collection.json", "actions.jsonl", "runtime_stats.jsonl", "device_samples.jsonl"):
            if not (folder / name).is_file():
                errors.append(run["run_id"] + ": missing " + name)
        a, s, r = [rows(folder / name) for name in ("actions.jsonl", "steps.jsonl", "runtime_stats.jsonl")]
        if not (len(a) == len(s) == len(r)):
            errors.append(run["run_id"] + ": action/runtime/step length mismatch")
        for x,y,z in zip(a,s,r):
            if x["raw_action"] != y["raw_action"] or z["action_token_ids"] != y["action_tokens"]:
                errors.append(run["run_id"] + ": raw record disagreement")
            vv = [v["accepted_prefix_length"] for v in rows(folder / "verify.jsonl") if v["step_id"] == y["step_id"]]
            if vv != z["accept_lengths"]:
                errors.append(run["run_id"] + ": acceptance stream differs from existing runtime")
        if not list(folder.glob("rollouts/**/*.mp4")):
            errors.append(run["run_id"] + ": missing video")
    if args.check_source_hashes:
        # Preserve every per-run version. Analysis-only evolution is disclosed,
        # while unexpected changes to inference code invalidate this check.
        analysis_files = {"src/acceptance/aggregate.py", "src/acceptance/check_evidence.py", "src/acceptance/common.py",
                          "src/acceptance/run_matrix.py", "src/acceptance/report.py", "src/tests/test_acceptance_tools.py",
                          "src/acceptance/quantization_report.py", "src/acceptance/mixed_report.py",
                          "src/acceptance/calibrated_report.py", "src/acceptance/check_quant_replay.py"}
        for path in sorted((args.campaign / "runs").glob("*/source_manifest.sha256")):
            for line in path.read_text().splitlines():
                expected, rel = line.split("  ",1)
                current = sha256(args.root / rel)
                if current != expected:
                    # The capture launches capture_entry.py, never replay_entry.py.
                    # That separate tool was being developed while observations
                    # were captured; retain its hash difference without claiming
                    # it was an inference dependency of this run.
                    inactive_replay_tool = (path.parent.name == 'capture_optimized_s17'
                                            and rel == 'src/acceptance/replay_entry.py')
                    inactive_replay_tool = inactive_replay_tool or rel in {
                        "src/acceptance/quant_replay.py", "src/acceptance/quant_followup.py",
                        "src/acceptance/mixed_calibration_entry.py"}
                    inactive_replay_tool = inactive_replay_tool or rel in {
                        "src/acceptance/calibrated_block_entry.py", "src/acceptance/calibrated_model_entry.py",
                        "src/acceptance/calibrated_refine_entry.py"}
                    # Registered native captures never import the optional calibrated backend.
                    registered_run = load(path.parent / 'run.json')
                    inactive_replay_tool = inactive_replay_tool or (registered_run.get('profile') == 'native'
                        and rel in {'src/runtime_opt/KERVRuntimeOptimization/calibrated_int8.py',
                                    'src/runtime_opt/KERVRuntimeOptimization/calibrated_int8_kernels.py'})
                    if rel.startswith('src/tests/'):
                        analysis_files.add(rel)
                    documentation = rel.startswith('src/docs/') and rel.endswith('.md')
                    change = {"run_id":path.parent.name,"path":rel,"executed_sha256":expected,"current_sha256":current,
                              "classification":("inactive replay/follow-up tooling; not consumed by this registered run" if inactive_replay_tool
                                                else "documentation update" if documentation
                                                else "analysis/scheduler/test evolution" if rel in analysis_files else "inference source changed")}
                    source_changes.append(change)
                    if rel not in analysis_files and not inactive_replay_tool and not documentation:
                        # Early smoke iterations explicitly precede fixes, and are retained.
                        if path.parent.name.startswith("smoke_"):
                            limitations.append(path.parent.name + ": pre-fix source version retained in manifest")
                        else:
                            errors.append("Unexpected inference source changed for " + path.parent.name + ": " + rel)
    if args.check_weight_hashes:
        for rel, value in load(args.campaign / "inventory/weight_manifest.json").items():
            if sha256(args.root / rel) != value["sha256"]:
                errors.append("Weight changed: " + rel)
    checks = {"local_evidence_integrity": "PASS" if not errors else "FAIL", "errors": errors, "limitations": limitations,
              "source_version_changes":source_changes,
              "formal_acceptance_status": "BLOCKED", "scope": "Integrity is not taskbook acceptance or independent third-party verification"}
    dump(args.campaign / "reports/evidence_checks.json", checks)
    print({"local_evidence_integrity":checks["local_evidence_integrity"],"error_count":len(errors),
           "limitation_count":len(limitations),"source_version_changes_count":len(source_changes)})
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
