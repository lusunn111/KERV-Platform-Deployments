"""Execute pre-registered local MUSA cases serially, preserving every outcome."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time
from acceptance.common import dump, load, sha256


def utc():
    return datetime.now(timezone.utc).isoformat()


def make_command(root, case, context_path, container):
    inside = "/workspace/" + str(context_path.relative_to(root))
    out = "/workspace/" + str(context_path.parent.relative_to(root))
    if case["profile"] == "specvla":
        return ["docker", "exec", "-e", "PYTHONPATH=/workspace/ports/specvla_musa:/workspace/ports/specvla_musa/openvla:/workspace/src",
                "-e", "KERV_DEVICE_BACKEND=musa", "-e", "LIBERO_CONFIG_PATH=/workspace/.libero", "-e", "WANDB_MODE=disabled",
                "-e", "KERV_ACCEPTANCE_CONTEXT=" + inside, container,
                "timeout", "--signal=TERM", "--kill-after=30s", str(int(case["timeout_seconds"])) + "s",
                "/workspace/.venv/bin/python", "/workspace/src/acceptance/run_specvla_case.py",
                "--output", out, "--max-steps", case["max_steps"], "--task-id", case["task_id"],
                "--init-state-id", case["init_state_id"], "--seed", case["seed"]]
    if case["profile"] not in ("native", "optimized", "optimized_unfused", "quantized", "quantized_g32", "mixed", "lossless_bf16_12", "lossless_bf16_12_native", "calibrated", "precision"):
        raise ValueError("B0 requires a separately verified upstream adapter, not relabeled KERV")
    command = ["docker", "exec", "-e", "PATH=/workspace/.venv/bin:/usr/local/musa/bin:/usr/local/bin:/usr/bin:/bin",
               "-e", "KERV_DEVICE_BACKEND=musa", "-e", "LIBERO_CONFIG_PATH=/workspace/.libero", "-e", "WANDB_MODE=disabled",
               "-e", "KERV_ACCEPTANCE_CONTEXT=" + inside, container,
               "timeout", "--signal=TERM", "--kill-after=30s", str(int(case["timeout_seconds"])) + "s",
               "/workspace/.venv/bin/python", "/workspace/src/run_kerv.py", "--backend", "musa",
               "--musa-profile", {"precision": "native", "calibrated": "native", "lossless_bf16_12_native": "native", "lossless_bf16_12": "quantized", "quantized_g32": "quantized", "optimized_unfused": "optimized"}.get(case["profile"], case["profile"]), "--flagscale-root", "/workspace/third_party/FlagScale",
               "--base-checkpoint", "/workspace/checkpoints/openvla-libero-goal",
               "--draft-checkpoint", "/workspace/checkpoints/kerv-drafter", "--libero-config", "/workspace/.libero",
               "--max-tasks", "1", "--trials", "1", "--max-episode-steps", case["max_steps"], "--output-dir", out,
               "--override", "++inference.runtime.acceptance_task_id=" + case["task_id"],
               "--override", "++inference.runtime.acceptance_init_state_id=" + case["init_state_id"],
               "--override", "++inference.runtime.seed=" + case["seed"],
               "--override", "inference.runtime.max_planning_steps=" + case["planning_steps"]]
    if case["profile"] == "quantized_g32":
        command.extend(["--override", "inference.runtime.w8a16.group_size=32"])
    if case["profile"] == "optimized_unfused":
        command.extend(["--override", "inference.runtime.linear_fusion.enabled=false"])
    if case["profile"] == "mixed":
        policy = Path(case['policy_path'])
        policy_inside = '/workspace/' + str(policy.relative_to(root) if policy.is_absolute() else policy)
        command.extend(['--override', 'inference.runtime.w8a16.policy_path=' + policy_inside])
        group_size = case.get('policy_group_size') or load(policy if policy.is_absolute() else root/policy)['group_size']
        command.extend(['--override', 'inference.runtime.w8a16.group_size=' + str(group_size)])
    if case["profile"] in ("lossless_bf16_12", "lossless_bf16_12_native"):
        command.extend(["--override", "inference.runtime.entrypoint=/workspace/src/openvla/experiments/robot/libero/run_lossless_libero.py",
                        "--override", 'inference.runtime.w8a16.groups="verify_mlp,verify_attention"',
                        "--override", 'inference.runtime.w8a16.policy_path=""',
                        "--override", "inference.runtime.w8a16.group_size=128",
                        "--override", "inference.runtime.w8a16.enabled=true",
                        "--override", "inference.runtime.w8a16.backend=musa",
                        "--override", "inference.runtime.w8a16.strict=true",
                        "--override", "inference.runtime.linear_fusion.enabled=false"])
    if case["profile"] == "calibrated":
        kernel = case.get('kernel_backend') or 'reference'
        if kernel not in ('reference','rows','prefetch','tree_rows','tree_prefetch','safe_rows','shared_history'): raise ValueError('Unknown calibrated kernel backend')
        entry = ('run_calibrated_shared_libero.py' if kernel == 'shared_history' else
                 'run_calibrated_safe_libero.py' if kernel == 'safe_rows' else
                 'run_calibrated_prefetch_libero.py' if kernel in ('prefetch','tree_prefetch') else
                 'run_calibrated_tree_libero.py' if kernel == 'tree_rows' else
                 'run_calibrated_kernel_libero.py' if kernel == 'rows' else 'run_calibrated_libero.py')
        policy = Path(case["policy_path"])
        inside_policy = "/workspace/" + str(policy.relative_to(root) if policy.is_absolute() else policy)
        command.extend(["--override", "inference.runtime.entrypoint=/workspace/src/openvla/experiments/robot/libero/"+entry,
                        "--override", "inference.runtime.w8a16.enabled=true", "--override", "inference.runtime.w8a16.backend=musa",
                        "--override", "inference.runtime.w8a16.groups=verify_mlp", "--override", "inference.runtime.w8a16.strict=true",
                        "--override", "inference.runtime.w8a16.group_size=32", "--override", "inference.runtime.w8a16.policy_path="+inside_policy,
                        "--override", "inference.runtime.linear_fusion.enabled=false"])
        if kernel in ('tree_rows','tree_prefetch','safe_rows','shared_history'):
            command.extend(['--override','inference.runtime.tree_builder.enabled=true',
                            '--override','inference.runtime.tree_builder.static_runtime=false',
                            '--override','inference.runtime.tree_builder.kalman_batch='+('true' if kernel=='shared_history' else 'false')])
    if case['profile']=='precision' or case.get('precision_reference')=='true':
        entry='run_precision_libero.py' if case['profile']=='precision' else 'run_precision_reference_libero.py'
        command.extend(['--override','inference.runtime.entrypoint=/workspace/src/openvla/experiments/robot/libero/'+entry,
                        '--override','inference.runtime.linear_fusion.enabled=false',
                        '--override','inference.runtime.tree_builder.enabled=true',
                        '--override','inference.runtime.tree_builder.static_runtime=false',
                        '--override','inference.runtime.tree_builder.kalman_batch=true'])
        if case['profile']=='precision':
            policy=Path(case['policy_path'])
            inside_policy='/workspace/'+str(policy.relative_to(root) if policy.is_absolute() else policy)
            command.extend(['--override','inference.runtime.w8a16.enabled=true',
                            '--override','inference.runtime.w8a16.backend=musa',
                            '--override','inference.runtime.w8a16.strict=true',
                            '--override','inference.runtime.w8a16.groups="verify_mlp,verify_attention"',
                            '--override','inference.runtime.w8a16.policy_path='+inside_policy])
    if case.get("kind") == "capture":
        command.extend(["--override", "inference.runtime.entrypoint=/workspace/src/acceptance/capture_entry.py"])
    return command


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("/root/KERV"))
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--cases", type=Path, required=True)
    p.add_argument("--case-id", action="append", help="Run only these pre-registered IDs")
    p.add_argument("--container", default="kerv-musa-runtime")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true", help="Skip existing result IDs; never append/retry them")
    args = p.parse_args()
    root = args.root.resolve(); campaign = args.campaign.resolve()
    with args.cases.open() as stream:
        cases = list(csv.DictReader(stream))
    ids = [c["run_id"] for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate registered run IDs")
    frozen = campaign / "inventory" / args.cases.name if args.cases.name.startswith("cases_addendum_") else campaign / "inventory/cases.csv"
    if not frozen.is_file():
        raise ValueError("Freeze the registered cases inside campaign/inventory before execution")
    if frozen.exists() and sha256(frozen) != sha256(args.cases):
        raise ValueError("Cases differ from the pre-registered frozen manifest")
    if args.case_id and not set(args.case_id) <= set(ids):
        raise ValueError("Unknown requested case ID")
    failures = 0
    for case in cases:
        if args.case_id and case["run_id"] not in args.case_id:
            continue
        if case["chip_id"] != "H1" or case["physical_device_id"] != "0":
            raise ValueError("This runner is frozen for the existing physical GPU 0 container")
        folder = campaign / "runs" / case["run_id"]
        if folder.exists():
            if args.resume and (folder / "run.json").exists() and load(folder / "run.json").get("end_timestamp"):
                print("SKIP completed ID", case["run_id"], flush=True)
                continue
            raise FileExistsError("Refusing to append to existing run " + str(folder))
        context_path = folder / "context.json"
        command = make_command(root, case, context_path, args.container)
        if args.dry_run:
            print(json.dumps({"run_id": case["run_id"], "command": command}, ensure_ascii=False))
            continue
        # Resource check before every run, preserving raw process/device output.
        probes = campaign / "resource_checks"
        probes.mkdir(exist_ok=True)
        resource = None
        for attempt in range(16):
            resource = subprocess.run(["mthreads-gmi", "-i", "0"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
            (probes / (case["run_id"] + "_" + stamp + ".log")).write_text(resource.stdout)
            if resource.returncode == 0 and "No running processes found" in resource.stdout:
                break
            if attempt < 15:
                time.sleep(2)  # Driver context release may lag process exit; never kill another job.
        else:
            raise RuntimeError("Device 0 is not certified idle after 30s; no other jobs will be interrupted")
        folder.mkdir(parents=True, exist_ok=False)
        source_records = []
        for tree in (root / "src", root / "ports/specvla_musa"):
            for source in sorted(tree.rglob("*")):
                if source.is_file() and not any(part in (".git", "__pycache__") for part in source.relative_to(root).parts):
                    source_records.append(sha256(source) + "  " + str(source.relative_to(root)))
        (folder / "source_manifest.sha256").write_text("\n".join(source_records) + "\n")
        (folder / "device_before.log").write_text(resource.stdout)
        context = {**case, "campaign_id": campaign.name, "diagnostic": case["kind"] == "diagnostic"}
        dump(context_path, context)
        record = {**context, "command": command, "cwd": str(root), "start_timestamp": utc(),
                  "host": __import__("socket").gethostname(), "container": args.container,
                  "source_manifest": "runs/" + case["run_id"] + "/source_manifest.sha256",
                  "weight_manifest": "inventory/weight_manifest.json", "status": "RUNNING"}
        dump(folder / "run.json", record)
        start = time.monotonic(); next_sample = start
        print("RUN", case["run_id"], flush=True)
        with (folder / "stdout_stderr.log").open("w") as log, (folder / "device_samples.jsonl").open("w") as samples:
            process = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                current = time.monotonic()
                if current >= next_sample:
                    sampled = subprocess.run(["mthreads-gmi", "-q", "-i", "0", "--json"], text=True,
                                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=10)
                    samples.write(json.dumps({"timestamp": utc(), "elapsed_seconds": current-start,
                                              "requested_interval_seconds": .1, "exit_code": sampled.returncode,
                                              "raw": sampled.stdout}) + "\n")
                    samples.flush()
                    next_sample = current + .1
                if time.monotonic() - start > int(case["timeout_seconds"]) + 90:
                    # GNU timeout in the container owns termination of its process group.
                    process.terminate()
                    break
                time.sleep(.02)
            code = process.wait(timeout=30)
        record.update({"end_timestamp": utc(), "exit_code": code, "launcher_wall_seconds": time.monotonic()-start})
        episode_file = folder / "episodes.jsonl"
        from acceptance.common import rows
        episodes = rows(episode_file)
        collection = load(folder / "collection.json") if (folder / "collection.json").exists() else None
        complete = (code == 0 and len(episodes) == 1 and episodes[0]["steps"] > 0
                    and collection is not None and not collection["incomplete_episode"]
                    and not rows(folder / "errors.jsonl"))
        record["status"] = "PASS" if complete else "FAIL"
        record["task_success"] = episodes[0]["task_success"] if episodes else None
        record["status_scope"] = "execution completeness; task success separately recorded, smoke excluded from SR"
        record["error"] = None if complete else "Worker exit or missing/incomplete episode evidence"
        dump(folder / "run.json", record)
        print(record["status"], case["run_id"], "task_success=", record["task_success"], flush=True)
        if not complete:
            failures += 1
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
