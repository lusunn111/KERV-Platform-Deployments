"""Read-only local acceptance prerequisites. Does not download or install."""
import argparse
import shutil
import subprocess
from pathlib import Path
from acceptance.common import dump, load, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("/root/KERV"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--hash-weights", action="store_true")
    args = p.parse_args()
    root = args.root.resolve()
    checks = []
    def add(name, ok, detail):
        checks.append({"check": name, "status": "PASS" if ok else "BLOCKED", "detail": detail})
    for rel in ("src/run_kerv.py", "src/scripts/run_musa.sh", "TestOutline.md",
                "docs/acceptance/models.yaml", "docs/acceptance/platforms.yaml", "docs/acceptance/chips.yaml",
                "third_party/LIBERO/libero/libero/bddl_files", ".libero/config.yaml"):
        add(rel, (root / rel).exists(), str(root / rel))
    taskbook = root / "docs/acceptance/sources/2025-1215-课题任务书.pdf"
    add("taskbook_sha256", taskbook.exists() and sha256(taskbook) == "94c0d5812d9bf43b48cd5689083d408f41753010aa0bfac537b109fdca0f3f4c", str(taskbook))
    base = root / "checkpoints/openvla-libero-goal"
    index = base / "model.safetensors.index.json"
    shards = sorted(set(load(index)["weight_map"].values())) if index.exists() else []
    files = [base / x for x in shards] + [base / "config.json", base / "tokenizer_config.json", base / "preprocessor_config.json"]
    add("verifier_files", bool(shards) and all(x.is_file() for x in files), [str(x) for x in files])
    add("verifier_revision", (base / "source.json").exists() and load(base / "source.json").get("revision") == "fa5ae1e7509348889295bba8e08621d8b55e9baf", "fixed public OpenVLA revision")
    draft = root / "checkpoints/kerv-drafter/pytorch_model.bin"
    add("drafter_file", draft.exists() and draft.stat().st_size == 734546940, str(draft))
    if args.hash_weights:
        files += [draft] + sorted((root / "checkpoints/kerv-drafter").glob("*.json"))
        dump(args.output.parent / "weight_manifest.json", {str(x.relative_to(root)): {"bytes": x.stat().st_size, "sha256": sha256(x)} for x in files if x.is_file()})
        add("drafter_sha256", sha256(draft) == "d3f62ae500f27b356083f997b24fa7a67e5dda86e5083fa15985f45250009aae", "full file hashed")
    probe = subprocess.run(["docker", "inspect", "--format", "{{.State.Running}} {{.HostConfig.Runtime}} {{.Image}}", "kerv-musa-runtime"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    add("container", probe.returncode == 0 and probe.stdout.startswith("true mthreads"), probe.stdout.strip())
    device = subprocess.run(["mthreads-gmi", "-i", "0"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (args.output.parent / "preflight_device.log").write_text(device.stdout)
    add("device_tool", device.returncode == 0 and "MTT S5000" in device.stdout, "device 0; raw output separately retained")
    add("disk_headroom", shutil.disk_usage(root).free > 10 * 1024**3, {"free_bytes": shutil.disk_usage(root).free, "stage_minimum_bytes": 10 * 1024**3})
    report = {"scope": "H1 Moore local self-test only", "checks": checks,
              "local_prerequisites_passed": all(x["status"] == "PASS" for x in checks),
              "formal_acceptance_status": "BLOCKED", "formal_blockers": ["No A100 reference in local scope", "M2 not provided", "P2/P3 not approved/provided", "Decision records unsigned", "Independent third-party test not performed"]}
    dump(args.output, report)
    print(args.output)
    if not report["local_prerequisites_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
