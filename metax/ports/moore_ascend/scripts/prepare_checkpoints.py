"""Download the public verifier and the SpecVLA LIBERO-Goal drafter."""

import argparse
import json
import shutil
from pathlib import Path

import gdown
import requests
from huggingface_hub import HfApi, snapshot_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/workspace/checkpoints"))
    parser.add_argument("--only", choices=("base", "draft", "both"), default="both")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    if args.only in {"draft", "both"}:
        source = args.root / "specvla-libero-goal"
        files = gdown.download_folder(
            id="1W7nBHM9-bf9tq4NQDUfles583OUfrtLv", output=str(source),
            quiet=False, skip_download=True, timeout=30, retries=2,
        )
        if files is None:
            raise SystemExit("SpecVLA drafter download failed")
        for file in files:
            target_file = Path(file.local_path)
            target_file.parent.mkdir(parents=True, exist_ok=True)
            if target_file.is_file() and target_file.stat().st_size:
                continue
            # The Google Drive UC endpoint redirects to a host unavailable
            # through some institutional proxies. This official endpoint
            # serves the same public file without that legacy redirect.
            url = f"https://drive.usercontent.google.com/download?id={file.id}&export=download&confirm=t"
            partial = target_file.with_suffix(target_file.suffix + ".partial")
            with requests.get(url, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                if "text/html" in response.headers.get("content-type", ""):
                    raise RuntimeError(f"Google Drive returned a confirmation page for {file.path}")
                with partial.open("wb") as destination:
                    for chunk in response.iter_content(8 * 1024 * 1024):
                        destination.write(chunk)
            partial.replace(target_file)
            print(f"Downloaded {file.path}: {target_file.stat().st_size} bytes", flush=True)
        candidates = list(source.rglob("pytorch_model.bin"))
        if len(candidates) != 1:
            raise SystemExit(f"Expected one drafter checkpoint; found {candidates}")
        draft = args.root / "kerv-drafter"
        draft.mkdir(exist_ok=True)
        shutil.copyfile(candidates[0], draft / "pytorch_model.bin")
        config = candidates[0].parent / "config.json"
        if not config.exists():
            config = Path(__file__).resolve().parents[1] / "training/drafter_config.json"
        shutil.copyfile(config, draft / "config.json")
        (draft / "source.json").write_text(json.dumps({
            "repository": "https://github.com/PineTreeWss/SpecVLA",
            "folder_id": "1W7nBHM9-bf9tq4NQDUfles583OUfrtLv",
            "original_checkpoint": str(candidates[0]),
        }, indent=2) + "\n")
    if args.only in {"base", "both"}:
        repo = "openvla/openvla-7b-finetuned-libero-goal"
        revision = HfApi().model_info(repo).sha
        target = args.root / "openvla-libero-goal"
        snapshot_download(repo, revision=revision, local_dir=target, max_workers=4)
        (target / "source.json").write_text(json.dumps({"repository": repo, "revision": revision}, indent=2) + "\n")


if __name__ == "__main__":
    main()
