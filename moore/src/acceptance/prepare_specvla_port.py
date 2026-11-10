"""Create an isolated MUSA port of the pinned real SpecVLA source.

No KERV generation, Kalman, thresholds or fused operators are substituted.
"""
import argparse
import difflib
from pathlib import Path
import shutil
import subprocess
from acceptance.common import dump, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("/root/KERV"))
    args = p.parse_args()
    root = args.root.resolve()
    upstream = root / "third_party/SpecVLA"
    revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if revision != "56ef2985e25c7903d2759254efb754327fb975f7":
        raise ValueError("Unexpected SpecVLA source revision")
    target = root / "ports/specvla_musa"
    if target.exists():
        raise FileExistsError("Preserve existing port; use its manifest rather than overwriting")
    shutil.copytree(upstream / "openvla", target / "openvla", symlinks=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.egg-info", "wandb"))
    edits = []
    def update(rel, transform):
        path = target / rel
        original = path.read_text()
        result = transform(original)
        if result == original:
            raise ValueError("Expected compatibility edit absent: " + rel)
        path.write_text(result)
        edits.append({"file":rel, "original_sha256":sha256(upstream/rel), "ported_sha256":sha256(path)})
        return "".join(difflib.unified_diff(original.splitlines(True), result.splitlines(True), fromfile="upstream/"+rel, tofile="musa/"+rel))
    patches = []
    patches.append(update("openvla/prismatic/__init__.py", lambda _: (root/"src/openvla/prismatic/__init__.py").read_text()))
    shutil.copy2(root/"src/openvla/experiments/robot/device_runtime.py", target/"openvla/experiments/robot/device_runtime.py")
    # Local processor implementation with the same config/tokenizer, avoiding Hub dynamic-code fallback.
    kerv_utils = (root/"src/openvla/experiments/robot/openvla_utils.py").read_text()
    local_processor = kerv_utils[kerv_utils.index("def get_processor("):kerv_utils.index("def crop_and_resize(")]
    def utils(text):
        text = text.replace("import time\n", "import time\nfrom .device_runtime import get_device, synchronize\n")
        text = text.replace("AutoModelForVision2Seq, AutoProcessor", "AutoModelForVision2Seq, AutoProcessor, AutoTokenizer")
        text = text.replace('torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")', "get_device()")
        text = text.replace("            use_spec = True\n", '            use_spec = True,\n            attn_implementation="eager",\n')
        begin, end = text.index("def get_processor("), text.index("def crop_and_resize(")
        text = text[:begin] + local_processor + text[end:]
        text = text.replace("start_time = time.time()", "start_time = time.perf_counter()")
        text = text.replace("end_time = time.time()", "synchronize(DEVICE)\n    end_time = time.perf_counter()")
        return text
    patches.append(update("openvla/experiments/robot/openvla_utils.py", utils))
    def model(text):
        text = text.replace("MMModel(config, path=base_model_name_or_path,load_emb=True)", "MMModel(config, path=base_model_name_or_path,load_emb=False)")
        text = text.replace("torch.load(load_model_path)", 'torch.load(load_model_path, map_location="cpu", weights_only=True)')
        text = text.replace("self.ea_layer.load_state_dict(ea_layer_state_dict, strict=True)", '''# The checkpoint's redundant 32064-row embedding is immediately replaced below.
        ea_layer_state_dict.pop("embed_tokens.weight", None)
        incompatible = self.ea_layer.load_state_dict(ea_layer_state_dict, strict=False)
        if incompatible.missing_keys != ["embed_tokens.weight"] or incompatible.unexpected_keys:
            raise RuntimeError("SpecVLA drafter parameters do not match: " + str(incompatible))''')
        needle = "        discretized_actions = self.vocab_size - predicted_action_token_ids"
        text = text.replace(needle, "        self._last_action_token_ids_cpu = predicted_action_token_ids.tolist()\n" + needle)
        needle = "            input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, new_token,prompt_embeds,past_key_value_data,attention_mask = update_inference_inputs("
        start = text.index("    def eagenerate(")
        before, body = text[:start], text[start:]
        body = body.replace(needle, '''            audit = getattr(self, "_acceptance_collector", None)
            if audit is not None:
                control = torch.stack((best_candidate.long(), accept_length.long())).detach().cpu().tolist()
                audit_event = audit.verify(candidates, int(control[0]), int(control[1]), accept_threshold, False, input_ids.shape[1]-input_len-1)
            else:
                audit_event = None
            audit_length_before = int(input_ids.shape[1])
''' + needle, 1)
        needle = "            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():"
        body = body.replace(needle, '''            if audit_event is not None:
                committed = int(input_ids.shape[1]) - audit_length_before
                audit_event["committed_new_action_tokens"] = committed
                audit_event["committed_accepted_prefix_length"] = min(audit_event["accepted_prefix_length"], max(0, committed-1))
''' + needle, 1)
        return before + body
    patches.append(update("openvla/prismatic/extern/hf/modeling_speculation.py", model))
    (target/"musa_compatibility.patch").write_text("\n".join(patches))
    dump(target/"port_manifest.json", {"upstream":"https://github.com/PineTreeWss/SpecVLA", "revision":revision,
        "changes":edits, "added_device_helper_sha256":sha256(target/"openvla/experiments/robot/device_runtime.py"),
        "baseline_identity":"Real upstream SpecVLA eagenerate/cnets/utils; fixed relaxed threshold 9, original tree defaults; no KERV Kalman/dynamic acceptance/runtime fusion",
        "validation_status":"NOT_VALIDATED until import, fixed-input and real runs are archived",
        "limits":["MUSA BF16/eager port, not an A100 reference experiment", "Redundant drafter embedding replaced by shared verifier embedding as upstream already does after loading", "Extra control read for audit can affect timing; no official acceptance claim"]})
    print(target)


if __name__ == "__main__":
    main()
