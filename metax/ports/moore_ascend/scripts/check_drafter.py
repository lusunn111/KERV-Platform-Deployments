"""Validate a SpecVLA/KERV drafter against the inference model without a verifier."""

import argparse
import json
from pathlib import Path

import torch

from openvla.specdecoding.model.cnets import EConfig, MMModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    config = EConfig.from_pretrained(str(args.checkpoint))
    with torch.device("meta"):
        model = MMModel(config, load_emb=False)
    state = torch.load(args.checkpoint / "pytorch_model.bin", map_location="cpu", weights_only=True)
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state) - {"embed_tokens.weight"})
    unexpected = sorted(set(state) - set(expected))
    mismatched = {k: [list(state[k].shape), list(expected[k].shape)]
                  for k in set(state) & set(expected)
                  if k != "embed_tokens.weight" and state[k].shape != expected[k].shape}
    report = {"checkpoint": str(args.checkpoint), "parameter_count": len(state),
              "missing": missing, "unexpected": unexpected, "mismatched_shapes": mismatched,
              "embedding_source": "shared verifier embedding (KERV inference loader)",
              "checkpoint_embedding_shape": list(state["embed_tokens.weight"].shape)
              if "embed_tokens.weight" in state else None}
    print(json.dumps(report, indent=2))
    (args.checkpoint / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    if missing or unexpected or mismatched:
        raise SystemExit("Drafter does not match KERV; refusing partial parameter loading")


if __name__ == "__main__":
    main()
