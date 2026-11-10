import hashlib
import json
import math
import statistics
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text())


def rows(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
            for line in path.read_text().splitlines() if line.strip()]


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def distribution(values):
    values = list(values)
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None}
    if not all(math.isfinite(x) for x in values):
        raise ValueError("Non-finite observation")
    return {"count": len(values), "mean": statistics.mean(values),
            "median": statistics.median(values),
            "p95": sorted(values)[math.ceil(.95 * len(values)) - 1]}


def threshold_results(baseline_sr, tested_sr, delta_length, transfer_bytes_per_second,
                      memory_saved_bytes):
    # Decimal arithmetic avoids e.g. 0.55 - 0.5 being classified inconsistently.
    from decimal import Decimal
    loss = Decimal(str(baseline_sr)) - Decimal(str(tested_sr))
    return {
        "sr_loss_percentage_points": float(loss * 100),
        "sr_loss_relative": float(loss / Decimal(str(baseline_sr))) if baseline_sr else None,
        "sr_loss_pp_lt5": loss < Decimal("0.05"),
        "sr_loss_relative_lt5percent": loss / Decimal(str(baseline_sr)) < Decimal("0.05") if baseline_sr else None,
        "acceptance_improvement_gt0_5": Decimal(str(delta_length)) > Decimal("0.5"),
        "transfer_lt100_decimal_MBps": transfer_bytes_per_second < 100_000_000,
        "memory_saving_ge2_decimal_GB": memory_saved_bytes >= 2_000_000_000,
        "memory_saving_ge2_GiB": memory_saved_bytes >= 2 * 1024**3,
    }


def committed_candidate_prefix(event, action_dim=7):
    """Apply the real decoder's action-boundary clamp, excluding root and EOS.

    Posterior acceptance can exceed the remaining action budget. Direct KF
    completion uses the first action_dim accepted candidates, without that
    ordinary update's extra root-budget clamp.
    """
    accepted = int(event["accepted_prefix_length"])
    capacity = action_dim if event["compensation_triggered"] else max(0, action_dim - 1 - int(event["tokens_before"]))
    return min(accepted, capacity)
