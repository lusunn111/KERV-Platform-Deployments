"""Boundary and missing-evidence regressions required by TestOutline P02."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from acceptance.common import distribution, rows, threshold_results, dump, committed_candidate_prefix
from acceptance.aggregate import aggregate


class AcceptanceToolsTests(unittest.TestCase):
    def test_action_boundary_clamp_excludes_eos_and_unused_verified_tail(self):
        self.assertEqual(committed_candidate_prefix({"accepted_prefix_length":5,"tokens_before":4,"compensation_triggered":False}),2)
        self.assertEqual(committed_candidate_prefix({"accepted_prefix_length":0,"tokens_before":0,"compensation_triggered":False}),0)
        self.assertEqual(committed_candidate_prefix({"accepted_prefix_length":7,"tokens_before":0,"compensation_triggered":True}),7)
    def test_strict_thresholds_and_decimal_units(self):
        r = threshold_results(1, .95, .5, 100_000_000, 2_000_000_000)
        self.assertFalse(r["sr_loss_pp_lt5"])
        self.assertFalse(r["sr_loss_relative_lt5percent"])
        self.assertFalse(r["acceptance_improvement_gt0_5"])
        self.assertFalse(r["transfer_lt100_decimal_MBps"])
        self.assertTrue(r["memory_saving_ge2_decimal_GB"])
        self.assertFalse(r["memory_saving_ge2_GiB"])
        self.assertIsNone(threshold_results(0, 0, 1, 0, 0)["sr_loss_relative"])

    def test_zero_acceptances_and_nearest_rank(self):
        self.assertEqual(distribution([0, 0, 3])["mean"], 1)
        self.assertEqual(distribution(range(1, 21))["p95"], 19)
        with self.assertRaises(ValueError):
            distribution([1, float("inf")])

    def test_non_finite_json_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bad.jsonl"
            p.write_text('{"x":NaN}\n')
            with self.assertRaises(ValueError):
                rows(p)

    def fixture(self, tmp):
        root = Path(tmp)
        (root / "inventory").mkdir()
        (root / "runs/x").mkdir(parents=True)
        case = {"run_id":"x", "kind":"full", "profile":"native", "task_id":"0", "init_state_id":"0", "seed":"7", "repeat_id":"0", "model_id":"M1", "platform_id":"P1", "chip_id":"H1", "physical_device_id":"0"}
        with (root / "inventory/cases.csv").open("w") as f:
            w = csv.DictWriter(f, fieldnames=list(case)); w.writeheader(); w.writerow(case)
        return root

    def test_failed_episode_remains_in_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.fixture(tmp)
            dump(root / "runs/x/run.json", {"status":"FAIL", "task_success":False})
            report = aggregate(root)
            native = next(g for g in report["groups"] if g["profile"] == "native")
            self.assertEqual(native["executed_cases_including_errors"], 1)
            self.assertEqual(native["success_rate_over_executed_cases"], 0)
            self.assertEqual(report["formal_acceptance_status"], "BLOCKED")

    def test_duplicate_and_missing_step_ids_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.fixture(tmp)
            for ids in ([0,0], [0,2]):
                (root / "runs/x/steps.jsonl").write_text("\n".join(json.dumps({"episode_id":"0_1","step_id":i}) for i in ids)+"\n")
                with self.assertRaises(ValueError):
                    aggregate(root)

    def test_missing_verification_event_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.fixture(tmp)
            step = {"episode_id":"0_1", "step_id":0, "raw_action":[0]*7, "executed_action":[0]*7,
                    "verify_events":2,"model_seconds":1,"pipeline_seconds":1,"env_seconds":1,
                    "wall_seconds":3,"cpu_seconds":.2,"observation_hash":"x"}
            (root / "runs/x/steps.jsonl").write_text(json.dumps(step)+"\n")
            event = {"episode_id":"0_1","step_id":0,"verify_id":0,"accepted_prefix_length":0}
            (root / "runs/x/verify.jsonl").write_text(json.dumps(event)+"\n")
            with self.assertRaises(ValueError):
                aggregate(root)


if __name__ == "__main__":
    unittest.main()
