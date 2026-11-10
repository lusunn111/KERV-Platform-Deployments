"""The launcher must preserve a failing inference child's exit status."""

import subprocess
import tempfile
import unittest
from pathlib import Path

from omegaconf import OmegaConf
from flagscale.runner.backend.backend_vllm import VllmBackend
from flagscale.runner.runner_inference import _generate_run_script_inference


class FlagScaleExitStatusTests(unittest.TestCase):
    def test_legacy_and_current_inference_runners(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = OmegaConf.create({"experiment": {"runner": {}}, "inference": {"logging": {
                "log_dir": str(root / "logs"), "scripts_dir": str(root / "scripts"),
                "pids_dir": str(root / "pids"),
            }}})
            backend = object.__new__(VllmBackend)
            backend.task_type = "inference"
            for name, generate in [
                ("legacy", lambda cmd: _generate_run_script_inference(config, "localhost", 0, cmd)),
                ("current", lambda cmd: backend.generate_run_script(config, "localhost", 0, cmd)),
            ]:
                for expected in (0, 23):
                    with self.subTest(runner=name, exit_status=expected):
                        script = generate(f"exit {expected}")
                        result = subprocess.run(["bash", script], capture_output=True, text=True)
                        self.assertEqual(result.returncode, expected, result.stderr)


if __name__ == "__main__":
    unittest.main()
