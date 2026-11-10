"""Exercise the real YAML-based CLI parser used by the inference child."""

from dataclasses import dataclass
import unittest

import draccus
from FlagScale.inference_kerv import _bool, _draccus_value


@dataclass
class Options:
    mode: str = "auto"
    buckets: str = "1,2"
    enabled: bool = True


class CliValueTests(unittest.TestCase):
    def test_off_empty_and_boolean_survive_child_parser(self):
        cfg = draccus.parse(Options, args=[
            "--mode", _draccus_value("off"),
            "--buckets", _draccus_value(""),
            "--enabled", _draccus_value(_bool(False)),
        ])
        self.assertEqual(cfg.mode, "off")
        self.assertEqual(cfg.buckets, "")
        self.assertFalse(cfg.enabled)


if __name__ == "__main__":
    unittest.main()
