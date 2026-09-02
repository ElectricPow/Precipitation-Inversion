"""Tests for the resumable E0/N/I/W physical dR/dz backfill entry point."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


from scripts.backfill_stage1_drdz import (
    evaluation_command,
    reusable_physical_drdz_report,
)


class DrdzBackfillTests(unittest.TestCase):
    def test_reuse_requires_complete_coverage_metric_and_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.json"
            reusable, reason = reusable_physical_drdz_report(path)
            self.assertFalse(reusable)
            self.assertIn("does not exist", reason)

            value = {
                "split": "val",
                "patch_evaluation": {
                    "coverage": {"complete_patch_support": True},
                    "metrics": {},
                },
            }
            path.write_text(json.dumps(value), encoding="utf-8")
            reusable, reason = reusable_physical_drdz_report(path)
            self.assertFalse(reusable)
            self.assertIn("physical_drdz", reason)

            value["patch_evaluation"]["metrics"]["physical_drdz"] = {
                "support": {"sha256": "abc123"}
            }
            path.write_text(json.dumps(value), encoding="utf-8")
            self.assertEqual(
                reusable_physical_drdz_report(path),
                (True, "complete physical dR/dz report"),
            )

    def test_evaluation_command_is_complete_validation_without_orbits(self) -> None:
        command = evaluation_command(
            Path("best.pt"),
            Path("metrics.json"),
            device="cuda:0",
            num_workers=0,
            bootstrap_seed=17,
            bootstrap_replicates=25,
            bootstrap_confidence=0.9,
        )
        self.assertIn("--stratified", command)
        self.assertEqual(command[command.index("--split") + 1], "val")
        self.assertEqual(command[command.index("--full-orbits") + 1], "0")
        self.assertEqual(command[command.index("--num-workers") + 1], "0")
        self.assertEqual(command[command.index("--output") + 1], "metrics.json")


if __name__ == "__main__":
    unittest.main()
