"""Tests for Stage-2 threshold provenance and evaluation output helpers."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.evaluate_stage2_unet3d import (  # noqa: E402
    load_threshold_file,
    resolve_support_threshold,
)


class Stage2EvaluationTests(unittest.TestCase):
    def test_threshold_file_must_record_validation_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "threshold.json"
            path.write_text(
                json.dumps({"threshold": 0.35, "selected_on_split": "val"}),
                encoding="utf-8",
            )
            self.assertEqual(load_threshold_file(path), 0.35)
            path.write_text(
                json.dumps({"threshold": 0.35, "selected_on_split": "test"}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "split=val"):
                load_threshold_file(path)

    def test_explicit_and_file_threshold_are_mutually_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "threshold.json"
            path.write_text(
                json.dumps({"threshold": 0.4, "selected_on_split": "val"}),
                encoding="utf-8",
            )
            self.assertEqual(
                resolve_support_threshold(
                    explicit=None, threshold_file=path, configured=0.5
                ),
                0.4,
            )
            self.assertEqual(
                resolve_support_threshold(
                    explicit=None, threshold_file=None, configured=0.5
                ),
                0.5,
            )
            with self.assertRaisesRegex(ValueError, "either"):
                resolve_support_threshold(
                    explicit=0.2, threshold_file=path, configured=0.5
                )
            with self.assertRaisesRegex(ValueError, r"\(0,1\)"):
                resolve_support_threshold(
                    explicit=1.0, threshold_file=None, configured=0.5
                )


if __name__ == "__main__":
    unittest.main()
