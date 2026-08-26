"""Tests for strict cross-experiment physical dR/dz comparison."""

from __future__ import annotations

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path


from scripts.compare_stage1_drdz import _assert_same_support, _write_outputs


def regression(count: int, error: float) -> dict[str, float | int]:
    return {
        "count": count,
        "mae": error,
        "rmse": error * 1.5,
        "bias": -error / 2,
        "r2": 0.5,
        "pearson_r": 0.7,
    }


def gradient_metrics(error: float) -> dict:
    overall = {
        **regression(30, error),
        "mean_abs_prediction": 0.8,
        "mean_abs_target": 1.0,
        "mean_abs_gradient_ratio": 0.8,
        "sign_epsilon": 0.1,
        "sign_evaluated_count": 25,
        "sign_agreement_fraction": 0.75,
    }
    heights = {
        "z_0p25_km": regression(10, error),
        "z_0p75_km": regression(20, error * 1.2),
    }
    files = {
        "orbit_a": regression(10, error),
        "orbit_b": regression(20, error * 1.1),
    }
    return {
        "definitions": {"protocol": "same"},
        "all": overall,
        "by_midpoint_height_km": heights,
        "filewise": {"per_file": files},
    }


class DrdzComparisonTests(unittest.TestCase):
    def test_generates_comparison_and_rejects_different_support(self) -> None:
        runs = {"E0": gradient_metrics(1.0), "I": gradient_metrics(0.8)}
        _assert_same_support(runs)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            result = _write_outputs(
                runs,
                {"E0": Path("e0.json"), "I": Path("i.json")},
                baseline_label="E0",
                output_dir=output,
                seed=17,
                replicates=50,
                confidence=0.9,
                dpi=35,
            )
            self.assertEqual(result["support_pair_count"], 30)
            for name in (
                "metrics.csv",
                "paired_vs_baseline.csv",
                "comparison.png",
                "comparison_by_height.png",
                "summary.json",
                "summary.md",
            ):
                self.assertTrue((output / name).is_file(), name)
                self.assertGreater((output / name).stat().st_size, 0)

        mismatched = deepcopy(runs)
        mismatched["I"]["all"]["count"] = 29
        with self.assertRaisesRegex(ValueError, "pair count differs"):
            _assert_same_support(mismatched)

        fingerprint_mismatch = deepcopy(runs)
        fingerprint_mismatch["E0"]["support"] = {"sha256": "same-support"}
        fingerprint_mismatch["I"]["support"] = {"sha256": None}
        with self.assertRaisesRegex(ValueError, "exact reliable.*support differs"):
            _assert_same_support(fingerprint_mismatch)


if __name__ == "__main__":
    unittest.main()
