"""Tests for complete-validation stratified metric plotting."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.plot_stage1_stratified_metrics import (  # noqa: E402
    generate_stratified_analysis,
    load_stratified_metrics,
)


def regression(count: int, error: float) -> dict[str, float | int]:
    return {
        "count": count,
        "mae": error,
        "rmse": error * 1.5,
        "bias": -error / 2,
        "r2": 0.5,
        "pearson_r": 0.8,
    }


class StratifiedVisualizationTests(unittest.TestCase):
    def test_generates_png_csv_and_summary(self) -> None:
        heights = ("z_0p125_km", "z_0p375_km")
        intensities = (
            "lt_1_mm_h",
            "1_to_5_mm_h",
            "5_to_10_mm_h",
            "10_to_30_mm_h",
            "ge_30_mm_h",
        )
        stratified = {
            "definitions": {},
            "by_height_km": {
                label: regression((index + 1) * 10, 0.2 + index)
                for index, label in enumerate(heights)
            },
            "by_cfb_distance_km": {
                "lt_m1_km": regression(0, 0.0),
                "m1_to_0_km": regression(0, 0.0),
                "0_to_0p5_km": regression(12, 1.0),
                "0p5_to_2_km": regression(20, 0.5),
                "ge_2_km": regression(8, 0.2),
            },
            "by_height_and_intensity_mm_h": {
                height: {
                    intensity: regression(5 + position, 0.1 + position)
                    for position, intensity in enumerate(intensities)
                }
                for height in heights
            },
            "by_precipitation_type": {
                "stratiform": regression(20, 0.3),
                "convective": regression(10, 1.2),
                "other": regression(2, 0.5),
                "no_precipitation": regression(0, 0.0),
                "unclassified": regression(0, 0.0),
            },
        }
        below_cfb = {
            "definition": "diagnostic only",
            "log": regression(7, 0.2),
            "rain": {
                "all": regression(7, 1.0),
                "target_bins_mm_h": {
                    "lt_1": regression(2, 0.2),
                    "1_to_5": regression(2, 0.5),
                    "5_to_10": regression(1, 1.0),
                    "10_to_30": regression(1, 2.0),
                    "ge_30": regression(1, 4.0),
                },
            },
        }
        per_file = {
            "orbit_a": regression(10, 0.5),
            "orbit_b": regression(20, 1.0),
        }
        confidence_interval = {
            name: {"low": 0.1, "high": 1.2, "valid_replicates": 50}
            for name in ("mae", "rmse", "bias", "r2", "pearson_r")
        }
        filewise = {
            "definition": "macro by file",
            "file_count_total": 2,
            "file_count_nonempty": 2,
            "valid_voxel_count": 30,
            "macro_average": {
                "mae": 0.75,
                "rmse": 1.125,
                "bias": -0.375,
                "r2": 0.5,
                "pearson_r": 0.8,
            },
            "bootstrap": {
                "seed": 17,
                "replicates": 50,
                "confidence_level": 0.9,
                "confidence_interval": confidence_interval,
            },
            "per_file": per_file,
        }
        drdz_overall = {
            **regression(25, 1.0),
            "mean_abs_prediction": 0.8,
            "mean_abs_target": 1.0,
            "mean_abs_gradient_ratio": 0.8,
            "sign_epsilon": 0.1,
            "sign_evaluated_count": 20,
            "sign_agreement_fraction": 0.75,
        }
        drdz = {
            "definitions": {},
            "all": drdz_overall,
            "by_midpoint_height_km": {
                label: regression(10 + index, 0.5 + index)
                for index, label in enumerate(heights)
            },
            "by_midpoint_cfb_distance_km": stratified[
                "by_cfb_distance_km"
            ],
            "by_pair_mean_target_intensity_mm_h": {
                label: regression(5, 0.5) for label in intensities
            },
            "by_precipitation_type": stratified["by_precipitation_type"],
            "filewise": filewise,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "metrics.json"
            source.write_text(
                json.dumps(
                    {
                        "patch_evaluation": {
                            "metrics": {
                                "stratified": stratified,
                                "diagnostics": {
                                    "below_cfb_native_positive": below_cfb
                                },
                                "filewise": filewise,
                                "physical_drdz": drdz,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            summary = generate_stratified_analysis(source, dpi=35)
            destination = root / "stratified"
            self.assertEqual(summary["evaluation_section"], "patch_evaluation")
            self.assertTrue(summary["has_below_cfb_native_positive_diagnostic"])
            self.assertTrue(summary["has_filewise_macro_bootstrap"])
            self.assertTrue(summary["has_physical_drdz"])
            for name in (
                "metrics_by_height.png",
                "height_intensity_heatmaps.png",
                "cfb_and_precipitation_type.png",
                "metrics_by_height.csv",
                "metrics_by_cfb_distance.csv",
                "metrics_by_precipitation_type.csv",
                "below_cfb_native_positive.png",
                "below_cfb_native_positive.csv",
                "filewise_macro_bootstrap.png",
                "filewise_macro_bootstrap.csv",
                "metrics_by_file.csv",
                "drdz_overall.csv",
                "drdz_by_height.csv",
                "drdz_by_height.png",
                "drdz_by_cfb_distance.csv",
                "drdz_by_target_intensity.csv",
                "drdz_by_precipitation_type.csv",
                "drdz_grouped.png",
                "drdz_metrics_by_file.csv",
                "drdz_filewise_macro_bootstrap.csv",
                "drdz_summary.md",
                "summary.json",
                "summary.md",
            ):
                self.assertTrue((destination / name).is_file(), name)
                self.assertGreater((destination / name).stat().st_size, 0)

    def test_missing_stratified_metrics_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.json"
            path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no stratified metrics"):
                load_stratified_metrics(path)


if __name__ == "__main__":
    unittest.main()
