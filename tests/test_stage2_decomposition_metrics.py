"""Unit tests for Stage-2 R0 probability and structure diagnostics."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from precipitation_inversion.metrics.stage2_decomposition import (  # noqa: E402
    CentroidDisplacementAccumulator,
    EchoColumnAccumulator,
    MultiThresholdSpatialAccumulator,
    Stage2DecompositionDiagnostics,
    SupportProbabilityAccumulator,
)


class Stage2DecompositionMetricTests(unittest.TestCase):
    def test_probability_metrics_are_masked_and_calibrated(self) -> None:
        probability = np.array(
            [[[1.0, 0.0], [0.9, 0.1]]], dtype=np.float32
        )
        target = np.array([[[1, 0], [1, 0]]], dtype=bool)
        domain = np.ones_like(target)
        accumulator = SupportProbabilityAccumulator(
            histogram_bins=10, calibration_bins=5
        )
        accumulator.update(probability, target, domain)
        result = accumulator.compute()

        self.assertEqual(result["count"], 4)
        self.assertEqual(result["positive_count"], 2)
        self.assertAlmostEqual(result["average_precision_histogram"], 1.0)
        self.assertAlmostEqual(result["brier_score"], 0.005, places=7)
        self.assertAlmostEqual(result["expected_calibration_error"], 0.05, places=7)

        invalid = probability.copy()
        invalid[0, 0, 0] = 1.1
        with self.assertRaisesRegex(ValueError, r"\[0,1\]"):
            accumulator.update(invalid, target, domain)

    def test_nested_threshold_skill_uses_physical_dbz_and_support(self) -> None:
        target_dbz = np.array([[[10.0, 20.0], [30.0, 40.0]]])
        prediction_dbz = target_dbz.copy()
        support = np.ones_like(target_dbz, dtype=bool)
        domain = np.ones_like(support)
        accumulator = MultiThresholdSpatialAccumulator(
            (15.0, 25.0, 35.0), fss_radii=(0, 1)
        )
        accumulator.update(
            prediction_dbz, support, target_dbz, support, domain
        )
        result = accumulator.compute()

        self.assertEqual(result["15"]["support"]["target_positive_count"], 3)
        self.assertEqual(result["25"]["support"]["target_positive_count"], 2)
        self.assertEqual(result["35"]["support"]["target_positive_count"], 1)
        self.assertAlmostEqual(result["35"]["support"]["csi"], 1.0)
        self.assertAlmostEqual(result["35"]["fss"]["1"]["fss"], 1.0)

    def test_column_top_and_base_keep_height_semantics(self) -> None:
        heights = np.array([0.25, 0.75, 1.25], dtype=np.float32)
        target = np.zeros((2, 1, 3), dtype=bool)
        predicted = np.zeros_like(target)
        target[0, 0, 0:2] = True       # base=.25, top=.75
        predicted[0, 0, 1:3] = True    # base=.75, top=1.25
        target[1, 0, 1] = True
        predicted[1, 0, 1] = True
        accumulator = EchoColumnAccumulator(heights)
        accumulator.update(predicted, target, np.ones_like(target))
        result = accumulator.compute()

        self.assertEqual(result["column_support"]["true_positive"], 2)
        self.assertAlmostEqual(result["paired_echo_top"]["bias_km"], 0.25)
        self.assertAlmostEqual(result["paired_echo_base"]["bias_km"], 0.25)

    def test_centroid_reports_known_horizontal_shift(self) -> None:
        target_dbz = np.full((3, 3, 1), np.nan, dtype=np.float32)
        prediction_dbz = np.full_like(target_dbz, np.nan)
        target_support = np.zeros_like(target_dbz, dtype=bool)
        predicted_support = np.zeros_like(target_support)
        target_support[0, 1, 0] = True
        predicted_support[1, 2, 0] = True
        target_dbz[target_support] = 30.0
        prediction_dbz[predicted_support] = 30.0
        accumulator = CentroidDisplacementAccumulator()
        accumulator.update(
            prediction_dbz,
            predicted_support,
            target_dbz,
            target_support,
            np.ones_like(target_support),
        )
        result = accumulator.compute()

        self.assertEqual(result["paired_orbit_height_count"], 1)
        self.assertAlmostEqual(result["mean_scan_offset_grid_cells"], 1.0)
        self.assertAlmostEqual(result["mean_ray_offset_grid_cells"], 1.0)
        self.assertAlmostEqual(result["mean_distance_grid_cells"], math.sqrt(2.0))

    def test_centroid_does_not_mutate_caller_support_arrays(self) -> None:
        target_dbz = np.full((2, 2, 1), 30.0, dtype=np.float32)
        prediction_dbz = target_dbz.copy()
        target_support = np.ones_like(target_dbz, dtype=bool)
        predicted_support = np.ones_like(target_support)
        domain = np.zeros_like(target_support)
        domain[0, 0, 0] = True
        target_before = target_support.copy()
        predicted_before = predicted_support.copy()

        CentroidDisplacementAccumulator().update(
            prediction_dbz,
            predicted_support,
            target_dbz,
            target_support,
            domain,
        )

        np.testing.assert_array_equal(target_support, target_before)
        np.testing.assert_array_equal(predicted_support, predicted_before)

    def test_combined_diagnostics_keep_region_probability_and_cfad(self) -> None:
        heights = np.array([0.25, 0.75], dtype=np.float32)
        target_dbz = np.array([[[20.0, 40.0], [np.nan, np.nan]]], dtype=np.float32)
        prediction_dbz = np.array([[[21.0, 38.0], [10.0, 10.0]]], dtype=np.float32)
        target_support = np.isfinite(target_dbz)
        predicted_support = np.array([[[True, True], [False, False]]])
        probability = np.array([[[0.9, 0.8], [0.1, 0.2]]], dtype=np.float32)
        domain = np.ones_like(target_support)
        diagnostics = Stage2DecompositionDiagnostics.create(
            heights,
            thresholds_dbz=(15.0, 35.0),
            fss_radii=(0,),
            cfad_edges_dbz=(0.0, 25.0, 50.0),
            probability_histogram_bins=20,
            probability_calibration_bins=5,
        )
        diagnostics.update(
            probability,
            prediction_dbz,
            predicted_support,
            target_dbz,
            target_support,
            domain,
            region_masks={"observed": np.array([[[True, True], [False, False]]])},
        )
        result = diagnostics.compute()

        self.assertEqual(result["support_probability"]["count"], 4)
        self.assertEqual(
            result["support_probability_by_region"]["observed"]["count"], 2
        )
        self.assertEqual(len(diagnostics.cfad.rows()), 4)
        self.assertEqual(
            sum(row["target_count"] for row in diagnostics.cfad.rows()), 2
        )


if __name__ == "__main__":
    unittest.main()
