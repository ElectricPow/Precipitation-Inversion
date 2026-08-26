"""Tests for streaming masked precipitation regression metrics."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from precipitation_inversion.metrics.regression import (  # noqa: E402
    PrecipitationRegressionMetrics,
    RegressionAccumulator,
)


class RegressionAccumulatorTests(unittest.TestCase):
    def test_masked_metrics_match_direct_values(self) -> None:
        # Arrays represent (B=1,C=1,D=1,H=1,Z=3). The final height is padding.
        prediction = torch.tensor([2.0, 4.0, 999.0]).reshape(1, 1, 1, 1, 3)
        target = torch.tensor([1.0, 5.0, 0.0]).reshape_as(prediction)
        mask = torch.tensor([True, True, False]).reshape_as(prediction)
        accumulator = RegressionAccumulator()
        accumulator.update(prediction, target, mask)
        result = accumulator.compute()
        self.assertEqual(result["count"], 2)
        self.assertAlmostEqual(result["mae"], 1.0)
        self.assertAlmostEqual(result["rmse"], 1.0)
        self.assertAlmostEqual(result["bias"], 0.0)
        self.assertAlmostEqual(result["r2"], 0.75)
        self.assertAlmostEqual(result["pearson_r"], 1.0)

    def test_streaming_updates_and_merge_equal_one_shot(self) -> None:
        prediction = np.array([1.0, 4.0, 8.0, 3.0], dtype=np.float32)
        target = np.array([2.0, 2.0, 7.0, 5.0], dtype=np.float32)
        mask = np.array([True, False, True, True])
        expected = RegressionAccumulator()
        expected.update(prediction, target, mask)

        first = RegressionAccumulator()
        second = RegressionAccumulator()
        first.update(prediction[:2], target[:2], mask[:2])
        second.update(prediction[2:], target[2:], mask[2:])
        first.merge(second)
        for key, expected_value in expected.compute().items():
            actual = first.compute()[key]
            if isinstance(expected_value, float):
                self.assertAlmostEqual(actual, expected_value)
            else:
                self.assertEqual(actual, expected_value)

    def test_empty_mask_and_invalid_selected_values(self) -> None:
        accumulator = RegressionAccumulator()
        accumulator.update(torch.tensor([math.nan]), torch.tensor([1.0]), torch.tensor([False]))
        result = accumulator.compute()
        self.assertEqual(result["count"], 0)
        self.assertTrue(math.isnan(result["mae"]))
        with self.assertRaisesRegex(ValueError, "finite"):
            accumulator.update(
                torch.tensor([math.nan]), torch.tensor([1.0]), torch.tensor([True])
            )
        with self.assertRaises(TypeError):
            accumulator.update(torch.ones(1), torch.ones(1), torch.ones(1))


class PrecipitationRegressionMetricsTests(unittest.TestCase):
    def test_log_conversion_physical_metrics_and_target_bins(self) -> None:
        target_rain = torch.tensor([0.5, 2.0, 7.0, 20.0, 40.0])
        prediction_rain = target_rain + 1.0
        mask = torch.ones_like(target_rain, dtype=torch.bool)
        metrics = PrecipitationRegressionMetrics((1, 5, 10, 30))
        metrics.update_log(
            torch.log1p(prediction_rain), torch.log1p(target_rain), mask
        )
        result = metrics.compute()
        self.assertEqual(result["rain"]["all"]["count"], 5)
        self.assertAlmostEqual(result["rain"]["all"]["mae"], 1.0, places=5)
        bins = result["rain"]["target_bins_mm_h"]
        self.assertEqual(bins["lt_1"]["count"], 1)
        self.assertEqual(bins["1_to_5"]["count"], 1)
        self.assertEqual(bins["5_to_10"]["count"], 1)
        self.assertEqual(bins["10_to_30"]["count"], 1)
        self.assertEqual(bins["ge_30"]["count"], 1)

    def test_negative_log_prediction_is_clamped_only_in_physical_space(self) -> None:
        metrics = PrecipitationRegressionMetrics((1,))
        metrics.update_log(
            torch.tensor([-2.0]), torch.tensor([math.log(2.0)]), torch.tensor([True])
        )
        result = metrics.compute()
        self.assertAlmostEqual(result["rain"]["all"]["bias"], -1.0)
        self.assertAlmostEqual(result["log"]["bias"], -2.0 - math.log(2.0))

    def test_update_rain_and_reset(self) -> None:
        metrics = PrecipitationRegressionMetrics((1, 5))
        metrics.update_rain(
            torch.tensor([1.0, 3.0]),
            torch.tensor([1.0, 2.0]),
            torch.tensor([True, False]),
        )
        self.assertEqual(metrics.compute()["rain"]["all"]["count"], 1)
        metrics.reset()
        self.assertEqual(metrics.compute()["rain"]["all"]["count"], 0)

    def test_invalid_thresholds_and_shape_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            PrecipitationRegressionMetrics((5, 1))
        metrics = PrecipitationRegressionMetrics()
        with self.assertRaisesRegex(ValueError, "shapes differ"):
            metrics.update_log(torch.ones(2), torch.ones(3), torch.ones(2, dtype=torch.bool))


if __name__ == "__main__":
    unittest.main()
