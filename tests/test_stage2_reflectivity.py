"""Tests for stage-two support, dBZ, and neighborhood metrics."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from precipitation_inversion.metrics.stage2_reflectivity import (  # noqa: E402
    FractionSkillAccumulator,
    ReflectivityRegressionAccumulator,
    Stage2ReflectivityMetrics,
    SupportConfusionAccumulator,
    finite_metrics_for_json,
    horizontal_window_sum,
)


class SupportMetricTests(unittest.TestCase):
    def test_confusion_metrics_respect_explicit_domain(self) -> None:
        predicted = np.array([True, True, False, False, True], dtype=bool)
        target = np.array([True, False, True, False, True], dtype=bool)
        domain = np.array([True, True, True, True, False], dtype=bool)
        metric = SupportConfusionAccumulator()
        metric.update(predicted, target, domain)
        result = metric.compute()
        self.assertEqual(
            (result["true_positive"], result["false_positive"],
             result["false_negative"], result["true_negative"]),
            (1, 1, 1, 1),
        )
        self.assertAlmostEqual(result["precision"], 0.5)
        self.assertAlmostEqual(result["recall"], 0.5)
        self.assertAlmostEqual(result["f1"], 0.5)
        self.assertAlmostEqual(result["csi"], 1.0 / 3.0)

    def test_invalid_mask_dtype_or_shape_is_rejected(self) -> None:
        metric = SupportConfusionAccumulator()
        with self.assertRaises(TypeError):
            metric.update([1, 0], np.array([True, False]), np.ones(2, bool))
        with self.assertRaises(ValueError):
            metric.update(
                np.ones(2, bool), np.ones(3, bool), np.ones(2, bool)
            )


class ReflectivityMetricTests(unittest.TestCase):
    def test_regression_values_and_merge(self) -> None:
        first = ReflectivityRegressionAccumulator()
        second = ReflectivityRegressionAccumulator()
        first.update([11.0, 18.0], [10.0, 20.0], np.ones(2, dtype=bool))
        second.update([30.0], [30.0], np.ones(1, dtype=bool))
        first.merge(second)
        result = first.compute()
        self.assertEqual(result["count"], 3)
        self.assertAlmostEqual(result["mae_dbz"], 1.0)
        self.assertAlmostEqual(result["rmse_dbz"], math.sqrt(5.0 / 3.0))
        self.assertAlmostEqual(result["bias_dbz"], -1.0 / 3.0)
        self.assertGreater(result["pearson_r"], 0.98)

    def test_selected_nonfinite_value_is_rejected(self) -> None:
        metric = ReflectivityRegressionAccumulator()
        with self.assertRaisesRegex(ValueError, "finite"):
            metric.update([np.nan], [1.0], np.array([True]))


class FractionSkillTests(unittest.TestCase):
    def test_horizontal_windows_do_not_mix_height(self) -> None:
        values = np.zeros((3, 3, 2), dtype=np.float32)
        values[1, 1, 0] = 1.0
        result = horizontal_window_sum(values, radius=1)
        np.testing.assert_array_equal(result[:, :, 0], np.ones((3, 3)))
        np.testing.assert_array_equal(result[:, :, 1], np.zeros((3, 3)))

    def test_fss_rewards_nearby_displacement_at_larger_scale(self) -> None:
        target = np.zeros((5, 5, 1), dtype=bool)
        prediction = np.zeros_like(target)
        target[2, 2, 0] = True
        prediction[2, 3, 0] = True
        domain = np.ones_like(target)
        exact = FractionSkillAccumulator(radius=0)
        neighborhood = FractionSkillAccumulator(radius=1)
        exact.update(prediction, target, domain)
        neighborhood.update(prediction, target, domain)
        self.assertAlmostEqual(exact.compute()["fss"], 0.0)
        self.assertGreater(neighborhood.compute()["fss"], 0.0)

    def test_joint_metric_uses_common_support_for_dbz(self) -> None:
        target_support = np.array([True, True, False, False], dtype=bool)
        predicted_support = np.array([True, False, True, False], dtype=bool)
        domain = np.ones(4, dtype=bool)
        prediction = np.array([12.0, np.nan, 99.0, np.nan])
        target = np.array([10.0, 20.0, np.nan, np.nan])
        metric = Stage2ReflectivityMetrics(fss_radii=())
        metric.update(
            prediction, predicted_support, target, target_support, domain
        )
        result = metric.compute()
        self.assertEqual(result["support"]["true_positive"], 1)
        self.assertEqual(
            result["reflectivity_on_common_support"]["count"], 1
        )
        self.assertAlmostEqual(
            result["reflectivity_on_common_support"]["mae_dbz"], 2.0
        )
        self.assertIsNone(
            finite_metrics_for_json({"undefined": math.nan})["undefined"]
        )


if __name__ == "__main__":
    unittest.main()
