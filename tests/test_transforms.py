"""Tests for online per-height statistics and reversible transforms."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from precipitation_inversion.data.transforms import (  # noqa: E402
    PerLevelRunningStats,
    PerLevelStandardizer,
    clip_below_threshold,
    fill_missing_with_mask,
    inverse_log1p_rain,
    log1p_rain,
)


class RunningStatsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.values = np.array(
            [
                [[1.0, 10.0, np.nan], [3.0, 14.0, np.nan]],
                [[5.0, 18.0, np.nan], [7.0, np.nan, np.nan]],
            ]
        )

    def test_online_statistics_match_direct_numpy_calculation(self) -> None:
        stats = PerLevelRunningStats.empty(3).update(self.values)
        np.testing.assert_array_equal(stats.count, [4, 3, 0])
        np.testing.assert_allclose(stats.mean[:2], [4.0, 14.0])
        np.testing.assert_allclose(
            stats.std()[:2],
            [np.std([1, 3, 5, 7]), np.std([10, 14, 18])],
        )
        np.testing.assert_allclose(stats.minimum[:2], [1.0, 10.0])
        np.testing.assert_allclose(stats.maximum[:2], [7.0, 18.0])
        self.assertTrue(np.isnan(stats.std()[2]))

    def test_chunked_and_merged_statistics_equal_single_update(self) -> None:
        direct = PerLevelRunningStats.empty(3).update(self.values)
        chunked = PerLevelRunningStats.empty(3)
        chunked.update(self.values[:1]).update(self.values[1:])
        left = PerLevelRunningStats.empty(3).update(self.values[:1])
        right = PerLevelRunningStats.empty(3).update(self.values[1:])
        left.merge(right)
        for candidate in (chunked, left):
            np.testing.assert_array_equal(candidate.count, direct.count)
            np.testing.assert_allclose(candidate.mean, direct.mean)
            np.testing.assert_allclose(candidate.m2, direct.m2)
            np.testing.assert_allclose(candidate.minimum, direct.minimum)
            np.testing.assert_allclose(candidate.maximum, direct.maximum)

    def test_selection_mask_is_combined_with_finite_values(self) -> None:
        selected = np.zeros(self.values.shape, dtype=bool)
        selected[0] = True
        stats = PerLevelRunningStats.empty(3).update(
            self.values, valid_mask=selected
        )
        np.testing.assert_array_equal(stats.count, [2, 2, 0])
        np.testing.assert_allclose(stats.mean[:2], [2.0, 12.0])

    def test_json_dictionary_uses_null_for_empty_levels(self) -> None:
        stats = PerLevelRunningStats.empty(3).update(self.values)
        value = stats.to_dict(heights_km=np.array([0.125, 0.375, 0.625]))
        self.assertIsNone(value["mean"][2])
        self.assertIsNone(value["std"][2])
        self.assertEqual(value["empty_level_count"], 1)
        # Strict JSON serialization must not depend on non-standard NaN tokens.
        encoded = json.dumps(value, allow_nan=False)
        self.assertIn("null", encoded)

    def test_invalid_shapes_are_rejected(self) -> None:
        stats = PerLevelRunningStats.empty(3)
        with self.assertRaisesRegex(ValueError, "height levels"):
            stats.update(np.zeros((2, 4)))
        with self.assertRaisesRegex(ValueError, "valid_mask shape"):
            stats.update(np.zeros((2, 3)), valid_mask=np.ones((2, 2), dtype=bool))


class TransformTests(unittest.TestCase):
    def test_standardization_preserves_missingness_and_is_reversible(self) -> None:
        standardizer = PerLevelStandardizer(
            mean=np.array([2.0, 10.0, np.nan]),
            std=np.array([1.0, 2.0, np.nan]),
        )
        values = np.array([[1.0, np.nan, 5.0], [3.0, 12.0, 6.0]])
        transformed, valid = standardizer.transform(values)
        np.testing.assert_array_equal(
            valid, [[True, False, False], [True, True, False]]
        )
        np.testing.assert_allclose(
            transformed[valid], [-1.0, 1.0, 1.0]
        )
        self.assertTrue(np.isnan(transformed[~valid]).all())
        restored = standardizer.inverse(transformed)
        np.testing.assert_allclose(restored[valid], values[valid])
        self.assertTrue(np.isnan(restored[~valid]).all())

    def test_standardization_can_fill_only_after_returning_mask(self) -> None:
        standardizer = PerLevelStandardizer(mean=[0.0, 0.0], std=[1.0, 0.0])
        transformed, valid = standardizer.transform(
            [[2.0, 3.0], [np.nan, 4.0]], fill_value=-9.0
        )
        np.testing.assert_array_equal(valid, [[True, False], [False, False]])
        np.testing.assert_allclose(transformed, [[2.0, -9.0], [-9.0, -9.0]])

    def test_log1p_rain_round_trip_and_mask(self) -> None:
        rain = np.array([0.0, 1.0, 10.0, np.nan, -1.0])
        transformed, valid = log1p_rain(rain)
        np.testing.assert_array_equal(valid, [True, True, True, False, False])
        restored = inverse_log1p_rain(transformed)
        np.testing.assert_allclose(restored[valid], rain[valid], rtol=1e-6)
        self.assertTrue(np.isnan(restored[~valid]).all())

    def test_threshold_clipping_does_not_fill_missing_values(self) -> None:
        values = np.array([-5.0, 8.9, 9.0, 20.0, np.nan])
        clipped = clip_below_threshold(values, threshold=9.0, replacement=0.0)
        np.testing.assert_allclose(clipped[:4], [0.0, 0.0, 9.0, 20.0])
        self.assertTrue(np.isnan(clipped[4]))

    def test_fill_missing_returns_original_observation_mask(self) -> None:
        filled, observed = fill_missing_with_mask([1.0, np.nan, np.inf], fill_value=0)
        np.testing.assert_array_equal(observed, [True, False, False])
        np.testing.assert_allclose(filled, [1.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()

