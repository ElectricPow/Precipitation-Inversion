"""Unit tests for the canonical GR--DPR mask rules."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from precipitation_inversion.data.masks import (  # noqa: E402
    clutter_mask_from_cfb,
    gr_observation_mask,
    precipitation_label_mask,
    positive_rain_mask,
    profile_has_observation,
    to_float_array,
    valid_cfb_mask,
    zero_rain_mask,
)


class MaskTests(unittest.TestCase):
    def test_fill_values_are_missing_but_weak_dbz_is_valid(self) -> None:
        source = np.ma.array(
            [-9999.9, -30.0, 0.0, 12.0, 8.0],
            mask=[False, False, True, False, False],
        )
        converted = to_float_array(source)
        self.assertTrue(np.isnan(converted[0]))
        self.assertTrue(np.isnan(converted[2]))
        np.testing.assert_array_equal(
            gr_observation_mask(source), [False, True, False, True, True]
        )

    def test_masked_integer_array_can_represent_missing_values(self) -> None:
        source = np.ma.array([0, 1, -9999], mask=[False, True, False])
        converted = to_float_array(source)
        np.testing.assert_array_equal(np.isfinite(converted), [True, False, False])

    def test_zero_and_positive_rain_are_distinct(self) -> None:
        rain = np.array([np.nan, -9999.0, -1.0, 0.0, 0.1, 10.0])
        valid = precipitation_label_mask(rain, exclude_clutter=False)
        np.testing.assert_array_equal(valid, [False, False, False, True, True, True])
        np.testing.assert_array_equal(
            zero_rain_mask(rain, valid_mask=valid),
            [False, False, False, True, False, False],
        )
        np.testing.assert_array_equal(
            positive_rain_mask(rain, threshold=1.0, valid_mask=valid),
            [False, False, False, False, False, True],
        )

    def test_cfb_masks_only_bins_below_boundary(self) -> None:
        z = np.array([0.125, 0.375, 0.625, 0.875])
        cfb = np.array([[2, np.nan, 4, 1.5]])
        clutter = clutter_mask_from_cfb(cfb, z)
        np.testing.assert_array_equal(
            clutter,
            [[[True, True, False, False], [False] * 4, [False] * 4, [False] * 4]],
        )
        np.testing.assert_array_equal(
            valid_cfb_mask(cfb, z.size), [[True, False, False, False]]
        )

    def test_precipitation_mask_can_exclude_clutter(self) -> None:
        rain = np.zeros((1, 2, 4), dtype=float)
        z = np.array([0.125, 0.375, 0.625, 0.875])
        valid = precipitation_label_mask(rain, cfb=np.array([[2, 1]]), z=z)
        np.testing.assert_array_equal(
            valid,
            [[[False, False, True, True], [False, True, True, True]]],
        )

    def test_profile_observation_collapses_vertical_axis(self) -> None:
        mask = np.array([[[False, True], [False, False]]])
        np.testing.assert_array_equal(
            profile_has_observation(mask), [[True, False]]
        )


if __name__ == "__main__":
    unittest.main()
