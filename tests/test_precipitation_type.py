"""Tests for label-only DPR precipitation-type targets and class balancing."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from precipitation_inversion.data.precipitation_type import (  # noqa: E402
    TYPE_IGNORE_INDEX,
    build_type_target_and_mask,
    inverse_sqrt_class_weights,
)


class PrecipitationTypeDataTests(unittest.TestCase):
    def test_target_requires_class_core_and_native_dpr_support(self) -> None:
        codes = np.array([[1, 2, 3], [-1111, -9999, 1]])
        core = np.array([[True, True, False], [True, True, True]])
        support = np.array([[True, False, True], [True, True, True]])
        target, mask = build_type_target_and_mask(
            codes, core_profile_mask=core, dpr_profile_support=support
        )
        np.testing.assert_array_equal(
            mask, [[True, False, False], [False, False, True]]
        )
        np.testing.assert_array_equal(
            target,
            [[0, TYPE_IGNORE_INDEX, TYPE_IGNORE_INDEX],
             [TYPE_IGNORE_INDEX, TYPE_IGNORE_INDEX, 0]],
        )

    def test_inverse_sqrt_weights_have_count_weighted_mean_one(self) -> None:
        counts = np.asarray([100, 25, 4], dtype=np.float64)
        weights = inverse_sqrt_class_weights(counts)
        self.assertGreater(weights[2], weights[1])
        self.assertGreater(weights[1], weights[0])
        self.assertAlmostEqual(float(np.sum(weights * counts) / counts.sum()), 1.0, places=6)
        with self.assertRaisesRegex(ValueError, "at least one"):
            inverse_sqrt_class_weights([100, 0, 5])


if __name__ == "__main__":
    unittest.main()
