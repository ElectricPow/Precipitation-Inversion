"""Tests for deployable Stage-2 sparse-GR geometry features."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from precipitation_inversion.data.stage2_geometry import (  # noqa: E402
    DEFAULT_GR_LOCAL_DENSITY_RADIUS,
    clipped_horizontal_chebyshev_distance,
    dilate_horizontal_once,
    horizontal_observation_count,
    scaled_horizontal_observation_density,
    scaled_horizontal_distance_to_observation,
)


class Stage2GeometryTests(unittest.TestCase):
    def test_distance_is_horizontal_per_height_and_clipped(self) -> None:
        # (nscan=5,nray=5,z=3). Each of the first two levels has a different
        # observation; the third level has none and must stay maximally far.
        observed = np.zeros((5, 5, 3), dtype=bool)
        observed[2, 2, 0] = True
        observed[0, 0, 1] = True
        original = observed.copy()

        distance = clipped_horizontal_chebyshev_distance(
            observed, max_distance=3
        )
        self.assertEqual(distance.dtype, np.uint16)
        self.assertEqual(distance.shape, observed.shape)
        self.assertEqual(int(distance[2, 2, 0]), 0)
        self.assertEqual(int(distance[2, 3, 0]), 1)
        self.assertEqual(int(distance[0, 0, 0]), 2)
        self.assertEqual(int(distance[4, 4, 1]), 3)
        np.testing.assert_array_equal(distance[:, :, 2], 3)
        np.testing.assert_array_equal(observed, original)

    def test_scaled_distance_maps_observed_to_zero_and_unknown_to_one(self) -> None:
        observed = np.zeros((4, 4, 2), dtype=bool)
        observed[1, 1, 0] = True
        scaled = scaled_horizontal_distance_to_observation(
            observed, max_distance=2
        )
        self.assertEqual(scaled.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(scaled)))
        self.assertGreaterEqual(float(scaled.min()), 0.0)
        self.assertLessEqual(float(scaled.max()), 1.0)
        self.assertEqual(float(scaled[1, 1, 0]), 0.0)
        self.assertEqual(float(scaled[1, 2, 0]), 0.5)
        self.assertEqual(float(scaled[3, 3, 0]), 1.0)
        np.testing.assert_array_equal(scaled[:, :, 1], 1.0)

    def test_one_step_dilation_does_not_cross_height(self) -> None:
        observed = np.zeros((3, 3, 2), dtype=bool)
        observed[1, 1, 0] = True
        dilated = dilate_horizontal_once(observed)
        self.assertTrue(bool(dilated[:, :, 0].all()))
        self.assertFalse(bool(dilated[:, :, 1].any()))

    def test_invalid_mask_and_distance_are_rejected(self) -> None:
        with self.assertRaisesRegex(TypeError, "boolean"):
            scaled_horizontal_distance_to_observation(
                np.zeros((2, 2, 2), dtype=np.float32)
            )
        with self.assertRaisesRegex(ValueError, "shape"):
            scaled_horizontal_distance_to_observation(
                np.zeros((2, 2), dtype=bool)
            )
        with self.assertRaisesRegex(ValueError, "positive"):
            scaled_horizontal_distance_to_observation(
                np.zeros((2, 2, 2), dtype=bool), max_distance=0
            )
        with self.assertRaisesRegex(TypeError, "integer"):
            scaled_horizontal_distance_to_observation(
                np.zeros((2, 2, 2), dtype=bool), max_distance=1.5
            )

    def test_local_density_is_fixed_area_horizontal_and_per_height(self) -> None:
        # Source mask: (nscan=5,nray=5,z=2). The first height has two direct
        # observations; the second has none. Radius 1 means a fixed 3x3 area.
        observed = np.zeros((5, 5, 2), dtype=bool)
        observed[2, 2, 0] = True
        observed[2, 3, 0] = True
        original = observed.copy()

        count = horizontal_observation_count(observed, radius=1)
        density = scaled_horizontal_observation_density(observed, radius=1)

        self.assertEqual(count.dtype, np.uint32)
        self.assertEqual(density.dtype, np.float32)
        self.assertEqual(count.shape, observed.shape)
        self.assertEqual(int(count[2, 2, 0]), 2)
        self.assertEqual(int(count[2, 3, 0]), 2)
        self.assertEqual(int(count[0, 0, 0]), 0)
        self.assertAlmostEqual(float(density[2, 2, 0]), 2.0 / 9.0)
        np.testing.assert_array_equal(count[:, :, 1], 0)
        np.testing.assert_array_equal(density[:, :, 1], 0.0)
        np.testing.assert_array_equal(observed, original)

    def test_default_density_uses_five_by_five_and_fixed_boundary_denominator(self) -> None:
        self.assertEqual(DEFAULT_GR_LOCAL_DENSITY_RADIUS, 2)
        observed = np.zeros((3, 3, 1), dtype=bool)
        observed[0, 0, 0] = True
        density = scaled_horizontal_observation_density(observed)

        # Even at the physical corner, the single observation is divided by
        # 25 rather than only by the in-domain portion of the 5x5 window.
        self.assertAlmostEqual(float(density[0, 0, 0]), 1.0 / 25.0)
        self.assertAlmostEqual(float(density[2, 2, 0]), 1.0 / 25.0)
        self.assertTrue(np.all(np.isfinite(density)))
        self.assertGreaterEqual(float(density.min()), 0.0)
        self.assertLessEqual(float(density.max()), 1.0)

    def test_density_radius_zero_and_invalid_arguments(self) -> None:
        observed = np.zeros((2, 2, 1), dtype=bool)
        observed[1, 0, 0] = True
        density = scaled_horizontal_observation_density(observed, radius=0)
        np.testing.assert_array_equal(density, observed.astype(np.float32))

        with self.assertRaisesRegex(TypeError, "boolean"):
            scaled_horizontal_observation_density(
                np.zeros((2, 2, 1), dtype=np.float32)
            )
        with self.assertRaisesRegex(TypeError, "integer"):
            scaled_horizontal_observation_density(observed, radius=1.5)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            scaled_horizontal_observation_density(observed, radius=-1)

    def test_integral_image_density_matches_brute_force_at_edges(self) -> None:
        rng = np.random.default_rng(2026)
        observed = rng.random((6, 4, 3)) < 0.3
        for radius in (0, 1, 2):
            expected = np.zeros(observed.shape, dtype=np.uint32)
            for scan in range(observed.shape[0]):
                for ray in range(observed.shape[1]):
                    section = observed[
                        max(0, scan - radius) : scan + radius + 1,
                        max(0, ray - radius) : ray + radius + 1,
                        :,
                    ]
                    expected[scan, ray] = section.sum(axis=(0, 1))
            actual = horizontal_observation_count(observed, radius=radius)
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_allclose(
                scaled_horizontal_observation_density(observed, radius=radius),
                expected.astype(np.float32) / float((2 * radius + 1) ** 2),
            )


if __name__ == "__main__":
    unittest.main()
