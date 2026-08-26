"""Tests for deterministic selection and prediction diagnostic plots."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.visualize_stage1_test_predictions import (  # noqa: E402
    paired_metrics,
    plot_sample_diagnostics,
    select_file_ids,
)


class PredictionVisualizationTests(unittest.TestCase):
    def test_selection_is_fixed_unique_and_limited_to_eligible_orbits(self) -> None:
        eligible = [1, 3, 5, 8]
        first = select_file_ids(10, 3, 2026, eligible_ids=eligible)
        second = select_file_ids(10, 3, 2026, eligible_ids=eligible)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        self.assertEqual(len(set(first)), 3)
        self.assertTrue(set(first).issubset(eligible))
        with self.assertRaises(ValueError):
            select_file_ids(10, 2, 1, eligible_ids=[])

    def test_masked_metrics_use_prediction_minus_target_bias(self) -> None:
        target = np.array([1.0, 2.0, 99.0])
        prediction = np.array([2.0, 0.0, -99.0])
        result = paired_metrics(target, prediction, np.array([True, True, False]))
        self.assertEqual(result["count"], 2)
        self.assertAlmostEqual(result["mae"], 1.5)
        self.assertAlmostEqual(result["rmse"], np.sqrt(2.5))
        self.assertAlmostEqual(result["bias"], -0.5)

    def test_synthetic_orbit_produces_complete_diagnostic_figure(self) -> None:
        nscan, nray, z_size = 5, 4, 6
        z = np.linspace(0.5, 5.5, z_size, dtype=np.float32)
        lat = np.linspace(20, 21, nscan)[:, None] + np.zeros((nscan, nray))
        lon = np.linspace(110, 111, nray)[None, :] + np.zeros((nscan, nray))
        target = np.zeros((nscan, nray, z_size), dtype=np.float32)
        target[1:4, 1:3, :4] = np.linspace(0.1, 8.0, 24).reshape(3, 2, 4)
        prediction = target * 0.8 + 0.05
        evaluation = np.ones_like(target, dtype=bool)
        positive = evaluation & (target > 0)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "diagnostics.png"
            result = plot_sample_diagnostics(
                target=target,
                prediction=prediction,
                evaluation_mask=evaluation,
                positive_mask=positive,
                z=z,
                lat=lat,
                lon=lon,
                sample_id="synthetic",
                height_km=2.0,
                max_scatter_points=1000,
                rng=np.random.default_rng(7),
                destination=destination,
                dpi=35,
            )
            self.assertTrue(destination.is_file())
            self.assertGreater(destination.stat().st_size, 0)
            self.assertEqual(result["positive_target_metrics"]["count"], 24)
            self.assertEqual(len(result["vertical_rmse_positive"]), z_size)


if __name__ == "__main__":
    unittest.main()
