"""Tests for deterministic Stage-2 orbit diagnostic visualizations."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.visualize_stage2_test_predictions import (  # noqa: E402
    plot_stage2_orbit_diagnostics,
    select_file_ids,
)


class Stage2PredictionVisualizationTests(unittest.TestCase):
    def test_file_selection_is_fixed_unique_and_eligible(self) -> None:
        eligible = [0, 2, 4, 6]
        first = select_file_ids(8, 3, 2026, eligible_ids=eligible)
        second = select_file_ids(8, 3, 2026, eligible_ids=eligible)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        self.assertTrue(set(first).issubset(eligible))
        with self.assertRaises(ValueError):
            select_file_ids(8, 2, 1, eligible_ids=[])

    def test_synthetic_orbit_draws_maps_sections_distribution_and_metrics(self) -> None:
        shape = (7, 5, 6)
        z = np.linspace(0.125, 2.625, shape[-1])
        domain = np.ones(shape, dtype=bool)
        target_support = np.zeros(shape, dtype=bool)
        target_support[1:6, 1:4, 1:5] = True
        probability = np.full(shape, 0.1, dtype=np.float32)
        probability[target_support] = 0.8
        target_dbz = np.zeros(shape, dtype=np.float32)
        target_dbz[target_support] = np.linspace(
            10.0, 40.0, int(target_support.sum())
        )
        prediction_dbz = target_dbz * 0.9 + 1.0
        gr = np.zeros(shape, dtype=bool)
        gr[1:6:2, 1:4, 1:5] = True
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "stage2.png"
            result = plot_stage2_orbit_diagnostics(
                support_probability=probability,
                prediction_dbz=prediction_dbz,
                target_support=target_support,
                target_dbz=target_dbz,
                support_domain=domain,
                gr_value_mask=gr,
                heights_km=z,
                threshold=0.5,
                sample_id="synthetic",
                height_km=1.5,
                max_scatter_points=1000,
                rng=np.random.default_rng(3),
                destination=destination,
                dpi=35,
            )
            self.assertTrue(destination.is_file())
            self.assertGreater(destination.stat().st_size, 0)
            self.assertEqual(
                result["target_support_voxels"], int(target_support.sum())
            )
            self.assertAlmostEqual(result["metrics"]["support"]["csi"], 1.0)
            self.assertEqual(len(result["vertical_mae_dbz"]), shape[-1])


if __name__ == "__main__":
    unittest.main()
