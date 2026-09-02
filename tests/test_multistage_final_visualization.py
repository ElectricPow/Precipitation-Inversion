"""Synthetic tests for the unified Stage-1/Stage-2/cascade report figures.

The real report operates on complete orbits and several large checkpoints.  These
tests deliberately use one tiny in-memory orbit so that bin boundaries, mask
semantics, validation, metric direction, and figure generation can be checked
without accessing the experiment outputs or a GPU.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import matplotlib
import numpy as np


# Tests may run on a headless training server.  Select the non-interactive backend
# before importing the report script (which imports matplotlib.pyplot).
matplotlib.use("Agg", force=True)
from matplotlib import pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.visualize_multistage_final_models import (  # noqa: E402
    RAIN_INTENSITY_BINS,
    compute_regression_metrics,
    plot_rain_intensity_analysis,
    plot_stage2_audit,
    plot_vertical_structure,
    target_intensity_masks,
    validate_orbit_bundle,
)


def _synthetic_bundle() -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    """Return a small but physically consistent unified orbit bundle.

    All volumetric fields use ``(nscan, nray, z)``.  The values deliberately span
    all five target-rain bins, all three Stage-2 audit regions, and every height.
    """

    nscan, nray, z_size = 6, 5, 6
    shape = (nscan, nray, z_size)
    heights = np.linspace(0.125, 2.625, z_size, dtype=np.float32)

    # Broadcasted coordinates retain the horizontal (nscan, nray) geometry.
    lat = (
        np.linspace(20.0, 21.0, nscan, dtype=np.float32)[:, None]
        + np.zeros((nscan, nray), dtype=np.float32)
    )
    lon = (
        np.linspace(110.0, 111.0, nray, dtype=np.float32)[None, :]
        + np.zeros((nscan, nray), dtype=np.float32)
    )

    rain_levels = np.asarray(
        [0.25, 0.75, 1.0, 3.0, 5.0, 8.0, 10.0, 20.0, 30.0, 45.0],
        dtype=np.float32,
    )
    target_rain = np.resize(rain_levels, int(np.prod(shape))).reshape(shape)
    qc = np.ones(shape, dtype=bool)
    # Exclude one otherwise valid voxel to verify that plots respect the supplied
    # common evaluation domain rather than silently using every finite value.
    qc[0, 0, 0] = False
    positive = qc & (target_rain > 0.0)

    dpr_support = np.ones(shape, dtype=bool)
    target_dbz = 12.0 + 0.75 * target_rain
    target_dbz = target_dbz.astype(np.float32)

    gr_value_mask = np.zeros(shape, dtype=bool)
    gr_value_mask[::2, :, ::2] = True
    gr_interp_mask = gr_value_mask.copy()
    gr_interp_mask[:, :, 1::2] = True
    anchor = gr_value_mask & dpr_support
    gap = (~anchor) & gr_interp_mask & dpr_support
    outside = (~gr_interp_mask) & dpr_support

    gr_sparse = np.full(shape, np.nan, dtype=np.float32)
    gr_sparse[gr_value_mask] = target_dbz[gr_value_mask] - 2.0
    gr_interp = np.full(shape, np.nan, dtype=np.float32)
    gr_interp[gr_interp_mask] = target_dbz[gr_interp_mask] - 1.0

    w1p25_dbz = target_dbz * np.float32(0.82) + np.float32(2.0)
    r1_o_dbz = target_dbz * np.float32(0.94) + np.float32(0.5)
    support_probability = np.where(dpr_support, 0.9, 0.1).astype(np.float32)
    predicted_support = support_probability >= 0.8

    perfect_rain = target_rain.copy()
    biased_rain = target_rain * np.float32(0.5)
    fields: dict[str, np.ndarray] = {
        "target_rain_mm_h": target_rain,
        "target_dbz": target_dbz,
        "dbz_gr_sparse": gr_sparse,
        "dbz_gr_interp": gr_interp,
        "stage2_w1p25_dbz": w1p25_dbz,
        "stage2_w1p25_support_probability": support_probability,
        "stage2_r1_o_dbz": r1_o_dbz,
        "reliable_positive_mask": positive,
        "qc_label_mask": qc,
        "dpr_support": dpr_support,
        "stage2_support_domain": np.ones(shape, dtype=bool),
        "gr_value_mask": gr_value_mask,
        "gr_interp_mask": gr_interp_mask,
        "anchor_mask": anchor,
        "gap_mask": gap,
        "outside_mask": outside,
        "w1p25_predicted_support": predicted_support,
        "heights_km": heights,
        "lat": lat,
        "lon": lon,
        "cfb": np.full((nscan, nray), 1, dtype=np.int16),
        "precipitation_type": np.resize(
            np.asarray([0, 1, 2], dtype=np.int16), nscan * nray
        ).reshape(nscan, nray),
        "rain__perfect": perfect_rain,
        "rain__biased": biased_rain,
        "input_support__perfect": dpr_support.copy(),
        "output_support__perfect": dpr_support.copy(),
        "input_support__biased": predicted_support.copy(),
        "output_support__biased": predicted_support.copy(),
    }
    modes: list[dict[str, object]] = [
        {
            "slug": "perfect",
            "display_name": "DPR -> Stage 1",
            "rain_field": "rain__perfect",
            "input_support_field": "input_support__perfect",
            "output_support_field": "output_support__perfect",
            "deployable": False,
        },
        {
            "slug": "biased",
            "display_name": "Stage 2 -> Stage 1",
            "rain_field": "rain__biased",
            "input_support_field": "input_support__biased",
            "output_support_field": "output_support__biased",
            "deployable": True,
        },
    ]
    return fields, modes


class MultiStageFinalVisualizationTests(unittest.TestCase):
    def test_target_intensity_bins_are_disjoint_and_include_boundaries(self) -> None:
        # Boundary convention is [0,1), [1,5), [5,10), [10,30), [30,+inf).
        target = np.asarray(
            [0.0, 0.999, 1.0, 4.999, 5.0, 9.999, 10.0, 29.999, 30.0, 50.0,
             np.nan, 2.0],
            dtype=np.float32,
        ).reshape(2, 2, 3)
        evaluation_mask = np.ones_like(target, dtype=bool)
        evaluation_mask.reshape(-1)[-1] = False

        masks = target_intensity_masks(target, evaluation_mask)

        self.assertEqual(len(RAIN_INTENSITY_BINS), 5)
        self.assertEqual(len(masks), 5)
        ordered_masks = list(masks.values())
        self.assertTrue(all(mask.dtype == np.bool_ for mask in ordered_masks))
        self.assertTrue(all(mask.shape == target.shape for mask in ordered_masks))
        np.testing.assert_array_equal(
            [int(mask.sum()) for mask in ordered_masks], [2, 2, 2, 2, 2]
        )
        stacked = np.stack(ordered_masks, axis=0)
        self.assertFalse(np.any(stacked.sum(axis=0) > 1))
        expected_union = evaluation_mask & np.isfinite(target)
        np.testing.assert_array_equal(np.any(stacked, axis=0), expected_union)

        with self.assertRaises(TypeError):
            target_intensity_masks(target, evaluation_mask.astype(np.uint8))

    def test_regression_metrics_use_only_finite_masked_pairs(self) -> None:
        target = np.asarray([0.0, 1.0, 2.0, 3.0, np.nan], dtype=np.float64)
        prediction = np.asarray([1.0, 1.0, 4.0, 1.0, 8.0], dtype=np.float64)
        mask = np.ones(target.shape, dtype=bool)

        metrics = compute_regression_metrics(target, prediction, mask)

        self.assertEqual(metrics["count"], 4)
        self.assertAlmostEqual(metrics["mae"], 1.25)
        self.assertAlmostEqual(metrics["rmse"], 1.5)
        # Bias is prediction minus target: (1 + 0 + 2 - 2) / 4.
        self.assertAlmostEqual(metrics["bias"], 0.25)
        self.assertAlmostEqual(
            metrics["pearson_r"],
            float(np.corrcoef(target[:4], prediction[:4])[0, 1]),
        )

        with self.assertRaises(ValueError):
            compute_regression_metrics(target, prediction[:-1], mask)
        with self.assertRaises(TypeError):
            compute_regression_metrics(target, prediction, mask.astype(np.int8))

    def test_bundle_validation_rejects_shape_and_mask_dtype_errors(self) -> None:
        fields, _ = _synthetic_bundle()
        validate_orbit_bundle(fields)

        bad_shape = dict(fields)
        bad_shape["stage2_r1_o_dbz"] = fields["stage2_r1_o_dbz"][:-1]
        with self.assertRaises(ValueError):
            validate_orbit_bundle(bad_shape)

        bad_mask = dict(fields)
        bad_mask["gap_mask"] = fields["gap_mask"].astype(np.uint8)
        with self.assertRaises(TypeError):
            validate_orbit_bundle(bad_mask)

        missing = dict(fields)
        del missing["heights_km"]
        with self.assertRaises((KeyError, ValueError)):
            validate_orbit_bundle(missing)

    def test_synthetic_orbit_generates_both_complete_figures(self) -> None:
        fields, modes = _synthetic_bundle()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            intensity_figure, intensity_metrics = plot_rain_intensity_analysis(
                fields,
                modes,
                sample_id="synthetic",
            )
            audit_figure, audit_metrics = plot_stage2_audit(
                fields,
                sample_id="synthetic",
                height_km=1.5,
                max_points=500,
                rng=np.random.default_rng(7),
            )
            try:
                intensity_path = root / "rain_intensity.png"
                audit_path = root / "stage2_audit.png"
                intensity_figure.savefig(intensity_path, dpi=30)
                audit_figure.savefig(audit_path, dpi=30)
                self.assertTrue(intensity_path.is_file())
                self.assertTrue(audit_path.is_file())
                self.assertGreater(intensity_path.stat().st_size, 0)
                self.assertGreater(audit_path.stat().st_size, 0)

                self.assertEqual(set(intensity_metrics), {"perfect", "biased"})
                self.assertAlmostEqual(
                    intensity_metrics["perfect"]["overall"]["mae"], 0.0
                )
                self.assertEqual(
                    len(intensity_metrics["perfect"]["by_target_intensity"]), 5
                )
                self.assertIn("regions", audit_metrics)
                self.assertEqual(
                    set(audit_metrics["regions"]), {"anchor", "gap", "outside"}
                )
                self.assertGreater(audit_metrics["regions"]["anchor"]["count"], 0)
                self.assertGreater(audit_metrics["regions"]["gap"]["count"], 0)
                self.assertGreater(audit_metrics["regions"]["outside"]["count"], 0)
            finally:
                plt.close(intensity_figure)
            plt.close(audit_figure)

            vertical_figure, vertical_metrics = plot_vertical_structure(
                fields, modes, sample_id="synthetic-orbit"
            )
            vertical_path = root / "vertical.png"
            vertical_figure.savefig(vertical_path, dpi=50)
            self.assertTrue(vertical_path.is_file())
            self.assertGreater(vertical_path.stat().st_size, 0)
            self.assertIn("physical_drdz", vertical_metrics)
            plt.close(vertical_figure)


if __name__ == "__main__":
    unittest.main()
