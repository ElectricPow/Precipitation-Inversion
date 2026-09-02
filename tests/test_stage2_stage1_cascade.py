"""Tests for label-free Stage-2 -> Stage-1 orbit conversion."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from precipitation_inversion.data.transforms import PerLevelStandardizer  # noqa: E402
from precipitation_inversion.inference.stage2_stage1_cascade import (  # noqa: E402
    iter_stage1_cascade_windows,
    predict_stage1_from_reflectivity_orbit,
)
from scripts.evaluate_stage2_stage1_cascade import (  # noqa: E402
    FACTORIAL_AUDIT_FORMAT,
    build_factorial_2x2_audit,
    build_mode_definitions,
    factorial_2x2_rows,
    normalize_mode_slug,
    parse_stage2_run_specs,
    prepare_true_dpr_for_predicted_support,
)


class ConstantRainModel(nn.Module):
    """Return log1p(1 mm/h) at every model voxel."""

    def __init__(self) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(()))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.ones(
            (inputs.shape[0], 1, *inputs.shape[2:]),
            device=inputs.device,
            dtype=inputs.dtype,
        ) * (float(np.log(2.0)) + self.dummy)


class Stage2Stage1CascadeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.z = np.array([0.5, 1.0], dtype=np.float32)
        self.standardizer = PerLevelStandardizer(
            mean=np.array([10.0, 20.0]), std=np.array([2.0, 4.0])
        )
        self.dbz = np.zeros((5, 3, 2), dtype=np.float32)
        for scan in range(5):
            self.dbz[scan, :, 0] = 10.0 + 2.0 * scan
            self.dbz[scan, :, 1] = 20.0 + 4.0 * scan
        self.valid = np.ones_like(self.dbz, dtype=bool)
        self.valid[1, 1, :] = False
        self.dbz[~self.valid] = np.nan
        self.clutter = np.zeros_like(self.valid)
        self.clutter[:, :, 0] = True

    def test_windows_preserve_height_and_use_neutral_missing_fill(self) -> None:
        windows = list(
            iter_stage1_cascade_windows(
                self.dbz,
                self.valid,
                heights_km=self.z,
                standardizer=self.standardizer,
                cfb_clutter=self.clutter,
                core_size=2,
                halo_size=1,
                horizontal_multiple=2,
            )
        )
        self.assertEqual(len(windows), 3)
        self.assertEqual(windows[0].inputs.shape, (3, 4, 4, 2))
        self.assertEqual(windows[0].output_mask.shape, (1, 4, 4, 2))
        self.assertEqual([item.core_length for item in windows], [2, 2, 1])
        # First core's first physical scan is at local index=halo=1.
        np.testing.assert_allclose(windows[0].inputs[0, 1, :3], 0.0)
        self.assertTrue(np.all(windows[0].inputs[1, 1, :3] == 1.0))
        # Orbit-external halo and padded ray both use neutral zero, with mask=0.
        self.assertTrue(np.all(windows[0].inputs[:, 0] == 0.0))
        self.assertTrue(np.all(windows[0].inputs[:, :, 3] == 0.0))

    def test_complete_orbit_keeps_misses_as_zero_and_excludes_clutter(self) -> None:
        result = predict_stage1_from_reflectivity_orbit(
            ConstantRainModel(),
            self.dbz,
            self.valid,
            heights_km=self.z,
            standardizer=self.standardizer,
            cfb_clutter=self.clutter,
            core_size=2,
            halo_size=1,
            horizontal_multiple=2,
            device="cpu",
            batch_size=2,
            use_amp=False,
        )
        self.assertEqual(result.rain_rate_mm_h.shape, self.dbz.shape)
        expected_support = self.valid & ~self.clutter
        np.testing.assert_array_equal(result.input_support, self.valid)
        np.testing.assert_array_equal(result.output_support, expected_support)
        np.testing.assert_allclose(result.rain_rate_mm_h[expected_support], 1.0)
        np.testing.assert_allclose(result.rain_rate_mm_h[~expected_support], 0.0)

    def test_invalid_selected_dbz_is_rejected_before_model_inference(self) -> None:
        invalid = self.dbz.copy()
        invalid[0, 0, 1] = np.nan
        with self.assertRaisesRegex(ValueError, "selected reflectivity"):
            predict_stage1_from_reflectivity_orbit(
                ConstantRainModel(),
                invalid,
                self.valid,
                heights_km=self.z,
                standardizer=self.standardizer,
                device="cpu",
                use_amp=False,
            )

    def test_true_dbz_predicted_support_uses_heightwise_neutral_false_positive(self) -> None:
        predicted_support = self.valid.copy()
        predicted_support[1, 1, :] = True  # False-positive DPR support.
        prepared = prepare_true_dpr_for_predicted_support(
            self.dbz,
            self.valid,
            predicted_support,
            self.standardizer,
        )
        # True observed dBZ values remain untouched.
        np.testing.assert_allclose(prepared[self.valid], self.dbz[self.valid])
        # Missing physical values use mu(z), hence transform to neutral zero.
        np.testing.assert_allclose(prepared[1, 1], self.standardizer.mean)
        standardized, _ = self.standardizer.transform(
            prepared, valid_mask=predicted_support, fill_value=0.0
        )
        np.testing.assert_allclose(standardized[1, 1], 0.0)

    def test_run_spec_requires_validation_threshold_and_safe_unique_slug(self) -> None:
        self.assertEqual(normalize_mode_slug("W1.25 Best"), "w1_25_best")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "best.pt"
            checkpoint.write_bytes(b"placeholder")
            threshold = root / "threshold.json"
            threshold.write_text(
                json.dumps({"threshold": 0.8, "selected_on_split": "val"}),
                encoding="utf-8",
            )
            result = parse_stage2_run_specs(
                [["D", str(checkpoint), str(threshold)]]
            )
            self.assertEqual(result[0].slug, "d")
            invalid = root / "test_threshold.json"
            invalid.write_text(
                json.dumps({"threshold": 0.8, "selected_on_split": "test"}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "split=val"):
                parse_stage2_run_specs([["bad", str(checkpoint), str(invalid)]])

    def test_strict_2x2_modes_and_effects_are_complete(self) -> None:
        metric_names = (
            "mae",
            "rmse",
            "bias",
            "r2",
            "pearson_r",
            "ccc",
            "mean_abs_gradient_ratio",
            "sign_agreement_fraction",
        )

        def result(value: float) -> dict[str, object]:
            rain = {name: value for name in metric_names[:6]}
            drdz = {name: value for name in metric_names}
            return {
                "reliable_positive": {"rain": {"all": dict(rain)}},
                "qc_label_domain_including_zero": {"rain": {"all": dict(rain)}},
                "physical_drdz_reliable_positive": {"all": drdz},
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "best.pt"
            checkpoint.write_bytes(b"placeholder")
            threshold = root / "threshold.json"
            threshold.write_text(
                json.dumps({"threshold": 0.8, "selected_on_split": "val"}),
                encoding="utf-8",
            )
            spec = parse_stage2_run_specs(
                [["W1.25", str(checkpoint), str(threshold)]]
            )[0]
            modes = build_mode_definitions([spec], include_gr_interp=False)
            self.assertEqual(
                list(modes),
                [
                    "dpr_oracle",
                    "w1_25_oracle_mask",
                    "w1_25_true_dbz_predicted_mask",
                    "w1_25_predicted_mask",
                ],
            )
            self.assertEqual(
                modes["w1_25_true_dbz_predicted_mask"]["factorial_axes"],
                {"reflectivity_value": "true", "support": "predicted"},
            )
            computed = {
                "dpr_oracle": result(1.0),
                "w1_25_oracle_mask": result(2.0),
                "w1_25_true_dbz_predicted_mask": result(3.0),
                "w1_25_predicted_mask": result(5.0),
            }
            audit = build_factorial_2x2_audit([spec], computed)
            self.assertEqual(audit["format"], FACTORIAL_AUDIT_FORMAT)
            rmse = audit["runs"]["w1_25"]["metrics"][
                "reliable_positive_rain"
            ]["rmse"]
            self.assertEqual(
                rmse["effects"][
                    "interaction_PP_minus_PT_minus_TP_plus_TT"
                ],
                1.0,
            )
            rows = factorial_2x2_rows(audit)
            self.assertEqual(len(rows), 20)


if __name__ == "__main__":
    unittest.main()
