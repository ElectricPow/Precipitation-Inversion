"""Tests for the controlled ``S2-R1-P-PartialConv`` experiment."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from precipitation_inversion.models.stage2_partial_completion_unet3d import (  # noqa: E402
    PartialConv3D,
    Stage2PartialCompletionUNet3D,
)
from scripts.train_stage2_r1_oracle_sparse_value import (  # noqa: E402
    R1_P_ARCHITECTURE,
    build_model,
    load_config,
    stage2_completion_contract,
    validate_r1_config,
)
from scripts.compare_stage2_r1_partial_conv import build_gate_comparison  # noqa: E402


class Stage2R1PartialConvTests(unittest.TestCase):
    @staticmethod
    def _region_rows(
        *, outside_rmse: float, outside_r: float, gap_rmse: float,
        unanchored_rmse: float, unanchored_r: float, strong_rmse: float,
        strong_bias: float,
    ) -> list[dict[str, object]]:
        return [
            {"region": "dpr_outside_proxy", "rmse_dbz": outside_rmse,
             "pearson_r": outside_r, "bias_dbz": -1.0},
            {"region": "dpr_gap_proxy", "rmse_dbz": gap_rmse,
             "pearson_r": 0.8, "bias_dbz": -0.5},
            {"region": "dpr_unanchored", "rmse_dbz": unanchored_rmse,
             "pearson_r": unanchored_r, "bias_dbz": -0.8},
            {"region": "dpr_dbz_ge35", "rmse_dbz": strong_rmse,
             "pearson_r": 0.2, "bias_dbz": strong_bias},
        ]

    @staticmethod
    def _cascade_row(mode: str, rain_r: float, drdz_r: float) -> list[dict[str, object]]:
        return [
            {"mode": "dpr_oracle", "positive_pearson_r": 0.86, "drdz_pearson_r": 0.70},
            {"mode": mode, "positive_pearson_r": rain_r, "drdz_pearson_r": drdz_r},
        ]

    def test_partial_convolution_ignores_invalid_values_and_propagates_mask(self) -> None:
        layer = PartialConv3D(
            1, 1, kernel_size=(3, 3, 1), padding=(1, 1, 0), bias=False
        )
        with torch.no_grad():
            layer.convolution.weight.fill_(1.0)
        mask = torch.zeros(1, 1, 5, 5, 1)
        mask[..., 2, 2, 0] = 1.0
        clean = torch.zeros_like(mask)
        clean[..., 2, 2, 0] = 1.0
        contaminated = clean.clone()
        contaminated[mask == 0] = 1.0e6

        clean_output, clean_mask = layer(clean, mask)
        dirty_output, dirty_mask = layer(contaminated, mask)

        torch.testing.assert_close(clean_output, dirty_output)
        self.assertTrue(torch.equal(clean_mask, dirty_mask))
        self.assertEqual(int(clean_mask.sum()), 9)
        # One valid unit in a 3x3 window is renormalized to the full-window sum.
        torch.testing.assert_close(
            clean_output[clean_mask.expand_as(clean_output)], torch.full((9,), 9.0)
        )

        empty_output, empty_mask = layer(torch.randn_like(clean), torch.zeros_like(mask))
        self.assertFalse(bool(empty_mask.any()))
        self.assertTrue(torch.equal(empty_output, torch.zeros_like(empty_output)))
        # Reproduce the low-precision arithmetic used by CUDA AMP.  Empty
        # windows must stay finite instead of evaluating 0 * infinity.
        half_layer = layer.half()
        half_output, _ = half_layer(clean.half(), torch.zeros_like(mask).half())
        self.assertTrue(bool(torch.isfinite(half_output).all()))

    def test_dense_geometry_branch_is_not_erased_by_empty_anchor_mask(self) -> None:
        model = Stage2PartialCompletionUNet3D(
            base_channels=2,
            channel_multipliers=(1, 2, 4),
            bottleneck_dropout=0.0,
        ).eval()
        first = torch.zeros(1, 4, 16, 16, 5)
        second = first.clone()
        second[:, 2:4] = torch.randn_like(second[:, 2:4])
        with torch.inference_mode():
            sparse_features, sparse_mask = model.sparse_stem(first[:, 0:1], first[:, 1:2])
            first_features, first_mask = model.forward_input_stem(first)
            second_features, second_mask = model.forward_input_stem(second)
        self.assertFalse(bool(sparse_mask.any()))
        self.assertTrue(torch.equal(sparse_features, torch.zeros_like(sparse_features)))
        self.assertFalse(bool(first_mask.any()))
        self.assertFalse(bool(second_mask.any()))
        self.assertFalse(torch.equal(first_features, second_features))
        self.assertTrue(bool(torch.isfinite(second_features).all()))

    def test_model_preserves_height_and_blocks_invalid_value_gradient(self) -> None:
        model = Stage2PartialCompletionUNet3D(
            base_channels=2,
            channel_multipliers=(1, 2, 4),
            bottleneck_dropout=0.0,
        )
        inputs = torch.randn(1, 4, 16, 16, 5, requires_grad=True)
        with torch.no_grad():
            inputs[:, 1].zero_()
            inputs[:, 1, 4:12, 4:12, 1:4] = 1.0
        output = model(inputs)["reflectivity"]
        self.assertEqual(tuple(output.shape), (1, 1, 16, 16, 5))
        output.square().mean().backward()
        self.assertIsNotNone(inputs.grad)
        anchor = inputs[:, 1:2] > 0.5
        invalid_value_gradient = inputs.grad[:, 0:1][~anchor]
        self.assertTrue(torch.equal(invalid_value_gradient, torch.zeros_like(invalid_value_gradient)))
        self.assertGreater(float(inputs.grad[:, 2:4].abs().sum()), 0.0)
        self.assertGreater(float(model.sparse_stem.horizontal.convolution.weight.grad.abs().sum()), 0.0)

    def test_config_diff_is_restricted_to_identity_and_partial_architecture(self) -> None:
        baseline = load_config(PROJECT_ROOT / "configs" / "stage2_r1_o_dpr_sparse_value.yaml")
        candidate = load_config(PROJECT_ROOT / "configs" / "stage2_r1_p_partial_conv.yaml")
        channels = validate_r1_config(candidate)
        task, architecture, checkpoint_format = stage2_completion_contract(candidate)
        model = build_model(candidate)

        self.assertEqual(len(channels), 4)
        self.assertEqual(task, "stage2_r1_p_partial_conv")
        self.assertEqual(architecture, R1_P_ARCHITECTURE)
        self.assertEqual(checkpoint_format, "stage2_r1_p_partial_conv_v1")
        self.assertIsInstance(model, Stage2PartialCompletionUNet3D)
        baseline_model = build_model(baseline)
        baseline_parameters = sum(value.numel() for value in baseline_model.parameters())
        candidate_parameters = sum(value.numel() for value in model.parameters())
        # The comparison changes mask handling, not overall model capacity.
        self.assertLess(
            abs(candidate_parameters - baseline_parameters) / baseline_parameters,
            0.001,
        )
        for section in (
            "data", "loss", "optimizer", "scheduler", "training",
            "evaluation", "postprocessing", "runtime",
        ):
            self.assertEqual(candidate[section], baseline[section], section)
        candidate_model = copy.deepcopy(candidate["model"])
        self.assertEqual(candidate_model.pop("architecture"), R1_P_ARCHITECTURE)
        self.assertEqual(candidate_model, baseline["model"])
        self.assertFalse(candidate["training"]["early_stopping"]["enabled"])
        self.assertEqual(candidate["training"]["epochs"], 60)
        self.assertEqual(candidate["training"]["checkpoint_every"], 10)

    def test_task_and_architecture_cannot_be_mixed(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "stage2_r1_p_partial_conv.yaml")
        config["model"]["architecture"] = "standard_unet3d"
        with self.assertRaises(ValueError):
            validate_r1_config(config)

    def test_pre_registered_gate_comparison_uses_all_subtasks(self) -> None:
        baseline = self._region_rows(
            outside_rmse=10.0, outside_r=0.30, gap_rmse=5.0,
            unanchored_rmse=7.0, unanchored_r=0.50,
            strong_rmse=12.0, strong_bias=-6.0,
        )
        candidate = self._region_rows(
            outside_rmse=9.6, outside_r=0.31, gap_rmse=5.1,
            unanchored_rmse=6.8, unanchored_r=0.52,
            strong_rmse=11.3, strong_bias=-5.5,
        )
        result = build_gate_comparison(
            baseline,
            candidate,
            self._cascade_row("r1_o_oracle_support", 0.6234, 0.16),
            self._cascade_row("r1_p_partial_conv_oracle_support", 0.645, 0.17),
        )
        self.assertTrue(result["all_gates_passed"])
        self.assertTrue(result["gates"]["outside_primary"]["passed"])
        self.assertTrue(result["gates"]["gap_non_regression"]["passed"])

        failed = copy.deepcopy(candidate)
        failed[0]["rmse_dbz"] = 9.9
        failed[0]["pearson_r"] = 0.31
        result = build_gate_comparison(
            baseline,
            failed,
            self._cascade_row("r1_o_oracle_support", 0.6234, 0.16),
            self._cascade_row("r1_p_partial_conv_oracle_support", 0.645, 0.17),
        )
        self.assertFalse(result["gates"]["outside_primary"]["passed"])
        self.assertFalse(result["all_gates_passed"])


if __name__ == "__main__":
    unittest.main()
