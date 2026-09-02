"""Tests for independent Stage-2 support and reflectivity objectives."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from precipitation_inversion.losses.stage2_losses import (  # noqa: E402
    MaskedSupportLoss,
    Stage2CompositeLoss,
    build_stage2_loss,
    masked_support_bce_with_logits,
    masked_support_focal_loss,
)


def volume(values: list[float], *, requires_grad: bool = False) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32, requires_grad=requires_grad).reshape(
        1, 1, 1, 1, -1
    )


class MaskedSupportLossTests(unittest.TestCase):
    def test_bce_pos_weight_and_mask_match_manual_value(self) -> None:
        logits = volume([0.0, 0.0, 1000.0])
        target = volume([1.0, 0.0, 1.0])
        mask = torch.tensor([True, True, False]).reshape_as(logits)
        actual = masked_support_bce_with_logits(
            logits, target, mask, pos_weight=2.0
        )
        self.assertAlmostEqual(float(actual), 1.5 * math.log(2.0), places=6)

    def test_focal_gamma_zero_without_alpha_equals_bce(self) -> None:
        logits = volume([-1.0, 0.5, 2.0])
        target = volume([0.0, 1.0, 1.0])
        mask = torch.ones_like(logits, dtype=torch.bool)
        focal = masked_support_focal_loss(
            logits, target, mask, gamma=0.0, alpha=None
        )
        bce = F.binary_cross_entropy_with_logits(logits, target)
        torch.testing.assert_close(focal, bce)

    def test_ignored_nan_and_inf_are_neutralized_before_loss(self) -> None:
        logits = volume([0.0, float("inf"), float("nan")], requires_grad=True)
        logits.retain_grad()
        target = volume([1.0, float("nan"), float("nan")])
        mask = torch.tensor([True, False, False]).reshape_as(logits)
        loss = masked_support_bce_with_logits(logits, target, mask)
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        self.assertEqual(float(logits.grad[..., 1]), 0.0)
        self.assertEqual(float(logits.grad[..., 2]), 0.0)

    def test_empty_mask_returns_graph_connected_zero(self) -> None:
        logits = torch.randn(1, 1, 2, 2, 3, requires_grad=True)
        target = torch.zeros_like(logits)
        mask = torch.zeros_like(logits, dtype=torch.bool)
        loss = MaskedSupportLoss()(logits, target, mask)
        self.assertEqual(float(loss.detach()), 0.0)
        loss.backward()
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits))


class Stage2CompositeLossTests(unittest.TestCase):
    def test_components_counts_weights_and_exact_total(self) -> None:
        support_logits = volume([0.0, 0.0, 0.0, 99.0])
        reflectivity = volume([1.0, 99.0, 4.0, 99.0])
        support_target = volume([1.0, 0.0, 1.0, 0.0])
        target_dbz = volume([0.0, 0.0, 2.0, 0.0])
        support_mask = torch.tensor([True, True, True, False]).reshape_as(
            support_logits
        )
        regression_mask = torch.tensor([True, False, True, False]).reshape_as(
            reflectivity
        )
        criterion = Stage2CompositeLoss(
            support_weight=0.5,
            reflectivity_weight=2.0,
            reflectivity_beta=1.0,
        )
        parts = criterion.compute_components(
            support_logits,
            reflectivity,
            support_target,
            target_dbz,
            support_mask,
            regression_mask,
        )
        expected_support = math.log(2.0)
        # Smooth-L1 errors 1 and 2 with beta=1 -> 0.5 and 1.5, mean=1.
        self.assertAlmostEqual(float(parts.support), expected_support, places=6)
        self.assertAlmostEqual(float(parts.reflectivity), 1.0, places=6)
        self.assertAlmostEqual(float(parts.total), 0.5 * expected_support + 2.0)
        self.assertEqual(parts.support_count, 3)
        self.assertEqual(parts.support_positive_count, 2)
        self.assertEqual(parts.reflectivity_count, 2)
        self.assertEqual(parts.reflectivity_weight_sum, 2.0)

    def test_weighted_reflectivity_uses_weight_sum_and_masks_nan_gradients(self) -> None:
        support_logits = volume([0.0, 0.0, 0.0], requires_grad=True)
        reflectivity = volume([1.0, 2.0, float("nan")], requires_grad=True)
        reflectivity.retain_grad()
        support_target = volume([1.0, 1.0, 0.0])
        target_dbz = volume([0.0, 0.0, float("nan")])
        support_mask = torch.tensor([True, True, False]).reshape_as(support_logits)
        regression_mask = torch.tensor([True, True, False]).reshape_as(reflectivity)
        weights = volume([1.0, 3.0, float("nan")])
        parts = Stage2CompositeLoss(reflectivity_beta=1.0).compute_components(
            support_logits,
            reflectivity,
            support_target,
            target_dbz,
            support_mask,
            regression_mask,
            regression_weights=weights,
        )
        # Smooth-L1(beta=1): [0.5,1.5], hence (1*0.5+3*1.5)/(1+3)=1.25.
        self.assertAlmostEqual(float(parts.reflectivity.detach()), 1.25, places=6)
        self.assertEqual(parts.reflectivity_count, 2)
        self.assertEqual(parts.reflectivity_weight_sum, 4.0)
        parts.total.backward()
        self.assertTrue(bool(torch.isfinite(reflectivity.grad).all()))
        self.assertEqual(float(reflectivity.grad[..., 2]), 0.0)

        bad_weights = volume([1.0, -1.0, 0.0])
        with self.assertRaisesRegex(ValueError, "non-negative"):
            Stage2CompositeLoss(reflectivity_beta=1.0).compute_components(
                volume([0.0, 0.0, 0.0]),
                volume([1.0, 2.0, 0.0]),
                support_target,
                volume([0.0, 0.0, 0.0]),
                support_mask,
                regression_mask,
                regression_weights=bad_weights,
            )

    def test_masks_isolate_gradients_and_support_does_not_gate_regression(self) -> None:
        support_logits = volume([-100.0, 5.0, 5.0], requires_grad=True)
        reflectivity = volume([2.0, 50.0, 50.0], requires_grad=True)
        support_logits.retain_grad()
        reflectivity.retain_grad()
        support_target = volume([1.0, 0.0, 0.0])
        target_dbz = volume([0.0, 0.0, 0.0])
        support_mask = torch.tensor([True, True, False]).reshape_as(support_logits)
        regression_mask = torch.tensor([True, False, False]).reshape_as(reflectivity)
        loss = Stage2CompositeLoss()(
            support_logits,
            reflectivity,
            support_target,
            target_dbz,
            support_mask,
            regression_mask,
        )
        loss.backward()
        # Despite an almost-zero predicted support probability, dBZ receives
        # its own gradient because no hard predicted-support gate is applied.
        self.assertNotEqual(float(reflectivity.grad[..., 0]), 0.0)
        self.assertEqual(float(reflectivity.grad[..., 1]), 0.0)
        self.assertEqual(float(reflectivity.grad[..., 2]), 0.0)
        self.assertEqual(float(support_logits.grad[..., 2]), 0.0)

    def test_invalid_masks_targets_and_configuration_are_rejected(self) -> None:
        prediction = volume([0.0, 0.0])
        support_target = volume([0.0, 2.0])
        target_dbz = volume([0.0, 0.0])
        mask = torch.ones_like(prediction, dtype=torch.bool)
        criterion = Stage2CompositeLoss()
        with self.assertRaisesRegex(ValueError, r"in \[0,1\]"):
            criterion(
                prediction,
                prediction,
                support_target,
                target_dbz,
                mask,
                mask,
            )
        valid_target = volume([1.0, 0.0])
        regression = torch.tensor([False, True]).reshape_as(mask)
        with self.assertRaisesRegex(ValueError, "positive support"):
            criterion(
                prediction,
                prediction,
                valid_target,
                target_dbz,
                mask,
                regression,
            )
        support_subset = torch.tensor([True, False]).reshape_as(mask)
        regression = torch.tensor([False, True]).reshape_as(mask)
        with self.assertRaisesRegex(ValueError, "subset of support mask"):
            criterion(
                prediction,
                prediction,
                valid_target,
                target_dbz,
                support_subset,
                regression,
            )
        with self.assertRaisesRegex(ValueError, "cannot also use"):
            Stage2CompositeLoss(support_loss="focal", support_pos_weight=2.0)
        with self.assertRaisesRegex(ValueError, "at least one"):
            Stage2CompositeLoss(support_weight=0.0, reflectivity_weight=0.0)

    def test_config_builder_and_optional_focal(self) -> None:
        criterion = build_stage2_loss(
            {
                "support": {
                    "name": "focal",
                    "weight": 0.75,
                    "gamma": 1.5,
                    "alpha": 0.4,
                },
                "reflectivity": {"weight": 1.25, "beta": 0.5},
            }
        )
        self.assertEqual(criterion.support_criterion.name, "focal")
        self.assertEqual(criterion.support_weight, 0.75)
        self.assertEqual(criterion.reflectivity_weight, 1.25)
        self.assertEqual(criterion.reflectivity_beta, 0.5)

        # Dataset-side E3 fields are part of the strict reflectivity-loss
        # schema even though Stage2CompositeLoss only consumes the resulting
        # regression_weights tensor.
        weighted = build_stage2_loss(
            {
                "support": {"name": "bce"},
                "reflectivity": {
                    "weight": 1.0,
                    "beta": 0.2,
                    "intensity_bin_edges_dbz": [25.0, 35.0],
                    "intensity_bin_weights": [1.0, 1.25, 1.5],
                },
            }
        )
        self.assertEqual(weighted.reflectivity_beta, 0.2)
        with self.assertRaisesRegex(ValueError, "unknown.*reflectivity"):
            build_stage2_loss(
                {
                    "support": {},
                    "reflectivity": {"intensity_bin_weight": [1.0]},
                }
            )


if __name__ == "__main__":
    unittest.main()
