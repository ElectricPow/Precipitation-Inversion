"""Tests for masked log-rain regression objectives."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from precipitation_inversion.losses.masked_losses import (  # noqa: E402
    MaskedSmoothL1Loss,
    Stage1CompositeLoss,
    build_stage1_loss,
    masked_mae_loss,
    masked_mse_loss,
    masked_physical_gradient_smooth_l1_loss,
    masked_smooth_l1_loss,
)


class MaskedLossTests(unittest.TestCase):
    def test_mse_and_mae_ignore_halo_and_padding_values(self) -> None:
        prediction = torch.tensor([0.0, 2.0, 1000.0, -1000.0]).reshape(1, 1, 1, 1, 4)
        target = torch.tensor([1.0, 4.0, 0.0, 0.0]).reshape_as(prediction)
        mask = torch.tensor([True, True, False, False]).reshape_as(prediction)
        self.assertAlmostEqual(float(masked_mse_loss(prediction, target, mask)), 2.5)
        self.assertAlmostEqual(float(masked_mae_loss(prediction, target, mask)), 1.5)

    def test_smooth_l1_matches_pytorch_on_selected_voxels(self) -> None:
        prediction = torch.tensor([0.5, 2.0, 99.0]).reshape(1, 1, 1, 1, 3)
        target = torch.zeros_like(prediction)
        mask = torch.tensor([True, True, False]).reshape_as(prediction)
        expected = F.smooth_l1_loss(
            torch.tensor([0.5, 2.0]), torch.zeros(2), beta=1.0
        )
        actual = masked_smooth_l1_loss(
            prediction, target, mask, beta=1.0
        )
        self.assertAlmostEqual(float(actual), float(expected))

    def test_weighted_mean_uses_selected_weight_sum(self) -> None:
        prediction = torch.tensor([1.0, 3.0, 100.0]).reshape(1, 1, 1, 1, 3)
        target = torch.zeros_like(prediction)
        mask = torch.tensor([True, True, False]).reshape_as(prediction)
        weights = torch.tensor([1.0, 3.0, -99.0]).reshape_as(prediction)
        # Weighted absolute mean=(1*1 + 3*3)/(1+3)=2.5. Invalid unselected
        # weight is irrelevant because padding/halo cannot affect the loss.
        actual = masked_mae_loss(
            prediction, target, mask, weights=weights
        )
        self.assertAlmostEqual(float(actual), 2.5)

    def test_masked_out_voxels_receive_exactly_zero_gradient(self) -> None:
        prediction = torch.tensor([1.0, 5.0, 9.0], requires_grad=True).reshape(
            1, 1, 1, 1, 3
        )
        prediction.retain_grad()
        target = torch.zeros_like(prediction)
        mask = torch.tensor([True, False, True]).reshape_as(prediction)
        loss = masked_mse_loss(prediction, target, mask)
        loss.backward()
        self.assertEqual(float(prediction.grad[0, 0, 0, 0, 1]), 0.0)
        self.assertNotEqual(float(prediction.grad[0, 0, 0, 0, 0]), 0.0)

    def test_empty_mask_returns_connected_zero_or_can_raise(self) -> None:
        prediction = torch.randn(1, 1, 2, 2, 3, requires_grad=True)
        target = torch.zeros_like(prediction)
        mask = torch.zeros_like(prediction, dtype=torch.bool)
        loss = masked_smooth_l1_loss(prediction, target, mask)
        self.assertEqual(float(loss.detach()), 0.0)
        loss.backward()
        self.assertTrue(torch.equal(prediction.grad, torch.zeros_like(prediction)))
        with self.assertRaisesRegex(ValueError, "no positive-weight"):
            masked_smooth_l1_loss(prediction, target, mask, empty="raise")

    def test_none_reduction_retains_shape_and_zeros_unselected_values(self) -> None:
        prediction = torch.tensor([1.0, 2.0]).reshape(1, 1, 1, 1, 2)
        target = torch.zeros_like(prediction)
        mask = torch.tensor([True, False]).reshape_as(prediction)
        result = masked_mae_loss(prediction, target, mask, reduction="none")
        self.assertEqual(tuple(result.shape), tuple(prediction.shape))
        torch.testing.assert_close(result, torch.tensor([1.0, 0.0]).reshape_as(result))

    def test_module_wrapper_and_validation(self) -> None:
        criterion = MaskedSmoothL1Loss(beta=0.5)
        prediction = torch.ones(1, 1, 1, 1, 2)
        target = torch.zeros_like(prediction)
        mask = torch.ones_like(prediction, dtype=torch.bool)
        self.assertEqual(
            float(criterion(prediction, target, mask)),
            float(masked_smooth_l1_loss(prediction, target, mask, beta=0.5)),
        )
        with self.assertRaises(TypeError):
            masked_mse_loss(prediction, target, mask.float())
        with self.assertRaisesRegex(ValueError, "shapes differ"):
            masked_mse_loss(prediction, target[..., :1], mask)
        with self.assertRaises(ValueError):
            masked_smooth_l1_loss(prediction, target, mask, beta=0.0)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            masked_mae_loss(prediction, target, mask, weights=-torch.ones_like(target))

    def test_physical_gradient_loss_uses_real_nonuniform_height_spacing(self) -> None:
        # Physical rain profiles are prediction=[1,3,7], target=[1,2,8] mm/h.
        # At z=[0,0.5,1.5] km their upward gradients are [4,4] and [2,6],
        # so both errors have magnitude 2. Smooth-L1(beta=1) is therefore 1.5.
        prediction = torch.log1p(torch.tensor([1.0, 3.0, 7.0])).reshape(
            1, 1, 1, 1, 3
        )
        target = torch.log1p(torch.tensor([1.0, 2.0, 8.0])).reshape_as(
            prediction
        )
        reliable = torch.ones_like(prediction, dtype=torch.bool)
        height = torch.tensor([0.0, 0.5, 1.5]).reshape(1, 1, 1, 1, 3)
        loss = masked_physical_gradient_smooth_l1_loss(
            prediction,
            target,
            reliable,
            height_km=height,
            beta=1.0,
        )
        self.assertAlmostEqual(float(loss), 1.5, places=6)

    def test_physical_gradient_requires_two_reliable_adjacent_endpoints(self) -> None:
        prediction = torch.log1p(torch.tensor([1.0, 100.0, 3.0])).reshape(
            1, 1, 1, 1, 3
        )
        prediction.requires_grad_()
        target = torch.log1p(torch.tensor([1.0, 2.0, 3.0])).reshape_as(
            prediction
        )
        reliable = torch.tensor([True, False, True]).reshape_as(prediction)
        height = torch.tensor([0.0, 0.25, 0.5])
        loss = masked_physical_gradient_smooth_l1_loss(
            prediction, target, reliable, height_km=height
        )
        self.assertEqual(float(loss.detach()), 0.0)
        loss.backward()
        torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction))

    def test_unselected_extreme_log_value_cannot_overflow_gradient_loss(self) -> None:
        prediction = torch.tensor([0.0, 1.0, 1000.0], requires_grad=True).reshape(
            1, 1, 1, 1, 3
        )
        prediction.retain_grad()
        target = torch.zeros_like(prediction)
        reliable = torch.tensor([True, True, False]).reshape_as(prediction)
        loss = masked_physical_gradient_smooth_l1_loss(
            prediction,
            target,
            reliable,
            height_km=torch.tensor([0.0, 0.5, 1.0]),
        )
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        self.assertEqual(float(prediction.grad[..., 2]), 0.0)
        self.assertTrue(bool(torch.isfinite(prediction.grad).all()))

    def test_physical_gradient_clamps_negative_log_and_is_autocast_safe(self) -> None:
        # clamp([-1, log1p(2)]) -> physical rain [0,2] mm/h; target is [0,1].
        # Over dz=0.5 km, gradient error is 2 and Smooth-L1(beta=1)=1.5.
        prediction = torch.tensor(
            [-1.0, math.log1p(2.0)], requires_grad=True
        ).reshape(1, 1, 1, 1, 2)
        prediction.retain_grad()
        target = torch.tensor([0.0, math.log1p(1.0)]).reshape_as(prediction)
        reliable = torch.ones_like(prediction, dtype=torch.bool)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            loss = masked_physical_gradient_smooth_l1_loss(
                prediction,
                target,
                reliable,
                height_km=torch.tensor([0.0, 0.5]),
            )
        self.assertEqual(loss.dtype, torch.float32)
        self.assertAlmostEqual(float(loss.detach()), 1.5, places=5)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(prediction.grad).all()))

    def test_composite_loss_is_exact_i_plus_point_zero_two_g(self) -> None:
        prediction = torch.log1p(torch.tensor([1.0, 3.0, 7.0])).reshape(
            1, 1, 1, 1, 3
        )
        target = torch.log1p(torch.tensor([1.0, 2.0, 8.0])).reshape_as(
            prediction
        )
        mask = torch.ones_like(prediction, dtype=torch.bool)
        # Deliberately non-uniform I weights. G must remain the unweighted 1.5
        # calculated in the preceding hand-worked physical example.
        weights = torch.tensor([1.0, 2.0, 3.0]).reshape_as(prediction)
        criterion = Stage1CompositeLoss(
            primary_beta=0.2,
            physical_gradient_weight=0.02,
            physical_gradient_beta=1.0,
        )
        parts = criterion.compute_components(
            prediction,
            target,
            mask,
            weights,
            reliable_mask=mask,
            height_km=torch.tensor([0.0, 0.5, 1.5]),
        )
        expected_primary = masked_smooth_l1_loss(
            prediction, target, mask, beta=0.2, weights=weights
        )
        self.assertEqual(parts.physical_gradient_pair_count, 2)
        self.assertAlmostEqual(float(parts.physical_gradient), 1.5, places=6)
        torch.testing.assert_close(parts.primary, expected_primary)
        torch.testing.assert_close(
            parts.total, expected_primary + torch.tensor(0.02 * 1.5)
        )

        changed_weights = weights.flip(-1) * 10.0
        changed = criterion.compute_components(
            prediction,
            target,
            mask,
            changed_weights,
            reliable_mask=mask,
            height_km=torch.tensor([0.0, 0.5, 1.5]),
        )
        torch.testing.assert_close(
            changed.physical_gradient, parts.physical_gradient
        )

    def test_composite_excludes_weak_endpoint_and_validates_configuration(self) -> None:
        prediction = torch.zeros(1, 1, 1, 1, 2)
        target = torch.tensor([0.5, 1.0]).reshape_as(prediction)
        loss_mask = torch.ones_like(prediction, dtype=torch.bool)
        reliable = torch.tensor([True, False]).reshape_as(prediction)
        criterion = Stage1CompositeLoss(physical_gradient_weight=0.02)
        parts = criterion.compute_components(
            prediction,
            target,
            loss_mask,
            torch.tensor([1.0, 0.1]).reshape_as(prediction),
            reliable_mask=reliable,
            height_km=torch.tensor([0.0, 0.25]),
        )
        self.assertEqual(parts.physical_gradient_pair_count, 0)
        self.assertEqual(float(parts.physical_gradient), 0.0)
        with self.assertRaisesRegex(KeyError, "height_km"):
            criterion.compute_components(
                prediction,
                target,
                loss_mask,
                reliable_mask=reliable,
            )
        with self.assertRaisesRegex(ValueError, "strictly ascending"):
            criterion.compute_components(
                prediction,
                target,
                loss_mask,
                reliable_mask=loss_mask,
                height_km=torch.tensor([0.5, 0.0]),
            )

        legacy = build_stage1_loss({"name": "masked_smooth_l1", "beta": 0.2})
        self.assertFalse(legacy.physical_gradient_enabled)
        configured = build_stage1_loss(
            {
                "name": "masked_smooth_l1",
                "beta": 0.2,
                "physical_gradient": {
                    "enabled": True,
                    "weight": 0.02,
                    "beta": 1.0,
                },
            }
        )
        self.assertTrue(configured.physical_gradient_enabled)
        self.assertEqual(configured.physical_gradient_weight, 0.02)
        with self.assertRaisesRegex(ValueError, "positive weight"):
            build_stage1_loss(
                {
                    "name": "masked_smooth_l1",
                    "beta": 0.2,
                    "physical_gradient": {"enabled": True, "weight": 0.0},
                }
            )


if __name__ == "__main__":
    unittest.main()
