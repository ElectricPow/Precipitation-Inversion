"""Tests for masked log-rain regression objectives."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from precipitation_inversion.losses.masked_losses import (  # noqa: E402
    MaskedSmoothL1Loss,
    masked_mae_loss,
    masked_mse_loss,
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


if __name__ == "__main__":
    unittest.main()
