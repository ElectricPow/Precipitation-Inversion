"""Shape, initialization, and gradient tests for the Stage-2 dual-head U-Net."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from precipitation_inversion.losses.stage2_losses import Stage2CompositeLoss  # noqa: E402
from precipitation_inversion.models.stage2_unet3d import (  # noqa: E402
    Stage2UNet3D,
    stage2_predictions_from_output,
)


class Stage2UNet3DTests(unittest.TestCase):
    def test_real_patch_shape_and_every_encoder_height_remains_60(self) -> None:
        model = Stage2UNet3D(
            base_channels=2,
            bottleneck_dropout=0.0,
            support_prior_probability=0.04,
        ).eval()
        encoder_shapes: list[tuple[int, ...]] = []
        handles = [
            block.register_forward_hook(
                lambda _module, _args, output: encoder_shapes.append(
                    tuple(output.shape)
                )
            )
            for block in model.encoder
        ]
        try:
            inputs = torch.randn(1, 4, 64, 64, 60)
            with torch.inference_mode():
                output = model(inputs)
        finally:
            for handle in handles:
                handle.remove()
        support, reflectivity = stage2_predictions_from_output(output)
        self.assertEqual(tuple(support.shape), (1, 1, 64, 64, 60))
        self.assertEqual(tuple(reflectivity.shape), tuple(support.shape))
        self.assertEqual(
            encoder_shapes,
            [
                (1, 4, 32, 32, 60),
                (1, 8, 16, 16, 60),
                (1, 16, 8, 8, 60),
                (1, 32, 4, 4, 60),
            ],
        )

    def test_head_biases_start_near_support_prior_and_mean_dbz(self) -> None:
        prior = 0.04
        model = Stage2UNet3D(
            base_channels=2,
            channel_multipliers=(1, 2),
            bottleneck_dropout=0.0,
            support_prior_probability=prior,
        )
        expected_logit = math.log(prior / (1.0 - prior))
        self.assertAlmostEqual(
            float(model.support_head.bias.detach()), expected_logit, places=6
        )
        self.assertEqual(float(model.reflectivity_head.bias.detach()), 0.0)
        self.assertLess(float(model.support_head.weight.detach().std()), 0.01)
        self.assertLess(float(model.reflectivity_head.weight.detach().std()), 0.01)

    def test_composite_backward_reaches_backbone_and_both_heads(self) -> None:
        model = Stage2UNet3D(
            base_channels=2,
            channel_multipliers=(1, 2, 4),
            bottleneck_dropout=0.0,
        )
        inputs = torch.randn(1, 4, 16, 16, 5, requires_grad=True)
        output = model(inputs)
        support, reflectivity = stage2_predictions_from_output(output)
        target_support = torch.zeros_like(support)
        target_support[..., 1:4] = 1.0
        target_dbz = torch.zeros_like(reflectivity)
        target_dbz[..., 1:4] = 0.5
        support_mask = torch.ones_like(support, dtype=torch.bool)
        regression_mask = target_support.bool()
        loss = Stage2CompositeLoss()(
            support,
            reflectivity,
            target_support,
            target_dbz,
            support_mask,
            regression_mask,
        )
        loss.backward()
        self.assertIsNotNone(inputs.grad)
        self.assertTrue(bool(torch.isfinite(inputs.grad).all()))
        self.assertIsNotNone(model.support_head.weight.grad)
        self.assertIsNotNone(model.reflectivity_head.weight.grad)
        self.assertTrue(any(
            parameter.grad is not None
            for name, parameter in model.named_parameters()
            if name.startswith("stem")
        ))

    def test_output_is_unbounded_and_not_hard_gated(self) -> None:
        model = Stage2UNet3D(
            base_channels=2,
            channel_multipliers=(1, 2),
            bottleneck_dropout=0.0,
            support_prior_probability=0.01,
        ).eval()
        with torch.inference_mode():
            output = model(torch.zeros(1, 4, 8, 8, 3))
        support, reflectivity = stage2_predictions_from_output(output)
        self.assertTrue(bool((support < 0.0).all()))
        # Reflectivity is returned independently, not multiplied by sigmoid.
        torch.testing.assert_close(reflectivity, torch.zeros_like(reflectivity))

    def test_configurable_input_width_preserves_dual_head_shape(self) -> None:
        # Exercise the three-channel parent, four-channel distance experiment,
        # and dormant five-channel interpolation capability through the model.
        for in_channels in (3, 4, 5):
            with self.subTest(in_channels=in_channels):
                model = Stage2UNet3D(
                    in_channels=in_channels,
                    base_channels=2,
                    channel_multipliers=(1, 2),
                    bottleneck_dropout=0.0,
                ).eval()
                with torch.inference_mode():
                    support, reflectivity = stage2_predictions_from_output(
                        model(torch.randn(1, in_channels, 8, 8, 3))
                    )
                self.assertEqual(tuple(support.shape), (1, 1, 8, 8, 3))
                self.assertEqual(tuple(reflectivity.shape), tuple(support.shape))

    def test_invalid_input_prior_and_output_contract_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            Stage2UNet3D(in_channels=0)
        with self.assertRaisesRegex(ValueError, r"\(0,1\)"):
            Stage2UNet3D(support_prior_probability=1.0)
        model = Stage2UNet3D(
            base_channels=2,
            channel_multipliers=(1, 2),
            bottleneck_dropout=0.0,
        )
        with self.assertRaisesRegex(ValueError, "input channels"):
            model(torch.randn(1, 3, 8, 8, 3))
        with self.assertRaises(TypeError):
            stage2_predictions_from_output(torch.zeros(1))
        with self.assertRaisesRegex(ValueError, "share shape"):
            stage2_predictions_from_output(
                {
                    "support_logits": torch.zeros(1, 1, 2, 2, 3),
                    "reflectivity": torch.zeros(1, 1, 2, 2, 2),
                }
            )


if __name__ == "__main__":
    unittest.main()
