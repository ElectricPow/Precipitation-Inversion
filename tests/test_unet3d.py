"""Shape and gradient tests for the height-preserving anisotropic 3D U-Net."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from precipitation_inversion.models.blocks3d import (  # noqa: E402
    AnisotropicResidualBlock3D,
    HorizontalDownBlock3D,
    HorizontalUpBlock3D,
    compatible_group_count,
)
from precipitation_inversion.models.unet3d import Stage1UNet3D  # noqa: E402


class AnisotropicBlockTests(unittest.TestCase):
    def test_group_count_and_invalid_configuration(self) -> None:
        self.assertEqual(compatible_group_count(16), 8)
        self.assertEqual(compatible_group_count(12), 6)
        self.assertEqual(compatible_group_count(3), 3)
        with self.assertRaises(ValueError):
            compatible_group_count(0)
        with self.assertRaises(ValueError):
            AnisotropicResidualBlock3D(2, 4, dropout=1.0)

    def test_residual_down_and_up_shapes_never_resample_height(self) -> None:
        # Meaning: B=2, C=3, scan=16, ray=16, seven physical height levels.
        inputs = torch.randn(2, 3, 16, 16, 7)
        residual = AnisotropicResidualBlock3D(3, 4)
        skip = residual(inputs)
        self.assertEqual(tuple(skip.shape), (2, 4, 16, 16, 7))

        down = HorizontalDownBlock3D(4, 8)
        coarse = down(skip)
        self.assertEqual(tuple(coarse.shape), (2, 8, 8, 8, 7))

        up = HorizontalUpBlock3D(8, 4, 4)
        restored = up(coarse, skip)
        self.assertEqual(tuple(restored.shape), (2, 4, 16, 16, 7))

    def test_up_block_rejects_different_height_coordinates(self) -> None:
        up = HorizontalUpBlock3D(8, 4, 4)
        with self.assertRaisesRegex(ValueError, "height"):
            up(torch.randn(1, 8, 4, 4, 6), torch.randn(1, 4, 8, 8, 7))


class Stage1UNet3DTests(unittest.TestCase):
    def test_real_patch_shape_is_preserved_and_all_levels_keep_height_60(self) -> None:
        model = Stage1UNet3D(base_channels=2, bottleneck_dropout=0.0).eval()
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
            # Current Dataset batch: three input channels and no height padding.
            inputs = torch.randn(1, 3, 64, 64, 60)
            with torch.inference_mode():
                output = model(inputs)
        finally:
            for handle in handles:
                handle.remove()

        self.assertEqual(tuple(output.shape), (1, 1, 64, 64, 60))
        self.assertEqual(
            encoder_shapes,
            [
                (1, 4, 32, 32, 60),
                (1, 8, 16, 16, 60),
                (1, 16, 8, 8, 60),
                (1, 32, 4, 4, 60),
            ],
        )

    def test_small_model_backward_reaches_input_and_parameters(self) -> None:
        model = Stage1UNet3D(
            base_channels=2,
            channel_multipliers=(1, 2, 4),
            bottleneck_dropout=0.0,
        )
        inputs = torch.randn(1, 3, 16, 16, 7, requires_grad=True)
        output = model(inputs)
        self.assertEqual(tuple(output.shape), (1, 1, 16, 16, 7))
        output.square().mean().backward()
        self.assertIsNotNone(inputs.grad)
        self.assertTrue(torch.isfinite(inputs.grad).all())
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_invalid_channel_and_horizontal_shape_are_rejected(self) -> None:
        model = Stage1UNet3D(base_channels=2, bottleneck_dropout=0.0)
        with self.assertRaisesRegex(ValueError, "input channels"):
            model(torch.randn(1, 2, 16, 16, 7))
        with self.assertRaisesRegex(ValueError, "divisible"):
            model(torch.randn(1, 3, 63, 64, 7))


if __name__ == "__main__":
    unittest.main()
