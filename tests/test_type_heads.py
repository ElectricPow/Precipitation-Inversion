"""Shape, height-order, and gradient tests for precipitation-type heads."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from precipitation_inversion.models.multitask_unet3d import (  # noqa: E402
    Stage1MultiTaskUNet3D,
)
from precipitation_inversion.models.type_heads import (  # noqa: E402
    GlobalPoolTypeHead,
    Ordered3DMorphologyHead,
)


class PrecipitationTypeHeadTests(unittest.TestCase):
    def test_pool_control_is_height_permutation_invariant(self) -> None:
        head = GlobalPoolTypeHead(4).eval()
        features = torch.randn(2, 4, 8, 8, 12)
        permutation = torch.randperm(12)
        with torch.inference_mode():
            original = head(features)
            shuffled = head(features.index_select(-1, permutation))
        torch.testing.assert_close(original, shuffled, rtol=1e-5, atol=1e-6)
        self.assertEqual(tuple(original.shape), (2, 3, 8, 8))

    def test_ordered_head_retains_shape_and_responds_to_height_order(self) -> None:
        torch.manual_seed(3)
        head = Ordered3DMorphologyHead(
            4, compressed_channels=4, height_levels=12, hidden_2d_channels=8
        ).eval()
        features = torch.randn(1, 4, 8, 8, 12, requires_grad=True)
        output = head(features)
        reverse = torch.arange(11, -1, -1)
        reversed_output = head(features, height_permutation=reverse)
        self.assertEqual(tuple(output.shape), (1, 3, 8, 8))
        self.assertFalse(torch.allclose(output, reversed_output))
        output.square().mean().backward()
        self.assertIsNotNone(features.grad)

    def test_multitask_unet_returns_separate_3d_rain_and_2d_type(self) -> None:
        model = Stage1MultiTaskUNet3D(
            base_channels=2,
            channel_multipliers=(1, 2),
            bottleneck_dropout=0.0,
            type_head_kind="ordered_3d",
            type_head_config={
                "compressed_channels": 2,
                "height_levels": 12,
                "hidden_2d_channels": 4,
            },
        )
        inputs = torch.randn(1, 3, 8, 8, 12)
        output = model(inputs)
        self.assertEqual(tuple(output["rain"].shape), (1, 1, 8, 8, 12))
        self.assertEqual(tuple(output["type_logits"].shape), (1, 3, 8, 8))
        (output["rain"].mean() + output["type_logits"].mean()).backward()
        self.assertIsNotNone(model.output_head.weight.grad)
        self.assertTrue(any(parameter.grad is not None for parameter in model.type_head.parameters()))


if __name__ == "__main__":
    unittest.main()
