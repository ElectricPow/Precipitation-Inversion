"""Tests for the C1-O normalization bridge and freeze contract."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from precipitation_inversion.models.stage3_cascade import (  # noqa: E402
    Stage3C1OracleCascade,
    assert_c1_freeze_contract,
)
from precipitation_inversion.models.unet3d import Stage1UNet3D  # noqa: E402


class DummyStage2(nn.Module):
    in_channels = 2

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        shape = (inputs.shape[0], 1, *inputs.shape[2:])
        reflectivity = torch.ones(shape, dtype=inputs.dtype, device=inputs.device)
        return {
            "support_logits": torch.zeros_like(reflectivity) + self.scale * 0.0,
            "reflectivity": reflectivity * (2.0 * self.scale),
        }


def make_cascade() -> Stage3C1OracleCascade:
    return Stage3C1OracleCascade(
        DummyStage2(),  # type: ignore[arg-type]
        Stage1UNet3D(
            in_channels=3,
            out_channels=1,
            base_channels=2,
            channel_multipliers=(1, 2),
            max_groups=1,
            bottleneck_dropout=0.0,
        ),
        stage2_dbz_mean=[10.0, 20.0],
        stage2_dbz_std=[2.0, 4.0],
        stage1_dbz_mean=[12.0, 16.0],
        stage1_dbz_std=[4.0, 2.0],
        stage2_in_channels=2,
    )


class Stage3CascadeTests(unittest.TestCase):
    def test_bridge_changes_normalization_space_and_neutral_fills_support(self) -> None:
        model = make_cascade()
        packed = torch.zeros((1, 4, 4, 4, 2))
        packed[:, 2] = 1.0  # true support
        packed[:, 2, 1, 1, 0] = 0.0
        packed[:, 3, ..., 0] = -1.0
        packed[:, 3, ..., 1] = 1.0
        stage1_inputs, diagnostics = model.build_stage1_inputs(packed)
        self.assertEqual(tuple(stage1_inputs.shape), (1, 3, 4, 4, 2))
        # S2 standardized 2 -> physical [14,28] -> S1 standardized [0.5,6].
        self.assertAlmostEqual(float(stage1_inputs[0, 0, 0, 0, 0]), 0.5)
        self.assertAlmostEqual(float(stage1_inputs[0, 0, 0, 0, 1]), 6.0)
        self.assertEqual(float(stage1_inputs[0, 0, 1, 1, 0]), 0.0)
        self.assertEqual(float(stage1_inputs[0, 1, 1, 1, 0]), 0.0)
        self.assertTrue(
            torch.equal(diagnostics["true_dpr_support"], packed[:, 2:3] > 0.5)
        )

    def test_backward_updates_only_stage1_rain_path(self) -> None:
        model = make_cascade().train()
        assert_c1_freeze_contract(model)
        self.assertFalse(model.stage2_model.training)
        packed = torch.zeros((1, 4, 4, 4, 2))
        packed[:, 2] = 1.0
        output = model(packed)
        output.square().mean().backward()
        self.assertTrue(all(p.grad is None for p in model.stage2_model.parameters()))
        self.assertTrue(
            any(
                p.grad is not None
                for p in model.stage1_model.parameters()
                if p.requires_grad
            )
        )

    def test_invalid_packed_channel_count_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected 4 packed channels"):
            make_cascade()(torch.zeros((1, 3, 4, 4, 2)))


if __name__ == "__main__":
    unittest.main()
