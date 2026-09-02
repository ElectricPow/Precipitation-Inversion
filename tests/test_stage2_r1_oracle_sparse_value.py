"""Tests for the non-deployable S2-R1-O spatial-completion upper bound."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from precipitation_inversion.inference.stage2_completion_sliding_window import (  # noqa: E402
    predict_stage2_completion_full_orbit,
)
from precipitation_inversion.losses.stage2_completion_losses import (  # noqa: E402
    Stage2CompletionLoss,
)
from precipitation_inversion.models.stage2_completion_unet3d import (  # noqa: E402
    Stage2CompletionUNet3D,
    stage2_completion_prediction_from_output,
)
from precipitation_inversion.training.stage2_completion_engine import (  # noqa: E402
    evaluate_stage2_completion_one_epoch,
    train_stage2_completion_one_epoch,
)
from scripts.train_stage2_r1_oracle_sparse_value import (  # noqa: E402
    build_model,
    load_config,
    validate_r1_config,
)


class TinyCompletionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Conv3d(4, 1, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"reflectivity": self.head(inputs)}


def make_batch() -> dict[str, torch.Tensor]:
    inputs = torch.randn(2, 4, 4, 4, 3)
    target = torch.zeros(2, 1, 4, 4, 3)
    mask = torch.zeros_like(target, dtype=torch.bool)
    mask[:, :, 1:3, 1:3, 1:] = True
    target[mask] = 0.5
    weights = torch.zeros_like(target)
    weights[mask] = 1.25
    return {
        "inputs": inputs,
        "target_dbz": target,
        "regression_mask": mask,
        "regression_weights": weights,
    }


class SyntheticCompletionDataset(torch.utils.data.Dataset):
    halo_size = 1

    def __init__(self) -> None:
        self.files = [{"nscan": 5, "nray": 3, "z_size": 2}]
        self.dpr_standardizer = SimpleNamespace(
            mean=np.array([10.0, 20.0], dtype=np.float32),
            std=np.array([2.0, 4.0], dtype=np.float32),
        )

    def __len__(self) -> int:
        return 2

    def file_index_range(self, file_id: int) -> range:
        if file_id != 0:
            raise IndexError(file_id)
        return range(2)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start, length = ((0, 3), (3, 2))[index]
        inputs = torch.zeros(4, 4, 4, 2)
        for local in range(length):
            inputs[0, 1 + local, :3] = start + local + 1
        return {
            "inputs": inputs,
            "core_start": torch.tensor(start),
            "core_length": torch.tensor(length),
        }


class EchoCompletionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(()))

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"reflectivity": inputs[:, :1] + self.dummy}


class Stage2R1OracleSparseValueTests(unittest.TestCase):
    def test_single_head_unet_preserves_height_and_has_no_support_head(self) -> None:
        model = Stage2CompletionUNet3D(
            base_channels=2, channel_multipliers=(1, 2, 4),
            bottleneck_dropout=0.0,
        ).eval()
        with torch.inference_mode():
            output = model(torch.randn(1, 4, 16, 16, 5))
        prediction = stage2_completion_prediction_from_output(output)
        self.assertEqual(tuple(prediction.shape), (1, 1, 16, 16, 5))
        self.assertFalse(hasattr(model, "support_head"))
        self.assertFalse(hasattr(model, "output_head"))

    def test_weighted_masked_loss_and_backward_use_only_selected_dpr(self) -> None:
        prediction_values = torch.tensor([0.0, 10.0, 0.0], requires_grad=True)
        prediction = prediction_values.reshape(1, 1, 1, 1, 3)
        target = torch.tensor([1.0, -999.0, 2.0]).reshape_as(prediction)
        mask = torch.tensor([True, False, True]).reshape_as(prediction)
        weights = torch.tensor([1.0, 0.0, 3.0]).reshape_as(prediction)
        parts = Stage2CompletionLoss(beta=1.0).compute_components(
            prediction, target, mask, regression_weights=weights
        )
        # Smooth-L1 is 0.5 and 1.5; weighted mean=(0.5+4.5)/4=1.25.
        self.assertAlmostEqual(float(parts.total.detach()), 1.25)
        self.assertEqual(parts.reflectivity_count, 2)
        self.assertEqual(parts.reflectivity_weight_sum, 4.0)
        parts.total.backward()
        self.assertAlmostEqual(float(prediction_values.grad[1]), 0.0)
        self.assertTrue(bool(torch.isfinite(prediction_values.grad).all()))

    def test_training_engine_updates_single_head_and_restores_eval_mode(self) -> None:
        model = TinyCompletionModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = Stage2CompletionLoss(beta=0.2)
        before = model.head.weight.detach().clone()
        train = train_stage2_completion_one_epoch(
            model, [make_batch()], optimizer, criterion, "cpu",
            dpr_mean=[10, 20, 30], dpr_std=[2, 4, 8], use_amp=False,
        )
        self.assertEqual(train.optimizer_steps, 1)
        self.assertEqual(train.reflectivity_voxels, 16)
        self.assertFalse(torch.equal(before, model.head.weight.detach()))
        model.train()
        val = evaluate_stage2_completion_one_epoch(
            model, [make_batch()], criterion, "cpu",
            dpr_mean=[10, 20, 30], dpr_std=[2, 4, 8], use_amp=False,
        )
        self.assertTrue(model.training)
        self.assertEqual(val.optimizer_steps, 0)
        self.assertIn("reflectivity_on_oracle_support", val.metrics)

    def test_complete_orbit_reconstruction_discards_halo_and_inverts_stats(self) -> None:
        result = predict_stage2_completion_full_orbit(
            EchoCompletionModel(), SyntheticCompletionDataset(), 0,
            device="cpu", use_amp=False,
        )
        self.assertEqual(result.reflectivity_dbz.shape, (5, 3, 2))
        np.testing.assert_allclose(
            result.reflectivity_dbz[:, 0, 0], (np.arange(5) + 1) * 2 + 10
        )
        np.testing.assert_allclose(
            result.reflectivity_dbz[:, 0, 1], (np.arange(5) + 1) * 4 + 20
        )

    def test_formal_config_seals_non_deployable_four_channel_contract(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "stage2_r1_o_dpr_sparse_value.yaml")
        channels = validate_r1_config(config)
        model = build_model(config)
        self.assertEqual(len(channels), 4)
        self.assertEqual(model.in_channels, 4)
        self.assertFalse(config["experiment"]["deployable"])
        self.assertFalse(config["training"]["early_stopping"]["enabled"])
        self.assertEqual(config["training"]["checkpoint_every"], 10)
        # A value-only model receives no gradient from dpr_count=0 patches.
        # Preserve the W1.25 relative 0.10:0.30:0.40 target ratios after
        # removing its old 0.20 support/background quota.
        self.assertEqual(config["data"]["sampler"]["stratum_weights"], {
            "background": 0.0,
            "ordinary_target": 0.125,
            "fill_dominant_target": 0.375,
            "strong_target": 0.50,
        })


if __name__ == "__main__":
    unittest.main()
