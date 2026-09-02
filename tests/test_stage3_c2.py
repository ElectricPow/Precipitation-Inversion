"""Unit tests for C2-O frozen-Stage1 task-aware Stage-2 adaptation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from precipitation_inversion.losses.stage3_losses import (  # noqa: E402
    build_stage3_c2_loss,
)
from precipitation_inversion.models.stage2_unet3d import Stage2UNet3D  # noqa: E402
from precipitation_inversion.models.stage3_cascade import (  # noqa: E402
    Stage3C2OracleCascade,
    assert_c2_freeze_contract,
)
from precipitation_inversion.models.unet3d import Stage1UNet3D  # noqa: E402
from precipitation_inversion.training.stage3_engine import (  # noqa: E402
    audit_stage3_c2_gradient_scale,
    evaluate_stage3_c2_one_epoch,
    train_stage3_c2_one_epoch,
)


def make_c2_cascade() -> Stage3C2OracleCascade:
    stage2 = Stage2UNet3D(
        in_channels=2,
        base_channels=2,
        channel_multipliers=(1, 2),
        max_groups=1,
        bottleneck_dropout=0.0,
        support_prior_probability=0.2,
    )
    stage1 = Stage1UNet3D(
        in_channels=3,
        out_channels=1,
        base_channels=2,
        channel_multipliers=(1, 2),
        max_groups=1,
        bottleneck_dropout=0.0,
    )
    return Stage3C2OracleCascade(
        stage2,
        stage1,
        stage2_dbz_mean=[10.0, 20.0],
        stage2_dbz_std=[2.0, 4.0],
        stage1_dbz_mean=[12.0, 16.0],
        stage1_dbz_std=[4.0, 2.0],
        stage2_in_channels=2,
    )


def make_c2_loss(*, rain_weight: float = 0.1):
    return build_stage3_c2_loss(
        {
            "support": {"name": "bce", "weight": 1.0, "pos_weight": 2.0},
            "reflectivity": {"weight": 1.0, "beta": 0.2},
        },
        {
            "name": "masked_smooth_l1",
            "beta": 0.2,
            "physical_gradient": {"enabled": True, "weight": 0.02, "beta": 1.0},
        },
        rain_weight=rain_weight,
    )


def make_c2_batch() -> dict[str, torch.Tensor]:
    shape = (1, 1, 4, 4, 2)
    packed = torch.zeros((1, 4, 4, 4, 2), dtype=torch.float32)
    packed[:, 0] = torch.linspace(-1.0, 1.0, 4).view(1, 4, 1, 1)
    packed[:, 1] = 1.0
    packed[:, 2] = 1.0  # oracle DPR support
    packed[:, 3, ..., 0] = -1.0
    packed[:, 3, ..., 1] = 1.0
    mask = torch.ones(shape, dtype=torch.bool)
    return {
        "inputs": packed,
        "target": torch.full(shape, 0.6),
        "loss_mask": mask,
        "loss_weights": torch.ones(shape),
        "reliable_loss_mask": mask,
        "height_km": torch.tensor([[[[[0.5, 1.0]]]]]),
        "stage2_target_support": torch.ones(shape),
        "stage2_target_dbz": torch.full(shape, 0.25),
        "stage2_support_loss_mask": mask,
        "stage2_regression_mask": mask,
        "stage2_regression_weights": torch.ones(shape),
    }


class Stage3C2CascadeTests(unittest.TestCase):
    def test_freeze_scope_and_differentiable_stage1_bridge(self) -> None:
        model = make_c2_cascade().train()
        assert_c2_freeze_contract(model)
        self.assertFalse(model.stage1_model.training)
        self.assertFalse(model.stage2_model.encoder[-1].training)
        self.assertTrue(model.stage2_model.decoder[-1].training)
        output = model(make_c2_batch()["inputs"])
        output["rain"].square().mean().backward()
        self.assertTrue(all(p.grad is None for p in model.stage1_model.parameters()))
        self.assertTrue(
            any(
                p.grad is not None
                for p in model.stage2_model.parameters()
                if p.requires_grad
            )
        )

    def test_loss_and_engine_keep_three_tasks_visible(self) -> None:
        model = make_c2_cascade()
        criterion = make_c2_loss()
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=1e-3)
        frozen_before = {
            name: value.detach().clone()
            for name, value in model.stage1_model.state_dict().items()
        }
        result = train_stage3_c2_one_epoch(
            model,
            [make_c2_batch()],
            optimizer,
            criterion,
            "cpu",
            dpr_mean=[10.0, 20.0],
            dpr_std=[2.0, 4.0],
            support_threshold=0.5,
            thresholds_mm_h=(1.0, 5.0),
            scaler=None,
            use_amp=False,
            grad_clip_norm=1.0,
            accumulation_steps=1,
            max_batches=None,
        )
        self.assertEqual(result.optimizer_steps, 1)
        self.assertIn("stage2_anchor", result.loss_components)
        self.assertIn("rain_task", result.loss_components)
        self.assertIn("stage2", result.metrics)
        for name, value in model.stage1_model.state_dict().items():
            self.assertTrue(torch.equal(value, frozen_before[name]))
        validation = evaluate_stage3_c2_one_epoch(
            model,
            [make_c2_batch()],
            criterion,
            "cpu",
            dpr_mean=[10.0, 20.0],
            dpr_std=[2.0, 4.0],
            support_threshold=0.5,
            thresholds_mm_h=(1.0, 5.0),
            use_amp=False,
            max_batches=None,
        )
        self.assertEqual(validation.batch_count, 1)

    def test_train_only_gradient_audit_selects_bounded_weight(self) -> None:
        model = make_c2_cascade()
        report = audit_stage3_c2_gradient_scale(
            model,
            [make_c2_batch()],
            make_c2_loss(rain_weight=1.0),
            "cpu",
            target_gradient_ratio=0.25,
            min_rain_weight=1e-4,
            max_rain_weight=1.0,
            valid_batches_per_rank=1,
            max_candidate_batches_per_rank=1,
        )
        self.assertEqual(report["selection_scope"], "train_batches_only")
        self.assertEqual(report["valid_batch_count_global"], 1)
        self.assertGreaterEqual(report["selected_rain_weight"], 1e-4)
        self.assertLessEqual(report["selected_rain_weight"], 1.0)


if __name__ == "__main__":
    unittest.main()
