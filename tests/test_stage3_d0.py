"""Tests for deployable S3-D0 DirectMultiHead contracts."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from precipitation_inversion.losses.stage3_losses import build_stage3_d0_loss  # noqa: E402
from precipitation_inversion.models.stage2_unet3d import Stage2UNet3D  # noqa: E402
from precipitation_inversion.models.stage3_direct import (  # noqa: E402
    STAGE3_D0_DECODER_AND_HEADS,
    STAGE3_D0_RAIN_HEAD_ONLY,
    Stage3DirectMultiHeadUNet3D,
    assert_d0_trainable_contract,
    expected_d0_trainable_parameter_names,
    load_stage2_source_into_d0,
)
from precipitation_inversion.training.stage3_engine import (  # noqa: E402
    audit_stage3_d0_gradient_scale,
    evaluate_stage3_d0_one_epoch,
    train_stage3_d0_one_epoch,
)


def make_model(scope: str = STAGE3_D0_RAIN_HEAD_ONLY) -> Stage3DirectMultiHeadUNet3D:
    return Stage3DirectMultiHeadUNet3D(
        in_channels=4,
        base_channels=2,
        channel_multipliers=(1, 2),
        max_groups=1,
        bottleneck_dropout=0.0,
        support_prior_probability=0.2,
        trainable_scope=scope,
    )


def make_loss(rain_weight: float = 0.2, stage2_weight: float = 1.0):
    return build_stage3_d0_loss(
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
        stage2_weight=stage2_weight,
    )


def make_batch() -> dict[str, torch.Tensor]:
    shape = (1, 1, 4, 4, 2)
    inputs = torch.zeros((1, 4, 4, 4, 2), dtype=torch.float32)
    inputs[:, 0] = torch.linspace(-1.0, 1.0, 4).view(1, 4, 1, 1)
    inputs[:, 1] = 1.0
    inputs[:, 2] = 0.5
    inputs[:, 3, ..., 0] = -1.0
    inputs[:, 3, ..., 1] = 1.0
    mask = torch.ones(shape, dtype=torch.bool)
    return {
        "inputs": inputs,
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


class Stage3D0Tests(unittest.TestCase):
    def test_three_heads_preserve_grid_and_scope_is_exact(self) -> None:
        inputs = make_batch()["inputs"]
        for scope in (STAGE3_D0_RAIN_HEAD_ONLY, STAGE3_D0_DECODER_AND_HEADS):
            model = make_model(scope)
            output = model(inputs)
            self.assertEqual(set(output), {"support_logits", "reflectivity", "rain"})
            for value in output.values():
                self.assertEqual(tuple(value.shape), (1, 1, 4, 4, 2))
            assert_d0_trainable_contract(model)
            actual = {name for name, value in model.named_parameters() if value.requires_grad}
            self.assertEqual(actual, expected_d0_trainable_parameter_names(model))

    def test_stage2_source_loading_only_leaves_new_rain_head(self) -> None:
        source = Stage2UNet3D(
            in_channels=4, base_channels=2, channel_multipliers=(1, 2),
            max_groups=1, bottleneck_dropout=0.0, support_prior_probability=0.2,
        )
        model = make_model()
        load_stage2_source_into_d0(model, source.state_dict())
        for name, value in source.state_dict().items():
            self.assertTrue(torch.equal(value, model.state_dict()[name]))

    def test_head_probe_updates_only_rain_head_and_engine_reports_all_tasks(self) -> None:
        model = make_model()
        before = {name: value.detach().clone() for name, value in model.state_dict().items()}
        optimizer = torch.optim.AdamW(
            [value for value in model.parameters() if value.requires_grad], lr=1e-2
        )
        result = train_stage3_d0_one_epoch(
            model, [make_batch()], optimizer, make_loss(), "cpu",
            dpr_mean=[10.0, 20.0], dpr_std=[2.0, 4.0], support_threshold=0.5,
            thresholds_mm_h=(1.0, 5.0), scaler=None, use_amp=False,
            grad_clip_norm=1.0, accumulation_steps=1, max_batches=None,
        )
        self.assertEqual(result.optimizer_steps, 1)
        self.assertIn("stage2_anchor", result.loss_components)
        self.assertIn("rain_task", result.loss_components)
        self.assertIn("stage2", result.metrics)
        for name, value in model.state_dict().items():
            if name.startswith("rain_head."):
                self.assertFalse(torch.equal(value, before[name]))
            else:
                self.assertTrue(torch.equal(value, before[name]), name)
        validation = evaluate_stage3_d0_one_epoch(
            model, [make_batch()], make_loss(), "cpu",
            dpr_mean=[10.0, 20.0], dpr_std=[2.0, 4.0], support_threshold=0.5,
            thresholds_mm_h=(1.0, 5.0), use_amp=False, max_batches=None,
        )
        self.assertEqual(validation.batch_count, 1)

    def test_decoder_gradient_audit_uses_shared_parameters_and_is_bounded(self) -> None:
        model = make_model(STAGE3_D0_DECODER_AND_HEADS)
        report = audit_stage3_d0_gradient_scale(
            model, [make_batch()], make_loss(1.0), "cpu",
            target_gradient_ratio=0.25, min_rain_weight=1e-4,
            max_rain_weight=1.0, valid_batches_per_rank=1,
            max_candidate_batches_per_rank=1,
        )
        self.assertEqual(report["gradient_parameter_scope"], "trainable_decoder_only_excluding_task_heads")
        self.assertEqual(report["valid_batch_count_global"], 1)
        self.assertGreaterEqual(report["selected_rain_weight"], 1e-4)
        self.assertLessEqual(report["selected_rain_weight"], 1.0)

    def test_rain_primary_audit_scales_physics_not_rain(self) -> None:
        model = make_model(STAGE3_D0_DECODER_AND_HEADS)
        report = audit_stage3_d0_gradient_scale(
            model, [make_batch()], make_loss(1.0), "cpu",
            target_gradient_ratio=0.25, min_rain_weight=1e-4,
            max_rain_weight=1.0, valid_batches_per_rank=1,
            max_candidate_batches_per_rank=1, scaled_task="stage2",
        )
        self.assertEqual(report["objective_mode"], "rain_primary")
        self.assertEqual(report["scaled_task"], "stage2")
        self.assertEqual(report["selected_rain_weight"], 1.0)
        self.assertGreaterEqual(report["selected_stage2_weight"], 1e-4)
        self.assertLessEqual(report["selected_stage2_weight"], 1.0)
        ratio = report["geometric_mean_anchor_to_rain_gradient_ratio"]
        expected = min(max(0.25 / ratio, 1e-4), 1.0)
        self.assertAlmostEqual(report["selected_stage2_weight"], expected)

    def test_rain_primary_total_has_explicit_task_coefficients(self) -> None:
        model = make_model(STAGE3_D0_DECODER_AND_HEADS)
        batch = make_batch()
        output = model(batch["inputs"])
        criterion = make_loss(rain_weight=1.0, stage2_weight=0.2)
        parts = criterion.compute_components(
            rain_prediction=output["rain"],
            support_logits=output["support_logits"],
            reflectivity_prediction=output["reflectivity"],
            rain_target=batch["target"],
            rain_mask=batch["loss_mask"],
            rain_weights=batch["loss_weights"],
            reliable_rain_mask=batch["reliable_loss_mask"],
            height_km=batch["height_km"],
            target_support=batch["stage2_target_support"],
            target_dbz=batch["stage2_target_dbz"],
            support_mask=batch["stage2_support_loss_mask"],
            regression_mask=batch["stage2_regression_mask"],
            regression_weights=batch["stage2_regression_weights"],
        )
        self.assertTrue(
            torch.allclose(parts.total, parts.rain.total + 0.2 * parts.stage2.total)
        )
        self.assertEqual(parts.rain_weight, 1.0)
        self.assertEqual(parts.stage2_weight, 0.2)


if __name__ == "__main__":
    unittest.main()
