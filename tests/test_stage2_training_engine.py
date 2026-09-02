"""Tests for Stage-2 AMP/DDP-compatible train and validation loops."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from precipitation_inversion.losses.stage2_losses import Stage2CompositeLoss  # noqa: E402
from precipitation_inversion.data.stage2_patch_dataset import (  # noqa: E402
    stage2_patch_dataset_kwargs,
)
from precipitation_inversion.training.stage2_engine import (  # noqa: E402
    evaluate_stage2_one_epoch,
    standardized_to_physical_dbz,
    train_stage2_one_epoch,
)
from scripts.train_stage2_unet3d import (  # noqa: E402
    build_model,
    load_config,
    run_postprocessing,
)


class TinyStage2Model(nn.Module):
    def __init__(self, in_channels: int = 4) -> None:
        super().__init__()
        self.shared = nn.Conv3d(in_channels, 2, kernel_size=1)
        self.support = nn.Conv3d(2, 1, kernel_size=1)
        self.reflectivity = nn.Conv3d(2, 1, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        features = torch.tanh(self.shared(inputs))
        return {
            "support_logits": self.support(features),
            "reflectivity": self.reflectivity(features),
        }


def make_batch(
    *, empty_regression: bool = False, in_channels: int = 4
) -> dict[str, torch.Tensor]:
    inputs = torch.randn(2, in_channels, 4, 4, 3)
    support_mask = torch.ones(2, 1, 4, 4, 3, dtype=torch.bool)
    target_support = torch.zeros(2, 1, 4, 4, 3)
    if not empty_regression:
        target_support[:, :, 1:3, 1:3, 1:] = 1.0
    regression_mask = target_support.bool()
    target_dbz = torch.zeros_like(target_support)
    target_dbz[regression_mask] = 0.5
    return {
        "inputs": inputs,
        "target_support": target_support,
        "target_dbz": target_dbz,
        "support_loss_mask": support_mask,
        "regression_mask": regression_mask,
    }


def make_weighted_scalar_batch(
    *, target_dbz: float, regression_weight: float
) -> dict[str, torch.Tensor]:
    """One supervised voxel shaped ``(B=1,C=4,D=H=Z=1)``."""

    return {
        "inputs": torch.zeros(1, 4, 1, 1, 1),
        "target_support": torch.ones(1, 1, 1, 1, 1),
        "target_dbz": torch.full((1, 1, 1, 1, 1), target_dbz),
        "support_loss_mask": torch.ones(1, 1, 1, 1, 1, dtype=torch.bool),
        "regression_mask": torch.ones(1, 1, 1, 1, 1, dtype=torch.bool),
        "regression_weights": torch.full(
            (1, 1, 1, 1, 1), regression_weight
        ),
    }


class Stage2TrainingEngineTests(unittest.TestCase):
    def test_train_and_evaluate_keep_task_counts_and_physical_metrics(self) -> None:
        torch.manual_seed(7)
        model = TinyStage2Model()
        criterion = Stage2CompositeLoss(
            support_pos_weight=2.0,
            reflectivity_beta=0.2,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        batch = make_batch()
        before = model.shared.weight.detach().clone()
        train = train_stage2_one_epoch(
            model,
            [batch],
            optimizer,
            criterion,
            "cpu",
            dpr_mean=[10.0, 20.0, 30.0],
            dpr_std=[2.0, 4.0, 8.0],
            support_threshold=0.5,
            use_amp=False,
        )
        self.assertEqual(train.batch_count, 1)
        self.assertEqual(train.optimizer_steps, 1)
        self.assertEqual(train.support_voxels, 2 * 4 * 4 * 3)
        self.assertEqual(train.support_positive_voxels, 2 * 2 * 2 * 2)
        self.assertEqual(train.reflectivity_voxels, train.support_positive_voxels)
        self.assertEqual(
            train.metrics["reflectivity_on_target_support"]["count"],
            train.reflectivity_voxels,
        )
        self.assertFalse(torch.equal(before, model.shared.weight.detach()))
        self.assertAlmostEqual(
            train.loss,
            train.loss_components["weighted_support"]
            + train.loss_components["weighted_reflectivity"],
        )

        model.train()
        validation = evaluate_stage2_one_epoch(
            model,
            [batch],
            criterion,
            "cpu",
            dpr_mean=[10.0, 20.0, 30.0],
            dpr_std=[2.0, 4.0, 8.0],
            support_threshold=0.4,
            fss_radii=(1,),
            use_amp=False,
        )
        self.assertTrue(model.training)
        self.assertEqual(validation.optimizer_steps, 0)
        self.assertEqual(validation.metrics["support_threshold"], 0.4)
        self.assertIn("1", validation.metrics["fss"])

    def test_empty_regression_mask_keeps_support_training_finite(self) -> None:
        model = TinyStage2Model()
        criterion = Stage2CompositeLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        result = train_stage2_one_epoch(
            model,
            [make_batch(empty_regression=True)],
            optimizer,
            criterion,
            "cpu",
            dpr_mean=[0.0, 0.0, 0.0],
            dpr_std=[1.0, 1.0, 1.0],
            use_amp=False,
        )
        self.assertEqual(result.reflectivity_voxels, 0)
        self.assertEqual(result.loss_components["reflectivity_standardized_dbz"], 0.0)
        self.assertTrue(torch.isfinite(torch.tensor(result.loss)))

    def test_epoch_reflectivity_loss_aggregates_by_selected_weight_sum(self) -> None:
        model = TinyStage2Model()
        for parameter in model.parameters():
            nn.init.zeros_(parameter)
        criterion = Stage2CompositeLoss(reflectivity_beta=1.0)
        result = evaluate_stage2_one_epoch(
            model,
            [
                make_weighted_scalar_batch(target_dbz=1.0, regression_weight=1.0),
                make_weighted_scalar_batch(target_dbz=2.0, regression_weight=3.0),
            ],
            criterion,
            "cpu",
            dpr_mean=[0.0],
            dpr_std=[1.0],
            use_amp=False,
        )
        # Smooth-L1 values are 0.5 and 1.5. Global weighted mean is
        # (0.5*1 + 1.5*3) / 4 = 1.25, not their old count mean 1.0.
        self.assertAlmostEqual(
            result.loss_components["reflectivity_standardized_dbz"], 1.25
        )
        self.assertEqual(result.loss_components["reflectivity_count"], 2)
        self.assertEqual(result.loss_components["reflectivity_weight_sum"], 4.0)

    def test_engine_rejects_invalid_regression_weight_contract(self) -> None:
        model = TinyStage2Model()
        criterion = Stage2CompositeLoss()
        for mutation, message in (
            (lambda batch: batch.update(regression_weights=torch.ones(1)), "shapes"),
            (
                lambda batch: batch.update(
                    regression_weights=torch.ones_like(batch["target_dbz"], dtype=torch.int64)
                ),
                "floating-point",
            ),
            (
                lambda batch: batch.update(
                    regression_weights=torch.where(
                        batch["regression_mask"],
                        -torch.ones_like(batch["target_dbz"]),
                        torch.zeros_like(batch["target_dbz"]),
                    )
                ),
                "non-negative",
            ),
        ):
            with self.subTest(message=message):
                batch = make_batch()
                mutation(batch)
                with self.assertRaisesRegex((TypeError, ValueError), message):
                    evaluate_stage2_one_epoch(
                        model,
                        [batch],
                        criterion,
                        "cpu",
                        dpr_mean=[0.0, 0.0, 0.0],
                        dpr_std=[1.0, 1.0, 1.0],
                        use_amp=False,
                    )

    def test_three_channel_batch_runs_through_training_contract(self) -> None:
        model = TinyStage2Model(in_channels=3)
        criterion = Stage2CompositeLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        result = train_stage2_one_epoch(
            model,
            [make_batch(in_channels=3)],
            optimizer,
            criterion,
            "cpu",
            dpr_mean=[10.0, 20.0, 30.0],
            dpr_std=[2.0, 4.0, 8.0],
            use_amp=False,
        )
        self.assertEqual(result.batch_count, 1)
        self.assertTrue(torch.isfinite(torch.tensor(result.loss)))

    def test_five_channel_batch_runs_through_training_contract(self) -> None:
        model = TinyStage2Model(in_channels=5)
        criterion = Stage2CompositeLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        result = train_stage2_one_epoch(
            model,
            [make_batch(in_channels=5)],
            optimizer,
            criterion,
            "cpu",
            dpr_mean=[10.0, 20.0, 30.0],
            dpr_std=[2.0, 4.0, 8.0],
            use_amp=False,
        )
        self.assertEqual(result.batch_count, 1)
        self.assertTrue(torch.isfinite(torch.tensor(result.loss)))

    def test_inverse_standardization_broadcasts_only_over_height(self) -> None:
        values = torch.ones(2, 1, 3, 4, 2)
        physical = standardized_to_physical_dbz(
            values, mean=[10.0, 20.0], std=[2.0, 4.0]
        )
        torch.testing.assert_close(physical[..., 0], torch.full_like(physical[..., 0], 12.0))
        torch.testing.assert_close(physical[..., 1], torch.full_like(physical[..., 1], 24.0))
        with self.assertRaisesRegex(ValueError, "shape"):
            standardized_to_physical_dbz(values, [1.0], [1.0])

    def test_invalid_batch_mask_and_threshold_are_rejected(self) -> None:
        model = TinyStage2Model()
        criterion = Stage2CompositeLoss()
        batch = make_batch()
        batch["support_loss_mask"] = batch["support_loss_mask"].float()
        with self.assertRaisesRegex(TypeError, "boolean"):
            evaluate_stage2_one_epoch(
                model,
                [batch],
                criterion,
                "cpu",
                dpr_mean=[0.0, 0.0, 0.0],
                dpr_std=[1.0, 1.0, 1.0],
                use_amp=False,
            )
        with self.assertRaisesRegex(ValueError, r"\[0,1\]"):
            evaluate_stage2_one_epoch(
                model,
                [],
                criterion,
                "cpu",
                dpr_mean=[0.0, 0.0, 0.0],
                dpr_std=[1.0, 1.0, 1.0],
                support_threshold=2.0,
                use_amp=False,
            )

    def test_formal_configurations_build_matching_channel_models(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "stage2_unet3d.yaml")
        model = build_model(config)
        self.assertEqual(model.in_channels, 4)
        self.assertEqual(config["training"]["early_stopping"]["monitor"], "val_loss")
        self.assertEqual(config["loss"]["support"]["pos_weight"], 5.0)
        self.assertEqual(
            config["data"]["sampler"]["stratum_weights"]["background"], 0.20
        )

        three_channel = load_config(
            PROJECT_ROOT / "configs" / "stage2_unet3d_3ch_native.yaml"
        )
        three_channel_model = build_model(three_channel)
        self.assertEqual(three_channel_model.in_channels, 3)
        self.assertEqual(
            three_channel["data"]["input_channels"],
            [
                "dbz_gr_sparse_standardized",
                "gr_native_available",
                "height_scaled",
            ],
        )
        baseline_comparable = dict(config)
        three_comparable = dict(three_channel)
        self.assertEqual(
            three_comparable["data"]["sampler"],
            baseline_comparable["data"]["sampler"],
        )
        self.assertEqual(three_comparable["loss"], baseline_comparable["loss"])

        five_channel = load_config(
            PROJECT_ROOT / "configs" / "stage2_unet3d_5ch_interp.yaml"
        )
        five_channel_model = build_model(five_channel)
        self.assertEqual(five_channel_model.in_channels, 5)
        self.assertEqual(
            five_channel["data"]["input_channels"],
            [
                "dbz_gr_sparse_standardized",
                "gr_value_mask",
                "dbz_gr_interp_standardized",
                "gr_interp_value_mask",
                "height_scaled",
            ],
        )
        # The five-channel experiment changes only its GR-derived inputs and
        # model input width; sampler, losses and training policy remain 3V's.
        value_config = load_config(
            PROJECT_ROOT / "configs" / "stage2_unet3d_3ch_value.yaml"
        )
        self.assertEqual(five_channel["data"]["sampler"], value_config["data"]["sampler"])
        self.assertEqual(five_channel["loss"], value_config["loss"])
        self.assertEqual(five_channel["optimizer"], value_config["optimizer"])
        self.assertEqual(five_channel["scheduler"], value_config["scheduler"])
        self.assertEqual(five_channel["training"], value_config["training"])

        distance_config = load_config(
            PROJECT_ROOT / "configs" / "stage2_unet3d_4ch_distance.yaml"
        )
        distance_model = build_model(distance_config)
        self.assertEqual(distance_model.in_channels, 4)
        self.assertEqual(
            distance_config["data"]["input_channels"],
            [
                "dbz_gr_sparse_standardized",
                "gr_value_mask",
                "gr_nearest_distance_scaled",
                "height_scaled",
            ],
        )
        # S2-3V-D is a strict single-factor experiment: only the new distance
        # channel, model input width, experiment name and output directory may
        # differ from the three-channel value-mask parent.
        self.assertEqual(
            distance_config["data"]["sampler"], value_config["data"]["sampler"]
        )
        self.assertEqual(distance_config["loss"], value_config["loss"])
        self.assertEqual(distance_config["optimizer"], value_config["optimizer"])
        self.assertEqual(distance_config["scheduler"], value_config["scheduler"])
        self.assertEqual(distance_config["training"], value_config["training"])
        self.assertFalse(distance_config["postprocessing"]["evaluate_test"])
        self.assertFalse(distance_config["postprocessing"]["visualize_test"])

        density_config = load_config(
            PROJECT_ROOT / "configs" / "stage2_unet3d_4ch_density.yaml"
        )
        density_model = build_model(density_config)
        self.assertEqual(density_model.in_channels, 4)
        self.assertEqual(
            density_config["data"]["input_channels"],
            [
                "dbz_gr_sparse_standardized",
                "gr_value_mask",
                "gr_local_density_scaled",
                "height_scaled",
            ],
        )
        # S2-3V-rho is independent of S2-3V-D: the sole additional input is
        # same-height 5x5 GR value density. It inherits every optimization and
        # model-capacity choice from 3V and cannot automatically access test.
        density_data_comparable = dict(density_config["data"])
        value_data_comparable = dict(value_config["data"])
        density_data_comparable.pop("input_channels")
        value_data_comparable.pop("input_channels")
        self.assertEqual(density_data_comparable, value_data_comparable)
        self.assertEqual(density_config["loss"], value_config["loss"])
        self.assertEqual(density_config["optimizer"], value_config["optimizer"])
        self.assertEqual(density_config["scheduler"], value_config["scheduler"])
        self.assertEqual(density_config["training"], value_config["training"])
        density_model_comparable = dict(density_config["model"])
        value_model_comparable = dict(value_config["model"])
        density_model_comparable.pop("in_channels")
        value_model_comparable.pop("in_channels")
        self.assertEqual(density_model_comparable, value_model_comparable)
        self.assertEqual(density_config["evaluation"], value_config["evaluation"])
        self.assertEqual(
            density_config["postprocessing"], value_config["postprocessing"]
        )
        self.assertEqual(
            density_config["runtime"], value_config["runtime"]
        )
        self.assertFalse(density_config["postprocessing"]["evaluate_test"])
        self.assertFalse(density_config["postprocessing"]["visualize_test"])

        e3 = load_config(
            PROJECT_ROOT
            / "configs"
            / "stage2_unet3d_4ch_distance_intensity_w1p5.yaml"
        )
        e3_model = build_model(e3)
        self.assertEqual(e3_model.in_channels, 4)
        # S2-E3-W1.5 is a strict loss-only child of distance D. Every data,
        # model, optimizer, schedule, and training setting remains identical.
        self.assertEqual(e3["data"], distance_config["data"])
        self.assertEqual(e3["model"], distance_config["model"])
        self.assertEqual(e3["optimizer"], distance_config["optimizer"])
        self.assertEqual(e3["scheduler"], distance_config["scheduler"])
        self.assertEqual(e3["training"], distance_config["training"])
        self.assertEqual(e3["evaluation"], distance_config["evaluation"])
        # E3 additionally requests the already-existing three-checkpoint
        # validation comparison after training; this changes no optimization.
        e3_postprocessing = dict(e3["postprocessing"])
        distance_postprocessing = dict(distance_config["postprocessing"])
        self.assertTrue(e3_postprocessing.pop("compare_task_checkpoints"))
        self.assertFalse(distance_postprocessing.pop("compare_task_checkpoints"))
        self.assertEqual(e3_postprocessing, distance_postprocessing)
        self.assertEqual(e3["runtime"], distance_config["runtime"])
        self.assertEqual(e3["loss"]["support"], distance_config["loss"]["support"])
        e3_reflectivity = dict(e3["loss"]["reflectivity"])
        self.assertEqual(
            e3_reflectivity.pop("intensity_bin_edges_dbz"), [25.0, 35.0]
        )
        self.assertEqual(
            e3_reflectivity.pop("intensity_bin_weights"), [1.0, 1.25, 1.5]
        )
        self.assertEqual(e3_reflectivity, distance_config["loss"]["reflectivity"])
        self.assertEqual(
            stage2_patch_dataset_kwargs(e3["loss"]),
            {
                "reflectivity_intensity_bin_edges_dbz": (25.0, 35.0),
                "reflectivity_intensity_bin_weights": (1.0, 1.25, 1.5),
            },
        )

        e3_w1p25 = load_config(
            PROJECT_ROOT
            / "configs"
            / "stage2_unet3d_4ch_distance_intensity_w1p25.yaml"
        )
        e3_w1p25_model = build_model(e3_w1p25)
        self.assertEqual(e3_w1p25_model.in_channels, 4)
        self.assertEqual(
            e3_w1p25["experiment"],
            {
                "name": "stage2_four_channel_distance_intensity_w1p25",
                "seed": 2026,
                "output_dir": (
                    "outputs/stage2_ablations/"
                    "four_channel_distance_intensity_w1p25"
                ),
            },
        )
        # W1.25 is a strict dose-response experiment against W1.5: input,
        # architecture, sampling, optimization, checkpoint cadence and
        # validation protocol are identical. Only the two non-unit physical
        # dBZ bin weights and auditable experiment identity may differ.
        for key in (
            "data",
            "model",
            "optimizer",
            "scheduler",
            "training",
            "evaluation",
            "postprocessing",
            "runtime",
        ):
            self.assertEqual(e3_w1p25[key], e3[key])
        self.assertEqual(
            e3_w1p25["loss"]["support"], distance_config["loss"]["support"]
        )
        e3_w1p25_reflectivity = dict(e3_w1p25["loss"]["reflectivity"])
        self.assertEqual(
            e3_w1p25_reflectivity.pop("intensity_bin_edges_dbz"),
            [25.0, 35.0],
        )
        self.assertEqual(
            e3_w1p25_reflectivity.pop("intensity_bin_weights"),
            [1.0, 1.10, 1.25],
        )
        self.assertEqual(
            e3_w1p25_reflectivity,
            distance_config["loss"]["reflectivity"],
        )
        self.assertEqual(
            stage2_patch_dataset_kwargs(e3_w1p25["loss"]),
            {
                "reflectivity_intensity_bin_edges_dbz": (25.0, 35.0),
                "reflectivity_intensity_bin_weights": (1.0, 1.10, 1.25),
            },
        )

    def test_postprocessing_selects_val_threshold_before_test_and_visualization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "best.pt").touch()
            config = {
                "postprocessing": {
                    "enabled": True,
                    "device": "cpu",
                    "test_sample_count": 2,
                }
            }
            with mock.patch.dict(
                "os.environ", {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "3"}
            ), mock.patch("scripts.train_stage2_unet3d.subprocess.run") as runner:
                run_postprocessing(output, config)
            self.assertEqual(runner.call_count, 3)
            commands = [call.args[0] for call in runner.call_args_list]
            self.assertIn("--select-threshold", commands[0])
            self.assertIn("--threshold-file", commands[1])
            self.assertIn("visualize_stage2_test_predictions.py", commands[2][1])
            for call in runner.call_args_list:
                self.assertNotIn("RANK", call.kwargs["env"])
                self.assertNotIn("WORLD_SIZE", call.kwargs["env"])


if __name__ == "__main__":
    unittest.main()
