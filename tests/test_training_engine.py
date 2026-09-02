"""Tests for one-epoch training, validation, and checkpoint restoration."""

from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from precipitation_inversion.losses.masked_losses import (  # noqa: E402
    MaskedSmoothL1Loss,
    Stage1CompositeLoss,
)
from precipitation_inversion.metrics.regression import (  # noqa: E402
    FilewisePrecipitationMetrics,
    PhysicalRainGradientMetrics,
)
from precipitation_inversion.training.engine import (  # noqa: E402
    evaluate_one_epoch,
    load_checkpoint,
    save_checkpoint,
    seed_everything,
    should_save_periodic_checkpoint,
    train_one_epoch,
    validate_checkpoint_every,
    validate_training_output_directory,
)


class TinyPatchDataset(torch.utils.data.Dataset):
    """Return miniature tensors with the same channel-first contract as NC data."""

    def __init__(self, size: int = 3) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        # inputs=(C=3,D=4,H=4,Z=3); target/mask=(1,4,4,3).
        inputs = torch.zeros(3, 4, 4, 3, dtype=torch.float32)
        inputs[0].fill_(float(index + 1))
        inputs[1].fill_(1.0)
        inputs[2] = torch.linspace(-1.0, 1.0, 3).view(1, 1, 3)
        target = torch.zeros(1, 4, 4, 3, dtype=torch.float32)
        target[:, 1:3, 1:3, :] = 0.5 * (index + 1)
        mask = torch.zeros_like(target, dtype=torch.bool)
        mask[:, 1:3, 1:3, :] = True
        return {
            "inputs": inputs,
            "target": target,
            "loss_mask": mask,
            "file_id": torch.tensor(index, dtype=torch.int64),
            "height_km": torch.tensor([0.0, 0.5, 1.5]).reshape(1, 1, 1, 3),
            "precipitation_type": torch.ones(4, 4),
        }


class WeightedTinyDataset(torch.utils.data.Dataset):
    """One batch containing reliable and weak-CFB supervision."""

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        inputs = torch.zeros(3, 1, 1, 2, dtype=torch.float32)
        target = torch.tensor([[[[0.5, 1.0]]]], dtype=torch.float32)
        loss_mask = torch.ones_like(target, dtype=torch.bool)
        reliable = torch.tensor([[[[True, False]]]])
        weights = torch.tensor([[[[1.0, 0.1]]]], dtype=torch.float32)
        return {
            "inputs": inputs,
            "target": target,
            "loss_mask": loss_mask,
            "reliable_loss_mask": reliable,
            "loss_weights": weights,
            # The second voxel is a native-positive target one level below CFB.
            # These fields are evaluation diagnostics and do not change loss.
            "diagnostic_target": target.clone(),
            "native_positive_mask": torch.tensor([[[[False, True]]]]),
            "cfb_distance_km": torch.tensor([[[[0.0, -0.25]]]]),
            "height_km": torch.tensor([0.0, 0.25]).reshape(1, 1, 1, 2),
            "precipitation_type": torch.ones(1, 1),
            "file_id": torch.tensor(0, dtype=torch.int64),
        }


class DDPStyleWrapper(torch.nn.Module):
    """Expose ``module`` like DDP and fail if its wrapper forward is used."""

    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        raise AssertionError("evaluation must bypass the DDP-style wrapper")


class TrainingEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        seed_everything(123)
        self.loader = torch.utils.data.DataLoader(
            TinyPatchDataset(), batch_size=1, shuffle=False
        )
        self.model = torch.nn.Conv3d(3, 1, kernel_size=1)
        self.criterion = MaskedSmoothL1Loss(beta=0.2)

    def test_train_updates_parameters_and_supports_accumulation(self) -> None:
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-2)
        before = self.model.weight.detach().clone()
        result = train_one_epoch(
            self.model,
            self.loader,
            optimizer,
            self.criterion,
            "cpu",
            use_amp=False,
            accumulation_steps=2,
            grad_clip_norm=1.0,
        )
        self.assertEqual(result.batch_count, 3)
        self.assertEqual(result.optimizer_steps, 2)
        self.assertEqual(result.valid_voxels, 3 * 12)
        self.assertTrue(torch.isfinite(torch.tensor(result.loss)))
        self.assertEqual(result.metrics["rain"]["all"]["count"], 36)
        self.assertFalse(torch.equal(before, self.model.weight.detach()))

    def test_evaluation_does_not_update_parameters_and_restores_mode(self) -> None:
        self.model.train()
        before = [parameter.detach().clone() for parameter in self.model.parameters()]
        result = evaluate_one_epoch(
            self.model,
            self.loader,
            self.criterion,
            "cpu",
            use_amp=False,
            max_batches=2,
        )
        self.assertEqual(result.batch_count, 2)
        self.assertEqual(result.optimizer_steps, 0)
        self.assertEqual(result.valid_voxels, 24)
        self.assertTrue(self.model.training)
        for old, new in zip(before, self.model.parameters()):
            torch.testing.assert_close(old, new)

    def test_evaluation_uses_unwrapped_module_for_uneven_ddp_shards(self) -> None:
        wrapped = DDPStyleWrapper(torch.nn.Conv3d(3, 1, kernel_size=1))
        wrapped.train()
        result = evaluate_one_epoch(
            wrapped,
            self.loader,
            self.criterion,
            "cpu",
            use_amp=False,
            max_batches=1,
        )
        self.assertEqual(result.batch_count, 1)
        self.assertTrue(wrapped.training)

    def test_weighted_loss_keeps_primary_metrics_on_reliable_mask(self) -> None:
        model = torch.nn.Conv3d(3, 1, kernel_size=1)
        with torch.no_grad():
            model.weight.zero_()
            model.bias.zero_()
        loader = torch.utils.data.DataLoader(WeightedTinyDataset(), batch_size=1)
        result = evaluate_one_epoch(
            model, loader, self.criterion, "cpu", use_amp=False
        )
        batch = next(iter(loader))
        expected = self.criterion(
            model(batch["inputs"]),
            batch["target"],
            batch["loss_mask"],
            batch["loss_weights"],
        )
        self.assertAlmostEqual(result.loss, float(expected.detach()), places=7)
        self.assertEqual(result.valid_voxels, 2)  # reliable + weak loss locations
        self.assertEqual(result.metrics["rain"]["all"]["count"], 1)
        below = result.metrics["diagnostics"]["below_cfb_native_positive"]
        self.assertEqual(below["rain"]["all"]["count"], 1)
        self.assertAlmostEqual(
            below["rain"]["all"]["bias"], -math.expm1(1.0), places=6
        )

    def test_filewise_macro_metrics_are_optional_and_use_file_ids(self) -> None:
        filewise = FilewisePrecipitationMetrics(
            ("first", "second", "third"), bootstrap_replicates=25
        )
        result = evaluate_one_epoch(
            self.model,
            self.loader,
            self.criterion,
            "cpu",
            use_amp=False,
            max_batches=2,
            filewise_metrics=filewise,
        )
        values = result.metrics["filewise"]
        self.assertEqual(values["file_count_nonempty"], 2)
        self.assertEqual(values["valid_voxel_count"], 24)
        self.assertEqual(values["per_file"]["first"]["count"], 12)
        self.assertEqual(values["per_file"]["second"]["count"], 12)
        self.assertEqual(values["per_file"]["third"]["count"], 0)

    def test_physical_drdz_metrics_use_reliable_adjacent_pairs(self) -> None:
        gradients = PhysicalRainGradientMetrics(
            (0.0, 0.5, 1.5),
            file_labels=("first", "second", "third"),
            bootstrap_replicates=25,
        )
        result = evaluate_one_epoch(
            self.model,
            self.loader,
            self.criterion,
            "cpu",
            use_amp=False,
            max_batches=2,
            physical_gradient_metrics=gradients,
        )
        values = result.metrics["physical_drdz"]
        # Each sample has four selected horizontal profiles and two adjacent
        # vertical pairs: 2 batches * 4 profiles * 2 pairs = 16.
        self.assertEqual(values["all"]["count"], 16)
        self.assertEqual(values["filewise"]["valid_voxel_count"], 16)

        weak_loader = torch.utils.data.DataLoader(
            WeightedTinyDataset(), batch_size=1
        )
        weak_gradients = PhysicalRainGradientMetrics((0.0, 0.25))
        weak_result = evaluate_one_epoch(
            torch.nn.Conv3d(3, 1, kernel_size=1),
            weak_loader,
            self.criterion,
            "cpu",
            use_amp=False,
            physical_gradient_metrics=weak_gradients,
        )
        # The total loss mask has two endpoints, but only the first is reliable;
        # W's weak below-CFB endpoint must therefore not create a primary pair.
        self.assertEqual(
            weak_result.metrics["physical_drdz"]["all"]["count"], 0
        )

    def test_composite_gradient_loss_is_logged_with_its_own_pair_support(self) -> None:
        criterion = Stage1CompositeLoss(
            primary_beta=0.2,
            physical_gradient_weight=0.02,
            physical_gradient_beta=1.0,
        )
        result = evaluate_one_epoch(
            self.model,
            self.loader,
            criterion,
            "cpu",
            use_amp=False,
            max_batches=2,
        )
        components = result.loss_components
        # Each batch has four horizontal profiles and two valid vertical pairs.
        self.assertEqual(components["physical_drdz_pair_count"], 16)
        self.assertTrue(components["physical_drdz_enabled"])
        self.assertEqual(components["physical_drdz_weight"], 0.02)
        self.assertAlmostEqual(
            result.loss,
            components["primary_log_smooth_l1"]
            + components["weighted_physical_drdz"],
            places=7,
        )

        weak_loader = torch.utils.data.DataLoader(
            WeightedTinyDataset(), batch_size=1
        )
        weak_result = evaluate_one_epoch(
            torch.nn.Conv3d(3, 1, kernel_size=1),
            weak_loader,
            criterion,
            "cpu",
            use_amp=False,
        )
        # loss_mask contains reliable+weak endpoints, but G receives the
        # reliable mask and therefore cannot form a pair through W's label.
        self.assertEqual(
            weak_result.loss_components["physical_drdz_pair_count"], 0
        )
        self.assertEqual(
            weak_result.loss_components["physical_drdz_smooth_l1"], 0.0
        )

    def test_composite_gradient_loss_trains_and_backpropagates(self) -> None:
        criterion = Stage1CompositeLoss(
            primary_beta=0.2,
            physical_gradient_weight=0.02,
            physical_gradient_beta=1.0,
        )
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        before = self.model.weight.detach().clone()
        result = train_one_epoch(
            self.model,
            self.loader,
            optimizer,
            criterion,
            "cpu",
            use_amp=False,
            max_batches=1,
        )
        self.assertEqual(result.loss_components["physical_drdz_pair_count"], 8)
        self.assertTrue(math.isfinite(result.loss))
        self.assertFalse(torch.equal(before, self.model.weight.detach()))

    def test_checkpoint_round_trip_restores_model_optimizer_and_metadata(self) -> None:
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
        train_one_epoch(
            self.model,
            self.loader,
            optimizer,
            self.criterion,
            "cpu",
            use_amp=False,
            max_batches=1,
        )
        expected = {
            name: value.detach().clone() for name, value in self.model.state_dict().items()
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(
                path,
                self.model,
                epoch=4,
                global_step=17,
                optimizer=optimizer,
                scheduler=scheduler,
                config={"model": {"base_channels": 16}},
                metrics={"val_loss": 0.25},
            )
            with torch.no_grad():
                for parameter in self.model.parameters():
                    parameter.zero_()
            restored_optimizer = torch.optim.AdamW(self.model.parameters(), lr=9e-1)
            restored_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                restored_optimizer, T_max=5
            )
            checkpoint = load_checkpoint(
                path,
                self.model,
                optimizer=restored_optimizer,
                scheduler=restored_scheduler,
            )
        self.assertEqual(checkpoint["epoch"], 4)
        self.assertEqual(checkpoint["global_step"], 17)
        self.assertEqual(checkpoint["metrics"]["val_loss"], 0.25)
        self.assertAlmostEqual(restored_optimizer.param_groups[0]["lr"], 1e-2)
        for name, value in self.model.state_dict().items():
            torch.testing.assert_close(value, expected[name])

    def test_seed_and_argument_validation(self) -> None:
        seed_everything(7)
        first = torch.rand(3)
        seed_everything(7)
        torch.testing.assert_close(first, torch.rand(3))
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        with self.assertRaises(ValueError):
            train_one_epoch(
                self.model,
                self.loader,
                optimizer,
                self.criterion,
                "cpu",
                accumulation_steps=0,
            )
        with self.assertRaises(ValueError):
            evaluate_one_epoch(
                self.model,
                self.loader,
                self.criterion,
                "cpu",
                max_batches=0,
            )

    def test_periodic_checkpoint_policy_uses_zero_based_0009_cadence(self) -> None:
        self.assertEqual(validate_checkpoint_every(), 10)
        due = [
            epoch
            for epoch in range(30)
            if should_save_periodic_checkpoint(epoch)
        ]
        self.assertEqual(due, [9, 19, 29])
        self.assertTrue(should_save_periodic_checkpoint(19, 20))
        self.assertFalse(should_save_periodic_checkpoint(9, 20))
        self.assertFalse(should_save_periodic_checkpoint(999, 0))

    def test_periodic_checkpoint_policy_rejects_unsafe_intervals(self) -> None:
        for value in (-1, 1, 9):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_checkpoint_every(value)
        for value in (True, 10.0, "10"):
            with self.subTest(value=value), self.assertRaises(TypeError):
                validate_checkpoint_every(value)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            should_save_periodic_checkpoint(-1)
        with self.assertRaises(TypeError):
            should_save_periodic_checkpoint(True)  # type: ignore[arg-type]

    def test_output_directory_guard_only_allows_existing_history_on_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "run"
            output_dir.mkdir()
            self.assertEqual(
                validate_training_output_directory(output_dir),
                output_dir.resolve(),
            )
            (output_dir / "metrics.jsonl").write_text("{}\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                validate_training_output_directory(output_dir)
            with self.assertRaises(FileExistsError):
                validate_training_output_directory(
                    output_dir, initialize_from="source_best.pt"
                )
            self.assertEqual(
                validate_training_output_directory(output_dir, resume="last.pt"),
                output_dir.resolve(),
            )


if __name__ == "__main__":
    unittest.main()
