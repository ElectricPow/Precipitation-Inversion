"""Tests for one-epoch training, validation, and checkpoint restoration."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from precipitation_inversion.losses.masked_losses import (  # noqa: E402
    MaskedSmoothL1Loss,
)
from precipitation_inversion.training.engine import (  # noqa: E402
    evaluate_one_epoch,
    load_checkpoint,
    save_checkpoint,
    seed_everything,
    train_one_epoch,
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
        return {"inputs": inputs, "target": target, "loss_mask": mask}


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


if __name__ == "__main__":
    unittest.main()
