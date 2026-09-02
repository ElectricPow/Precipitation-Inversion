"""Tests for masked type loss, confusion metrics, and multitask engine wiring."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from precipitation_inversion.losses.masked_classification import (  # noqa: E402
    MaskedCrossEntropyLoss,
)
from precipitation_inversion.losses.masked_losses import MaskedSmoothL1Loss  # noqa: E402
from precipitation_inversion.metrics.classification import (  # noqa: E402
    MulticlassConfusionMetrics,
)
from precipitation_inversion.training.engine import (  # noqa: E402
    evaluate_one_epoch,
    train_one_epoch,
)


class TinyMultitaskDataset(torch.utils.data.Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        inputs = torch.ones(3, 4, 4, 3) * (index + 1)
        target = torch.full((1, 4, 4, 3), 0.3)
        loss_mask = torch.ones_like(target, dtype=torch.bool)
        type_target = torch.arange(16).reshape(4, 4) % 3
        type_mask = torch.ones(4, 4, dtype=torch.bool)
        type_mask[0, 0] = False
        type_target[0, 0] = -100
        return {
            "inputs": inputs,
            "target": target,
            "loss_mask": loss_mask,
            "type_target": type_target,
            "type_loss_mask": type_mask,
        }


class TinyMultitaskModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rain = torch.nn.Conv3d(3, 1, 1)
        self.types = torch.nn.Conv2d(3, 3, 1)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "rain": self.rain(inputs),
            "type_logits": self.types(inputs.mean(dim=-1)),
        }


class ClassificationTests(unittest.TestCase):
    def test_masked_cross_entropy_ignores_unselected_target(self) -> None:
        logits = torch.tensor([[[[4.0, 0.0]], [[0.0, 4.0]], [[0.0, 0.0]]]])
        target = torch.tensor([[[0, -100]]])
        mask = torch.tensor([[[True, False]]])
        criterion = MaskedCrossEntropyLoss([1.0, 2.0, 3.0])
        loss = criterion(logits, target, mask)
        expected = torch.nn.functional.cross_entropy(logits[:, :, :, :1].permute(0, 2, 3, 1).reshape(1, 3), torch.tensor([0]), weight=torch.tensor([1.0, 2.0, 3.0]))
        torch.testing.assert_close(loss, expected)

    def test_confusion_metrics_report_macro_and_convective_recall(self) -> None:
        metrics = MulticlassConfusionMetrics(("stratiform", "convective", "other"))
        target = torch.tensor([[[0, 1, 1, 2]]])
        prediction = torch.tensor([[[0, 0, 1, 2]]])
        metrics.update(prediction, target, torch.ones_like(target, dtype=torch.bool))
        result = metrics.compute()
        self.assertEqual(result["confusion_matrix"], [[1, 0, 0], [1, 1, 0], [0, 0, 1]])
        self.assertAlmostEqual(result["per_class"]["convective"]["recall"], 0.5)
        self.assertAlmostEqual(result["balanced_accuracy"], (1.0 + 0.5 + 1.0) / 3.0)

    def test_engine_optimizes_and_reports_both_tasks(self) -> None:
        loader = torch.utils.data.DataLoader(TinyMultitaskDataset(), batch_size=1)
        model = TinyMultitaskModel()
        rain_criterion = MaskedSmoothL1Loss(beta=0.2)
        type_criterion = MaskedCrossEntropyLoss([1.0, 1.0, 1.0])
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        result = train_one_epoch(
            model,
            loader,
            optimizer,
            rain_criterion,
            "cpu",
            use_amp=False,
            type_criterion=type_criterion,
            type_loss_weight=0.01,
        )
        self.assertIn("precipitation_type", result.metrics)
        self.assertEqual(result.metrics["precipitation_type"]["count"], 30)
        self.assertEqual(result.loss_components["type_valid_profiles"], 30)
        validation = evaluate_one_epoch(
            model,
            loader,
            rain_criterion,
            "cpu",
            use_amp=False,
            type_criterion=type_criterion,
            type_loss_weight=0.01,
        )
        self.assertIn("weighted_type_cross_entropy", validation.loss_components)


if __name__ == "__main__":
    unittest.main()
