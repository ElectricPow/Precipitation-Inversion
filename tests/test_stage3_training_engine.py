"""Tests that the C1 engine updates Stage 1 and never Stage 2."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from precipitation_inversion.losses.stage3_losses import (  # noqa: E402
    build_stage3_c1_loss,
)
from precipitation_inversion.training.stage3_engine import (  # noqa: E402
    evaluate_stage3_c1_one_epoch,
    train_stage3_c1_one_epoch,
)
from tests.test_stage3_cascade import make_cascade  # noqa: E402


class Stage3TrainingEngineTests(unittest.TestCase):
    def _batch(self) -> dict[str, torch.Tensor]:
        shape = (1, 1, 4, 4, 2)
        packed = torch.zeros((1, 4, 4, 4, 2))
        packed[:, 2] = 1.0
        mask = torch.ones(shape, dtype=torch.bool)
        return {
            "inputs": packed,
            "target": torch.full(shape, 0.5),
            "loss_mask": mask,
            "loss_weights": torch.ones(shape),
            "output_mask": mask,
            "reliable_loss_mask": mask,
            "height_km": torch.tensor([[[[[0.5, 1.0]]]]]),
        }

    def test_one_epoch_preserves_frozen_parameters(self) -> None:
        model = make_cascade()
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-3
        )
        criterion = build_stage3_c1_loss(
            {
                "name": "masked_smooth_l1",
                "beta": 0.2,
                "physical_gradient": {"enabled": True, "weight": 0.02, "beta": 1.0},
            }
        )
        stage2_before = {
            name: value.detach().clone()
            for name, value in model.stage2_model.state_dict().items()
        }
        result = train_stage3_c1_one_epoch(
            model,
            [self._batch()],
            optimizer,
            criterion,
            "cpu",
            use_amp=False,
            thresholds_mm_h=(1.0, 5.0),
        )
        self.assertEqual(result.optimizer_steps, 1)
        for name, value in model.stage2_model.state_dict().items():
            self.assertTrue(torch.equal(value, stage2_before[name]))
        validation = evaluate_stage3_c1_one_epoch(
            model,
            [self._batch()],
            criterion,
            "cpu",
            use_amp=False,
            thresholds_mm_h=(1.0, 5.0),
        )
        self.assertEqual(validation.batch_count, 1)


if __name__ == "__main__":
    unittest.main()
