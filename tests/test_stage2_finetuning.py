"""Tests for controlled Stage-2 low-learning-rate fine-tuning."""

from __future__ import annotations

import json
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

from precipitation_inversion.training.engine import save_checkpoint  # noqa: E402
from scripts.train_stage2_unet3d import (  # noqa: E402
    BestValidationLosses,
    initialize_model_weights,
    load_config,
    parse_args,
    run_postprocessing,
    write_validation_candidate_comparison,
)


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(3, 2)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layer(values)


def _candidate_metrics(epoch: int, csi: float, mae: float) -> dict[str, object]:
    return {
        "checkpoint_epoch": epoch,
        "metrics": {
            "support": {
                "csi": csi,
                "precision": 0.8,
                "recall": 0.7,
                "f1": 0.75,
            },
            "fss": {
                "1": {"fss": 0.71},
                "2": {"fss": 0.75},
                "4": {"fss": 0.80},
            },
            "reflectivity_on_target_support": {
                "mae_dbz": mae,
                "rmse_dbz": mae + 1.0,
                "bias_dbz": -0.5,
                "pearson_r": 0.72,
                "ccc": 0.70,
            },
            "reflectivity_on_common_support": {
                "mae_dbz": mae - 0.5,
                "pearson_r": 0.80,
            },
        },
    }


class Stage2FineTuningTests(unittest.TestCase):
    def test_finetune_configuration_changes_only_training_control(self) -> None:
        baseline = load_config(PROJECT_ROOT / "configs" / "stage2_unet3d.yaml")
        finetune = load_config(
            PROJECT_ROOT / "configs" / "stage2_unet3d_finetune_lr1e5.yaml"
        )
        self.assertEqual(finetune["optimizer"]["learning_rate"], 1e-5)
        self.assertEqual(finetune["scheduler"]["eta_min"], 1e-6)
        self.assertEqual(finetune["training"]["epochs"], 20)
        self.assertEqual(finetune["training"]["early_stopping"]["patience"], 8)
        self.assertEqual(finetune["loss"], baseline["loss"])
        self.assertEqual(finetune["model"], baseline["model"])
        self.assertEqual(
            finetune["data"]["sampler"], baseline["data"]["sampler"]
        )
        self.assertTrue(finetune["postprocessing"]["compare_task_checkpoints"])
        self.assertFalse(finetune["postprocessing"]["evaluate_test"])

    def test_resume_and_weights_only_initialization_are_mutually_exclusive(self) -> None:
        with mock.patch.object(
            sys,
            "argv",
            ["train", "--resume", "last.pt", "--initialize-from", "best.pt"],
        ), self.assertRaises(SystemExit):
            parse_args()

    def test_weights_only_initialization_cannot_restore_optimizer_state(self) -> None:
        torch.manual_seed(13)
        source = TinyModel()
        source_optimizer = torch.optim.AdamW(source.parameters(), lr=3e-4)
        loss = source(torch.ones(2, 3)).sum()
        loss.backward()
        source_optimizer.step()
        self.assertTrue(source_optimizer.state)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "source.pt"
            save_checkpoint(
                checkpoint,
                source,
                epoch=20,
                global_step=1234,
                optimizer=source_optimizer,
            )
            target = TinyModel()
            metadata = initialize_model_weights(checkpoint, target)
            # The fine-tuning optimizer is constructed after loading weights,
            # exactly as in the DDP training script.
            target_optimizer = torch.optim.AdamW(target.parameters(), lr=1e-5)

        for source_value, target_value in zip(
            source.state_dict().values(), target.state_dict().values()
        ):
            torch.testing.assert_close(source_value, target_value)
        self.assertFalse(target_optimizer.state)
        self.assertEqual(target_optimizer.param_groups[0]["lr"], 1e-5)
        self.assertEqual(metadata["mode"], "weights_only")
        self.assertEqual(metadata["source_epoch"], 20)
        self.assertEqual(metadata["source_global_step"], 1234)

    def test_three_validation_optima_update_independently(self) -> None:
        best = BestValidationLosses()
        first = best.update(
            joint=0.62,
            support=0.16,
            reflectivity=0.46,
            joint_min_delta=1e-4,
            task_min_delta=0.0,
        )
        self.assertEqual(first, {"joint": True, "support": True, "reflectivity": True})
        second = best.update(
            joint=0.63,
            support=0.17,
            reflectivity=0.45,
            joint_min_delta=1e-4,
            task_min_delta=0.0,
        )
        self.assertEqual(
            second, {"joint": False, "support": False, "reflectivity": True}
        )
        restored = BestValidationLosses.from_metrics(best.to_dict())
        self.assertEqual(restored, best)

    def test_candidate_summary_uses_each_independent_val_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = {
                "joint": root / "best_joint.pt",
                "support": root / "best_support.pt",
                "reflectivity": root / "best_dbz.pt",
            }
            for index, (role, checkpoint) in enumerate(candidates.items()):
                checkpoint.touch()
                role_output = root / "validation" / role
                role_output.mkdir(parents=True)
                (role_output / "metrics.json").write_text(
                    json.dumps(_candidate_metrics(2 + index, 0.5 + index / 10, 4.0 - index / 10)),
                    encoding="utf-8",
                )
                (role_output / "support_threshold.json").write_text(
                    json.dumps(
                        {
                            "threshold": 0.6 + index / 10,
                            "selected_on_split": "val",
                        }
                    ),
                    encoding="utf-8",
                )
            payload = write_validation_candidate_comparison(
                root / "validation", candidates
            )
            self.assertEqual(payload["selection_status"], "pending_analysis")
            self.assertEqual(len(payload["candidates"]), 3)
            self.assertEqual(payload["candidates"][2]["support_threshold"], 0.8)
            self.assertTrue((root / "validation" / "comparison.csv").is_file())

    def test_finetune_postprocessing_runs_only_three_complete_val_evaluations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for name in ("best_joint.pt", "best_support.pt", "best_dbz.pt"):
                (output / name).touch()
            config = {
                "postprocessing": {
                    "enabled": True,
                    "device": "cpu",
                    "compare_task_checkpoints": True,
                    "evaluate_test": False,
                    "visualize_test": False,
                }
            }
            with mock.patch(
                "scripts.train_stage2_unet3d.subprocess.run"
            ) as runner, mock.patch(
                "scripts.train_stage2_unet3d.write_validation_candidate_comparison"
            ) as writer:
                run_postprocessing(output, config)
            self.assertEqual(runner.call_count, 3)
            for call in runner.call_args_list:
                command = call.args[0]
                self.assertIn("--split", command)
                self.assertEqual(command[command.index("--split") + 1], "val")
                self.assertIn("--select-threshold", command)
                self.assertNotIn("test", command)
            writer.assert_called_once()
            self.assertNotIn("test/metrics", (output / "analysis" / "README.md").read_text())


if __name__ == "__main__":
    unittest.main()
