"""Tests for epoch-history extraction and plotting."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.plot_stage1_training_history import (  # noqa: E402
    generate_training_analysis,
    load_history,
)


def metric(value: float) -> dict[str, float | int]:
    return {
        "count": 10,
        "mae": value,
        "rmse": value * 2,
        "bias": -value / 2,
        "r2": 1 - value,
        "pearson_r": 1 - value / 2,
    }


def epoch_row(epoch: int, train_value: float, val_value: float) -> dict:
    bins = {
        name: metric(val_value * multiplier)
        for name, multiplier in zip(
            ("lt_1", "1_to_5", "5_to_10", "10_to_30", "ge_30"),
            (1, 2, 3, 4, 5),
        )
    }
    return {
        "epoch": epoch,
        "global_step": (epoch + 1) * 3,
        "learning_rate": 1e-4 / (epoch + 1),
        "train": {
            "loss": train_value,
            "loss_components": {
                "primary_log_smooth_l1": train_value * 0.9,
                "physical_drdz_smooth_l1": train_value * 5.0,
                "weighted_physical_drdz": train_value * 0.1,
                "weighted_physical_drdz_fraction": 0.1,
            },
            "duration_seconds": 2.0,
            "metrics": {"log": metric(train_value), "rain": {"all": metric(train_value)}},
        },
        "val": {
            "loss": val_value,
            "loss_components": {
                "primary_log_smooth_l1": val_value * 0.9,
                "physical_drdz_smooth_l1": val_value * 5.0,
                "weighted_physical_drdz": val_value * 0.1,
                "weighted_physical_drdz_fraction": 0.1,
            },
            "duration_seconds": 0.5,
            "metrics": {
                "log": metric(val_value),
                "rain": {"all": metric(val_value), "target_bins_mm_h": bins},
            },
        },
    }


class TrainingVisualizationTests(unittest.TestCase):
    def test_generates_figures_csv_and_best_epoch_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            # Input order is deliberately shuffled; loader must sort by epoch.
            rows = [epoch_row(2, 0.2, 0.4), epoch_row(0, 0.8, 0.9), epoch_row(1, 0.4, 0.3)]
            (output / "metrics.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
            )
            config = {
                "model": {"in_channels": 3, "out_channels": 1, "base_channels": 4},
                "data": {"batch_size": 1},
                "training": {"accumulation_steps": 1, "amp": False},
                "optimizer": {"name": "adamw", "learning_rate": 1e-4},
                "loss": {"name": "masked_smooth_l1"},
            }
            (output / "resolved_config.json").write_text(json.dumps(config), encoding="utf-8")
            summary = generate_training_analysis(output, dpi=35)
            analysis = output / "analysis" / "training_history"
            self.assertEqual(summary["best_epoch_by_validation_rain_rmse"], 1)
            for name in (
                "training_overview.png",
                "loss_components.png",
                "validation_intensity_bins.png",
                "generalization_gap.png",
                "epoch_metrics.csv",
                "summary.json",
                "summary.md",
            ):
                self.assertTrue((analysis / name).is_file(), name)
                self.assertGreater((analysis / name).stat().st_size, 0)
            self.assertEqual(
                summary["best_validation_loss_components"][
                    "weighted_physical_drdz_fraction"
                ],
                0.1,
            )

    def test_invalid_json_line_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            path.write_text("not-json\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid JSON"):
                load_history(path)


if __name__ == "__main__":
    unittest.main()
