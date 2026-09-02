"""Strict-control checks for the I+G-0.02 versus T3D configuration."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from train_stage1_unet3d import build_model  # noqa: E402
from precipitation_inversion.models.multitask_unet3d import (  # noqa: E402
    Stage1MultiTaskUNet3D,
)


class TypeTaskConfigurationTests(unittest.TestCase):
    def test_t3d_changes_only_auxiliary_task_and_reporting_configuration(self) -> None:
        baseline = json.loads(
            (PROJECT_ROOT / "configs/stage1_ablation_e0_n_i_g_drdz_002.yaml").read_text(
                encoding="utf-8"
            )
        )
        t3d = json.loads(
            (PROJECT_ROOT / "configs/stage1_ablation_i_g002_t3d.yaml").read_text(
                encoding="utf-8"
            )
        )
        for section in ("data", "model", "loss", "optimizer", "scheduler", "training"):
            self.assertEqual(t3d[section], baseline[section], section)
        task = t3d["type_task"]
        self.assertTrue(task["enabled"])
        self.assertFalse(task["rain_feedback"])
        self.assertEqual(task["head"]["kind"], "ordered_3d")
        self.assertEqual(task["head"]["height_levels"], 60)
        self.assertEqual(task["loss_weight"], 0.01)
        self.assertEqual(task["class_weights"], "inverse_sqrt_frequency")

    def test_factory_constructs_multitask_model_with_small_auxiliary_head(self) -> None:
        config = json.loads(
            (PROJECT_ROOT / "configs/stage1_ablation_i_g002_t3d.yaml").read_text(
                encoding="utf-8"
            )
        )
        model = build_model(config)
        self.assertIsInstance(model, Stage1MultiTaskUNet3D)
        head_parameters = sum(parameter.numel() for parameter in model.type_head.parameters())
        backbone_parameters = sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if not name.startswith("type_head.")
        )
        self.assertLess(head_parameters / backbone_parameters, 0.02)


if __name__ == "__main__":
    unittest.main()
