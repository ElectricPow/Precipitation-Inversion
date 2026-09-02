"""Tests for Stage-3 checkpoint metadata and evaluation command routing."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_stage3_cascade import (  # noqa: E402
    STAGE3_C2_CHECKPOINT_FORMAT,
    STAGE3_CHECKPOINT_FORMAT,
    build_evaluation_command,
    load_stage3_sources,
)


class Stage3EvaluationTests(unittest.TestCase):
    def test_launcher_requires_explicit_gpus_and_fixed_loopback_rendezvous(self) -> None:
        text = (PROJECT_ROOT / "scripts" / "launch_stage3_c1_ddp.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("CUDA_VISIBLE_DEVICES", text)
        self.assertIn('STAGE3_MASTER_ADDR:-127.0.0.1', text)
        self.assertIn('--master-port="${STAGE3_MASTER_PORT}"', text)
        self.assertNotIn("--standalone", text)
        c2_text = (PROJECT_ROOT / "scripts" / "launch_stage3_c2_ddp.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("CUDA_VISIBLE_DEVICES", c2_text)
        self.assertIn('STAGE3_MASTER_ADDR:-127.0.0.1', c2_text)
        self.assertIn("train_stage3_c2.py", c2_text)
        d0_text = (PROJECT_ROOT / "scripts" / "launch_stage3_d0_ddp.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("CUDA_VISIBLE_DEVICES", d0_text)
        self.assertIn('STAGE3_MASTER_ADDR:-127.0.0.1', d0_text)
        self.assertIn("train_stage3_d0.py", d0_text)
        self.assertNotIn("--standalone", d0_text)

    def test_d0_configs_preregister_head_probe_before_decoder_experiment(self) -> None:
        from scripts.train_stage1_unet3d import load_config

        head = load_config(PROJECT_ROOT / "configs" / "stage3_d0_h_frozen_feature_probe.yaml")
        decoder = load_config(PROJECT_ROOT / "configs" / "stage3_d0_d_decoder_multihead.yaml")
        rain_primary = load_config(
            PROJECT_ROOT / "configs" / "stage3_d0_d_rain_primary.yaml"
        )
        self.assertEqual(head["adaptation"]["trainable_scope"], "rain_head_only")
        self.assertEqual(decoder["adaptation"]["trainable_scope"], "decoder_and_all_heads")
        self.assertEqual(head["training"]["epochs"], 5)
        self.assertEqual(decoder["training"]["epochs"], 20)
        self.assertEqual(rain_primary["experiment"]["method"], "S3-D0-D-RainPrimary")
        self.assertEqual(rain_primary["adaptation"]["objective_mode"], "rain_primary")
        self.assertEqual(
            rain_primary["adaptation"]["initialization_checkpoint"],
            "outputs/stage3/d0_h_frozen_feature_probe/best_rain.pt",
        )
        self.assertEqual(
            rain_primary["gradient_audit"][
                "target_weighted_stage2_to_rain_gradient_ratio"
            ],
            0.25,
        )
        self.assertEqual(rain_primary["training"]["epochs"], 20)
        self.assertEqual(head["training"]["checkpoint_every"], 10)
        self.assertEqual(decoder["training"]["checkpoint_every"], 10)
        for config in (head, decoder, rain_primary):
            self.assertFalse(config["adaptation"]["satellite_inputs"])
            self.assertFalse(config["training"]["early_stopping"]["enabled"])
            self.assertTrue(config["runtime"]["distributed"]["static_graph"])

    def test_checkpoint_routes_to_recorded_stage2_and_oracle_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage2 = root / "stage2.pt"
            threshold = root / "threshold.json"
            stage2.write_bytes(b"stage2")
            threshold.write_text("{}", encoding="utf-8")
            checkpoint = root / "best.pt"
            torch.save(
                {
                    "stage3_format": STAGE3_CHECKPOINT_FORMAT,
                    "stage3_sources": {
                        "stage2_checkpoint": str(stage2),
                        "stage2_threshold_file": str(threshold),
                    },
                },
                checkpoint,
            )
            sources = load_stage3_sources(checkpoint)
            self.assertEqual(sources["stage2_checkpoint"], str(stage2))
            command = build_evaluation_command(
                checkpoint=checkpoint,
                output_dir=root / "evaluation",
                split="val",
                device="cpu",
                save_orbits=2,
                max_files=1,
                overwrite=True,
            )
            self.assertIn("C1Adapt-W1.25", command)
            self.assertIn("--stage1-checkpoint", command)
            self.assertIn("--max-files", command)
            self.assertIn("--overwrite", command)

    def test_c2_checkpoint_routes_current_stage2_and_frozen_stage1(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage1 = root / "stage1.pt"
            old_threshold = root / "old_threshold.json"
            new_threshold = root / "new_threshold.json"
            stage1.write_bytes(b"stage1")
            old_threshold.write_text("{}", encoding="utf-8")
            new_threshold.write_text("{}", encoding="utf-8")
            checkpoint = root / "best.pt"
            torch.save(
                {
                    "stage3_format": STAGE3_C2_CHECKPOINT_FORMAT,
                    "stage3_sources": {
                        "stage1_checkpoint": str(stage1),
                        "stage2_threshold_file": str(old_threshold),
                    },
                },
                checkpoint,
            )
            command = build_evaluation_command(
                checkpoint=checkpoint,
                output_dir=root / "evaluation",
                split="val",
                device="cpu",
                save_orbits=2,
                max_files=1,
                overwrite=True,
                threshold_file=new_threshold,
            )
            self.assertIn("C2TaskAware-W1.25", command)
            self.assertIn(str(stage1), command)
            self.assertIn(str(checkpoint.resolve()), command)
            self.assertIn(str(new_threshold.resolve()), command)


if __name__ == "__main__":
    unittest.main()
