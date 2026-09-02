"""Fast tests for multi-process launch safety without opening network sockets."""

from __future__ import annotations

import copy
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_stage1_unet3d import (  # noqa: E402
    _initialize_distributed,
    _resolve_device,
    build_loader,
    load_config,
    run_postprocessing,
)


class DistributedConfigurationTests(unittest.TestCase):
    def test_validation_loader_preserves_uneven_tail_batches(self) -> None:
        class TinyFileDataset:
            files = [
                {
                    "file_id": 0,
                    "index_start": 0,
                    "index_stop": 5,
                    "sample_count": 5,
                }
            ]

            def __len__(self) -> int:
                return 5

            def __getitem__(self, index: int) -> int:
                return index

        config = {
            "data": {
                "batch_size": 1,
                "block_size": 1,
                "num_workers": 0,
                "pin_memory": False,
                "persistent_workers": False,
            }
        }
        _, validation_sampler = build_loader(
            TinyFileDataset(), config, training=False, seed=3
        )
        _, training_sampler = build_loader(
            TinyFileDataset(), config, training=True, seed=3
        )
        self.assertFalse(validation_sampler.even_batches)
        self.assertTrue(training_sampler.even_batches)

    def test_cpu_device_is_valid_for_single_or_multi_process_use(self) -> None:
        self.assertEqual(str(_resolve_device("cpu", local_rank=0, world_size=1)), "cpu")
        self.assertEqual(str(_resolve_device("cpu", local_rank=1, world_size=2)), "cpu")

    def test_explicit_cuda_index_is_rejected_for_torchrun(self) -> None:
        # This check happens before CUDA initialization, so it also runs on CI.
        with self.assertRaisesRegex(ValueError, "LOCAL_RANK"):
            _resolve_device("cuda:5", local_rank=1, world_size=2)

    def test_nccl_cannot_be_used_with_cpu(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires CUDA"):
            _initialize_distributed(
                __import__("torch").device("cpu"),
                2,
                backend="nccl",
                timeout_seconds=5,
            )

    def test_default_config_contains_bounded_ddp_settings(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "stage1_unet3d.yaml")
        ddp = config["runtime"]["distributed"]
        self.assertEqual(ddp["backend"], "auto")
        self.assertGreater(ddp["timeout_seconds"], 0)
        self.assertTrue(ddp["static_graph"])
        self.assertFalse(ddp["find_unused_parameters"])

    def test_formal_configs_use_disk_safe_checkpoint_cadence(self) -> None:
        paths = sorted((PROJECT_ROOT / "configs").glob("stage[12]*.yaml"))
        self.assertTrue(paths)
        for path in paths:
            with self.subTest(config=path.name):
                interval = load_config(path)["training"]["checkpoint_every"]
                expected = 0 if path.name == "stage1_ablation_smoke.yaml" else 10
                self.assertEqual(interval, expected)

    def test_ablation_configs_change_only_declared_experimental_factors(self) -> None:
        paths = {
            "e0": PROJECT_ROOT / "configs" / "stage1_ablation_e0_baseline.yaml",
            "e1": PROJECT_ROOT / "configs" / "stage1_ablation_e1_mask_cfb.yaml",
            "e2": PROJECT_ROOT / "configs" / "stage1_ablation_e2_cfb_distance.yaml",
            "e3e1": PROJECT_ROOT
            / "configs"
            / "stage1_ablation_e3_weighted_from_e1.yaml",
            "e3e2": PROJECT_ROOT
            / "configs"
            / "stage1_ablation_e3_weighted_from_e2.yaml",
            "e4e1": PROJECT_ROOT
            / "configs"
            / "stage1_ablation_e4_weak_from_e1.yaml",
            "e4e2": PROJECT_ROOT
            / "configs"
            / "stage1_ablation_e4_weak_from_e2.yaml",
        }
        configs = {name: load_config(path) for name, path in paths.items()}
        baseline = configs["e0"]

        # All runs use the same split, seed, optimizer, schedule and U-Net
        # capacity. E2 alone has one extra diagnostic input channel.
        self.assertEqual(
            {value["experiment"]["seed"] for value in configs.values()}, {2026}
        )
        self.assertEqual(
            len({value["experiment"]["output_dir"] for value in configs.values()}),
            len(configs),
        )
        for value in configs.values():
            for key in ("train_index", "val_index", "test_index", "normalization"):
                self.assertEqual(value["data"][key], baseline["data"][key])
            self.assertEqual(value["optimizer"], baseline["optimizer"])
            self.assertEqual(value["scheduler"], baseline["scheduler"])
            self.assertEqual(value["training"], baseline["training"])
            self.assertEqual(value["runtime"], baseline["runtime"])
            comparable_model = dict(value["model"])
            comparable_model.pop("in_channels")
            baseline_model = dict(baseline["model"])
            baseline_model.pop("in_channels")
            self.assertEqual(comparable_model, baseline_model)

        self.assertEqual(
            {name: value["data"]["cfb_input_mode"] for name, value in configs.items()},
            {
                "e0": "baseline",
                "e1": "mask_below_cfb",
                "e2": "signed_distance",
                "e3e1": "mask_below_cfb",
                "e3e2": "signed_distance",
                "e4e1": "mask_below_cfb",
                "e4e2": "signed_distance",
            },
        )
        for name in ("e2", "e3e2", "e4e2"):
            self.assertEqual(configs[name]["model"]["in_channels"], 4)
        for name in ("e0", "e1", "e3e1", "e4e1"):
            self.assertEqual(configs[name]["model"]["in_channels"], 3)

        # Weighted training is introduced only after the phase-one comparison;
        # weak CFB labels are introduced only in the final ablation.
        for name in ("e0", "e1", "e2"):
            self.assertEqual(configs[name]["loss"]["height_loss_weighting"], "none")
            self.assertEqual(configs[name]["loss"]["intensity_loss_bin_edges"], [])
        for name in ("e3e1", "e3e2", "e4e1", "e4e2"):
            self.assertEqual(
                configs[name]["loss"]["height_loss_weighting"],
                "inverse_sqrt_frequency",
            )
            self.assertEqual(
                len(configs[name]["loss"]["intensity_loss_bin_weights"]),
                len(configs[name]["loss"]["intensity_loss_bin_edges"]) + 1,
            )
        for name in ("e0", "e1", "e2", "e3e1", "e3e2"):
            self.assertEqual(configs[name]["data"]["weak_cfb_layer_weights"], [])
        for name in ("e4e1", "e4e2"):
            self.assertEqual(
                configs[name]["data"]["weak_cfb_layer_weights"], [0.1, 0.05]
            )

        early_stopping = baseline["training"]["early_stopping"]
        self.assertEqual(baseline["training"]["epochs"], 60)
        self.assertTrue(early_stopping["enabled"])
        self.assertEqual(early_stopping["patience"], 12)
        self.assertEqual(early_stopping["monitor"], "val_rain_rmse")

    def test_ablation_suite_defaults_to_printing_phase_one_sequentially(self) -> None:
        environment = dict(os.environ)
        environment.pop("STAGE1_ABLATIONS", None)
        environment.update(
            {"STAGE1_EXECUTE": "0", "STAGE1_ABLATION_PHASE": "phase1"}
        )
        result = subprocess.run(
            ["bash", str(PROJECT_ROOT / "scripts" / "launch_stage1_ablation_suite.sh")],
            cwd=PROJECT_ROOT,
            env=environment,
            check=True,
            text=True,
            capture_output=True,
        )
        self.assertIn("print-only mode", result.stdout)
        self.assertLess(result.stdout.index("e0:"), result.stdout.index("e1:"))
        self.assertLess(result.stdout.index("e1:"), result.stdout.index("e2:"))
        self.assertNotIn("e3:", result.stdout)
        self.assertNotIn("e4:", result.stdout)

    def test_ablation_suite_requires_and_selects_the_winning_parent(self) -> None:
        script = str(PROJECT_ROOT / "scripts" / "launch_stage1_ablation_suite.sh")
        for phase, parent, expected in (
            ("weighted", "e1", "e3_from_e1"),
            ("weighted", "e2", "e3_from_e2"),
            ("weak", "e1", "e4_from_e1"),
            ("weak", "e2", "e4_from_e2"),
        ):
            with self.subTest(phase=phase, parent=parent):
                environment = dict(os.environ)
                environment.pop("STAGE1_ABLATIONS", None)
                environment.update(
                    {
                        "STAGE1_EXECUTE": "0",
                        "STAGE1_ABLATION_PHASE": phase,
                        "STAGE1_ABLATION_PARENT": parent,
                    }
                )
                result = subprocess.run(
                    ["bash", script],
                    cwd=PROJECT_ROOT,
                    env=environment,
                    check=True,
                    text=True,
                    capture_output=True,
                )
                self.assertIn(f"experiments={expected}", result.stdout)

    def test_e0_new_normalization_ablation_configs_isolate_factors(self) -> None:
        paths = {
            "e0_n": PROJECT_ROOT
            / "configs"
            / "stage1_ablation_e0_n_dbz_valid.yaml",
            "e0_n_i": PROJECT_ROOT
            / "configs"
            / "stage1_ablation_e0_n_i_intensity.yaml",
            "e0_n_w": PROJECT_ROOT
            / "configs"
            / "stage1_ablation_e0_n_w_weak_cfb.yaml",
            "e0_n_iw": PROJECT_ROOT
            / "configs"
            / "stage1_ablation_e0_n_iw_combined.yaml",
        }
        configs = {name: load_config(path) for name, path in paths.items()}
        baseline = load_config(
            PROJECT_ROOT / "configs" / "stage1_ablation_e0_baseline.yaml"
        )
        expected_stats = "metadata/normalization/stage1_dbz_valid.json"
        expected_intensity = [1.0, 1.0, 1.5, 2.0, 3.0]

        self.assertEqual(
            len({value["experiment"]["output_dir"] for value in configs.values()}),
            len(configs),
        )
        for value in configs.values():
            self.assertEqual(value["experiment"]["seed"], baseline["experiment"]["seed"])
            self.assertEqual(value["data"]["normalization"], expected_stats)
            self.assertEqual(value["data"]["cfb_input_mode"], "baseline")
            self.assertEqual(value["model"], baseline["model"])
            self.assertEqual(value["optimizer"], baseline["optimizer"])
            self.assertEqual(value["scheduler"], baseline["scheduler"])
            self.assertEqual(value["training"], baseline["training"])
            self.assertEqual(value["runtime"], baseline["runtime"])
            self.assertEqual(value["loss"]["height_loss_weighting"], "none")
            self.assertEqual(
                value["loss"]["height_loss_weight_min"],
                baseline["loss"]["height_loss_weight_min"],
            )
            self.assertEqual(
                value["loss"]["height_loss_weight_max"],
                baseline["loss"]["height_loss_weight_max"],
            )
            for key in ("train_index", "val_index", "test_index"):
                self.assertEqual(value["data"][key], baseline["data"][key])

        for name in ("e0_n", "e0_n_i"):
            self.assertEqual(configs[name]["data"]["weak_cfb_layer_weights"], [])
        for name in ("e0_n_w", "e0_n_iw"):
            self.assertEqual(
                configs[name]["data"]["weak_cfb_layer_weights"], [0.1, 0.05]
            )
        for name in ("e0_n", "e0_n_w"):
            self.assertEqual(configs[name]["loss"]["intensity_loss_bin_edges"], [])
            self.assertEqual(configs[name]["loss"]["intensity_loss_bin_weights"], [])
        for name in ("e0_n_i", "e0_n_iw"):
            self.assertEqual(
                configs[name]["loss"]["intensity_loss_bin_edges"],
                [1.0, 5.0, 10.0, 30.0],
            )
            self.assertEqual(
                configs[name]["loss"]["intensity_loss_bin_weights"],
                expected_intensity,
            )

    def test_ablation_suite_prints_e0n_factorial_in_safe_order(self) -> None:
        environment = dict(os.environ)
        environment.pop("STAGE1_ABLATIONS", None)
        environment.update({"STAGE1_EXECUTE": "0", "STAGE1_ABLATION_PHASE": "e0n"})
        result = subprocess.run(
            ["bash", str(PROJECT_ROOT / "scripts" / "launch_stage1_ablation_suite.sh")],
            cwd=PROJECT_ROOT,
            env=environment,
            check=True,
            text=True,
            capture_output=True,
        )
        self.assertIn("print-only mode", result.stdout)
        positions = [
            result.stdout.index(f"{name}:")
            for name in ("e0_n", "e0_n_i", "e0_n_w", "e0_n_iw")
        ]
        self.assertEqual(positions, sorted(positions))

    def test_i_plus_g_configuration_is_a_strict_single_factor_ablation(self) -> None:
        parent = load_config(
            PROJECT_ROOT / "configs" / "stage1_ablation_e0_n_i_intensity.yaml"
        )
        child = load_config(
            PROJECT_ROOT
            / "configs"
            / "stage1_ablation_e0_n_i_g_drdz_002.yaml"
        )
        physical = child["loss"]["physical_gradient"]
        self.assertEqual(
            physical, {"enabled": True, "weight": 0.02, "beta": 1.0}
        )
        self.assertEqual(child["model"]["in_channels"], 3)
        self.assertEqual(child["data"]["weak_cfb_layer_weights"], [])
        self.assertEqual(
            child["training"]["early_stopping"]["monitor"], "val_rain_rmse"
        )

        # The only permitted differences are run identity/output and the new
        # G block. Everything controlling data, I weights, model and optimizer
        # must remain byte-for-byte equivalent after those fields are removed.
        parent_comparable = copy.deepcopy(parent)
        child_comparable = copy.deepcopy(child)
        for value in (parent_comparable, child_comparable):
            value["experiment"].pop("name")
            value["experiment"].pop("output_dir")
        child_comparable["loss"].pop("physical_gradient")
        self.assertEqual(child_comparable, parent_comparable)

    def test_ablation_suite_exposes_single_i_plus_g_launch_id(self) -> None:
        environment = dict(os.environ)
        environment.pop("STAGE1_ABLATIONS", None)
        environment.update({"STAGE1_EXECUTE": "0", "STAGE1_ABLATION_PHASE": "ig"})
        result = subprocess.run(
            ["bash", str(PROJECT_ROOT / "scripts" / "launch_stage1_ablation_suite.sh")],
            cwd=PROJECT_ROOT,
            env=environment,
            check=True,
            text=True,
            capture_output=True,
        )
        self.assertIn("experiments=e0_n_i_g", result.stdout)
        self.assertIn("stage1_ablation_e0_n_i_g_drdz_002.yaml", result.stdout)
        self.assertIn("stage1_e0_n_i_g_drdz_002", result.stdout)

    def test_postprocessing_runs_once_with_torchrun_ranks_removed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "best.pt").touch()
            config = {
                "postprocessing": {
                    "enabled": True,
                    "test_sample_count": 2,
                    "selection_seed": 9,
                    "device": "cpu",
                }
            }
            with mock.patch.dict(
                "os.environ", {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "3"}
            ), mock.patch("scripts.train_stage1_unet3d.subprocess.run") as runner:
                run_postprocessing(output, config)
            self.assertEqual(runner.call_count, 4)
            for call in runner.call_args_list:
                environment = call.kwargs["env"]
                self.assertNotIn("RANK", environment)
                self.assertNotIn("LOCAL_RANK", environment)
                self.assertNotIn("WORLD_SIZE", environment)
                self.assertTrue(call.kwargs["check"])


if __name__ == "__main__":
    unittest.main()
