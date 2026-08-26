"""Fast tests for multi-process launch safety without opening network sockets."""

from __future__ import annotations

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
    load_config,
    run_postprocessing,
)


class DistributedConfigurationTests(unittest.TestCase):
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
            self.assertEqual(runner.call_count, 2)
            for call in runner.call_args_list:
                environment = call.kwargs["env"]
                self.assertNotIn("RANK", environment)
                self.assertNotIn("LOCAL_RANK", environment)
                self.assertNotIn("WORLD_SIZE", environment)
                self.assertTrue(call.kwargs["check"])


if __name__ == "__main__":
    unittest.main()
