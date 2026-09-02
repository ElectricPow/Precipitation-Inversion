"""Tests for Stage-2 complete-orbit stitching and threshold selection."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from precipitation_inversion.inference.stage2_sliding_window import (  # noqa: E402
    SupportThresholdSweep,
    predict_stage2_full_orbit,
    predict_stage3_d0_full_orbit,
    reconstruct_stage2_fields,
    reconstruct_stage2_targets,
    select_support_threshold,
)


FIELD_NAMES = (
    "target_dbz",
    "target_support",
    "support_loss_mask",
    "regression_mask",
    "overlap_mask",
    "dpr_only_mask",
    "gr_only_mask",
    "neither_mask",
    "gap_proxy_mask",
    "outside_proxy_mask",
    "below_cfb_target_mask",
    "gr_value_mask",
    "gr_native_available",
    "dpr_sparse_anchor_mask",
)


class SyntheticStage2Dataset(torch.utils.data.Dataset):
    halo_size = 1

    def __init__(self) -> None:
        self.files = [
            {
                "nscan": 5,
                "nray": 3,
                "z_size": 2,
                "file_name": "synthetic.nc",
                "sample_id": "synthetic",
            }
        ]
        self.dpr_standardizer = SimpleNamespace(
            mean=np.array([10.0, 20.0], dtype=np.float32),
            std=np.array([2.0, 4.0], dtype=np.float32),
        )

    def __len__(self) -> int:
        return 2

    def file_index_range(self, file_id: int) -> range:
        if file_id != 0:
            raise IndexError(file_id)
        return range(2)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        starts = (0, 3)
        lengths = (3, 2)
        start, length = starts[index], lengths[index]
        inputs = torch.zeros(4, 4, 4, 2)
        for local in range(length):
            global_scan = start + local
            inputs[0, 1 + local, :3] = float(global_scan)
            inputs[1, 1 + local, :3] = float(global_scan + 1)
        sample: dict[str, torch.Tensor] = {
            "inputs": inputs,
            "core_start": torch.tensor(start),
            "core_length": torch.tensor(length),
        }
        target = torch.zeros(1, 4, 4, 2)
        target[:, 1 : 1 + length, :3] = inputs[1:2, 1 : 1 + length, :3]
        support = target > 0
        for name in FIELD_NAMES:
            if name == "target_dbz":
                sample[name] = target.clone()
            elif name == "target_support":
                sample[name] = support.float()
            elif name in {"support_loss_mask", "regression_mask", "gr_value_mask", "gr_native_available", "dpr_sparse_anchor_mask"}:
                sample[name] = support.clone()
            else:
                sample[name] = torch.zeros_like(support)
        return sample


class EchoStage2Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(()))

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "support_logits": inputs[:, 0:1] + self.dummy,
            "reflectivity": inputs[:, 1:2] + self.dummy,
        }


class EchoD0Model(EchoStage2Model):
    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        output = super().forward(inputs)
        output["rain"] = torch.log1p(torch.relu(inputs[:, 1:2])) + self.dummy
        return output


class Stage2SlidingWindowTests(unittest.TestCase):
    def test_prediction_and_target_fields_cover_each_scan_once(self) -> None:
        dataset = SyntheticStage2Dataset()
        result = predict_stage2_full_orbit(
            EchoStage2Model(), dataset, 0, device="cpu", use_amp=False
        )
        self.assertEqual(result.support_logits.shape, (5, 3, 2))
        np.testing.assert_allclose(result.support_logits[:, 0, 0], np.arange(5))
        # standardized=(scan+1); physical=standardized*std+mean per height.
        np.testing.assert_allclose(
            result.reflectivity_dbz[:, 0, 0],
            (np.arange(5) + 1) * 2 + 10,
        )
        np.testing.assert_allclose(
            result.reflectivity_dbz[:, 0, 1],
            (np.arange(5) + 1) * 4 + 20,
        )
        fields = reconstruct_stage2_fields(dataset, 0, ["target_support"])
        self.assertEqual(fields["target_support"].shape, (5, 3, 2))
        targets = reconstruct_stage2_targets(dataset, 0)
        self.assertEqual(targets["target_support"].dtype, np.bool_)
        np.testing.assert_allclose(targets["target_dbz"][:, 0, 0], (np.arange(5) + 1) * 2 + 10)

    def test_threshold_sweep_is_streaming_and_uses_csi(self) -> None:
        probabilities = np.array([0.9, 0.7, 0.4, 0.1])
        target = np.array([True, True, False, False])
        domain = np.ones(4, dtype=bool)
        sweep = SupportThresholdSweep([0.3, 0.5, 0.8], objective="csi")
        sweep.update(probabilities[:2], target[:2], domain[:2])
        sweep.update(probabilities[2:], target[2:], domain[2:])
        result = sweep.compute()
        self.assertEqual(result.threshold, 0.5)
        self.assertEqual(result.objective_value, 1.0)
        wrapped = select_support_threshold(
            probabilities,
            target,
            domain,
            candidates=[0.3, 0.5, 0.8],
        )
        self.assertEqual(wrapped.threshold, result.threshold)

    def test_threshold_search_rejects_test_leakage_inputs_and_invalid_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "candidates"):
            SupportThresholdSweep([0.0, 1.0])
        sweep = SupportThresholdSweep([0.5])
        with self.assertRaisesRegex(TypeError, "boolean"):
            sweep.update(np.ones(2), np.ones(2), np.ones(2, dtype=bool))
        with self.assertRaisesRegex(IndexError, "1"):
            predict_stage2_full_orbit(EchoStage2Model(), SyntheticStage2Dataset(), 1)

    def test_d0_reconstructs_ungated_rain_for_later_support_comparison(self) -> None:
        result = predict_stage3_d0_full_orbit(
            EchoD0Model(), SyntheticStage2Dataset(), 0, device="cpu", use_amp=False
        )
        self.assertEqual(result.rain_rate.shape, (5, 3, 2))
        np.testing.assert_allclose(result.rain_rate[:, 0, 0], np.arange(5) + 1)
        self.assertEqual(result.support_probability.shape, result.rain_rate.shape)


if __name__ == "__main__":
    unittest.main()
