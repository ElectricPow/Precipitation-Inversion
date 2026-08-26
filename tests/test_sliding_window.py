"""Tests for non-overlapping-core reconstruction and full-orbit inference."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from precipitation_inversion.inference.sliding_window import (  # noqa: E402
    CoreWindowReconstructor,
    predict_full_orbit,
    stitch_core_predictions,
)
from tests.test_patch_dataset import make_patch_dataset_fixture  # noqa: E402


class CoreWindowReconstructionTests(unittest.TestCase):
    def _patch(self, start: int, length: int) -> tuple[np.ndarray, np.ndarray]:
        # Model output/mask shape=(C=1,Dp=64,Hp=16,Z=6); height is unpadded.
        prediction = np.zeros((1, 64, 16, 6), dtype=np.float32)
        mask = np.zeros((1, 64, 16, 6), dtype=bool)
        for local_scan in range(length):
            prediction[0, 16 + local_scan, :5, :6] = start + local_scan + 1
        mask[0, 16 : 16 + length, :5, :6] = True
        return prediction, mask

    def test_stitching_crops_halo_and_exactly_recovers_variable_nscan(self) -> None:
        starts = [0, 32, 64]
        lengths = [32, 32, 6]
        values_and_masks = [self._patch(start, length) for start, length in zip(starts, lengths)]
        result = stitch_core_predictions(
            [value for value, _ in values_and_masks],
            core_starts=starts,
            core_lengths=lengths,
            original_shape=(70, 5, 6),
            halo_size=16,
            output_masks=[mask for _, mask in values_and_masks],
        )
        self.assertEqual(result.shape, (1, 70, 5, 6))
        for scan in range(70):
            np.testing.assert_allclose(result[0, scan], scan + 1)

    def test_masked_voxels_are_zero_and_overlap_is_rejected(self) -> None:
        prediction, mask = self._patch(0, 32)
        mask[0, 16, 0, 0] = False
        reconstructor = CoreWindowReconstructor((40, 5, 6), halo_size=16)
        reconstructor.add(
            prediction, core_start=0, core_length=32, output_mask=mask
        )
        self.assertEqual(float(reconstructor.output[0, 0, 0, 0]), 0.0)
        with self.assertRaisesRegex(ValueError, "overlaps"):
            reconstructor.add(prediction, core_start=16, core_length=16)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            reconstructor.finalize()


class FullOrbitInferenceTests(unittest.TestCase):
    def test_constant_log_prediction_is_cropped_masked_and_inverted(self) -> None:
        import torch

        class ConstantLogModel(torch.nn.Module):
            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                # Before inputs=(B,C,D,H,Z); output=(B,1,D,H,Z).
                return torch.full_like(inputs[:, :1], np.log(3.0))

        with tempfile.TemporaryDirectory() as directory:
            dataset, _, _ = make_patch_dataset_fixture(Path(directory))
            model = ConstantLogModel()
            model.train()
            prediction = predict_full_orbit(
                model,
                dataset,
                file_id=0,
                device="cpu",
                batch_size=2,
                num_workers=0,
                use_amp=False,
            )
        self.assertEqual(prediction.shape, (70, 5, 6))
        # Five above-CFB DPR cells form the output support; exp(log(3))-1=2 mm/h.
        self.assertEqual(int(np.count_nonzero(prediction)), 5)
        np.testing.assert_allclose(prediction[prediction > 0], 2.0, rtol=1e-6)
        self.assertEqual(float(prediction[10, 0, 0]), 0.0)  # CFB clutter removed
        self.assertTrue(model.training)  # original train/eval state is restored

    def test_positive_only_dataset_cannot_reconstruct_complete_orbit(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as directory:
            _, metadata, normalization = make_patch_dataset_fixture(Path(directory))
            from precipitation_inversion.data.patch_dataset import Stage1PatchDataset

            dataset = Stage1PatchDataset(metadata, normalization, positive_only=True)
            with self.assertRaisesRegex(ValueError, "positive_only=False"):
                predict_full_orbit(torch.nn.Identity(), dataset, 0, use_amp=False)


if __name__ == "__main__":
    unittest.main()
