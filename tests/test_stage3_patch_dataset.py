"""Tests for aligned C1-O Stage-3 patch construction."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from netCDF4 import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from precipitation_inversion.data.patch_dataset import (  # noqa: E402
    PATCH_INDEX_FORMAT_VERSION,
    PATCH_INPUT_CHANNELS,
    PATCH_INPUT_VARIABLE,
    PATCH_LABEL_VARIABLE,
    build_stage1_patch_index_records,
    save_patch_index,
)
from precipitation_inversion.data.stage2_patch_dataset import (  # noqa: E402
    STAGE2_FOUR_CHANNEL_DISTANCE_INPUT_CHANNELS,
    STAGE2_INPUT_CHANNELS,
    STAGE2_INPUT_VARIABLE,
    STAGE2_PATCH_INDEX_FORMAT_VERSION,
    STAGE2_TARGET_VARIABLE,
    build_stage2_patch_index_records,
    save_stage2_patch_index,
)
from precipitation_inversion.data.stage3_patch_dataset import (  # noqa: E402
    Stage3C1PatchDataset,
    Stage3C2PatchDataset,
    Stage3D0PatchDataset,
)


HEIGHTS = np.array([0.125, 0.375, 0.625], dtype=np.float32)


def create_combined_nc(path: Path) -> None:
    shape = (6, 3, 3)
    gr = np.full(shape, np.nan, dtype=np.float32)
    gr[0, 0] = [10.0, 20.0, 30.0]
    gr[1, 1] = [14.0, 24.0, 34.0]
    dpr = np.full(shape, np.nan, dtype=np.float32)
    dpr[0, 0] = [12.0, 22.0, 32.0]
    dpr[1, 1] = [16.0, 26.0, 36.0]
    rain = np.zeros(shape, dtype=np.float32)
    rain[np.isfinite(dpr)] = 1.0
    with Dataset(path, "w") as dataset:
        dataset.createDimension("nscan", shape[0])
        dataset.createDimension("nray", shape[1])
        dataset.createDimension("z", shape[2])
        z = dataset.createVariable("z", "f4", ("z",), fill_value=np.nan)
        z[:] = HEIGHTS
        for name, values in (
            ("dbz_gr_sparse", gr),
            ("dbz_gr_interp", gr),
            ("dbz_dpr", dpr),
            ("pre_dpr", rain),
        ):
            variable = dataset.createVariable(
                name, "f4", ("nscan", "nray", "z"), fill_value=np.nan
            )
            variable[:] = values
        cfb = dataset.createVariable("cfb", "i2", ("nscan", "nray"))
        cfb[:] = 0


def write_stage1_normalization(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "scope": "training_split_only",
                "selection_mask": "variable_valid",
                "label_qc_applied_to_input_statistics": False,
                "variables": {
                    "dbz_dpr": {
                        "heights_km": HEIGHTS.tolist(),
                        "mean": [14.0, 24.0, 34.0],
                        "std": [2.0, 2.0, 2.0],
                        "count": [2, 2, 2],
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def write_stage2_normalization(path: Path) -> None:
    def values(mean: list[float]) -> dict[str, object]:
        return {
            "heights_km": HEIGHTS.tolist(),
            "mean": mean,
            "std": [2.0, 2.0, 2.0],
            "count": [2, 2, 2],
        }

    path.write_text(
        json.dumps(
            {
                "stage": 2,
                "scope": "training_split_only",
                "selection_mask": "reflectivity_storage_value",
                "processed_file_count": 1,
                "validated_file_count": 1,
                "variables": {
                    "dbz_gr_sparse": values([12.0, 22.0, 32.0]),
                    "dbz_dpr": values([14.0, 24.0, 34.0]),
                },
            }
        ),
        encoding="utf-8",
    )


def make_dataset(root: Path) -> Stage3C1PatchDataset:
    source = root / "sample.nc"
    create_combined_nc(source)
    records1, files1, z1, nray1, _ = build_stage1_patch_index_records(
        [source], core_size=4
    )
    save_patch_index(
        root / "stage1.npy",
        root / "stage1.json",
        records1,
        {
            "format_version": PATCH_INDEX_FORMAT_VERSION,
            "input_variable": PATCH_INPUT_VARIABLE,
            "label_variable": PATCH_LABEL_VARIABLE,
            "label_transform": "log1p",
            "input_channels": list(PATCH_INPUT_CHANNELS),
            "core_size": 4,
            "halo_size": 2,
            "horizontal_multiple": 4,
            "height_padding": 0,
            "nray": nray1,
            "z_size": len(z1),
            "heights_km": z1.tolist(),
            "padded_patch_shape": [8, 4, len(z1)],
            "patch_count": len(records1),
            "files": files1,
        },
    )
    records2, files2, z2, nray2, _ = build_stage2_patch_index_records(
        [source], core_size=4
    )
    save_stage2_patch_index(
        root / "stage2.npy",
        root / "stage2.json",
        records2,
        {
            "format_version": STAGE2_PATCH_INDEX_FORMAT_VERSION,
            "stage": 2,
            "input_variable": STAGE2_INPUT_VARIABLE,
            "input_channels": list(STAGE2_INPUT_CHANNELS),
            "target_variable": STAGE2_TARGET_VARIABLE,
            "core_size": 4,
            "halo_size": 2,
            "horizontal_multiple": 4,
            "height_padding": 0,
            "nray": nray2,
            "z_size": len(z2),
            "heights_km": z2.tolist(),
            "padded_patch_shape": [8, 4, len(z2)],
            "patch_count": len(records2),
            "files": files2,
        },
    )
    write_stage1_normalization(root / "stage1_norm.json")
    write_stage2_normalization(root / "stage2_norm.json")
    return Stage3C1PatchDataset(
        stage1_index_metadata=root / "stage1.json",
        stage1_normalization_stats=root / "stage1_norm.json",
        stage2_index_metadata=root / "stage2.json",
        stage2_normalization_stats=root / "stage2_norm.json",
        stage2_input_channels=STAGE2_FOUR_CHANNEL_DISTANCE_INPUT_CHANNELS,
        positive_only=True,
    )


class Stage3PatchDatasetTests(unittest.TestCase):
    def test_packed_channels_align_and_true_dbz_never_enters_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = make_dataset(root)
            first = dataset[0]
            self.assertEqual(tuple(first["inputs"].shape), (6, 8, 4, 3))
            self.assertEqual(
                dataset.feature_names[-2:],
                ["true_dpr_support_oracle", "stage1_height_scaled_copy"],
            )
            self.assertTrue(
                np.array_equal(
                    first["inputs"][-2].numpy().astype(bool),
                    first["true_dpr_support"][0].numpy(),
                )
            )
            before = first["inputs"].clone()
            # Change only true DPR values, retaining exactly the same support.
            # GR features/support/height—and therefore packed model input—must
            # remain unchanged; only label-side Stage-1 decoding sees the dBZ.
            with Dataset(root / "sample.nc", "a") as source:
                values = source.variables["dbz_dpr"][:]
                finite = np.isfinite(values)
                values[finite] += 50.0
                source.variables["dbz_dpr"][:] = values
            dataset.clear_cache()
            after = dataset[0]["inputs"]
            self.assertTrue(bool((before == after).all()))

    def test_positive_only_uses_raw_index_to_find_stage2_patch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = make_dataset(Path(directory))
            self.assertEqual(len(dataset), 1)
            item = dataset[0]
            self.assertEqual(int(item["patch_index"]), 0)
            self.assertEqual(int(item["stage2_patch_index"]), 0)

    def test_c2_retains_full_stage2_population_and_physical_targets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_dataset(root)
            dataset = Stage3C2PatchDataset(
                stage1_index_metadata=root / "stage1.json",
                stage1_normalization_stats=root / "stage1_norm.json",
                stage2_index_metadata=root / "stage2.json",
                stage2_normalization_stats=root / "stage2_norm.json",
                stage2_input_channels=STAGE2_FOUR_CHANNEL_DISTANCE_INPUT_CHANNELS,
                stage2_options={
                    "reflectivity_intensity_bin_edges_dbz": [25.0, 35.0],
                    "reflectivity_intensity_bin_weights": [1.0, 1.1, 1.25],
                },
            )
            self.assertEqual(len(dataset), 2)
            self.assertIn("dpr_count", dataset.records.dtype.names)
            item = dataset[0]
            self.assertEqual(tuple(item["inputs"].shape), (6, 8, 4, 3))
            for name in (
                "stage2_target_support",
                "stage2_target_dbz",
                "stage2_support_loss_mask",
                "stage2_regression_mask",
                "stage2_regression_weights",
            ):
                self.assertEqual(tuple(item[name].shape), (1, 8, 4, 3))

    def test_d0_input_contains_only_gr_derived_stage2_channels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_dataset(root)
            dataset = Stage3D0PatchDataset(
                stage1_index_metadata=root / "stage1.json",
                stage1_normalization_stats=root / "stage1_norm.json",
                stage2_index_metadata=root / "stage2.json",
                stage2_normalization_stats=root / "stage2_norm.json",
                stage2_input_channels=STAGE2_FOUR_CHANNEL_DISTANCE_INPUT_CHANNELS,
                stage2_options={
                    "reflectivity_intensity_bin_edges_dbz": [25.0, 35.0],
                    "reflectivity_intensity_bin_weights": [1.0, 1.1, 1.25],
                },
            )
            first = dataset[0]
            self.assertEqual(tuple(first["inputs"].shape), (4, 8, 4, 3))
            self.assertEqual(tuple(dataset.feature_names), STAGE2_FOUR_CHANNEL_DISTANCE_INPUT_CHANNELS)
            before = first["inputs"].clone()
            with Dataset(root / "sample.nc", "a") as source:
                values = source.variables["dbz_dpr"][:]
                values[np.isfinite(values)] += 50.0
                source.variables["dbz_dpr"][:] = values
            dataset.clear_cache()
            self.assertTrue(torch.equal(before, dataset[0]["inputs"]))


if __name__ == "__main__":
    unittest.main()
