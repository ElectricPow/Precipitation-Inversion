"""Tests for core-with-halo patch indexing and tensor construction."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from precipitation_inversion.data.patch_dataset import (  # noqa: E402
    PATCH_INDEX_FORMAT_VERSION,
    PATCH_INPUT_CHANNELS,
    PATCH_INPUT_VARIABLE,
    PATCH_LABEL_VARIABLE,
    Stage1PatchDataset,
    build_stage1_patch_index_records,
    ceil_to_multiple,
    core_starts,
    save_patch_index,
)


HEIGHTS = np.array([0.125, 0.375, 0.625, 0.875, 1.125, 1.375], dtype=np.float32)
RAIN_CELLS = (
    (0, 0, 2, 1.0, 20.0),
    (31, 1, 2, 2.0, 22.0),
    (32, 2, 3, 3.0, 24.0),
    (63, 3, 4, 4.0, 35.0),
    (69, 4, 5, 5.0, 50.0),
)


def create_patch_nc(path: Path, *, nscan: int = 70, with_rain: bool = True) -> None:
    """Create source arrays shaped ``(nscan,5,6)`` with known edge rain cells."""

    shape = (nscan, 5, 6)
    with Dataset(path, "w") as dataset:
        dataset.createDimension("nscan", shape[0])
        dataset.createDimension("nray", shape[1])
        dataset.createDimension("z", shape[2])
        z = dataset.createVariable("z", "f4", ("z",), fill_value=np.nan)
        z[:] = HEIGHTS

        dbz_values = np.full(shape, np.nan, dtype=np.float32)
        rain_values = np.zeros(shape, dtype=np.float32)
        if with_rain:
            for scan, ray, level, rain, dbz in RAIN_CELLS:
                if scan < nscan:
                    rain_values[scan, ray, level] = rain
                    dbz_values[scan, ray, level] = dbz
            # Native positive/DPR-valid but below cfb, therefore excluded by QC.
            rain_values[10, 0, 0] = 9.0
            dbz_values[10, 0, 0] = 30.0

        dbz = dataset.createVariable(
            "dbz_dpr", "f4", ("nscan", "nray", "z"), fill_value=np.nan
        )
        dbz[:] = dbz_values
        pre = dataset.createVariable(
            "pre_dpr", "f4", ("nscan", "nray", "z"), fill_value=np.nan
        )
        pre[:] = rain_values
        cfb = dataset.createVariable("cfb", "i2", ("nscan", "nray"))
        cfb[:] = 1


def write_normalization(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "scope": "training_split_only",
                "selection_mask": "pre_positive_qc",
                "variables": {
                    "dbz_dpr": {
                        "heights_km": HEIGHTS.tolist(),
                        "mean": [None, None, 10.0, 20.0, 30.0, 40.0],
                        "std": [None, None, 2.0, 4.0, 5.0, 10.0],
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def make_patch_dataset_fixture(root: Path) -> tuple[Stage1PatchDataset, Path, Path]:
    """Create two files, five cores, and return the all-core Dataset."""

    rainy = root / "rainy.nc"
    dry = root / "dry.nc"
    create_patch_nc(rainy, nscan=70, with_rain=True)
    create_patch_nc(dry, nscan=33, with_rain=False)
    records, files, heights, nray, _ = build_stage1_patch_index_records(
        [rainy, dry], core_size=32
    )
    index_path = root / "train.npy"
    metadata_path = root / "train.json"
    save_patch_index(
        index_path,
        metadata_path,
        records,
        {
            "format_version": PATCH_INDEX_FORMAT_VERSION,
            "input_variable": PATCH_INPUT_VARIABLE,
            "label_variable": PATCH_LABEL_VARIABLE,
            "label_transform": "log1p",
            "input_channels": list(PATCH_INPUT_CHANNELS),
            "core_size": 32,
            "halo_size": 16,
            "horizontal_multiple": 16,
            "height_padding": 0,
            "nray": nray,
            "z_size": len(heights),
            "heights_km": heights.tolist(),
            "padded_patch_shape": [64, 16, len(heights)],
            "patch_count": len(records),
            "files": files,
        },
    )
    normalization_path = root / "normalization.json"
    write_normalization(normalization_path)
    return Stage1PatchDataset(metadata_path, normalization_path), metadata_path, normalization_path


class PatchIndexTests(unittest.TestCase):
    def test_core_starts_and_padding_multiple(self) -> None:
        self.assertEqual(core_starts(70, 32), (0, 32, 64))
        self.assertEqual(core_starts(32, 32), (0,))
        self.assertEqual(ceil_to_multiple(49, 16), 64)
        self.assertEqual(ceil_to_multiple(64, 16), 64)

    def test_index_counts_only_nonoverlapping_core_not_halo(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rainy.nc"
            create_patch_nc(path)
            records, files, heights, nray, count = build_stage1_patch_index_records(
                [path], core_size=32
            )
        self.assertEqual(count, 1)
        self.assertEqual(nray, 5)
        np.testing.assert_allclose(heights, HEIGHTS)
        np.testing.assert_array_equal(records["core_start"], [0, 32, 64])
        np.testing.assert_array_equal(records["core_length"], [32, 32, 6])
        np.testing.assert_array_equal(records["positive_count"], [2, 2, 1])
        np.testing.assert_array_equal(records["input_count"], [2, 2, 1])
        self.assertEqual(files[0]["positive_count"], 5)


class Stage1PatchDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.dataset, self.metadata_path, self.normalization_path = (
            make_patch_dataset_fixture(self.root)
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_shapes_channels_padding_and_first_core_values(self) -> None:
        item = self.dataset[0]
        self.assertEqual(tuple(item["inputs"].shape), (3, 64, 16, 6))
        for name in ("target", "loss_mask", "output_mask", "core_mask"):
            self.assertEqual(tuple(item[name].shape), (1, 64, 16, 6))
        self.assertEqual(item["inputs"].dtype, item["target"].dtype)
        self.assertEqual(item["loss_mask"].dtype, __import__("torch").bool)
        self.assertEqual(item["unpadded_shape"].tolist(), [64, 5, 6])
        self.assertEqual(item["padded_shape"].tolist(), [64, 16, 6])

        # Source scan 0 maps to destination scan halo=16. At z=2,
        # standardized dbz=(20-10)/2=5 and validity channel equals one.
        self.assertAlmostEqual(float(item["inputs"][0, 16, 0, 2]), 5.0)
        self.assertEqual(float(item["inputs"][1, 16, 0, 2]), 1.0)
        # Height is the third channel: six physical levels map exactly to
        # [-1,1], while orbit-boundary and ray-padding positions remain zero.
        self.assertAlmostEqual(float(item["inputs"][2, 16, 0, 0]), -1.0)
        self.assertAlmostEqual(float(item["inputs"][2, 16, 0, 5]), 1.0)
        self.assertEqual(float(item["inputs"][2, 0, 0, 0]), 0.0)
        self.assertEqual(float(item["inputs"][2, 16, 5, 0]), 0.0)
        self.assertAlmostEqual(float(item["target"][0, 16, 0, 2]), np.log(2.0))
        self.assertEqual(int(item["loss_mask"].sum()), 2)
        self.assertEqual(int(item["output_mask"].sum()), 2)
        self.assertEqual(int(item["core_mask"].sum()), 32 * 5 * 6)
        # Low-z clutter and every ray/z padding cell remain masked and zero-filled.
        self.assertEqual(float(item["inputs"][1, 26, 0, 0]), 0.0)
        self.assertFalse(bool(item["loss_mask"][0, 26, 0, 0]))
        self.assertFalse(bool(item["core_mask"][0, 16, 5, 0]))

    def test_halo_is_input_context_but_not_second_cores_loss(self) -> None:
        item = self.dataset[1]  # core [32,64), context window [16,80)
        # Global scan31 maps to local15 and is visible to the model.
        self.assertEqual(float(item["inputs"][1, 15, 1, 2]), 1.0)
        # It is halo context, hence excluded from the core loss.
        self.assertFalse(bool(item["loss_mask"][0, 15, 1, 2]))
        # Global scan32 maps to local16, the first scan of this core.
        self.assertTrue(bool(item["loss_mask"][0, 16, 2, 3]))
        self.assertEqual(int(item["loss_mask"].sum()), 2)

    def test_last_short_core_and_positive_only_filter(self) -> None:
        last_rainy = self.dataset[2]
        self.assertEqual(int(last_rainy["core_start"]), 64)
        self.assertEqual(int(last_rainy["core_length"]), 6)
        self.assertEqual(int(last_rainy["core_mask"].sum()), 6 * 5 * 6)
        self.assertEqual(len(self.dataset), 5)  # three rainy-file + two dry-file cores

        filtered = Stage1PatchDataset(
            self.metadata_path,
            self.normalization_path,
            positive_only=True,
        )
        self.assertEqual(len(filtered), 3)
        self.assertEqual(list(filtered.file_index_range(0)), [0, 1, 2])
        self.assertEqual(list(filtered.file_index_range(1)), [])
        self.assertTrue(all(int(filtered[i]["file_id"]) == 0 for i in range(3)))

    def test_dataloader_batch_has_channel_first_5d_tensors(self) -> None:
        import torch

        loader = torch.utils.data.DataLoader(self.dataset, batch_size=2, num_workers=0)
        batch = next(iter(loader))
        self.assertEqual(tuple(batch["inputs"].shape), (2, 3, 64, 16, 6))
        self.assertEqual(tuple(batch["target"].shape), (2, 1, 64, 16, 6))
        self.assertEqual(tuple(batch["loss_mask"].shape), (2, 1, 64, 16, 6))


if __name__ == "__main__":
    unittest.main()
