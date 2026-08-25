"""Tests for the compact stage-one index and PyTorch Dataset."""

from __future__ import annotations

import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from precipitation_inversion.data import dataset as dataset_module  # noqa: E402
from precipitation_inversion.data.dataset import (  # noqa: E402
    STAGE1_INDEX_DTYPE,
    STAGE1_INDEX_FORMAT_VERSION,
    STAGE1_INPUT_VARIABLES,
    Stage1IntensityDataset,
    atomic_save_json,
    atomic_save_npy,
    build_stage1_index_records,
    sha256_file,
)


HEIGHTS = np.array([0.125, 0.375, 0.625, 0.875], dtype=np.float32)


def create_stage1_nc(path: Path, *, second_file: bool = False) -> None:
    """Create a tiny file containing positive, cluttered, and unusable cells."""

    shape = (3, 2, 4)
    coordinate = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    offset = 100.0 if second_file else 0.0
    with Dataset(path, "w") as dataset:
        dataset.createDimension("nscan", shape[0])
        dataset.createDimension("nray", shape[1])
        dataset.createDimension("z", shape[2])
        z = dataset.createVariable("z", "f4", ("z",), fill_value=np.nan)
        z[:] = HEIGHTS

        values = {
            "dbz_dpr": 20.0 + offset + coordinate,
            "p": 900.0 + offset + coordinate,
            "t": 270.0 + offset + coordinate,
            "q": 0.01 + (offset + coordinate) / 1000.0,
        }
        # This positive label should be omitted because one model input is missing.
        if not second_file:
            values["q"][1, 1, 2] = np.nan
        for name, array in values.items():
            variable = dataset.createVariable(
                name, "f4", ("nscan", "nray", "z"), fill_value=np.nan
            )
            variable[:] = array

        rain = np.zeros(shape, dtype=np.float32)
        if second_file:
            rain[0, 1, 1] = 2.0
            rain[2, 1, 3] = 8.0
        else:
            rain[0, 0, 0] = 9.0  # below cfb: positive natively, cluttered after QC
            rain[0, 0, 1] = 1.0
            rain[1, 1, 2] = 3.0  # q is missing
            rain[2, 0, 3] = 7.0
        pre = dataset.createVariable(
            "pre_dpr", "f4", ("nscan", "nray", "z"), fill_value=np.nan
        )
        pre[:] = rain
        cfb = dataset.createVariable("cfb", "i2", ("nscan", "nray"))
        cfb[:] = 1


def write_normalization(path: Path) -> None:
    variables = {}
    for name in STAGE1_INPUT_VARIABLES:
        variables[name] = {
            "heights_km": HEIGHTS.tolist(),
            "mean": [0.0, 0.0, 0.0, 0.0],
            "std": [1.0, 2.0, 4.0, 8.0],
        }
    path.write_text(
        json.dumps(
            {
                "scope": "training_split_only",
                "selection_mask": "pre_positive_qc",
                "variables": variables,
            }
        ),
        encoding="utf-8",
    )


class Stage1IndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.paths = [self.root / "first.nc", self.root / "second.nc"]
        create_stage1_nc(self.paths[0])
        create_stage1_nc(self.paths[1], second_file=True)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_index_keeps_only_usable_positive_qc_cells(self) -> None:
        records, files, heights, chunks = build_stage1_index_records(
            self.paths, scan_chunk_size=2
        )
        self.assertEqual(records.dtype, STAGE1_INDEX_DTYPE)
        self.assertEqual(records.dtype.itemsize, 6)
        self.assertEqual(len(records), 4)
        self.assertEqual(chunks, 4)
        np.testing.assert_allclose(heights, HEIGHTS)
        self.assertEqual(files[0]["positive_qc_count"], 3)
        self.assertEqual(files[0]["excluded_missing_input_count"], 1)
        self.assertEqual((files[0]["index_start"], files[0]["index_stop"]), (0, 2))
        self.assertEqual((files[1]["index_start"], files[1]["index_stop"]), (2, 4))
        positions = [
            tuple(int(record[name]) for name in record.dtype.names)
            for record in records
        ]
        self.assertEqual(
            positions,
            [(0, 0, 0, 1), (0, 2, 0, 3), (1, 0, 1, 1), (1, 2, 1, 3)],
        )

    def test_invalid_index_build_arguments_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            build_stage1_index_records(self.paths, scan_chunk_size=0)
        with self.assertRaisesRegex(ValueError, "at least one"):
            build_stage1_index_records([], scan_chunk_size=2)


@unittest.skipIf(dataset_module.torch is None, "a usable PyTorch is not installed")
class Stage1DatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.paths = [self.root / "first.nc", self.root / "second.nc"]
        create_stage1_nc(self.paths[0])
        create_stage1_nc(self.paths[1], second_file=True)
        records, files, heights, _ = build_stage1_index_records(
            self.paths, scan_chunk_size=2
        )
        self.index_path = self.root / "train.npy"
        atomic_save_npy(self.index_path, records)
        self.metadata_path = self.root / "train.json"
        atomic_save_json(
            self.metadata_path,
            {
                "format_version": STAGE1_INDEX_FORMAT_VERSION,
                "selection_mask": "pre_positive_qc",
                "input_variables": list(STAGE1_INPUT_VARIABLES),
                "label_variable": "pre_dpr",
                "label_transform": "log1p",
                "index_file": self.index_path.name,
                "index_sha256": sha256_file(self.index_path),
                "sample_count": len(records),
                "heights_km": heights.tolist(),
                "files": files,
            },
        )
        self.normalization_path = self.root / "normalization.json"
        write_normalization(self.normalization_path)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_item_contains_standardized_features_log_target_and_location(self) -> None:
        torch = dataset_module.torch
        dataset = Stage1IntensityDataset(
            self.metadata_path, self.normalization_path, cache_size=1
        )
        item = dataset[0]  # first.nc, scan=0, ray=0, level=1
        self.assertEqual(dataset.feature_names, ["dbz_dpr", "p", "t", "q", "height_scaled"])
        expected = torch.tensor(
            [21.0 / 2.0, 901.0 / 2.0, 271.0 / 2.0, 0.011 / 2.0, -1.0 / 3.0],
            dtype=torch.float32,
        )
        torch.testing.assert_close(item["features"], expected)
        torch.testing.assert_close(
            item["target"], torch.tensor(np.log(2.0), dtype=torch.float32)
        )
        self.assertEqual(float(item["rain_rate"]), 1.0)
        self.assertEqual((int(item["file_id"]), int(item["level"])), (0, 1))

    def test_lru_cache_and_dataloader_batch(self) -> None:
        torch = dataset_module.torch
        dataset = Stage1IntensityDataset(
            self.metadata_path,
            self.normalization_path,
            cache_size=1,
            include_height=False,
        )
        _ = dataset[0]
        self.assertEqual(dataset.cached_file_ids, (0,))
        _ = dataset[2]
        self.assertEqual(dataset.cached_file_ids, (1,))
        loader = torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=0)
        batch = next(iter(loader))
        self.assertEqual(tuple(batch["features"].shape), (2, 4))
        self.assertEqual(tuple(batch["target"].shape), (2,))

    def test_multi_worker_dataloader_can_read_the_memory_mapped_index(self) -> None:
        torch = dataset_module.torch
        dataset = Stage1IntensityDataset(self.metadata_path, self.normalization_path)
        loader = torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=2)
        batch = next(iter(loader))
        self.assertEqual(tuple(batch["features"].shape), (2, 5))
        self.assertTrue(bool(torch.isfinite(batch["features"]).all()))

    def test_pickled_worker_copy_reopens_index_and_drops_file_cache(self) -> None:
        dataset = Stage1IntensityDataset(self.metadata_path, self.normalization_path)
        _ = dataset[0]
        restored = pickle.loads(pickle.dumps(dataset))
        self.assertEqual(restored.cached_file_ids, ())
        self.assertIsInstance(restored.records, np.memmap)
        self.assertEqual(len(restored), 4)
        self.assertEqual(float(restored[0]["rain_rate"]), 1.0)

    def test_corrupted_index_is_detected_by_hash(self) -> None:
        with self.index_path.open("ab") as handle:
            handle.write(b"corruption")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            Stage1IntensityDataset(self.metadata_path, self.normalization_path)


if __name__ == "__main__":
    unittest.main()
