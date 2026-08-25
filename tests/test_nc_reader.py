"""Unit and small integration tests for selective GR--DPR NetCDF reads."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from precipitation_inversion.data.nc_reader import (  # noqa: E402
    normalize_scan_slice,
    read_nc_sample,
)


TEST_VARIABLES = (
    "z",
    "lat",
    "lon",
    "dbz_gr_sparse",
    "dbz_gr_interp",
    "dbz_dpr",
    "pre_dpr",
    "cfb",
)


def create_test_nc(path: Path) -> None:
    """Create a tiny swath with known missing, zero, rain, and clutter cells."""

    with Dataset(path, "w") as dataset:
        dataset.createDimension("nscan", 4)
        dataset.createDimension("nray", 2)
        dataset.createDimension("z", 3)

        z = dataset.createVariable("z", "f8", ("z",), fill_value=np.nan)
        z.units = "km"
        z.long_name = "altitude"
        z[:] = [0.125, 0.375, 0.625]

        lat = dataset.createVariable(
            "lat", "f4", ("nscan", "nray"), fill_value=np.nan
        )
        lon = dataset.createVariable(
            "lon", "f4", ("nscan", "nray"), fill_value=np.nan
        )
        lat[:] = np.arange(8, dtype=np.float32).reshape(4, 2)
        lon[:] = 100 + np.arange(8, dtype=np.float32).reshape(4, 2)

        shape = (4, 2, 3)
        gr_values = np.full(shape, np.nan, dtype=np.float32)
        gr_values[1, 0, 1] = 15.0
        gr_values[1, 1, 2] = -5.0  # weak negative dBZ remains observed
        gr_values[2, 1, 2] = 18.0
        interp_values = gr_values.copy()
        interp_values[2, 0, 2] = 17.0

        pre_values = np.zeros(shape, dtype=np.float32)
        pre_values[1, 0, 1] = 2.0
        pre_values[1, 1, 0] = 3.0  # positive, but below CFB
        pre_values[2, 0, 2] = 5.0
        pre_values[2, 1, 2] = np.nan
        dpr_values = np.full(shape, np.nan, dtype=np.float32)
        dpr_values[1, 0, 1] = 20.0
        dpr_values[1, 1, 0] = 21.0
        dpr_values[2, 0, 2] = 22.0

        for name, values in (
            ("dbz_gr_sparse", gr_values),
            ("dbz_gr_interp", interp_values),
            ("dbz_dpr", dpr_values),
            ("pre_dpr", pre_values),
        ):
            variable = dataset.createVariable(
                name, "f4", ("nscan", "nray", "z"), fill_value=np.nan
            )
            variable[:] = values

        cfb = dataset.createVariable("cfb", "i2", ("nscan", "nray"))
        cfb[:] = 1  # z[0] is below clutter-free bottom for every profile


class NCReaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary_directory.name) / "sample.nc"
        create_test_nc(self.path)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_selective_scan_read_preserves_alignment_and_metadata(self) -> None:
        sample = read_nc_sample(
            self.path, variables=TEST_VARIABLES, scan_slice=slice(1, 3)
        )
        self.assertEqual(sample.source_dimensions["nscan"], 4)
        self.assertEqual(sample.dimensions["nscan"], 2)
        self.assertEqual(sample.shape_3d, (2, 2, 3))
        self.assertEqual((sample.scan_start, sample.scan_stop), (1, 3))
        self.assertEqual(sample.variables["z"].shape, (3,))
        self.assertEqual(sample.variables["lat"].shape, (2, 2))
        self.assertEqual(sample.variables["pre_dpr"].shape, (2, 2, 3))
        self.assertEqual(sample.variables["pre_dpr"].dtype, np.float32)
        self.assertEqual(sample.metadata["z"].source_shape, (3,))
        self.assertEqual(sample.metadata["lat"].source_shape, (4, 2))
        self.assertEqual(sample.metadata["lat"].returned_shape, (2, 2))

    def test_masks_distinguish_missing_zero_positive_and_clutter(self) -> None:
        sample = read_nc_sample(
            self.path, variables=TEST_VARIABLES, scan_slice=slice(1, 3)
        )
        masks = sample.masks
        self.assertEqual(int(masks["gr_sparse_observed"].sum()), 3)
        self.assertTrue(masks["gr_sparse_observed"][0, 1, 2])  # -5 dBZ
        self.assertEqual(int(masks["dpr_reflectivity_valid"].sum()), 3)
        self.assertEqual(int(masks["pre_valid_native"].sum()), 11)
        self.assertEqual(int(masks["pre_zero_native"].sum()), 8)
        self.assertEqual(int(masks["pre_positive_native"].sum()), 3)
        self.assertEqual(int(masks["cfb_clutter"].sum()), 4)
        self.assertEqual(int(masks["pre_valid_qc"].sum()), 7)
        self.assertEqual(int(masks["pre_positive_qc"].sum()), 2)
        self.assertEqual(int(masks["gr_dpr_overlap"].sum()), 1)
        np.testing.assert_array_equal(
            masks["dpr_reflectivity_valid"], masks["pre_positive_native"]
        )

    def test_only_requested_variables_are_loaded(self) -> None:
        sample = read_nc_sample(
            self.path, variables=("z", "lat"), scan_slice=slice(0, 1)
        )
        self.assertEqual(set(sample.variables), {"z", "lat"})
        self.assertEqual(sample.masks, {})
        with self.assertRaisesRegex(KeyError, "not requested"):
            sample.require_variable("pre_dpr")
        with self.assertRaisesRegex(KeyError, "cannot be built"):
            sample.require_mask("pre_valid_qc")

    def test_masks_can_be_disabled(self) -> None:
        sample = read_nc_sample(
            self.path,
            variables=("z", "pre_dpr", "cfb"),
            build_masks=False,
        )
        self.assertFalse(sample.masks)

    def test_missing_requested_variable_is_rejected(self) -> None:
        with self.assertRaisesRegex(KeyError, "missing requested"):
            read_nc_sample(self.path, variables=("z", "does_not_exist"))

    def test_non_floating_output_dtype_is_rejected(self) -> None:
        with self.assertRaisesRegex(TypeError, "floating-point"):
            read_nc_sample(self.path, variables=("z",), dtype=np.int16)

    def test_non_contiguous_or_empty_scan_slice_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "contiguous"):
            normalize_scan_slice(slice(None, None, 2), 4)
        with self.assertRaisesRegex(ValueError, "selects no scans"):
            normalize_scan_slice(slice(2, 2), 4)


if __name__ == "__main__":
    unittest.main()

