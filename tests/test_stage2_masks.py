"""Tests for storage-aware stage-two reflectivity masks."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from precipitation_inversion.data.stage2_masks import (  # noqa: E402
    build_stage2_spatial_masks,
    classify_reflectivity_storage,
    mask_counts,
    physical_reflectivity_values,
)


class Stage2StorageMaskTests(unittest.TestCase):
    def test_three_states_are_exhaustive_and_keep_weak_dbz(self) -> None:
        values = np.ma.array(
            [123.0, -9999.9, -5.0, 0.0, 20.0, np.nan, np.inf],
            mask=[True, False, False, False, False, False, False],
        )

        states = classify_reflectivity_storage(values)

        np.testing.assert_array_equal(
            states.native_missing,
            [True, False, False, False, False, True, True],
        )
        np.testing.assert_array_equal(
            states.sentinel,
            [False, True, False, False, False, False, False],
        )
        np.testing.assert_array_equal(
            states.value,
            [False, False, True, True, True, False, False],
        )
        np.testing.assert_array_equal(
            states.native_available,
            [False, True, True, True, True, False, False],
        )
        self.assertEqual(
            states.counts(),
            {
                "total": 7,
                "native_missing": 3,
                "sentinel": 1,
                "value": 3,
                "native_available": 4,
            },
        )

    def test_physical_values_replace_both_missing_states_without_mutation(self) -> None:
        values = np.ma.array(
            [-9999.9, -3.0, 12.0, 99.0],
            mask=[False, False, False, True],
            dtype=np.float32,
        )
        original_data = values.data.copy()
        original_mask = np.ma.getmaskarray(values).copy()

        physical = physical_reflectivity_values(values, fill_value=0.0)

        np.testing.assert_allclose(physical, [0.0, -3.0, 12.0, 0.0])
        self.assertEqual(physical.dtype, np.float32)
        np.testing.assert_array_equal(values.data, original_data)
        np.testing.assert_array_equal(np.ma.getmaskarray(values), original_mask)

    def test_netcdf_nan_fill_and_finite_sentinel_remain_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "states.nc"
            with Dataset(path, "w") as dataset:
                dataset.createDimension("x", 4)
                variable = dataset.createVariable(
                    "dbz", "f4", ("x",), fill_value=np.nan
                )
                variable[:] = [np.nan, -9999.9, -10.0, 15.0]
            with Dataset(path, "r") as dataset:
                states = classify_reflectivity_storage(dataset["dbz"][:])

        np.testing.assert_array_equal(
            states.native_missing, [True, False, False, False]
        )
        np.testing.assert_array_equal(states.sentinel, [False, True, False, False])
        np.testing.assert_array_equal(states.value, [False, False, True, True])

    def test_spatial_masks_form_four_zones_and_reachability_proxies(self) -> None:
        # Shape: (nscan=1, nray=2, z=3).
        gr = np.ma.array(
            [[[10.0, -9999.9, np.nan], [30.0, -9999.9, np.nan]]],
            mask=[[[False, False, True], [False, False, True]]],
        )
        interp = np.ma.array(
            [[[10.0, 15.0, np.nan], [30.0, -9999.9, np.nan]]],
            mask=[[[False, False, True], [False, False, True]]],
        )
        dpr = np.array(
            [[[11.0, 16.0, -9999.9], [-9999.9, -9999.9, 20.0]]]
        )
        precipitation = np.array([[[1.0, 1.0, 0.0], [0.0, 0.0, 2.0]]])

        masks = build_stage2_spatial_masks(
            gr,
            dpr,
            dbz_gr_interp=interp,
            pre_dpr=precipitation,
        )

        self.assertEqual(int(masks["q11_overlap"].sum()), 1)
        self.assertEqual(int(masks["q01_dpr_only"].sum()), 2)
        self.assertEqual(int(masks["q10_gr_only"].sum()), 1)
        self.assertEqual(int(masks["q00_neither"].sum()), 2)
        self.assertEqual(int(masks["gap_proxy"].sum()), 1)
        self.assertEqual(int(masks["outside_proxy"].sum()), 3)
        self.assertEqual(int(masks["dpr_only_gap_proxy"].sum()), 1)
        self.assertEqual(int(masks["dpr_only_outside_proxy"].sum()), 1)
        self.assertEqual(int(masks["occupancy_domain"].sum()), 6)
        self.assertEqual(int(masks["pre_positive"].sum()), 3)

        zones = {
            name: masks[name]
            for name in (
                "q11_overlap",
                "q01_dpr_only",
                "q10_gr_only",
                "q00_neither",
            )
        }
        self.assertEqual(sum(mask_counts(zones).values()), 6)

    def test_mismatched_shapes_and_non_boolean_count_inputs_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "shapes differ"):
            build_stage2_spatial_masks(np.zeros((2, 2)), np.zeros((2, 3)))
        with self.assertRaisesRegex(TypeError, "boolean"):
            mask_counts({"not_a_mask": np.ones((2, 2), dtype=np.float32)})


if __name__ == "__main__":
    unittest.main()
