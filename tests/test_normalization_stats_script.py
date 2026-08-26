"""Tests for train-only input-normalization statistic fitting semantics."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.fit_normalization_stats import fit_statistics  # noqa: E402


def create_normalization_nc(path: Path) -> None:
    """Create ``(2,2,3)`` input values with one below-CFB positive label."""

    with Dataset(path, "w") as dataset:
        dataset.createDimension("nscan", 2)
        dataset.createDimension("nray", 2)
        dataset.createDimension("z", 3)
        heights = dataset.createVariable("z", "f4", ("z",))
        heights[:] = [0.125, 0.375, 0.625]

        dbz_values = np.full((2, 2, 3), np.nan, dtype=np.float32)
        dbz_values[0, 0, 0] = 10.0  # finite input, below CFB
        dbz_values[0, 0, 1] = 20.0  # reliable positive label
        dbz_values[1, 0, 1] = 22.0  # finite input, zero-rain label
        dbz_values[1, 1, 2] = 30.0  # reliable positive label
        dbz = dataset.createVariable(
            "dbz_dpr", "f4", ("nscan", "nray", "z"), fill_value=np.nan
        )
        dbz[:] = dbz_values

        rain_values = np.zeros((2, 2, 3), dtype=np.float32)
        rain_values[0, 0, 0] = 9.0
        rain_values[0, 0, 1] = 1.0
        rain_values[1, 1, 2] = 2.0
        rain = dataset.createVariable(
            "pre_dpr", "f4", ("nscan", "nray", "z"), fill_value=np.nan
        )
        rain[:] = rain_values
        cfb = dataset.createVariable("cfb", "i2", ("nscan", "nray"))
        cfb[:] = 1


class NormalizationStatisticsScriptTests(unittest.TestCase):
    def test_variable_valid_input_and_reliable_height_counts_are_independent(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.nc"
            create_normalization_nc(path)
            accumulators, z, _, chunks, loss_counts = fit_statistics(
                [path],
                variables=("dbz_dpr",),
                selection_mask="variable_valid",
                height_loss_selection_mask="pre_positive_qc",
                scan_chunk_size=1,
            )

        self.assertEqual(chunks, 2)
        np.testing.assert_allclose(z, [0.125, 0.375, 0.625])
        np.testing.assert_array_equal(
            accumulators["dbz_dpr"].count, [1, 2, 1]
        )
        np.testing.assert_allclose(
            accumulators["dbz_dpr"].mean, [10.0, 21.0, 30.0]
        )
        # The level-0 rain value is positive but below CFB. It contributes to
        # native input normalization, never the reliable loss-height counts.
        np.testing.assert_array_equal(loss_counts, [0, 1, 1])

    def test_legacy_label_selection_would_remove_the_low_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.nc"
            create_normalization_nc(path)
            accumulators, _, _, _, loss_counts = fit_statistics(
                [path],
                variables=("dbz_dpr",),
                selection_mask="pre_positive_qc",
                height_loss_selection_mask="none",
                scan_chunk_size=2,
            )

        np.testing.assert_array_equal(
            accumulators["dbz_dpr"].count, [0, 1, 1]
        )
        self.assertIsNone(loss_counts)

    def test_invalid_height_loss_selection_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "height_loss_selection_mask"):
            fit_statistics(
                [Path("not-opened.nc")],
                variables=("dbz_dpr",),
                selection_mask="variable_valid",
                height_loss_selection_mask="pre_valid_native",
                scan_chunk_size=1,
            )


if __name__ == "__main__":
    unittest.main()
