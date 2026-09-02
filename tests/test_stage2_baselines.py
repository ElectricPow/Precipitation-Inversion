"""Integration tests for controlled Stage-2 baseline evaluation."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.evaluate_stage2_baselines import (  # noqa: E402
    BASELINE_NAMES,
    build_baseline_predictions,
    evaluate_rows,
    height_calibrate_reflectivity,
    prepare_output_directory,
    write_outputs,
)


HEIGHTS = [0.125, 0.375]


def _statistics(mean: list[float], std: list[float]) -> dict[str, object]:
    return {
        "heights_km": HEIGHTS,
        "mean": mean,
        "std": std,
        "count": [10, 10],
    }


def normalization() -> dict[str, object]:
    return {
        "stage": 2,
        "scope": "training_split_only",
        "selection_mask": "reflectivity_storage_value",
        "variables": {
            "dbz_gr_sparse": _statistics([10.0, 20.0], [2.0, 4.0]),
            "dbz_gr_interp": _statistics([12.0, 22.0], [2.0, 4.0]),
            "dbz_dpr": _statistics([14.0, 24.0], [4.0, 8.0]),
        },
    }


def create_baseline_nc(path: Path) -> None:
    shape = (3, 2, 2)
    gr = np.full(shape, np.nan, dtype=np.float32)
    gr[0, 0, :] = [10.0, 20.0]  # direct target support
    gr[0, 1, :] = [12.0, 24.0]  # false-positive GR support
    gr[2, 0, 0] = -9999.9
    interp = np.full(shape, np.nan, dtype=np.float32)
    interp[0, 0, :] = gr[0, 0, :]
    interp[0, 1, :] = gr[0, 1, :]
    interp[1, 0, :] = [12.0, 22.0]  # reaches one DPR-only profile
    dpr = np.full(shape, -9999.9, dtype=np.float32)
    dpr[0, 0, :] = [14.0, 24.0]
    dpr[1, 0, :] = [18.0, 32.0]
    dpr[2, 1, :] = [20.0, 36.0]  # outside interpolation proxy
    pre = np.zeros(shape, dtype=np.float32)
    pre[dpr > -9990.0] = 1.0

    with Dataset(path, "w") as dataset:
        dataset.createDimension("nscan", 3)
        dataset.createDimension("nray", 2)
        dataset.createDimension("z", 2)
        z = dataset.createVariable("z", "f4", ("z",), fill_value=np.nan)
        z[:] = HEIGHTS
        for name, values in (
            ("dbz_gr_sparse", gr),
            ("dbz_gr_interp", interp),
            ("dbz_dpr", dpr),
            ("pre_dpr", pre),
        ):
            variable = dataset.createVariable(
                name, "f4", ("nscan", "nray", "z"), fill_value=np.nan
            )
            variable[:] = values
        cfb = dataset.createVariable("cfb", "i2", ("nscan", "nray"))
        cfb[:] = 0


class Stage2BaselineTests(unittest.TestCase):
    def test_height_calibration_and_invalid_level_handling(self) -> None:
        values = np.array([[[10.0, 24.0]]])
        support = np.ones_like(values, dtype=bool)
        mapped, effective = height_calibrate_reflectivity(
            values,
            support,
            source_statistics=_statistics([10.0, 20.0], [2.0, 4.0]),
            target_statistics=_statistics([14.0, 24.0], [4.0, 8.0]),
        )
        np.testing.assert_array_equal(effective, support)
        np.testing.assert_allclose(mapped, [[[14.0, 32.0]]])
        with self.assertRaises(TypeError):
            height_calibrate_reflectivity(
                values,
                np.ones_like(values),
                source_statistics=_statistics([10.0, 20.0], [2.0, 4.0]),
                target_statistics=_statistics([14.0, 24.0], [4.0, 8.0]),
            )

    def test_prediction_set_contains_four_controlled_baselines(self) -> None:
        values = np.array([[[10.0, 20.0]]], dtype=np.float32)
        support = np.ones_like(values, dtype=bool)
        predictions = build_baseline_predictions(
            values, support, values + 2.0, support, normalization()
        )
        self.assertEqual(tuple(predictions), BASELINE_NAMES)
        np.testing.assert_allclose(
            predictions["gr_sparse_height_calibrated"][0],
            [[[14.0, 24.0]]],
        )

    def test_end_to_end_metrics_regions_and_atomic_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sample.nc"
            create_baseline_nc(source)
            rows = [
                {
                    "sample_id": "synthetic",
                    "file_path": str(source),
                    "split": "test",
                }
            ]
            result = evaluate_rows(rows, normalization(), fss_radii=(1,))
            sparse = result["summary"]["all"]["gr_sparse_raw"]
            interp = result["summary"]["all"]["gr_interp_raw"]
            self.assertEqual(sparse["support"]["true_positive"], 2)
            self.assertEqual(sparse["support"]["false_positive"], 2)
            self.assertEqual(sparse["support"]["false_negative"], 4)
            self.assertAlmostEqual(sparse["support"]["recall"], 1.0 / 3.0)
            self.assertAlmostEqual(interp["support"]["recall"], 2.0 / 3.0)
            calibrated = result["summary"]["all"][
                "gr_sparse_height_calibrated"
            ]
            self.assertAlmostEqual(
                calibrated["reflectivity_on_common_support"]["mae_dbz"], 0.0
            )
            region_names = {
                row["region"]
                for row in result["per_region"]
                if row["group"] == "all"
            }
            self.assertIn("dpr_gap_proxy", region_names)
            self.assertIn("dpr_outside_proxy", region_names)

            output = root / "outputs"
            prepare_output_directory(output, overwrite=False)
            normalization_path = root / "normalization.json"
            normalization_path.write_text(json.dumps(normalization()))
            manifest = root / "manifest.csv"
            manifest.write_text("sample_id,file_path,split\n")
            summary = write_outputs(
                output,
                result,
                rows=rows,
                split_manifest=manifest,
                normalization_path=normalization_path,
                fss_radii=(1,),
            )
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {"summary.json", "per_file.csv", "per_height.csv", "per_region.csv"},
            )
            self.assertEqual(summary["selected_file_count"], 1)
            with self.assertRaises(FileExistsError):
                prepare_output_directory(output, overwrite=False)


if __name__ == "__main__":
    unittest.main()
