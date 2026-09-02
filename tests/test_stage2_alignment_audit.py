"""Small integration tests for the stage-two GR/DPR alignment audit."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_stage2_gr_dpr_alignment import (  # noqa: E402
    AuditOptions,
    AuditTask,
    aggregate_results,
    analyze_file,
    best_shift_rows,
    load_tasks,
    prepare_output_directory,
    shift_rows,
    write_outputs,
)


def create_alignment_nc(path: Path) -> None:
    """Create one shifted DPR target and all three GR storage states."""

    shape = (3, 2, 2)
    with Dataset(path, "w") as dataset:
        dataset.createDimension("nscan", shape[0])
        dataset.createDimension("nray", shape[1])
        dataset.createDimension("z", shape[2])
        z = dataset.createVariable("z", "f4", ("z",), fill_value=np.nan)
        z[:] = [0.125, 0.375]

        gr_values = np.full(shape, np.nan, dtype=np.float32)
        gr_values[0, 0, 0] = 20.0
        gr_values[0, 1, 0] = -9999.9

        interp_values = np.full(shape, np.nan, dtype=np.float32)
        interp_values[0, 0, 0] = 20.0
        interp_values[1, 0, 0] = 24.0  # interpolation-reachable DPR-only point
        interp_values[0, 1, 0] = -9999.9

        # DPR uses a finite sentinel for every no-echo cell, as in the real NCs.
        dpr_values = np.full(shape, -9999.9, dtype=np.float32)
        dpr_values[1, 0, 0] = 25.0

        precipitation = np.zeros(shape, dtype=np.float32)
        precipitation[1, 0, 0] = 2.0

        for name, values in (
            ("dbz_gr_sparse", gr_values),
            ("dbz_gr_interp", interp_values),
            ("dbz_dpr", dpr_values),
            ("pre_dpr", precipitation),
        ):
            variable = dataset.createVariable(
                name, "f4", ("nscan", "nray", "z"), fill_value=np.nan
            )
            variable[:] = values

        cfb = dataset.createVariable("cfb", "i2", ("nscan", "nray"))
        cfb[:] = 0


class Stage2AlignmentAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.nc_path = self.root / "sample.nc"
        create_alignment_nc(self.nc_path)
        self.options = AuditOptions(
            max_shift=1,
            distance_radii=(0, 1, 2),
            density_radius=1,
        )
        self.task = AuditTask(
            path=self.nc_path,
            sample_id="synthetic",
            split="train",
            options=self.options,
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_file_audit_preserves_states_regions_and_shift_direction(self) -> None:
        result = analyze_file(self.task)
        row = result["row"]

        self.assertEqual(row["total_count"], 12)
        self.assertEqual(row["gr_native_missing_count"], 10)
        self.assertEqual(row["gr_sentinel_count"], 1)
        self.assertEqual(row["gr_value_count"], 1)
        self.assertEqual(row["dpr_native_missing_count"], 0)
        self.assertEqual(row["dpr_sentinel_count"], 11)
        self.assertEqual(row["dpr_value_count"], 1)
        self.assertEqual(row["q11_overlap_count"], 0)
        self.assertEqual(row["q01_dpr_only_count"], 1)
        self.assertEqual(row["q10_gr_only_count"], 1)
        self.assertEqual(row["q00_neither_count"], 10)
        self.assertEqual(row["dpr_only_gap_proxy_count"], 1)
        self.assertEqual(row["dpr_only_outside_proxy_count"], 0)
        self.assertEqual(row["dpr_pre_positive_mismatch_count"], 0)

        np.testing.assert_array_equal(
            result["distance"]["dpr_distance_bin_count"], [0, 1, 0, 0]
        )
        aggregate = aggregate_results([result])
        shifts = shift_rows(aggregate)
        best = best_shift_rows(shifts)
        height_zero = next(
            row
            for row in best
            if row["group"] == "all" and row["height_index"] == 0
        )
        self.assertEqual(height_zero["best_scan_shift"], 1)
        self.assertEqual(height_zero["best_ray_shift"], 0)
        self.assertAlmostEqual(height_zero["best_support_csi"], 1.0)
        self.assertAlmostEqual(height_zero["mae_dbz_at_best_support_shift"], 5.0)

    def test_manifest_loading_and_atomic_tabular_outputs(self) -> None:
        manifest = self.root / "split_manifest.csv"
        with manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=("sample_id", "file_path", "split")
            )
            writer.writeheader()
            writer.writerow(
                {
                    "sample_id": "synthetic",
                    "file_path": str(self.nc_path),
                    "split": "train",
                }
            )

        tasks = load_tasks(
            manifest,
            splits=("train",),
            count=None,
            options=self.options,
        )
        self.assertEqual(tasks, [self.task])
        result = analyze_file(tasks[0])
        aggregate = aggregate_results([result])
        output_dir = self.root / "audit"
        prepare_output_directory(output_dir, overwrite=False)
        summary = write_outputs(
            output_dir,
            [result],
            aggregate,
            tasks=tasks,
            manifest_path=manifest,
            plots=False,
        )

        expected = {
            "summary.json",
            "per_file.csv",
            "per_height.csv",
            "distance_to_gr.csv",
            "local_density.csv",
            "shift_metrics.csv",
            "best_shift_by_height.csv",
        }
        self.assertEqual({path.name for path in output_dir.iterdir()}, expected)
        loaded = json.loads((output_dir / "summary.json").read_text("utf-8"))
        self.assertEqual(loaded["selected_file_count"], 1)
        self.assertEqual(loaded["groups"]["all"]["totals"]["q01_dpr_only_count"], 1)
        self.assertEqual(summary["groups"]["all"]["ratios"]["q01_of_dpr"], 1.0)
        with self.assertRaisesRegex(FileExistsError, "already exist"):
            prepare_output_directory(output_dir, overwrite=False)

    def test_invalid_radius_and_missing_manifest_columns_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "start at 0"):
            AuditOptions(distance_radii=(1, 2))
        manifest = self.root / "bad.csv"
        manifest.write_text("sample_id,split\nx,train\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "must contain columns"):
            load_tasks(
                manifest,
                splits=("train",),
                count=None,
                options=self.options,
            )


if __name__ == "__main__":
    unittest.main()
