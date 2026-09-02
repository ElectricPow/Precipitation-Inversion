"""Independent tests for the validation-only S2-R0 local shift oracle."""

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
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.analyze_stage2_local_shift_audit import (  # noqa: E402
    SAMPLE_ID_HASH_CONTRACT,
    OrbitTask,
    analyze_task,
    load_validation_tasks,
    prepare_output_directory,
    write_outputs,
)
from precipitation_inversion.metrics.stage2_local_shift import (  # noqa: E402
    LocalShiftOptions,
    aggregate_local_shift_audits,
    audit_orbit_local_shifts,
    non_overlapping_windows,
)


def opposite_shift_volumes() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Two target windows require opposite scan shifts at the same height."""

    shape = (12, 7, 2)
    domain = np.ones(shape, dtype=bool)
    gr = np.zeros(shape, dtype=bool)
    dpr = np.zeros(shape, dtype=bool)
    # Window 0: GR(2,3) must move to DPR(3,3), hence +1 scan shift.
    gr[2, 3, 0] = True
    dpr[3, 3, 0] = True
    # Window 1: GR(9,3) must move to DPR(8,3), hence -1 scan shift.
    gr[9, 3, 0] = True
    dpr[8, 3, 0] = True
    heights = np.asarray([0.125, 0.375], dtype=np.float64)
    return gr, dpr, domain, heights


def create_opposite_shift_nc(path: Path) -> None:
    gr, dpr, domain, heights = opposite_shift_volumes()
    shape = gr.shape
    with Dataset(path, "w") as dataset:
        dataset.createDimension("nscan", shape[0])
        dataset.createDimension("nray", shape[1])
        dataset.createDimension("z", shape[2])
        z = dataset.createVariable("z", "f4", ("z",))
        z[:] = heights

        gr_values = np.full(shape, -9999.9, dtype=np.float32)
        dpr_values = np.full(shape, -9999.9, dtype=np.float32)
        gr_values[gr] = 20.0
        dpr_values[dpr] = 24.0
        pre_values = np.zeros(shape, dtype=np.float32)
        pre_values[~domain] = np.nan
        for name, values in (
            ("dbz_gr_sparse", gr_values),
            ("dbz_dpr", dpr_values),
            ("pre_dpr", pre_values),
        ):
            variable = dataset.createVariable(
                name, "f4", ("nscan", "nray", "z"), fill_value=np.nan
            )
            variable[:] = values


class LocalShiftCoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.options = LocalShiftOptions(
            window_scan=6,
            window_ray=7,
            max_shift=1,
            min_dpr_support=1,
            min_gr_support=1,
        )

    def test_non_overlapping_windows_retain_partial_tail(self) -> None:
        windows = non_overlapping_windows(
            13,
            7,
            window_scan=6,
            window_ray=7,
            include_partial=True,
        )
        self.assertEqual(
            [(row["scan_start"], row["scan_end"]) for row in windows],
            [(0, 6), (6, 12), (12, 13)],
        )
        full = non_overlapping_windows(
            13,
            7,
            window_scan=6,
            window_ray=7,
            include_partial=False,
        )
        self.assertEqual(len(full), 2)

    def test_local_oracle_exposes_opposite_shifts_hidden_by_orbit_pooling(self) -> None:
        gr, dpr, domain, heights = opposite_shift_volumes()
        result = audit_orbit_local_shifts(
            gr,
            dpr,
            domain,
            heights,
            sample_id="opposite",
            file_name="opposite.nc",
            options=self.options,
        )
        rows = result["window_height_rows"]
        self.assertEqual(len(rows), 2)
        self.assertEqual(
            {(row["best_scan_shift"], row["best_ray_shift"]) for row in rows},
            {(-1, 0), (1, 0)},
        )
        self.assertTrue(all(row["exact_support_csi"] == 0.0 for row in rows))
        self.assertTrue(all(row["best_support_csi"] == 1.0 for row in rows))

        orbit = result["orbit_row"]
        self.assertAlmostEqual(orbit["exact_support_csi"], 0.0)
        self.assertAlmostEqual(orbit["single_shift_oracle_support_csi"], 1.0 / 3.0)
        self.assertAlmostEqual(orbit["local_oracle_support_csi"], 1.0)
        self.assertTrue(orbit["opposite_scan_signs_present"])
        self.assertTrue(orbit["opposite_vector_pair_present"])
        self.assertGreater(orbit["local_minus_single_shift_csi"], 0.0)
        # The fixed target domain covers the global inner 10x5 cells once;
        # internal window boundaries introduce no additional crop.
        self.assertEqual(orbit["exact_domain_count"], 50)

        height_zero = result["per_height_rows"][0]
        self.assertEqual(height_zero["valid_window_count"], 2)
        self.assertTrue(height_zero["opposite_scan_signs_present"])
        self.assertEqual(result["per_height_rows"][1]["valid_window_count"], 0)

    def test_oracle_rejects_support_outside_common_occupancy_domain(self) -> None:
        gr, dpr, domain, heights = opposite_shift_volumes()
        domain[2, 3, 0] = False
        with self.assertRaisesRegex(ValueError, "subset of occupancy_domain"):
            audit_orbit_local_shifts(
                gr,
                dpr,
                domain,
                heights,
                sample_id="bad",
                file_name="bad.nc",
                options=self.options,
            )

    def test_validation_aggregate_reports_cancellation_evidence(self) -> None:
        gr, dpr, domain, heights = opposite_shift_volumes()
        audits = [
            audit_orbit_local_shifts(
                gr,
                dpr,
                domain,
                heights,
                sample_id=f"orbit-{index}",
                file_name=f"orbit-{index}.nc",
                options=self.options,
            )
            for index in range(2)
        ]
        aggregate = aggregate_local_shift_audits(audits)
        summary = aggregate["summary"]
        self.assertEqual(summary["orbit_count"], 2)
        self.assertEqual(summary["cancellation_evidence_orbit_count"], 2)
        self.assertAlmostEqual(summary["exact_support_csi"], 0.0)
        self.assertAlmostEqual(
            summary["one_shift_all_validation_oracle_support_csi"], 1.0 / 3.0
        )
        self.assertAlmostEqual(summary["local_window_height_oracle_support_csi"], 1.0)


class LocalShiftScriptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.nc_path = self.root / "opposite.nc"
        create_opposite_shift_nc(self.nc_path)
        self.options = LocalShiftOptions(
            window_scan=6, window_ray=7, max_shift=1
        )

    def test_manifest_is_validation_only_and_outputs_are_reproducible(self) -> None:
        manifest = self.root / "split_manifest.csv"
        with manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=("sample_id", "file_path", "split")
            )
            writer.writeheader()
            writer.writerow(
                {
                    "sample_id": "validation",
                    "file_path": str(self.nc_path),
                    "split": "val",
                }
            )
            writer.writerow(
                {
                    "sample_id": "forbidden-test",
                    "file_path": str(self.nc_path),
                    "split": "test",
                }
            )
        tasks, expected = load_validation_tasks(
            manifest, options=self.options, count=None
        )
        self.assertEqual(expected, 1)
        self.assertEqual([task.sample_id for task in tasks], ["validation"])

        audit = analyze_task(tasks[0])
        aggregate = aggregate_local_shift_audits([audit])
        output_dir = self.root / "output"
        prepare_output_directory(output_dir, overwrite=False)
        summary = write_outputs(
            output_dir,
            aggregate,
            manifest_path=manifest,
            tasks=tasks,
            expected_count=expected,
        )
        self.assertTrue(summary["formal_result"])
        self.assertFalse(summary["test_set_accessed"])
        expected_files = {
            "summary.json",
            "local_window_height.csv",
            "per_orbit_height.csv",
            "per_orbit_height_shift.csv",
            "per_orbit.csv",
            "per_orbit_shift.csv",
            "best_shift_histogram.csv",
            "all_validation_shift.csv",
            "report.md",
        }
        self.assertEqual({path.name for path in output_dir.iterdir()}, expected_files)
        loaded = json.loads((output_dir / "summary.json").read_text("utf-8"))
        self.assertEqual(loaded["format"], "stage2_r0_local_shift_audit_v1")
        self.assertEqual(loaded["metrics"]["cancellation_evidence_orbit_count"], 1)
        self.assertEqual(loaded["sample_ids"], ["validation"])
        self.assertEqual(loaded["sample_id_hash_contract"], SAMPLE_ID_HASH_CONTRACT)
        self.assertEqual(len(loaded["sample_ids_sha256"]), 64)
        self.assertEqual(len(loaded["split_manifest_sha256"]), 64)
        with self.assertRaisesRegex(FileExistsError, "already exist"):
            prepare_output_directory(output_dir, overwrite=False)


if __name__ == "__main__":
    unittest.main()
