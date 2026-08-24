#!/usr/bin/env python3
"""Audit GR--DPR NetCDF files and build a reproducible file-level manifest.

The script is intentionally read-only with respect to the source dataset.  It
summarizes data coverage, precipitation imbalance, GR--DPR overlap, mask
relationships, and schema consistency before any train/validation/test split is
chosen.  Outputs are first written as ``*.partial`` and atomically renamed only
after the selected scan completes.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from netCDF4 import Dataset, num2date


# Allow ``python scripts/build_dataset_manifest.py`` from a fresh checkout before
# the project is installed as a package.  The import remains the same after a
# future editable installation.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from precipitation_inversion.data.masks import (  # noqa: E402
    clutter_mask_from_cfb,
    dpr_reflectivity_mask,
    gr_observation_mask,
    precipitation_label_mask,
    positive_rain_mask,
    profile_has_observation,
    to_float_array,
    valid_cfb_mask,
    zero_rain_mask,
)


DEFAULT_DATASET_DIR = Path(
    "/storage/GR_DPR_3D/GRToDPRRes_V07_Pct_V1.2.1_sw_260412"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "metadata" / "manifests"

KEY_VARIABLES = (
    "z",
    "lat",
    "lon",
    "dbz_gr_sparse",
    "dbz_gr_interp",
    "dbz_dpr",
    "pre_dpr",
    "cfb",
    "typePrecip",
    "flagPrecip",
)
OPTIONAL_AUXILIARIES = ("p", "t", "q")
RAIN_THRESHOLDS = (1.0, 5.0, 10.0, 20.0, 50.0)

# This order makes the CSV stable and easy to diff across repeated runs.
MANIFEST_FIELDS = (
    "sample_id",
    "file_name",
    "file_path",
    "file_size_bytes",
    "date",
    "start_time",
    "end_time",
    "orbit_id",
    "scan_size",
    "ray_size",
    "z_size",
    "variable_count",
    "variables",
    "missing_key_variables",
    "has_p",
    "has_t",
    "has_q",
    "latitude_min",
    "latitude_max",
    "longitude_min",
    "longitude_max",
    "total_voxel_count",
    "gr_sparse_valid_count",
    "gr_sparse_valid_ratio",
    "gr_interp_valid_count",
    "gr_interp_valid_ratio",
    "gr_profiles_with_observation",
    "gr_below_9dbz_count",
    "dpr_dbz_valid_count",
    "dpr_dbz_valid_ratio",
    "pre_valid_count",
    "pre_missing_count",
    "pre_zero_count",
    "pre_positive_count",
    "pre_zero_ratio",
    "pre_qc_valid_count",
    "pre_qc_zero_count",
    "pre_qc_positive_count",
    "gr_dpr_overlap_count",
    "gr_valid_pre_positive_count",
    "gr_missing_pre_positive_count",
    "gr_missing_interp_valid_count",
    "pre_p50_positive",
    "pre_p90_positive",
    "pre_p95_positive",
    "pre_p99_positive",
    "pre_max",
    "pre_gt_1_count",
    "pre_gt_5_count",
    "pre_gt_10_count",
    "pre_gt_20_count",
    "pre_gt_50_count",
    "no_precip_profile_count",
    "stratiform_profile_count",
    "convective_profile_count",
    "other_precip_profile_count",
    "cfb_valid_profile_count",
    "cfb_invalid_profile_count",
    "cfb_clutter_cell_count",
    "dpr_mask_equals_positive_rain",
    "dpr_valid_pre_zero_count",
    "dpr_missing_pre_positive_count",
    "status",
    "warning_count",
    "warning_messages",
)

COUNT_FIELDS = (
    "total_voxel_count",
    "gr_sparse_valid_count",
    "gr_interp_valid_count",
    "gr_profiles_with_observation",
    "gr_below_9dbz_count",
    "dpr_dbz_valid_count",
    "pre_valid_count",
    "pre_missing_count",
    "pre_zero_count",
    "pre_positive_count",
    "pre_qc_valid_count",
    "pre_qc_zero_count",
    "pre_qc_positive_count",
    "gr_dpr_overlap_count",
    "gr_valid_pre_positive_count",
    "gr_missing_pre_positive_count",
    "gr_missing_interp_valid_count",
    "pre_gt_1_count",
    "pre_gt_5_count",
    "pre_gt_10_count",
    "pre_gt_20_count",
    "pre_gt_50_count",
    "no_precip_profile_count",
    "stratiform_profile_count",
    "convective_profile_count",
    "other_precip_profile_count",
    "cfb_valid_profile_count",
    "cfb_invalid_profile_count",
    "cfb_clutter_cell_count",
    "dpr_valid_pre_zero_count",
    "dpr_missing_pre_positive_count",
)

FILENAME_PATTERN = re.compile(
    r"\.(?P<date>\d{8})-S(?P<start>\d{6})-E(?P<end>\d{6})\."
    r"(?P<orbit>\d+)\."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a file-level audit manifest for the GR-DPR NetCDF dataset."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--count",
        type=int,
        help="Only inspect the first N files in sorted order (default: all files).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Independent worker processes (default: 1; increase cautiously on shared storage).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing complete manifest in --output-dir.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately if one NetCDF file cannot be audited.",
    )
    return parser.parse_args()


def safe_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def finite_bounds(values: np.ndarray) -> tuple[float | None, float | None]:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return None, None
    return float(finite.min()), float(finite.max())


def filename_metadata(file_name: str) -> tuple[str, str]:
    match = FILENAME_PATTERN.search(file_name)
    if not match:
        return "", ""
    date = datetime.strptime(match.group("date"), "%Y%m%d").date().isoformat()
    return date, match.group("orbit")


def iso_time_bounds(dataset: Dataset) -> tuple[str, str]:
    """Decode the first and last NetCDF time values without assuming a calendar."""

    if "time" not in dataset.variables:
        return "", ""
    variable = dataset.variables["time"]
    if not hasattr(variable, "units") or variable.size == 0:
        return "", ""
    values = to_float_array(variable)
    finite = values[np.isfinite(values)]
    if not finite.size:
        return "", ""
    calendar = getattr(variable, "calendar", "standard")
    decoded = num2date([finite[0], finite[-1]], variable.units, calendar=calendar)
    return tuple(str(value).replace(" ", "T") for value in decoded)  # type: ignore[return-value]


def require_same_shape(
    name: str, array: np.ndarray, expected: tuple[int, ...], warnings: list[str]
) -> bool:
    if array.shape == expected:
        return True
    warnings.append(f"{name} shape {array.shape} != expected {expected}")
    return False


def analyze_file(path: Path) -> dict[str, Any]:
    """Read one file and return one flat manifest row.

    The function is top-level and side-effect free so it can run in a separate
    process when ``--workers`` is greater than one.
    """

    warnings: list[str] = []
    date, orbit_id = filename_metadata(path.name)
    row: dict[str, Any] = {field: None for field in MANIFEST_FIELDS}
    row.update(
        {
            "sample_id": path.stem,
            "file_name": path.name,
            "file_path": str(path.resolve()),
            "file_size_bytes": path.stat().st_size,
            "date": date,
            "orbit_id": orbit_id,
        }
    )

    with Dataset(path, "r") as dataset:
        dimensions = {name: len(dim) for name, dim in dataset.dimensions.items()}
        variable_names = tuple(dataset.variables)
        variable_set = set(variable_names)
        missing = sorted(set(KEY_VARIABLES).difference(variable_set))
        if missing:
            warnings.append("missing key variables: " + ",".join(missing))

        row.update(
            {
                "scan_size": dimensions.get("nscan"),
                "ray_size": dimensions.get("nray"),
                "z_size": dimensions.get("z"),
                "variable_count": len(variable_names),
                "variables": ",".join(variable_names),
                "missing_key_variables": ",".join(missing),
                **{
                    f"has_{name}": name in variable_set
                    for name in OPTIONAL_AUXILIARIES
                },
            }
        )
        row["start_time"], row["end_time"] = iso_time_bounds(dataset)

        if "lat" in variable_set:
            row["latitude_min"], row["latitude_max"] = finite_bounds(
                to_float_array(dataset.variables["lat"])
            )
        if "lon" in variable_set:
            row["longitude_min"], row["longitude_max"] = finite_bounds(
                to_float_array(dataset.variables["lon"])
            )

        expected_3d: tuple[int, ...] | None = None
        if all(dimensions.get(name) is not None for name in ("nscan", "nray", "z")):
            expected_3d = (
                dimensions["nscan"],
                dimensions["nray"],
                dimensions["z"],
            )
            row["total_voxel_count"] = int(np.prod(expected_3d))

        arrays: dict[str, np.ndarray] = {}
        for name in (
            "z",
            "dbz_gr_sparse",
            "dbz_gr_interp",
            "dbz_dpr",
            "pre_dpr",
            "cfb",
            "typePrecip",
        ):
            if name in variable_set:
                arrays[name] = to_float_array(dataset.variables[name])

        if expected_3d is not None:
            for name in ("dbz_gr_sparse", "dbz_gr_interp", "dbz_dpr", "pre_dpr"):
                if name in arrays:
                    require_same_shape(name, arrays[name], expected_3d, warnings)

        gr_mask: np.ndarray | None = None
        interp_mask: np.ndarray | None = None
        dpr_mask: np.ndarray | None = None
        pre_valid: np.ndarray | None = None
        pre_positive: np.ndarray | None = None
        pre_zero: np.ndarray | None = None

        if "dbz_gr_sparse" in arrays:
            gr = arrays["dbz_gr_sparse"]
            gr_mask = gr_observation_mask(gr)
            row["gr_sparse_valid_count"] = int(gr_mask.sum())
            row["gr_sparse_valid_ratio"] = safe_ratio(int(gr_mask.sum()), gr.size)
            row["gr_profiles_with_observation"] = int(
                profile_has_observation(gr_mask).sum()
            )
            # This is diagnostic only: weak values remain observations in masks.
            row["gr_below_9dbz_count"] = int((gr_mask & (gr < 9.0)).sum())

        if "dbz_gr_interp" in arrays:
            interp = arrays["dbz_gr_interp"]
            interp_mask = gr_observation_mask(interp)
            row["gr_interp_valid_count"] = int(interp_mask.sum())
            row["gr_interp_valid_ratio"] = safe_ratio(
                int(interp_mask.sum()), interp.size
            )

        if "dbz_dpr" in arrays:
            dpr = arrays["dbz_dpr"]
            dpr_mask = dpr_reflectivity_mask(dpr)
            row["dpr_dbz_valid_count"] = int(dpr_mask.sum())
            row["dpr_dbz_valid_ratio"] = safe_ratio(int(dpr_mask.sum()), dpr.size)

        if "pre_dpr" in arrays:
            precipitation = arrays["pre_dpr"]
            pre_valid = precipitation_label_mask(
                precipitation, exclude_clutter=False
            )
            pre_zero = zero_rain_mask(precipitation, valid_mask=pre_valid)
            pre_positive = positive_rain_mask(
                precipitation, valid_mask=pre_valid
            )
            valid_count = int(pre_valid.sum())
            zero_count = int(pre_zero.sum())
            row.update(
                {
                    "pre_valid_count": valid_count,
                    "pre_missing_count": precipitation.size - valid_count,
                    "pre_zero_count": zero_count,
                    "pre_positive_count": int(pre_positive.sum()),
                    "pre_zero_ratio": safe_ratio(zero_count, valid_count),
                }
            )
            positive_values = precipitation[pre_positive]
            if positive_values.size:
                quantiles = np.percentile(positive_values, [50, 90, 95, 99])
                for key, value in zip(
                    (
                        "pre_p50_positive",
                        "pre_p90_positive",
                        "pre_p95_positive",
                        "pre_p99_positive",
                    ),
                    quantiles,
                ):
                    row[key] = float(value)
                row["pre_max"] = float(positive_values.max())
            for threshold in RAIN_THRESHOLDS:
                key = f"pre_gt_{int(threshold)}_count"
                row[key] = int(
                    positive_rain_mask(
                        precipitation,
                        threshold=threshold,
                        valid_mask=pre_valid,
                    ).sum()
                )

        if gr_mask is not None and dpr_mask is not None:
            if gr_mask.shape == dpr_mask.shape:
                row["gr_dpr_overlap_count"] = int((gr_mask & dpr_mask).sum())
            else:
                warnings.append("cannot compare GR and DPR masks with different shapes")

        if gr_mask is not None and pre_positive is not None:
            if gr_mask.shape == pre_positive.shape:
                row["gr_valid_pre_positive_count"] = int(
                    (gr_mask & pre_positive).sum()
                )
                row["gr_missing_pre_positive_count"] = int(
                    (~gr_mask & pre_positive).sum()
                )
            else:
                warnings.append("cannot compare GR and precipitation masks")

        if gr_mask is not None and interp_mask is not None:
            if gr_mask.shape == interp_mask.shape:
                row["gr_missing_interp_valid_count"] = int(
                    (~gr_mask & interp_mask).sum()
                )
            else:
                warnings.append("cannot compare sparse and interpolated GR masks")

        if dpr_mask is not None and pre_positive is not None and pre_zero is not None:
            if dpr_mask.shape == pre_positive.shape:
                equal = bool(np.array_equal(dpr_mask, pre_positive))
                row["dpr_mask_equals_positive_rain"] = equal
                row["dpr_valid_pre_zero_count"] = int((dpr_mask & pre_zero).sum())
                row["dpr_missing_pre_positive_count"] = int(
                    (~dpr_mask & pre_positive).sum()
                )
                if not equal:
                    warnings.append("DPR reflectivity mask != positive-rain mask")

        if "cfb" in arrays and "z" in arrays:
            cfb = arrays["cfb"]
            z = arrays["z"]
            cfb_valid = valid_cfb_mask(cfb, z.size)
            clutter = clutter_mask_from_cfb(cfb, z)
            row["cfb_valid_profile_count"] = int(cfb_valid.sum())
            row["cfb_invalid_profile_count"] = cfb.size - int(cfb_valid.sum())
            row["cfb_clutter_cell_count"] = int(clutter.sum())
            if "pre_dpr" in arrays and clutter.shape == arrays["pre_dpr"].shape:
                qc_valid = precipitation_label_mask(
                    arrays["pre_dpr"], cfb=cfb, z=z
                )
                row["pre_qc_valid_count"] = int(qc_valid.sum())
                row["pre_qc_zero_count"] = int(
                    zero_rain_mask(arrays["pre_dpr"], valid_mask=qc_valid).sum()
                )
                row["pre_qc_positive_count"] = int(
                    positive_rain_mask(
                        arrays["pre_dpr"], valid_mask=qc_valid
                    ).sum()
                )
            elif "pre_dpr" in arrays:
                warnings.append("cannot apply cfb mask to pre_dpr with different shape")

        if "typePrecip" in arrays:
            precipitation_type = arrays["typePrecip"]
            row["no_precip_profile_count"] = int(
                (precipitation_type == -1111).sum()
            )
            row["stratiform_profile_count"] = int((precipitation_type == 1).sum())
            row["convective_profile_count"] = int((precipitation_type == 2).sum())
            row["other_precip_profile_count"] = int((precipitation_type == 3).sum())

    row["status"] = "warning" if warnings else "ok"
    row["warning_count"] = len(warnings)
    row["warning_messages"] = " | ".join(warnings)
    return row


def analyze_file_safely(path: Path) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
    try:
        return analyze_file(path), None
    except Exception as exc:  # keep the full dataset audit moving by default
        return None, {
            "file_name": path.name,
            "file_path": str(path.resolve()),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }


def atomic_csv_write(
    path: Path, fieldnames: Iterable[str], rows: Iterable[dict[str, Any]]
) -> None:
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    partial.replace(path)


def atomic_json_write(path: Path, value: dict[str, Any]) -> None:
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    partial.replace(path)


def integer_sum(rows: list[dict[str, Any]], field: str) -> int:
    return sum(int(row[field]) for row in rows if row.get(field) is not None)


def build_summary(
    input_dir: Path,
    selected_count: int,
    rows: list[dict[str, Any]],
    failures: list[dict[str, str]],
) -> dict[str, Any]:
    totals = {field: integer_sum(rows, field) for field in COUNT_FIELDS}
    total_voxels = totals["total_voxel_count"]
    pre_valid = totals["pre_valid_count"]

    dimension_counts = Counter(
        f"{row.get('scan_size')}x{row.get('ray_size')}x{row.get('z_size')}"
        for row in rows
    )
    variable_presence = {
        name: sum(bool(row.get(f"has_{name}")) for row in rows)
        for name in OPTIONAL_AUXILIARIES
    }
    key_variable_presence = {
        name: sum(
            name in str(row.get("variables") or "").split(",") for row in rows
        )
        for name in KEY_VARIABLES
    }
    warning_files = [row["file_name"] for row in rows if row["status"] == "warning"]
    mask_mismatch_files = [
        row["file_name"]
        for row in rows
        if row.get("dpr_mask_equals_positive_rain") is False
    ]

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(input_dir.resolve()),
        "selected_file_count": selected_count,
        "successful_file_count": len(rows),
        "failed_file_count": len(failures),
        "warning_file_count": len(warning_files),
        "warning_files": warning_files,
        "dpr_mask_positive_rain_mismatch_count": len(mask_mismatch_files),
        "dpr_mask_positive_rain_mismatch_files": mask_mismatch_files,
        "dimension_shape_counts": dict(sorted(dimension_counts.items())),
        "key_variable_presence": key_variable_presence,
        "auxiliary_variable_presence": variable_presence,
        "totals": totals,
        "ratios": {
            "gr_sparse_valid_ratio": safe_ratio(
                totals["gr_sparse_valid_count"], total_voxels
            ),
            "gr_interp_valid_ratio": safe_ratio(
                totals["gr_interp_valid_count"], total_voxels
            ),
            "dpr_dbz_valid_ratio": safe_ratio(
                totals["dpr_dbz_valid_count"], total_voxels
            ),
            "pre_valid_ratio": safe_ratio(pre_valid, total_voxels),
            "pre_zero_ratio_among_valid": safe_ratio(
                totals["pre_zero_count"], pre_valid
            ),
            "pre_positive_ratio_among_valid": safe_ratio(
                totals["pre_positive_count"], pre_valid
            ),
            "gr_dpr_overlap_ratio_of_dpr": safe_ratio(
                totals["gr_dpr_overlap_count"], totals["dpr_dbz_valid_count"]
            ),
        },
    }


def selected_files(input_dir: Path, count: int | None) -> list[Path]:
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")
    files = sorted(input_dir.glob("*.nc"))
    if not files:
        raise FileNotFoundError(f"No .nc files found in: {input_dir}")
    if count is not None:
        if count <= 0:
            raise ValueError("--count must be positive")
        if count > len(files):
            raise ValueError(
                f"Requested {count} files, but only found {len(files)} NetCDF files"
            )
        files = files[:count]
    return files


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = (
        output_dir / "dataset_manifest.csv",
        output_dir / "dataset_summary.json",
        output_dir / "failed_files.csv",
    )
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"Output already exists ({names}); use --overwrite to replace it"
        )


def scan_files(
    files: list[Path], workers: int, fail_fast: bool
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    if workers <= 0:
        raise ValueError("--workers must be positive")
    if workers == 1:
        results = map(analyze_file_safely, files)
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        results = executor.map(analyze_file_safely, files)

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    try:
        for index, (row, failure) in enumerate(results, start=1):
            if failure is not None:
                failures.append(failure)
                print(
                    f"ERROR [{index}/{len(files)}] {failure['file_name']}: "
                    f"{failure['error_type']}: {failure['error_message']}",
                    flush=True,
                )
                if fail_fast:
                    raise RuntimeError(
                        f"Failed to inspect {failure['file_name']}: "
                        f"{failure['error_message']}"
                    )
            else:
                assert row is not None
                rows.append(row)
                print(
                    f"OK [{index}/{len(files)}] {row['file_name']} "
                    f"status={row['status']}",
                    flush=True,
                )
    finally:
        if workers > 1:
            executor.shutdown(wait=True, cancel_futures=fail_fast)
    return rows, failures


def main() -> None:
    args = parse_args()
    files = selected_files(args.input_dir, args.count)
    prepare_output_dir(args.output_dir, args.overwrite)
    print(f"Dataset: {args.input_dir.resolve()}")
    print(f"Selected files: {len(files)}")
    print(f"Workers: {args.workers}")

    rows, failures = scan_files(files, args.workers, args.fail_fast)
    # executor.map preserves input order, but sorting here also protects future
    # implementations from producing non-deterministic manifests.
    rows.sort(key=lambda row: str(row["file_name"]))
    failures.sort(key=lambda row: row["file_name"])
    summary = build_summary(args.input_dir, len(files), rows, failures)

    manifest_path = args.output_dir / "dataset_manifest.csv"
    summary_path = args.output_dir / "dataset_summary.json"
    failures_path = args.output_dir / "failed_files.csv"
    atomic_csv_write(manifest_path, MANIFEST_FIELDS, rows)
    atomic_csv_write(
        failures_path,
        ("file_name", "file_path", "error_type", "error_message"),
        failures,
    )
    atomic_json_write(summary_path, summary)

    print("\nDataset audit complete")
    print(f"  successful: {len(rows)}")
    print(f"  failed: {len(failures)}")
    print(f"  warnings: {summary['warning_file_count']}")
    print(f"  manifest: {manifest_path}")
    print(f"  summary: {summary_path}")
    print(f"  failures: {failures_path}")
    if failures:
        raise RuntimeError(
            f"{len(failures)} file(s) failed; inspect {failures_path} for details"
        )


if __name__ == "__main__":
    main()
