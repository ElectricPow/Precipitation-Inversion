#!/usr/bin/env python3
"""Audit stage-two GR/DPR storage states, recoverability, and alignment.

The source NetCDF files are opened read-only.  The audit deliberately reads the
raw masked arrays instead of calling ``to_float_array`` first, because stage two
must retain the distinction between native mask/NaN, finite ``-9999.9``
sentinels, and physical dBZ values.

Outputs
-------
``summary.json``
    Configuration, split-level totals, ratios, and overlap reflectivity metrics.
``per_file.csv``
    File-level three-state, four-zone, interpolation-reachability, and mismatch
    counts.
``per_height.csv``
    The same core coverage diagnostics at all 60 physical heights.
``distance_to_gr.csv``
    DPR targets grouped by truncated horizontal Chebyshev distance to the
    nearest direct GR value.
``local_density.csv``
    DPR targets grouped by direct-GR counts in a configurable horizontal window.
``shift_metrics.csv`` and ``best_shift_by_height.csv``
    Exact small-shift binary-support and overlapping-dBZ alignment diagnostics.
``*.png``
    Compact plots of the global height, region, distance, and shift summaries.

The neighborhood and interpolation groups are diagnostic proxies.  They do not
claim to recover the true radar beam geometry absent from the current NC files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from netCDF4 import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from precipitation_inversion.data.masks import (  # noqa: E402
    clutter_mask_from_cfb,
    to_float_array,
)
from precipitation_inversion.data.stage2_masks import (  # noqa: E402
    build_stage2_spatial_masks,
    classify_reflectivity_storage,
    physical_reflectivity_values,
)


DEFAULT_SPLIT_MANIFEST = PROJECT_ROOT / "metadata" / "splits" / "split_manifest.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stage2_alignment_audit"
DEFAULT_DISTANCE_RADII = (0, 1, 2, 3, 5, 8)
DEFAULT_DENSITY_RADIUS = 2
VALID_SPLITS = ("train", "val", "test")

PAIR_FIELDS = (
    "pair_count",
    "sum_error",
    "sum_abs_error",
    "sum_squared_error",
    "sum_gr",
    "sum_dpr",
    "sum_gr_squared",
    "sum_dpr_squared",
    "sum_product",
)

HEIGHT_COUNT_FIELDS = (
    "total_count",
    "occupancy_domain_count",
    "gr_native_missing_count",
    "gr_sentinel_count",
    "gr_value_count",
    "interp_native_missing_count",
    "interp_sentinel_count",
    "interp_value_count",
    "dpr_native_missing_count",
    "dpr_sentinel_count",
    "dpr_value_count",
    "q11_overlap_count",
    "q01_dpr_only_count",
    "q10_gr_only_count",
    "q00_neither_count",
    "dpr_only_gap_proxy_count",
    "dpr_only_outside_proxy_count",
    "dpr_below_cfb_count",
    "dpr_pre_positive_mismatch_count",
)

DENSITY_LABELS = ("0", "1", "2-4", "5-9", ">=10")


@dataclass(frozen=True)
class AuditOptions:
    """Numerical options shared by every independently audited file."""

    max_shift: int = 2
    distance_radii: tuple[int, ...] = DEFAULT_DISTANCE_RADII
    density_radius: int = DEFAULT_DENSITY_RADIUS

    def __post_init__(self) -> None:
        if self.max_shift < 0:
            raise ValueError("max_shift must be non-negative")
        radii = tuple(int(value) for value in self.distance_radii)
        if not radii or radii[0] != 0:
            raise ValueError("distance_radii must start at 0")
        if any(value < 0 for value in radii) or any(
            right <= left for left, right in zip(radii, radii[1:])
        ):
            raise ValueError("distance_radii must be non-negative and increasing")
        if self.density_radius < 0:
            raise ValueError("density_radius must be non-negative")
        object.__setattr__(self, "distance_radii", radii)


@dataclass(frozen=True)
class AuditTask:
    """Pickle-safe description of one file audit."""

    path: Path
    sample_id: str
    split: str
    options: AuditOptions


def parse_int_tuple(value: str) -> tuple[int, ...]:
    """Parse a comma-separated integer tuple for CLI radius arguments."""

    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    try:
        AuditOptions(distance_radii=result)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit stage-two GR-to-DPR storage states and alignment."
    )
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=VALID_SPLITS,
        default=list(VALID_SPLITS),
        help="Splits selected from split_manifest.csv (default: all).",
    )
    parser.add_argument(
        "--count",
        type=int,
        help="Inspect only the first N selected rows; intended for smoke tests.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Independent file workers (default: 1; shared storage may prefer 1-4).",
    )
    parser.add_argument(
        "--max-shift",
        type=int,
        default=2,
        help="Audit all horizontal shifts in [-N,N]^2 (default: 2).",
    )
    parser.add_argument(
        "--distance-radii",
        type=parse_int_tuple,
        default=DEFAULT_DISTANCE_RADII,
        help="Increasing Chebyshev radii beginning at 0 (default: 0,1,2,3,5,8).",
    )
    parser.add_argument(
        "--density-radius",
        type=int,
        default=DEFAULT_DENSITY_RADIUS,
        help="Horizontal GR-density window radius (default: 2 gives 5x5).",
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    """Return a JSON-friendly ratio or ``None`` for an empty denominator."""

    return float(numerator / denominator) if denominator else None


def load_tasks(
    manifest_path: Path,
    *,
    splits: Sequence[str],
    count: int | None,
    options: AuditOptions,
) -> list[AuditTask]:
    """Load selected, deterministic file tasks from the canonical split manifest."""

    if not manifest_path.is_file():
        raise FileNotFoundError(f"split manifest not found: {manifest_path}")
    selected_splits = set(splits)
    if not selected_splits or not selected_splits.issubset(VALID_SPLITS):
        raise ValueError(f"splits must be selected from {VALID_SPLITS}")
    if count is not None and count <= 0:
        raise ValueError("count must be positive")

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "file_path", "split"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"split manifest must contain columns {sorted(required)}: {manifest_path}"
            )
        records = [row for row in reader if row["split"] in selected_splits]

    records.sort(key=lambda row: (row["split"], row["sample_id"]))
    if count is not None:
        records = records[:count]
    if not records:
        raise ValueError("no files selected from split manifest")

    tasks: list[AuditTask] = []
    for record in records:
        path = Path(record["file_path"]).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"source NetCDF file not found: {path}")
        tasks.append(
            AuditTask(
                path=path,
                sample_id=record["sample_id"],
                split=record["split"],
                options=options,
            )
        )
    return tasks


def shift_offsets(max_shift: int) -> tuple[tuple[int, int], ...]:
    """Return deterministic ``(scan_shift, ray_shift)`` offsets."""

    if max_shift < 0:
        raise ValueError("max_shift must be non-negative")
    return tuple(
        (scan_shift, ray_shift)
        for scan_shift in range(-max_shift, max_shift + 1)
        for ray_shift in range(-max_shift, max_shift + 1)
    )


def _axis_alignment_slices(length: int, shift: int) -> tuple[slice, slice]:
    """Align source at index ``i`` with target at ``i + shift``."""

    if abs(shift) >= length:
        raise ValueError(f"shift {shift} leaves no overlap for axis length {length}")
    if shift >= 0:
        return slice(0, length - shift), slice(shift, length)
    return slice(-shift, length), slice(0, length + shift)


def _paired_statistics_by_height(
    gr_values: np.ndarray,
    dpr_values: np.ndarray,
    pair_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return sufficient paired-dBZ statistics along horizontal axes."""

    if gr_values.shape != dpr_values.shape or pair_mask.shape != gr_values.shape:
        raise ValueError("paired statistics inputs must have identical shapes")
    if gr_values.ndim != 3:
        raise ValueError("paired statistics expect (nscan,nray,z) arrays")
    mask = np.asarray(pair_mask, dtype=bool)
    gr = np.where(mask, gr_values, 0.0).astype(np.float64, copy=False)
    dpr = np.where(mask, dpr_values, 0.0).astype(np.float64, copy=False)
    error = gr - dpr
    axes = (0, 1)
    return {
        "pair_count": mask.sum(axis=axes, dtype=np.int64),
        "sum_error": error.sum(axis=axes, dtype=np.float64),
        "sum_abs_error": np.abs(error).sum(axis=axes, dtype=np.float64),
        "sum_squared_error": np.square(error).sum(axis=axes, dtype=np.float64),
        "sum_gr": gr.sum(axis=axes, dtype=np.float64),
        "sum_dpr": dpr.sum(axis=axes, dtype=np.float64),
        "sum_gr_squared": np.square(gr).sum(axis=axes, dtype=np.float64),
        "sum_dpr_squared": np.square(dpr).sum(axis=axes, dtype=np.float64),
        "sum_product": (gr * dpr).sum(axis=axes, dtype=np.float64),
    }


def paired_metrics(statistics: Mapping[str, Any]) -> dict[str, float | int | None]:
    """Derive MAE/RMSE/Bias/Pearson from additive sufficient statistics."""

    count = int(np.asarray(statistics["pair_count"]).sum())
    if count == 0:
        return {
            "count": 0,
            "mae_dbz": None,
            "rmse_dbz": None,
            "bias_gr_minus_dpr_dbz": None,
            "pearson_r": None,
        }
    sum_error = float(np.asarray(statistics["sum_error"]).sum())
    sum_abs = float(np.asarray(statistics["sum_abs_error"]).sum())
    sum_squared = float(np.asarray(statistics["sum_squared_error"]).sum())
    sum_gr = float(np.asarray(statistics["sum_gr"]).sum())
    sum_dpr = float(np.asarray(statistics["sum_dpr"]).sum())
    sum_gr_squared = float(np.asarray(statistics["sum_gr_squared"]).sum())
    sum_dpr_squared = float(np.asarray(statistics["sum_dpr_squared"]).sum())
    sum_product = float(np.asarray(statistics["sum_product"]).sum())

    covariance_numerator = sum_product - sum_gr * sum_dpr / count
    gr_variance_numerator = sum_gr_squared - sum_gr * sum_gr / count
    dpr_variance_numerator = sum_dpr_squared - sum_dpr * sum_dpr / count
    denominator = math.sqrt(
        max(gr_variance_numerator, 0.0) * max(dpr_variance_numerator, 0.0)
    )
    pearson = covariance_numerator / denominator if denominator > 0.0 else None
    return {
        "count": count,
        "mae_dbz": sum_abs / count,
        "rmse_dbz": math.sqrt(max(sum_squared / count, 0.0)),
        "bias_gr_minus_dpr_dbz": sum_error / count,
        "pearson_r": pearson,
    }


def _dilate_horizontal_once(mask: np.ndarray) -> np.ndarray:
    """One-cell 8-neighbor dilation without mixing the height dimension."""

    if mask.ndim != 3:
        raise ValueError("horizontal dilation expects (nscan,nray,z)")
    nscan, nray, z_size = mask.shape
    padded = np.pad(mask, ((1, 1), (1, 1), (0, 0)), constant_values=False)
    result = np.zeros((nscan, nray, z_size), dtype=bool)
    for scan_offset in range(3):
        for ray_offset in range(3):
            result |= padded[
                scan_offset : scan_offset + nscan,
                ray_offset : ray_offset + nray,
                :,
            ]
    return result


def horizontal_window_count(mask: np.ndarray, radius: int) -> np.ndarray:
    """Count direct observations in a square horizontal window at each height."""

    if mask.ndim != 3:
        raise ValueError("window counts expect (nscan,nray,z)")
    if radius < 0:
        raise ValueError("radius must be non-negative")
    nscan, nray, z_size = mask.shape
    padded = np.pad(
        mask,
        ((radius, radius), (radius, radius), (0, 0)),
        constant_values=False,
    )
    result = np.zeros((nscan, nray, z_size), dtype=np.uint16)
    width = 2 * radius + 1
    for scan_offset in range(width):
        for ray_offset in range(width):
            result += padded[
                scan_offset : scan_offset + nscan,
                ray_offset : ray_offset + nray,
                :,
            ]
    return result


def neighborhood_diagnostics(
    gr_mask: np.ndarray,
    dpr_mask: np.ndarray,
    dpr_values: np.ndarray,
    radii: Sequence[int],
) -> dict[str, np.ndarray]:
    """Compute cumulative relaxed matches and disjoint nearest-distance bins."""

    if gr_mask.shape != dpr_mask.shape or gr_mask.shape != dpr_values.shape:
        raise ValueError("neighborhood inputs must have identical shapes")
    radii_tuple = tuple(int(value) for value in radii)
    AuditOptions(distance_radii=radii_tuple)
    max_radius = radii_tuple[-1]
    radius_to_index = {radius: index for index, radius in enumerate(radii_tuple)}
    dpr_cumulative = np.zeros(len(radii_tuple), dtype=np.int64)
    gr_cumulative = np.zeros(len(radii_tuple), dtype=np.int64)

    gr_reach = np.asarray(gr_mask, dtype=bool).copy()
    dpr_reach = np.asarray(dpr_mask, dtype=bool).copy()
    for radius in range(max_radius + 1):
        if radius in radius_to_index:
            index = radius_to_index[radius]
            dpr_cumulative[index] = int((dpr_mask & gr_reach).sum())
            gr_cumulative[index] = int((gr_mask & dpr_reach).sum())
        if radius < max_radius:
            gr_reach = _dilate_horizontal_once(gr_reach)
            dpr_reach = _dilate_horizontal_once(dpr_reach)

    # Convert cumulative target hits to mutually exclusive distance bins.  The
    # last bin contains targets farther than the largest audited radius.
    bin_count = np.empty(len(radii_tuple) + 1, dtype=np.int64)
    bin_count[0] = dpr_cumulative[0]
    bin_count[1:-1] = np.diff(dpr_cumulative)
    bin_count[-1] = int(dpr_mask.sum()) - dpr_cumulative[-1]

    # Recover each disjoint ring once more to retain the mean target dBZ.  This
    # array is diagnostic only and is never used to train a model.
    bin_dbz_sum = np.zeros(len(radii_tuple) + 1, dtype=np.float64)
    previous_reach = np.zeros(gr_mask.shape, dtype=bool)
    gr_reach = np.asarray(gr_mask, dtype=bool).copy()
    radius_index = 0
    for radius in range(max_radius + 1):
        if radius == radii_tuple[radius_index]:
            ring = dpr_mask & gr_reach & ~previous_reach
            bin_dbz_sum[radius_index] = float(dpr_values[ring].sum(dtype=np.float64))
            previous_reach = gr_reach.copy()
            radius_index += 1
            if radius_index == len(radii_tuple):
                break
        gr_reach = _dilate_horizontal_once(gr_reach)
    outside = dpr_mask & ~previous_reach
    bin_dbz_sum[-1] = float(dpr_values[outside].sum(dtype=np.float64))
    return {
        "dpr_cumulative_count": dpr_cumulative,
        "gr_cumulative_count": gr_cumulative,
        "dpr_distance_bin_count": bin_count,
        "dpr_distance_bin_dbz_sum": bin_dbz_sum,
    }


def local_density_diagnostics(
    gr_mask: np.ndarray,
    dpr_mask: np.ndarray,
    dpr_values: np.ndarray,
    radius: int,
) -> dict[str, np.ndarray]:
    """Group DPR targets by direct-GR count in a local horizontal window."""

    density = horizontal_window_count(gr_mask, radius)
    selectors = (
        density == 0,
        density == 1,
        (density >= 2) & (density <= 4),
        (density >= 5) & (density <= 9),
        density >= 10,
    )
    counts = np.zeros(len(selectors), dtype=np.int64)
    dbz_sums = np.zeros(len(selectors), dtype=np.float64)
    for index, selector in enumerate(selectors):
        selected = dpr_mask & selector
        counts[index] = int(selected.sum())
        dbz_sums[index] = float(dpr_values[selected].sum(dtype=np.float64))
    return {"dpr_count": counts, "dpr_dbz_sum": dbz_sums}


def shifted_alignment_statistics(
    gr_mask: np.ndarray,
    dpr_mask: np.ndarray,
    gr_values: np.ndarray,
    dpr_values: np.ndarray,
    *,
    max_shift: int,
) -> tuple[tuple[tuple[int, int], ...], dict[str, np.ndarray]]:
    """Compute per-height binary and paired-dBZ statistics for small shifts.

    A positive ``scan_shift`` compares source GR at ``(i,j,z)`` with DPR at
    ``(i+scan_shift,j+ray_shift,z)``.  No circular wrapping is used.
    """

    shape = gr_mask.shape
    if (
        dpr_mask.shape != shape
        or gr_values.shape != shape
        or dpr_values.shape != shape
        or len(shape) != 3
    ):
        raise ValueError("shifted alignment inputs must share a 3-D shape")
    offsets = shift_offsets(max_shift)
    z_size = shape[-1]
    statistics: dict[str, np.ndarray] = {
        "gr_count": np.zeros((len(offsets), z_size), dtype=np.int64),
        "dpr_count": np.zeros((len(offsets), z_size), dtype=np.int64),
        "union_count": np.zeros((len(offsets), z_size), dtype=np.int64),
        **{
            field: np.zeros(
                (len(offsets), z_size),
                dtype=np.int64 if field == "pair_count" else np.float64,
            )
            for field in PAIR_FIELDS
        },
    }

    for offset_index, (scan_shift, ray_shift) in enumerate(offsets):
        gr_scan, dpr_scan = _axis_alignment_slices(shape[0], scan_shift)
        gr_ray, dpr_ray = _axis_alignment_slices(shape[1], ray_shift)
        gr_index = (gr_scan, gr_ray, slice(None))
        dpr_index = (dpr_scan, dpr_ray, slice(None))
        gr_aligned = gr_mask[gr_index]
        dpr_aligned = dpr_mask[dpr_index]
        pair = gr_aligned & dpr_aligned
        statistics["gr_count"][offset_index] = gr_aligned.sum(
            axis=(0, 1), dtype=np.int64
        )
        statistics["dpr_count"][offset_index] = dpr_aligned.sum(
            axis=(0, 1), dtype=np.int64
        )
        statistics["union_count"][offset_index] = (
            gr_aligned | dpr_aligned
        ).sum(axis=(0, 1), dtype=np.int64)
        pair_statistics = _paired_statistics_by_height(
            gr_values[gr_index], dpr_values[dpr_index], pair
        )
        for field in PAIR_FIELDS:
            statistics[field][offset_index] = pair_statistics[field]
    return offsets, statistics


def _storage_counts(prefix: str, states: Any) -> dict[str, int]:
    values = states.counts()
    return {
        f"{prefix}_native_missing_count": values["native_missing"],
        f"{prefix}_sentinel_count": values["sentinel"],
        f"{prefix}_value_count": values["value"],
        f"{prefix}_native_available_count": values["native_available"],
    }


def analyze_file(task: AuditTask) -> dict[str, Any]:
    """Audit one NetCDF file and return additive NumPy diagnostics."""

    with Dataset(task.path, "r") as dataset:
        required = {
            "z",
            "dbz_gr_sparse",
            "dbz_gr_interp",
            "dbz_dpr",
            "pre_dpr",
            "cfb",
        }
        missing = sorted(required.difference(dataset.variables))
        if missing:
            raise KeyError(f"{task.path.name} missing variables: {', '.join(missing)}")
        z = to_float_array(dataset["z"])
        gr_raw = dataset["dbz_gr_sparse"][:]
        interp_raw = dataset["dbz_gr_interp"][:]
        dpr_raw = dataset["dbz_dpr"][:]
        pre_raw = dataset["pre_dpr"][:]
        cfb_raw = dataset["cfb"][:]

    expected_shape = (gr_raw.shape[0], gr_raw.shape[1], z.size)
    for name, array in (
        ("dbz_gr_sparse", gr_raw),
        ("dbz_gr_interp", interp_raw),
        ("dbz_dpr", dpr_raw),
        ("pre_dpr", pre_raw),
    ):
        if array.shape != expected_shape:
            raise ValueError(
                f"{task.path.name} {name} shape {array.shape} != {expected_shape}"
            )
    if z.ndim != 1 or not np.all(np.isfinite(z)) or not np.all(np.diff(z) > 0):
        raise ValueError(f"{task.path.name} has an invalid height coordinate")

    gr_states = classify_reflectivity_storage(gr_raw)
    interp_states = classify_reflectivity_storage(interp_raw)
    dpr_states = classify_reflectivity_storage(dpr_raw)
    masks = build_stage2_spatial_masks(
        gr_raw,
        dpr_raw,
        dbz_gr_interp=interp_raw,
        pre_dpr=pre_raw,
    )
    gr_values = physical_reflectivity_values(gr_raw, masks=gr_states)
    dpr_values = physical_reflectivity_values(dpr_raw, masks=dpr_states)
    occupancy_domain = masks["occupancy_domain"]
    pre_positive = masks["pre_positive"]
    dpr_value = masks["dpr_value"]
    mismatch = dpr_value ^ pre_positive

    # The training four-zone contract is defined only inside the trustworthy
    # occurrence-label domain.  Q11/Q01 are unchanged in this dataset because
    # every retained DPR dBZ point has a positive precipitation label.
    q11 = occupancy_domain & masks["q11_overlap"]
    q01 = occupancy_domain & masks["q01_dpr_only"]
    q10 = occupancy_domain & masks["q10_gr_only"]
    q00 = occupancy_domain & masks["q00_neither"]
    dpr_gap = occupancy_domain & masks["dpr_only_gap_proxy"]
    dpr_outside = occupancy_domain & masks["dpr_only_outside_proxy"]

    cfb_clutter = clutter_mask_from_cfb(cfb_raw, z)
    dpr_below_cfb = dpr_value & cfb_clutter

    q11_pair = _paired_statistics_by_height(gr_values, dpr_values, q11)
    height: dict[str, np.ndarray] = {
        "total_count": np.full(z.shape, expected_shape[0] * expected_shape[1], dtype=np.int64),
        "occupancy_domain_count": occupancy_domain.sum(axis=(0, 1), dtype=np.int64),
        "gr_native_missing_count": gr_states.native_missing.sum(axis=(0, 1), dtype=np.int64),
        "gr_sentinel_count": gr_states.sentinel.sum(axis=(0, 1), dtype=np.int64),
        "gr_value_count": gr_states.value.sum(axis=(0, 1), dtype=np.int64),
        "interp_native_missing_count": interp_states.native_missing.sum(axis=(0, 1), dtype=np.int64),
        "interp_sentinel_count": interp_states.sentinel.sum(axis=(0, 1), dtype=np.int64),
        "interp_value_count": interp_states.value.sum(axis=(0, 1), dtype=np.int64),
        "dpr_native_missing_count": dpr_states.native_missing.sum(axis=(0, 1), dtype=np.int64),
        "dpr_sentinel_count": dpr_states.sentinel.sum(axis=(0, 1), dtype=np.int64),
        "dpr_value_count": dpr_value.sum(axis=(0, 1), dtype=np.int64),
        "q11_overlap_count": q11.sum(axis=(0, 1), dtype=np.int64),
        "q01_dpr_only_count": q01.sum(axis=(0, 1), dtype=np.int64),
        "q10_gr_only_count": q10.sum(axis=(0, 1), dtype=np.int64),
        "q00_neither_count": q00.sum(axis=(0, 1), dtype=np.int64),
        "dpr_only_gap_proxy_count": dpr_gap.sum(axis=(0, 1), dtype=np.int64),
        "dpr_only_outside_proxy_count": dpr_outside.sum(axis=(0, 1), dtype=np.int64),
        "dpr_below_cfb_count": dpr_below_cfb.sum(axis=(0, 1), dtype=np.int64),
        "dpr_pre_positive_mismatch_count": mismatch.sum(axis=(0, 1), dtype=np.int64),
        **{f"q11_{field}": values for field, values in q11_pair.items()},
    }

    distance = neighborhood_diagnostics(
        gr_states.value,
        dpr_value,
        dpr_values,
        task.options.distance_radii,
    )
    density = local_density_diagnostics(
        gr_states.value,
        dpr_value,
        dpr_values,
        task.options.density_radius,
    )
    offsets, shift = shifted_alignment_statistics(
        gr_states.value,
        dpr_value,
        gr_values,
        dpr_values,
        max_shift=task.options.max_shift,
    )

    row: dict[str, Any] = {
        "sample_id": task.sample_id,
        "file_name": task.path.name,
        "file_path": str(task.path),
        "split": task.split,
        "nscan": expected_shape[0],
        "nray": expected_shape[1],
        "z_size": expected_shape[2],
        "total_count": int(np.prod(expected_shape)),
        **_storage_counts("gr", gr_states),
        **_storage_counts("interp", interp_states),
        **_storage_counts("dpr", dpr_states),
        "occupancy_domain_count": int(occupancy_domain.sum()),
        "pre_positive_count": int(pre_positive.sum()),
        "dpr_pre_positive_mismatch_count": int(mismatch.sum()),
        "q11_overlap_count": int(q11.sum()),
        "q01_dpr_only_count": int(q01.sum()),
        "q10_gr_only_count": int(q10.sum()),
        "q00_neither_count": int(q00.sum()),
        "dpr_only_gap_proxy_count": int(dpr_gap.sum()),
        "dpr_only_outside_proxy_count": int(dpr_outside.sum()),
        "dpr_below_cfb_count": int(dpr_below_cfb.sum()),
        "interp_missing_direct_gr_count": int((gr_states.value & ~interp_states.value).sum()),
    }
    row.update({f"q11_{key}": value for key, value in paired_metrics(q11_pair).items()})
    return {
        "row": row,
        "z": z.astype(np.float64, copy=False),
        "height": height,
        "distance": distance,
        "density": density,
        "offsets": offsets,
        "shift": shift,
    }


def run_tasks(tasks: Sequence[AuditTask], *, workers: int) -> list[dict[str, Any]]:
    """Run deterministic file audits sequentially or in independent processes."""

    if workers <= 0:
        raise ValueError("workers must be positive")
    if workers == 1:
        return [analyze_file(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(analyze_file, tasks))


def _sum_mapping(
    results: Sequence[dict[str, Any]], key: str
) -> dict[str, np.ndarray]:
    fields = tuple(results[0][key])
    if any(tuple(result[key]) != fields for result in results):
        raise ValueError(f"inconsistent {key} diagnostic fields across files")
    return {
        field: sum(
            (np.asarray(result[key][field]) for result in results),
            np.zeros_like(np.asarray(results[0][key][field])),
        )
        for field in fields
    }


def aggregate_results(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate additive diagnostics for all files and each present split."""

    if not results:
        raise ValueError("results must not be empty")
    reference_z = np.asarray(results[0]["z"], dtype=np.float64)
    reference_offsets = tuple(results[0]["offsets"])
    for result in results[1:]:
        if not np.allclose(result["z"], reference_z, rtol=0.0, atol=1e-6):
            raise ValueError("height coordinates differ across audited files")
        if tuple(result["offsets"]) != reference_offsets:
            raise ValueError("shift offsets differ across audited files")

    grouped: dict[str, Any] = {}
    group_names = ["all"] + [
        split for split in VALID_SPLITS if any(r["row"]["split"] == split for r in results)
    ]
    for group in group_names:
        selected = (
            list(results)
            if group == "all"
            else [result for result in results if result["row"]["split"] == group]
        )
        grouped[group] = {
            "file_count": len(selected),
            "height": _sum_mapping(selected, "height"),
            "distance": _sum_mapping(selected, "distance"),
            "density": _sum_mapping(selected, "density"),
            "shift": _sum_mapping(selected, "shift"),
        }
    return {"z": reference_z, "offsets": reference_offsets, "groups": grouped}


def _pair_statistics_from_prefixed(
    mapping: Mapping[str, Any], prefix: str
) -> dict[str, Any]:
    return {field: mapping[f"{prefix}_{field}"] for field in PAIR_FIELDS}


def build_summary(
    aggregate: Mapping[str, Any],
    *,
    tasks: Sequence[AuditTask],
    manifest_path: Path,
) -> dict[str, Any]:
    """Build compact JSON totals from the additive height diagnostics."""

    group_summaries: dict[str, Any] = {}
    for group_name, group in aggregate["groups"].items():
        height = group["height"]
        totals = {
            field: int(np.asarray(height[field]).sum()) for field in HEIGHT_COUNT_FIELDS
        }
        total = totals["total_count"]
        domain = totals["occupancy_domain_count"]
        dpr = totals["dpr_value_count"]
        gr = totals["gr_value_count"]
        group_summaries[group_name] = {
            "file_count": group["file_count"],
            "totals": totals,
            "ratios": {
                "gr_native_missing_of_total": safe_ratio(
                    totals["gr_native_missing_count"], total
                ),
                "gr_sentinel_of_total": safe_ratio(totals["gr_sentinel_count"], total),
                "gr_value_of_total": safe_ratio(gr, total),
                "interp_value_of_total": safe_ratio(
                    totals["interp_value_count"], total
                ),
                "dpr_value_of_total": safe_ratio(dpr, total),
                "q11_of_dpr": safe_ratio(totals["q11_overlap_count"], dpr),
                "q01_of_dpr": safe_ratio(totals["q01_dpr_only_count"], dpr),
                "q10_of_gr": safe_ratio(totals["q10_gr_only_count"], gr),
                "q00_of_occupancy_domain": safe_ratio(
                    totals["q00_neither_count"], domain
                ),
                "dpr_gap_proxy_of_dpr": safe_ratio(
                    totals["dpr_only_gap_proxy_count"], dpr
                ),
                "dpr_outside_proxy_of_dpr": safe_ratio(
                    totals["dpr_only_outside_proxy_count"], dpr
                ),
                "dpr_below_cfb_of_dpr": safe_ratio(
                    totals["dpr_below_cfb_count"], dpr
                ),
            },
            "q11_reflectivity": paired_metrics(
                _pair_statistics_from_prefixed(height, "q11")
            ),
        }
    options = tasks[0].options
    return {
        "format_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split_manifest": str(manifest_path.resolve()),
        "selected_file_count": len(tasks),
        "selected_splits": sorted({task.split for task in tasks}),
        "configuration": {
            "sentinel_cutoff": -9990.0,
            "max_shift": options.max_shift,
            "distance_radii": list(options.distance_radii),
            "density_radius": options.density_radius,
            "shift_direction": (
                "positive shift compares GR(i,j,z) with "
                "DPR(i+scan_shift,j+ray_shift,z)"
            ),
            "distance_metric": "horizontal Chebyshev distance; height is not mixed",
        },
        "groups": group_summaries,
    }


def _distance_labels(radii: Sequence[int]) -> tuple[str, ...]:
    labels: list[str] = []
    previous = -1
    for radius in radii:
        lower = previous + 1
        labels.append(str(radius) if lower == radius else f"{lower}-{radius}")
        previous = radius
    labels.append(f">{radii[-1]}")
    return tuple(labels)


def per_height_rows(aggregate: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    z = aggregate["z"]
    for group_name, group in aggregate["groups"].items():
        height = group["height"]
        for index, height_km in enumerate(z):
            row: dict[str, Any] = {
                "group": group_name,
                "height_index": index,
                "height_km": float(height_km),
            }
            for field in HEIGHT_COUNT_FIELDS:
                row[field] = int(height[field][index])
            pair = {
                field: np.asarray(height[f"q11_{field}"])[index] for field in PAIR_FIELDS
            }
            row.update({f"q11_{key}": value for key, value in paired_metrics(pair).items()})
            row["gr_value_ratio"] = safe_ratio(
                row["gr_value_count"], row["total_count"]
            )
            row["interp_value_ratio"] = safe_ratio(
                row["interp_value_count"], row["total_count"]
            )
            row["dpr_value_ratio"] = safe_ratio(
                row["dpr_value_count"], row["total_count"]
            )
            rows.append(row)
    return rows


def distance_rows(
    aggregate: Mapping[str, Any], radii: Sequence[int]
) -> list[dict[str, Any]]:
    labels = _distance_labels(radii)
    rows: list[dict[str, Any]] = []
    for group_name, group in aggregate["groups"].items():
        distance = group["distance"]
        counts = distance["dpr_distance_bin_count"]
        dbz_sums = distance["dpr_distance_bin_dbz_sum"]
        total = int(np.asarray(counts).sum())
        for index, label in enumerate(labels):
            count = int(counts[index])
            rows.append(
                {
                    "group": group_name,
                    "distance_bin": label,
                    "count": count,
                    "fraction_of_dpr": safe_ratio(count, total),
                    "mean_dpr_dbz": safe_ratio(float(dbz_sums[index]), count),
                }
            )
        for index, radius in enumerate(radii):
            rows.append(
                {
                    "group": group_name,
                    "distance_bin": f"cumulative<={radius}",
                    "count": int(distance["dpr_cumulative_count"][index]),
                    "fraction_of_dpr": safe_ratio(
                        int(distance["dpr_cumulative_count"][index]), total
                    ),
                    "mean_dpr_dbz": None,
                }
            )
    return rows


def density_rows(aggregate: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group_name, group in aggregate["groups"].items():
        counts = group["density"]["dpr_count"]
        dbz_sums = group["density"]["dpr_dbz_sum"]
        total = int(np.asarray(counts).sum())
        for index, label in enumerate(DENSITY_LABELS):
            count = int(counts[index])
            rows.append(
                {
                    "group": group_name,
                    "local_gr_count_bin": label,
                    "dpr_count": count,
                    "fraction_of_dpr": safe_ratio(count, total),
                    "mean_dpr_dbz": safe_ratio(float(dbz_sums[index]), count),
                }
            )
    return rows


def shift_rows(aggregate: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offsets = aggregate["offsets"]
    z = aggregate["z"]
    for group_name, group in aggregate["groups"].items():
        shift = group["shift"]
        for offset_index, (scan_shift, ray_shift) in enumerate(offsets):
            for height_index, height_km in enumerate(z):
                union = int(shift["union_count"][offset_index, height_index])
                pair = {
                    field: shift[field][offset_index, height_index]
                    for field in PAIR_FIELDS
                }
                metrics = paired_metrics(pair)
                rows.append(
                    {
                        "group": group_name,
                        "height_index": height_index,
                        "height_km": float(height_km),
                        "scan_shift": scan_shift,
                        "ray_shift": ray_shift,
                        "gr_count": int(shift["gr_count"][offset_index, height_index]),
                        "dpr_count": int(shift["dpr_count"][offset_index, height_index]),
                        "overlap_count": metrics["count"],
                        "union_count": union,
                        "support_csi": safe_ratio(metrics["count"], union),
                        "mae_dbz": metrics["mae_dbz"],
                        "rmse_dbz": metrics["rmse_dbz"],
                        "bias_gr_minus_dpr_dbz": metrics[
                            "bias_gr_minus_dpr_dbz"
                        ],
                        "pearson_r": metrics["pearson_r"],
                    }
                )
    return rows


def best_shift_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Select maximum-CSI shift per group/height with deterministic tie breaks."""

    buckets: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        buckets.setdefault((str(row["group"]), int(row["height_index"])), []).append(row)
    output: list[dict[str, Any]] = []
    for (group, height_index), candidates in sorted(buckets.items()):
        best = max(
            candidates,
            key=lambda row: (
                -1.0 if row["support_csi"] is None else float(row["support_csi"]),
                -abs(int(row["scan_shift"])) - abs(int(row["ray_shift"])),
                -abs(int(row["scan_shift"])),
                -abs(int(row["ray_shift"])),
                -int(row["scan_shift"]),
                -int(row["ray_shift"]),
            ),
        )
        output.append(
            {
                "group": group,
                "height_index": height_index,
                "height_km": best["height_km"],
                "best_scan_shift": best["scan_shift"],
                "best_ray_shift": best["ray_shift"],
                "best_support_csi": best["support_csi"],
                "overlap_count": best["overlap_count"],
                "mae_dbz_at_best_support_shift": best["mae_dbz"],
                "rmse_dbz_at_best_support_shift": best["rmse_dbz"],
                "pearson_at_best_support_shift": best["pearson_r"],
            }
        )
    return output


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def atomic_json_write(path: Path, value: Mapping[str, Any]) -> None:
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=_json_default,
        )
        handle.write("\n")
    partial.replace(path)


def atomic_csv_write(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fieldnames.append(field)
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    partial.replace(path)


def prepare_output_directory(output_dir: Path, *, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = (
        "summary.json",
        "per_file.csv",
        "per_height.csv",
        "distance_to_gr.csv",
        "local_density.csv",
        "shift_metrics.csv",
        "best_shift_by_height.csv",
        "coverage_by_height.png",
        "regions_by_height.png",
        "distance_to_gr.png",
        "best_shift_by_height.png",
    )
    existing = [output_dir / name for name in targets if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "audit outputs already exist ("
            + ", ".join(path.name for path in existing[:4])
            + "); use --overwrite"
        )


def _save_figure(figure: Any, path: Path) -> None:
    partial = path.with_name(path.stem + ".partial" + path.suffix)
    figure.savefig(partial, dpi=160, bbox_inches="tight")
    partial.replace(path)


def write_plots(
    output_dir: Path,
    aggregate: Mapping[str, Any],
    distance_table: Sequence[Mapping[str, Any]],
    best_table: Sequence[Mapping[str, Any]],
) -> None:
    """Write compact global audit plots using a headless Matplotlib backend."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = aggregate["z"]
    height = aggregate["groups"]["all"]["height"]
    total = np.maximum(height["total_count"].astype(float), 1.0)

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(z, height["gr_value_count"] / total, label="GR direct value")
    axis.plot(z, height["interp_value_count"] / total, label="GR interpolated value")
    axis.plot(z, height["dpr_value_count"] / total, label="DPR value")
    axis.set(xlabel="Height (km)", ylabel="Fraction of horizontal cells", title="Reflectivity support by height")
    axis.grid(alpha=0.25)
    axis.legend()
    _save_figure(figure, output_dir / "coverage_by_height.png")
    plt.close(figure)

    domain = np.maximum(height["occupancy_domain_count"].astype(float), 1.0)
    figure, axis = plt.subplots(figsize=(8, 5))
    for field, label in (
        ("q11_overlap_count", "Q11 GR+DPR"),
        ("q01_dpr_only_count", "Q01 DPR only"),
        ("q10_gr_only_count", "Q10 GR only"),
    ):
        axis.plot(z, height[field] / domain, label=label)
    axis.set(xlabel="Height (km)", ylabel="Fraction of occurrence-label domain", title="Stage-two regions by height")
    axis.grid(alpha=0.25)
    axis.legend()
    _save_figure(figure, output_dir / "regions_by_height.png")
    plt.close(figure)

    disjoint = [
        row
        for row in distance_table
        if row["group"] == "all" and not str(row["distance_bin"]).startswith("cumulative")
    ]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(
        [str(row["distance_bin"]) for row in disjoint],
        [float(row["fraction_of_dpr"] or 0.0) for row in disjoint],
    )
    axis.set(xlabel="Horizontal Chebyshev distance to direct GR", ylabel="Fraction of DPR targets", title="DPR target recoverability proxy")
    axis.grid(axis="y", alpha=0.25)
    _save_figure(figure, output_dir / "distance_to_gr.png")
    plt.close(figure)

    selected_best = [row for row in best_table if row["group"] == "all"]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.step(
        [float(row["height_km"]) for row in selected_best],
        [int(row["best_scan_shift"]) for row in selected_best],
        where="mid",
        label="scan shift",
    )
    axis.step(
        [float(row["height_km"]) for row in selected_best],
        [int(row["best_ray_shift"]) for row in selected_best],
        where="mid",
        label="ray shift",
    )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set(xlabel="Height (km)", ylabel="Best binary-support shift (grid cells)", title="Best small GR-to-DPR shift by height")
    axis.grid(alpha=0.25)
    axis.legend()
    _save_figure(figure, output_dir / "best_shift_by_height.png")
    plt.close(figure)


def write_outputs(
    output_dir: Path,
    results: Sequence[dict[str, Any]],
    aggregate: Mapping[str, Any],
    *,
    tasks: Sequence[AuditTask],
    manifest_path: Path,
    plots: bool,
) -> dict[str, Any]:
    """Atomically write every tabular, JSON, and optional plot artifact."""

    summary = build_summary(aggregate, tasks=tasks, manifest_path=manifest_path)
    height_table = per_height_rows(aggregate)
    distance_table = distance_rows(aggregate, tasks[0].options.distance_radii)
    density_table = density_rows(aggregate)
    shift_table = shift_rows(aggregate)
    best_table = best_shift_rows(shift_table)
    atomic_json_write(output_dir / "summary.json", summary)
    atomic_csv_write(output_dir / "per_file.csv", [result["row"] for result in results])
    atomic_csv_write(output_dir / "per_height.csv", height_table)
    atomic_csv_write(output_dir / "distance_to_gr.csv", distance_table)
    atomic_csv_write(output_dir / "local_density.csv", density_table)
    atomic_csv_write(output_dir / "shift_metrics.csv", shift_table)
    atomic_csv_write(output_dir / "best_shift_by_height.csv", best_table)
    if plots:
        write_plots(output_dir, aggregate, distance_table, best_table)
    return summary


def main() -> None:
    args = parse_args()
    options = AuditOptions(
        max_shift=args.max_shift,
        distance_radii=tuple(args.distance_radii),
        density_radius=args.density_radius,
    )
    tasks = load_tasks(
        args.split_manifest,
        splits=args.splits,
        count=args.count,
        options=options,
    )
    prepare_output_directory(args.output_dir, overwrite=args.overwrite)
    results = run_tasks(tasks, workers=args.workers)
    aggregate = aggregate_results(results)
    summary = write_outputs(
        args.output_dir,
        results,
        aggregate,
        tasks=tasks,
        manifest_path=args.split_manifest,
        plots=not args.no_plots,
    )
    all_group = summary["groups"]["all"]
    print(f"Audited {summary['selected_file_count']} file(s) -> {args.output_dir}")
    print(
        "GR value / DPR value / Q01(DPR-only): "
        f"{all_group['totals']['gr_value_count']} / "
        f"{all_group['totals']['dpr_value_count']} / "
        f"{all_group['totals']['q01_dpr_only_count']}"
    )


if __name__ == "__main__":
    main()
