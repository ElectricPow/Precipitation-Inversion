#!/usr/bin/env python3
"""Evaluate controlled non-neural baselines for stage-two GR-to-DPR mapping.

Four baselines share exactly the same target masks and metrics:

``gr_sparse_raw``
    Direct physical sparse-GR values and their native support.
``gr_sparse_height_calibrated``
    Sparse GR mapped by train-only per-height mean/std to DPR dBZ space.
``gr_interp_raw``
    The legacy interpolated GR field supplied in each NetCDF file.
``gr_interp_height_calibrated``
    Interpolated GR with the same train-only per-height distribution mapping.

dBZ scores use common predicted/DPR support, while support recall/CSI/FSS
quantify how much of the dense target each sparse baseline actually covers.
The two must never be interpreted separately.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from netCDF4 import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from precipitation_inversion.data.masks import clutter_mask_from_cfb, to_float_array  # noqa: E402
from precipitation_inversion.data.stage2_masks import (  # noqa: E402
    build_stage2_spatial_masks,
    classify_reflectivity_storage,
    physical_reflectivity_values,
)
from precipitation_inversion.metrics.stage2_reflectivity import (  # noqa: E402
    Stage2ReflectivityMetrics,
    finite_metrics_for_json,
)


DEFAULT_SPLIT_MANIFEST = PROJECT_ROOT / "metadata" / "splits" / "split_manifest.csv"
DEFAULT_NORMALIZATION = (
    PROJECT_ROOT / "metadata" / "normalization" / "stage2_reflectivity.json"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stage2_baselines"
VALID_SPLITS = ("train", "val", "test")
BASELINE_NAMES = (
    "gr_sparse_raw",
    "gr_sparse_height_calibrated",
    "gr_interp_raw",
    "gr_interp_height_calibrated",
)
DEFAULT_FSS_RADII = (1, 2, 4)
DEFAULT_DBZ_BIN_EDGES = (15.0, 25.0, 35.0)


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if any(item < 0 for item in result) or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("radii must be unique and non-negative")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate sparse/interpolated Stage-2 reflectivity baselines."
    )
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--normalization", type=Path, default=DEFAULT_NORMALIZATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--splits", nargs="+", choices=VALID_SPLITS, default=list(VALID_SPLITS)
    )
    parser.add_argument("--max-files", type=int)
    parser.add_argument(
        "--fss-radii", type=_parse_int_tuple, default=DEFAULT_FSS_RADII
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_manifest_rows(
    path: Path, *, splits: Sequence[str], max_files: int | None
) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"split manifest not found: {path}")
    selected = set(splits)
    if not selected or not selected.issubset(VALID_SPLITS):
        raise ValueError(f"splits must be selected from {VALID_SPLITS}")
    if max_files is not None and max_files <= 0:
        raise ValueError("max_files must be positive")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "file_path", "split"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"split manifest must contain {sorted(required)}")
        rows = [row for row in reader if row["split"] in selected]
    rows.sort(key=lambda row: (row["split"], row["sample_id"]))
    if max_files is not None:
        rows = rows[:max_files]
    if not rows:
        raise ValueError("no files selected")
    for row in rows:
        source = Path(row["file_path"]).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"source file not found: {source}")
        row["file_path"] = str(source)
    return rows


def load_normalization(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"normalization JSON not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("stage") != 2 or value.get("scope") != "training_split_only":
        raise ValueError("baseline calibration requires train-only Stage-2 statistics")
    if value.get("selection_mask") != "reflectivity_storage_value":
        raise ValueError("normalization has unexpected selection mask")
    for name in ("dbz_gr_sparse", "dbz_gr_interp", "dbz_dpr"):
        if name not in value.get("variables", {}):
            raise KeyError(f"normalization missing {name!r}")
    return value


def _statistic_array(statistics: Mapping[str, Any], key: str) -> np.ndarray:
    return np.asarray(
        [np.nan if value is None else value for value in statistics[key]],
        dtype=np.float64,
    )


def height_calibrate_reflectivity(
    values: np.ndarray,
    support: np.ndarray,
    *,
    source_statistics: Mapping[str, Any],
    target_statistics: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Map source dBZ to target mean/std independently at every height."""

    source = np.asarray(values, dtype=np.float64)
    valid = np.asarray(support)
    if valid.dtype != np.bool_:
        raise TypeError("support must be boolean")
    if source.shape != valid.shape or source.ndim < 1:
        raise ValueError("values and support must have identical non-scalar shapes")
    source_mean = _statistic_array(source_statistics, "mean")
    source_std = _statistic_array(source_statistics, "std")
    target_mean = _statistic_array(target_statistics, "mean")
    target_std = _statistic_array(target_statistics, "std")
    if not (
        source_mean.shape
        == source_std.shape
        == target_mean.shape
        == target_std.shape
        == (source.shape[-1],)
    ):
        raise ValueError("normalization vectors must match the height axis")
    fitted = (
        np.isfinite(source_mean)
        & np.isfinite(source_std)
        & (source_std > 1e-6)
        & np.isfinite(target_mean)
        & np.isfinite(target_std)
        & (target_std > 1e-6)
    )
    shape = (1,) * (source.ndim - 1) + (-1,)
    effective = valid & np.isfinite(source) & fitted.reshape(shape)
    output = np.full(source.shape, np.nan, dtype=np.float32)
    mapped = (
        (source - source_mean.reshape(shape))
        / np.where(source_std > 1e-6, source_std, 1.0).reshape(shape)
        * target_std.reshape(shape)
        + target_mean.reshape(shape)
    )
    output[effective] = mapped[effective].astype(np.float32, copy=False)
    return output, effective


def build_baseline_predictions(
    gr_values: np.ndarray,
    gr_support: np.ndarray,
    interp_values: np.ndarray,
    interp_support: np.ndarray,
    normalization: Mapping[str, Any],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return four controlled baseline ``(dBZ,support)`` predictions."""

    variables = normalization["variables"]
    gr_calibrated = height_calibrate_reflectivity(
        gr_values,
        gr_support,
        source_statistics=variables["dbz_gr_sparse"],
        target_statistics=variables["dbz_dpr"],
    )
    interp_calibrated = height_calibrate_reflectivity(
        interp_values,
        interp_support,
        source_statistics=variables["dbz_gr_interp"],
        target_statistics=variables["dbz_dpr"],
    )
    return {
        "gr_sparse_raw": (gr_values, gr_support),
        "gr_sparse_height_calibrated": gr_calibrated,
        "gr_interp_raw": (interp_values, interp_support),
        "gr_interp_height_calibrated": interp_calibrated,
    }


def _read_file(path: Path) -> dict[str, Any]:
    with Dataset(path, "r") as dataset:
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
            raise KeyError(f"{path.name} missing variables: {', '.join(missing)}")
        z = to_float_array(dataset["z"][:]).astype(np.float64, copy=False)
        gr_raw = dataset["dbz_gr_sparse"][:]
        interp_raw = dataset["dbz_gr_interp"][:]
        dpr_raw = dataset["dbz_dpr"][:]
        pre_raw = dataset["pre_dpr"][:]
        cfb_raw = dataset["cfb"][:]
    gr_states = classify_reflectivity_storage(gr_raw)
    interp_states = classify_reflectivity_storage(interp_raw)
    dpr_states = classify_reflectivity_storage(dpr_raw)
    masks = build_stage2_spatial_masks(
        gr_raw, dpr_raw, dbz_gr_interp=interp_raw, pre_dpr=pre_raw
    )
    return {
        "z": z,
        "gr": physical_reflectivity_values(gr_raw, masks=gr_states),
        "interp": physical_reflectivity_values(interp_raw, masks=interp_states),
        "dpr": physical_reflectivity_values(dpr_raw, masks=dpr_states),
        "masks": masks,
        "cfb_clutter": clutter_mask_from_cfb(cfb_raw, z),
    }


def _region_masks(data: Mapping[str, Any]) -> dict[str, np.ndarray]:
    masks = data["masks"]
    domain = masks["occupancy_domain"]
    dpr = masks["dpr_value"]
    dpr_dbz = data["dpr"]
    clutter = data["cfb_clutter"]
    regions = {
        "all_domain": domain,
        "q11_direct_overlap": domain & masks["q11_overlap"],
        "q01_direct_missing": domain & masks["q01_dpr_only"],
        "q10_gr_only": domain & masks["q10_gr_only"],
        "dpr_gap_proxy": domain & masks["dpr_only_gap_proxy"],
        "dpr_outside_proxy": domain & masks["dpr_only_outside_proxy"],
        "dpr_above_cfb": domain & dpr & ~clutter,
        "dpr_below_cfb": domain & dpr & clutter,
    }
    edges = (-np.inf, *DEFAULT_DBZ_BIN_EDGES, np.inf)
    labels = ("lt15", "15to25", "25to35", "ge35")
    for label, lower, upper in zip(labels, edges, edges[1:]):
        regions[f"dpr_dbz_{label}"] = (
            domain & dpr & (dpr_dbz >= lower) & (dpr_dbz < upper)
        )
    return regions


def _flatten_metrics(prefix: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for section, values in prefix.items():
        if section == "fss":
            for radius, fss_values in values.items():
                for name, value in fss_values.items():
                    if name not in {"radius", "window_size"}:
                        result[f"fss_r{radius}_{name}"] = value
        else:
            for name, value in values.items():
                result[f"{section}_{name}"] = value
    return finite_metrics_for_json(result)


def evaluate_rows(
    rows: Sequence[Mapping[str, str]],
    normalization: Mapping[str, Any],
    *,
    fss_radii: tuple[int, ...] = DEFAULT_FSS_RADII,
) -> dict[str, Any]:
    """Evaluate selected files and return summary plus CSV-ready rows."""

    aggregate: dict[tuple[str, str], Stage2ReflectivityMetrics] = {}
    heights: dict[tuple[str, str, int], Stage2ReflectivityMetrics] = {}
    regions: dict[tuple[str, str, str], Stage2ReflectivityMetrics] = {}
    per_file: list[dict[str, Any]] = []
    reference_z: np.ndarray | None = None

    for file_index, row in enumerate(rows, start=1):
        path = Path(row["file_path"])
        data = _read_file(path)
        z = data["z"]
        if reference_z is None:
            reference_z = z.copy()
        elif z.shape != reference_z.shape or not np.allclose(
            z, reference_z, rtol=0.0, atol=1e-6
        ):
            raise ValueError(f"height coordinate differs from earlier files: {path}")
        masks = data["masks"]
        predictions = build_baseline_predictions(
            data["gr"],
            masks["gr_value"],
            data["interp"],
            masks["gr_interp_value"],
            normalization,
        )
        domain = masks["occupancy_domain"]
        target_support = masks["dpr_value"]
        region_values = _region_masks(data)

        for baseline, (prediction, predicted_support) in predictions.items():
            file_metric = Stage2ReflectivityMetrics(fss_radii=fss_radii)
            file_metric.update(
                prediction, predicted_support, data["dpr"], target_support, domain
            )
            computed = file_metric.compute()
            per_file.append(
                {
                    "sample_id": row["sample_id"],
                    "split": row["split"],
                    "file_name": path.name,
                    "baseline": baseline,
                    **_flatten_metrics(computed),
                }
            )
            for group in ("all", row["split"]):
                key = (group, baseline)
                if key not in aggregate:
                    aggregate[key] = Stage2ReflectivityMetrics(fss_radii=fss_radii)
                aggregate[key].merge(file_metric)

            for level_index in range(z.size):
                level_key = (row["split"], baseline, level_index)
                all_level_key = ("all", baseline, level_index)
                level_metric = Stage2ReflectivityMetrics(fss_radii=())
                section = (..., level_index)
                level_metric.update(
                    prediction[section],
                    predicted_support[section],
                    data["dpr"][section],
                    target_support[section],
                    domain[section],
                )
                for key in (level_key, all_level_key):
                    if key not in heights:
                        heights[key] = Stage2ReflectivityMetrics(fss_radii=())
                    heights[key].merge(level_metric)

            for region_name, region_domain in region_values.items():
                region_metric = Stage2ReflectivityMetrics(fss_radii=())
                region_metric.update(
                    prediction,
                    predicted_support,
                    data["dpr"],
                    target_support,
                    region_domain,
                )
                for group in ("all", row["split"]):
                    key = (group, baseline, region_name)
                    if key not in regions:
                        regions[key] = Stage2ReflectivityMetrics(fss_radii=())
                    regions[key].merge(region_metric)
        print(f"OK [{file_index}/{len(rows)}] {path.name}", flush=True)

    assert reference_z is not None
    summary: dict[str, Any] = defaultdict(dict)
    for (group, baseline), metric in aggregate.items():
        summary[group][baseline] = finite_metrics_for_json(metric.compute())
    height_rows = [
        {
            "group": group,
            "baseline": baseline,
            "height_index": level,
            "height_km": float(reference_z[level]),
            **_flatten_metrics(metric.compute()),
        }
        for (group, baseline, level), metric in sorted(heights.items())
    ]
    region_rows = [
        {
            "group": group,
            "baseline": baseline,
            "region": region,
            **_flatten_metrics(metric.compute()),
        }
        for (group, baseline, region), metric in sorted(regions.items())
    ]
    return {
        "summary": dict(summary),
        "per_file": per_file,
        "per_height": height_rows,
        "per_region": region_rows,
        "heights_km": reference_z.tolist(),
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    partial.replace(path)


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path.name}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    partial.replace(path)


def prepare_output_directory(path: Path, *, overwrite: bool) -> None:
    expected = (
        path / "summary.json",
        path / "per_file.csv",
        path / "per_height.csv",
        path / "per_region.csv",
    )
    existing = [item for item in expected if item.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "baseline outputs already exist; use --overwrite: "
            + ", ".join(str(item) for item in existing)
        )
    path.mkdir(parents=True, exist_ok=True)


def write_outputs(
    output_dir: Path,
    result: Mapping[str, Any],
    *,
    rows: Sequence[Mapping[str, str]],
    split_manifest: Path,
    normalization_path: Path,
    fss_radii: tuple[int, ...],
) -> dict[str, Any]:
    summary = {
        "format_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_file_count": len(rows),
        "selected_splits": sorted({row["split"] for row in rows}),
        "split_manifest": str(split_manifest.resolve()),
        "normalization": str(normalization_path.resolve()),
        "baseline_names": list(BASELINE_NAMES),
        "fss_radii": list(fss_radii),
        "dbz_bin_edges": list(DEFAULT_DBZ_BIN_EDGES),
        "metric_semantics": {
            "support": "binary DPR-value support inside the selected domain",
            "reflectivity": "physical dBZ on common predicted/target support",
            "fss": "domain-aware horizontal windows; height is never mixed",
        },
        "groups": result["summary"],
    }
    _atomic_json(output_dir / "summary.json", summary)
    _atomic_csv(output_dir / "per_file.csv", result["per_file"])
    _atomic_csv(output_dir / "per_height.csv", result["per_height"])
    _atomic_csv(output_dir / "per_region.csv", result["per_region"])
    return summary


def main() -> None:
    args = parse_args()
    rows = load_manifest_rows(
        args.split_manifest, splits=args.splits, max_files=args.max_files
    )
    normalization = load_normalization(args.normalization)
    prepare_output_directory(args.output_dir, overwrite=args.overwrite)
    result = evaluate_rows(rows, normalization, fss_radii=tuple(args.fss_radii))
    summary = write_outputs(
        args.output_dir,
        result,
        rows=rows,
        split_manifest=args.split_manifest,
        normalization_path=args.normalization,
        fss_radii=tuple(args.fss_radii),
    )
    print(f"Evaluated {len(rows)} file(s) -> {args.output_dir}")
    for baseline in BASELINE_NAMES:
        metrics = summary["groups"]["all"][baseline]
        support = metrics["support"]
        dbz = metrics["reflectivity_on_common_support"]
        print(
            f"  {baseline}: recall={support['recall']}, CSI={support['csi']}, "
            f"MAE={dbz['mae_dbz']} dBZ, r={dbz['pearson_r']}"
        )


if __name__ == "__main__":
    main()
