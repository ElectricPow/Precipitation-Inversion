#!/usr/bin/env python3
"""Fit per-height input statistics from the training split only.

Files are processed in contiguous scan chunks through ``read_nc_sample``.  By
default each variable is fitted from all of *its own* finite, non-fill training
values.  In particular, DPR input normalization is not selected by the
``pre_positive_qc`` label mask: finite echoes below CFB remain representable as
model inputs even though the reliable label loss still excludes them.

An independent per-height ``pre_positive_qc`` count is stored for optional
height-balanced loss.  Keeping that reference separate prevents input-frequency
statistics from silently becoming target-loss weights.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from netCDF4 import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from precipitation_inversion.data.nc_reader import read_nc_sample  # noqa: E402
from precipitation_inversion.data.transforms import (  # noqa: E402
    PerLevelRunningStats,
)


DEFAULT_FILE_LIST = PROJECT_ROOT / "metadata" / "splits" / "train_files.txt"
DEFAULT_SPLIT_MANIFEST = (
    PROJECT_ROOT / "metadata" / "splits" / "split_manifest.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "metadata"
    / "normalization"
    / "stage1_dbz_valid.json"
)
DEFAULT_VARIABLES = ("dbz_dpr", "p", "t", "q")
SELECTION_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "variable_valid": ("z",),
    "pre_valid_native": ("z", "pre_dpr"),
    "pre_positive_native": ("z", "pre_dpr"),
    "pre_valid_qc": ("z", "pre_dpr", "cfb"),
    "pre_positive_qc": ("z", "pre_dpr", "cfb"),
}
HEIGHT_LOSS_SELECTIONS = ("none", "pre_positive_qc")

SELECTION_DESCRIPTIONS = {
    "variable_valid": (
        "Each variable uses all of its own finite, non-fill values from the "
        "training split; no precipitation-label QC mask is applied."
    ),
    "pre_valid_native": "Valid native precipitation labels select every variable.",
    "pre_positive_native": (
        "Positive native precipitation labels select every variable."
    ),
    "pre_valid_qc": "Valid above-CFB precipitation labels select every variable.",
    "pre_positive_qc": (
        "Positive above-CFB precipitation labels select every variable."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit train-only per-height normalization statistics."
    )
    parser.add_argument("--file-list", type=Path, default=DEFAULT_FILE_LIST)
    parser.add_argument(
        "--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--variables", nargs="+", default=list(DEFAULT_VARIABLES)
    )
    parser.add_argument(
        "--selection-mask",
        choices=tuple(SELECTION_DEPENDENCIES),
        default="variable_valid",
        help=(
            "Values used to fit each input variable. The default 'variable_valid' "
            "decouples input normalization from precipitation-label QC."
        ),
    )
    parser.add_argument(
        "--height-loss-selection-mask",
        choices=HEIGHT_LOSS_SELECTIONS,
        default="pre_positive_qc",
        help=(
            "Independent mask whose per-height counts support loss weighting; "
            "use 'none' to omit this reference."
        ),
    )
    parser.add_argument(
        "--scan-chunk-size",
        type=int,
        default=64,
        help="Contiguous nscan rows per read (default: 64).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        help="Debug only: process the first N paths after validating the full list.",
    )
    parser.add_argument(
        "--skip-split-check",
        action="store_true",
        help="Do not verify that every path belongs to the train split.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output JSON.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_file_list(path: Path) -> list[Path]:
    if not path.is_file():
        raise FileNotFoundError(f"File list not found: {path}")
    paths = [
        Path(line.strip()).expanduser().resolve()
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    if not paths:
        raise ValueError(f"File list is empty: {path}")
    duplicates = len(paths) - len(set(paths))
    if duplicates:
        raise ValueError(f"File list contains {duplicates} duplicate path(s)")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} listed NetCDF file(s) do not exist; first: {missing[0]}"
        )
    return paths


def verify_training_membership(paths: Iterable[Path], split_manifest: Path) -> None:
    """Reject accidental normalization fitting on validation or test files."""

    if not split_manifest.is_file():
        raise FileNotFoundError(f"Split manifest not found: {split_manifest}")
    with split_manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    split_by_path = {
        str(Path(row["file_path"]).expanduser().resolve()): row["split"] for row in rows
    }
    unknown: list[str] = []
    non_train: list[str] = []
    for path in paths:
        split = split_by_path.get(str(path.resolve()))
        if split is None:
            unknown.append(str(path))
        elif split != "train":
            non_train.append(f"{path.name}:{split}")
    if unknown:
        raise ValueError(
            f"{len(unknown)} path(s) are absent from split_manifest; first: {unknown[0]}"
        )
    if non_train:
        raise ValueError(
            "Normalization input contains validation/test files: "
            + ", ".join(non_train[:5])
        )


def source_nscan(path: Path) -> int:
    with Dataset(path, "r") as dataset:
        if "nscan" not in dataset.dimensions:
            raise KeyError(f"NetCDF file has no nscan dimension: {path}")
        return len(dataset.dimensions["nscan"])


def unique_names(values: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        name = value.strip()
        if not name:
            raise ValueError("variable names must be non-empty")
        if name not in result:
            result.append(name)
    return tuple(result)


def atomic_json_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8") as handle:
        # allow_nan=False guarantees missing levels are serialized as null rather
        # than non-standard JSON NaN tokens.
        json.dump(
            value,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
    partial.replace(path)


def fit_statistics(
    paths: list[Path],
    *,
    variables: tuple[str, ...],
    selection_mask: str,
    height_loss_selection_mask: str = "pre_positive_qc",
    scan_chunk_size: int,
) -> tuple[
    dict[str, PerLevelRunningStats],
    np.ndarray,
    dict[str, str | None],
    int,
    np.ndarray | None,
]:
    if scan_chunk_size <= 0:
        raise ValueError("scan_chunk_size must be positive")
    if selection_mask not in SELECTION_DEPENDENCIES:
        raise ValueError(f"unsupported selection_mask: {selection_mask}")
    if height_loss_selection_mask not in HEIGHT_LOSS_SELECTIONS:
        raise ValueError(
            "height_loss_selection_mask must be 'none' or 'pre_positive_qc'"
        )
    dependencies = SELECTION_DEPENDENCIES[selection_mask]
    height_dependencies = (
        ()
        if height_loss_selection_mask == "none"
        else SELECTION_DEPENDENCIES[height_loss_selection_mask]
    )
    requested = unique_names((*variables, *dependencies, *height_dependencies))
    accumulators: dict[str, PerLevelRunningStats] = {}
    reference_z: np.ndarray | None = None
    units: dict[str, str | None] = {}
    chunk_count = 0
    height_loss_count: np.ndarray | None = None

    for file_index, path in enumerate(paths, start=1):
        nscan = source_nscan(path)
        for start in range(0, nscan, scan_chunk_size):
            stop = min(start + scan_chunk_size, nscan)
            sample = read_nc_sample(
                path,
                variables=requested,
                scan_slice=slice(start, stop),
                dtype=np.float32,
                build_masks=True,
            )
            z = sample.variables["z"].astype(np.float64)
            if reference_z is None:
                reference_z = z.copy()
                accumulators = {
                    name: PerLevelRunningStats.empty(z.size) for name in variables
                }
                units = {name: sample.metadata[name].units for name in variables}
                if height_loss_selection_mask != "none":
                    height_loss_count = np.zeros(z.size, dtype=np.int64)
            elif z.shape != reference_z.shape or not np.allclose(
                z, reference_z, rtol=0.0, atol=1e-6
            ):
                raise ValueError(f"Height coordinate differs from earlier files: {path}")

            selected = None if selection_mask == "variable_valid" else sample.masks[selection_mask]
            for name in variables:
                values = sample.variables[name]
                if values.ndim != 3 or values.shape[-1] != z.size:
                    raise ValueError(
                        f"Normalization variable {name!r} must have shape "
                        f"(nscan,nray,z), got {values.shape}"
                    )
                accumulators[name].update(values, valid_mask=selected)
            if height_loss_count is not None:
                height_mask = sample.masks[height_loss_selection_mask]
                if height_mask.shape != sample.shape_3d:
                    raise ValueError(
                        "height-loss selection mask must have shape "
                        f"{sample.shape_3d}, got {height_mask.shape}"
                    )
                height_loss_count += height_mask.sum(
                    axis=(0, 1), dtype=np.int64
                )
            chunk_count += 1
        print(
            f"OK [{file_index}/{len(paths)}] {path.name} nscan={nscan}",
            flush=True,
        )

    assert reference_z is not None
    return accumulators, reference_z, units, chunk_count, height_loss_count


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {args.output}; use --overwrite to replace it"
        )
    variables = unique_names(args.variables)
    paths = load_file_list(args.file_list)
    if not args.skip_split_check:
        verify_training_membership(paths, args.split_manifest)
    selected_paths = paths
    if args.max_files is not None:
        if args.max_files <= 0:
            raise ValueError("--max-files must be positive")
        selected_paths = paths[: args.max_files]

    print(f"Training file list: {args.file_list.resolve()}")
    print(f"Validated training paths: {len(paths)}")
    print(f"Files selected for this run: {len(selected_paths)}")
    print(f"Variables: {', '.join(variables)}")
    print(f"Selection mask: {args.selection_mask}")
    print(f"Height-loss selection mask: {args.height_loss_selection_mask}")
    print(f"Scan chunk size: {args.scan_chunk_size}")
    accumulators, z, units, chunk_count, height_loss_count = fit_statistics(
        selected_paths,
        variables=variables,
        selection_mask=args.selection_mask,
        height_loss_selection_mask=args.height_loss_selection_mask,
        scan_chunk_size=args.scan_chunk_size,
    )

    output = {
        "format_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "training_split_only",
        "file_list": str(args.file_list.resolve()),
        "file_list_sha256": sha256_file(args.file_list),
        "split_manifest": (
            None if args.skip_split_check else str(args.split_manifest.resolve())
        ),
        "split_manifest_sha256": (
            None if args.skip_split_check else sha256_file(args.split_manifest)
        ),
        "validated_file_count": len(paths),
        "processed_file_count": len(selected_paths),
        "scan_chunk_size": args.scan_chunk_size,
        "processed_chunk_count": chunk_count,
        "statistics_role": "model_input_normalization",
        "selection_mask": args.selection_mask,
        "selection_semantics": SELECTION_DESCRIPTIONS[args.selection_mask],
        "label_qc_applied_to_input_statistics": (
            args.selection_mask != "variable_valid"
        ),
        "height_loss_weight_reference": (
            None
            if height_loss_count is None
            else {
                "selection_mask": args.height_loss_selection_mask,
                "semantics": (
                    "Independent reliable-label counts by physical height; "
                    "not input-normalization counts."
                ),
                "heights_km": z.tolist(),
                "count": height_loss_count.tolist(),
                "total_count": int(height_loss_count.sum()),
            }
        ),
        "rain_transform": {
            "name": "log1p",
            "forward": "log(1 + R)",
            "inverse": "exp(y) - 1",
            "units": "mm/h",
        },
        "variables": {
            name: {
                "units": units[name],
                **accumulator.to_dict(heights_km=z),
            }
            for name, accumulator in accumulators.items()
        },
    }
    atomic_json_write(args.output, output)
    print(f"Normalization statistics: {args.output}")
    for name, accumulator in accumulators.items():
        print(
            f"  {name}: values={int(accumulator.count.sum()):,}, "
            f"empty_levels={int((accumulator.count == 0).sum())}"
        )
    if height_loss_count is not None:
        print(
            "  height-loss reference: "
            f"mask={args.height_loss_selection_mask}, "
            f"values={int(height_loss_count.sum()):,}, "
            f"empty_levels={int((height_loss_count == 0).sum())}"
        )


if __name__ == "__main__":
    main()
