#!/usr/bin/env python3
"""Fit train-only per-height physical reflectivity statistics for stage two.

Unlike the generic stage-one reader, this script classifies each raw NetCDF
masked array before numerical conversion.  Only the ``value`` state contributes
to moments; native missing values and finite legacy sentinels remain separately
counted and never become physical ``0 dBZ`` samples.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from netCDF4 import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from precipitation_inversion.data.stage2_masks import (  # noqa: E402
    classify_reflectivity_storage,
    physical_reflectivity_values,
)
from precipitation_inversion.data.transforms import PerLevelRunningStats  # noqa: E402


DEFAULT_FILE_LIST = PROJECT_ROOT / "metadata" / "splits" / "train_files.txt"
DEFAULT_SPLIT_MANIFEST = PROJECT_ROOT / "metadata" / "splits" / "split_manifest.csv"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "metadata" / "normalization" / "stage2_reflectivity.json"
)
DEFAULT_VARIABLES = ("dbz_gr_sparse", "dbz_gr_interp", "dbz_dpr")
FORMAT_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit train-only stage-two physical dBZ statistics."
    )
    parser.add_argument("--file-list", type=Path, default=DEFAULT_FILE_LIST)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--variables", nargs="+", default=list(DEFAULT_VARIABLES))
    parser.add_argument("--scan-chunk-size", type=int, default=64)
    parser.add_argument(
        "--max-files",
        type=int,
        help="Debug only; such partial statistics are rejected by the Dataset.",
    )
    parser.add_argument("--skip-split-check", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_file_list(path: Path) -> list[Path]:
    if not path.is_file():
        raise FileNotFoundError(f"file list not found: {path}")
    paths = [
        Path(line.strip()).expanduser().resolve()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not paths:
        raise ValueError(f"file list is empty: {path}")
    if len(paths) != len(set(paths)):
        raise ValueError(f"file list contains duplicate paths: {path}")
    missing = [source for source in paths if not source.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} source file(s) do not exist; first: {missing[0]}"
        )
    return paths


def verify_training_membership(paths: Iterable[Path], manifest_path: Path) -> None:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"split manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"file_path", "split"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"split manifest must contain {sorted(required)}")
        split_by_path = {
            str(Path(row["file_path"]).expanduser().resolve()): row["split"]
            for row in reader
        }
    for path in paths:
        split = split_by_path.get(str(path.resolve()))
        if split is None:
            raise ValueError(f"path is absent from split manifest: {path}")
        if split != "train":
            raise ValueError(f"normalization path belongs to {split}, not train: {path}")


def _unique_variables(values: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        name = str(value).strip()
        if not name:
            raise ValueError("variable names must be non-empty")
        if name not in result:
            result.append(name)
    if not result:
        raise ValueError("at least one variable is required")
    return tuple(result)


def fit_stage2_statistics(
    paths: Sequence[Path],
    *,
    variables: Sequence[str] = DEFAULT_VARIABLES,
    scan_chunk_size: int = 64,
) -> tuple[
    dict[str, PerLevelRunningStats],
    dict[str, dict[str, int]],
    np.ndarray,
    dict[str, str | None],
    int,
]:
    """Stream raw training reflectivity and return moments plus state counts."""

    if not paths:
        raise ValueError("at least one source path is required")
    if scan_chunk_size <= 0:
        raise ValueError("scan_chunk_size must be positive")
    names = _unique_variables(variables)
    accumulators: dict[str, PerLevelRunningStats] = {}
    storage_counts = {
        name: {"native_missing": 0, "sentinel": 0, "value": 0}
        for name in names
    }
    units: dict[str, str | None] = {}
    reference_z: np.ndarray | None = None
    chunk_count = 0

    for file_index, path_value in enumerate(paths, start=1):
        path = Path(path_value).expanduser().resolve()
        with Dataset(path, "r") as dataset:
            required = {"z", *names}
            missing = sorted(required.difference(dataset.variables))
            if missing:
                raise KeyError(f"{path.name} missing variables: {', '.join(missing)}")
            z = np.asarray(dataset["z"][:], dtype=np.float64)
            if (
                z.ndim != 1
                or not np.all(np.isfinite(z))
                or not np.all(np.diff(z) > 0.0)
            ):
                raise ValueError(f"invalid height coordinate: {path}")
            if reference_z is None:
                reference_z = z.copy()
                accumulators = {
                    name: PerLevelRunningStats.empty(z.size) for name in names
                }
                units = {
                    name: (
                        str(dataset[name].units)
                        if hasattr(dataset[name], "units")
                        else None
                    )
                    for name in names
                }
            elif z.shape != reference_z.shape or not np.allclose(
                z, reference_z, rtol=0.0, atol=1e-6
            ):
                raise ValueError(f"height coordinate differs from earlier files: {path}")

            nscan = len(dataset.dimensions["nscan"])
            nray = len(dataset.dimensions["nray"])
            for start in range(0, nscan, scan_chunk_size):
                stop = min(start + scan_chunk_size, nscan)
                for name in names:
                    variable = dataset[name]
                    if tuple(variable.dimensions) != ("nscan", "nray", "z"):
                        raise ValueError(
                            f"{path.name}:{name} dimensions {variable.dimensions} "
                            "must be (nscan,nray,z)"
                        )
                    raw = variable[start:stop, :, :]
                    expected = (stop - start, nray, z.size)
                    if raw.shape != expected:
                        raise ValueError(
                            f"{path.name}:{name} shape {raw.shape} != {expected}"
                        )
                    states = classify_reflectivity_storage(raw)
                    physical = physical_reflectivity_values(raw, masks=states)
                    accumulators[name].update(physical, valid_mask=states.value)
                    counts = states.counts()
                    for state in storage_counts[name]:
                        storage_counts[name][state] += int(counts[state])
                chunk_count += 1
        print(
            f"OK [{file_index}/{len(paths)}] {path.name} nscan={nscan}",
            flush=True,
        )

    assert reference_z is not None
    return accumulators, storage_counts, reference_z, units, chunk_count


def build_output(
    *,
    accumulators: dict[str, PerLevelRunningStats],
    storage_counts: dict[str, dict[str, int]],
    z: np.ndarray,
    units: dict[str, str | None],
    file_list: Path,
    split_manifest: Path | None,
    validated_file_count: int,
    processed_file_count: int,
    scan_chunk_size: int,
    chunk_count: int,
) -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": 2,
        "scope": "training_split_only",
        "statistics_role": "gr_input_and_dpr_target_reflectivity_normalization",
        "selection_mask": "reflectivity_storage_value",
        "selection_semantics": (
            "Per variable and height, only the storage-aware physical value "
            "state contributes; native missing and finite sentinels are excluded."
        ),
        "sentinel_cutoff": -9990.0,
        "file_list": str(file_list.resolve()),
        "file_list_sha256": sha256_file(file_list),
        "split_manifest": (
            str(split_manifest.resolve()) if split_manifest is not None else None
        ),
        "split_manifest_sha256": (
            sha256_file(split_manifest) if split_manifest is not None else None
        ),
        "validated_file_count": validated_file_count,
        "processed_file_count": processed_file_count,
        "scan_chunk_size": scan_chunk_size,
        "processed_chunk_count": chunk_count,
        "variables": {
            name: {
                "units": units[name],
                "storage_state_counts": storage_counts[name],
                **accumulator.to_dict(heights_km=z),
            }
            for name, accumulator in accumulators.items()
        },
    }


def atomic_json_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8") as handle:
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


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"output already exists: {args.output}; use --overwrite to replace it"
        )
    if args.max_files is not None and args.max_files <= 0:
        raise ValueError("--max-files must be positive")
    variables = _unique_variables(args.variables)
    paths = load_file_list(args.file_list)
    if not args.skip_split_check:
        verify_training_membership(paths, args.split_manifest)
    selected = paths if args.max_files is None else paths[: args.max_files]
    accumulators, counts, z, units, chunks = fit_stage2_statistics(
        selected,
        variables=variables,
        scan_chunk_size=args.scan_chunk_size,
    )
    output = build_output(
        accumulators=accumulators,
        storage_counts=counts,
        z=z,
        units=units,
        file_list=args.file_list,
        split_manifest=None if args.skip_split_check else args.split_manifest,
        validated_file_count=len(paths),
        processed_file_count=len(selected),
        scan_chunk_size=args.scan_chunk_size,
        chunk_count=chunks,
    )
    atomic_json_write(args.output, output)
    print(f"Stage-two normalization statistics: {args.output}")
    for name, accumulator in accumulators.items():
        print(
            f"  {name}: physical_values={int(accumulator.count.sum()):,}, "
            f"empty_levels={int(np.count_nonzero(accumulator.count == 0))}"
        )


if __name__ == "__main__":
    main()
