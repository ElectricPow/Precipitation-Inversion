#!/usr/bin/env python3
"""Build compact train/validation/test indices for stage-one intensity learning."""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from precipitation_inversion.data.dataset import (  # noqa: E402
    STAGE1_INDEX_DTYPE,
    STAGE1_INDEX_FORMAT_VERSION,
    STAGE1_INPUT_VARIABLES,
    atomic_save_json,
    atomic_save_npy,
    build_stage1_index_records,
    load_path_list,
    sha256_file,
)


DEFAULT_SPLITS_DIR = PROJECT_ROOT / "metadata" / "splits"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "metadata" / "stage1_indices"
SPLIT_NAMES = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build compact stage-one positive-rain sample indices."
    )
    parser.add_argument(
        "--split", choices=(*SPLIT_NAMES, "all"), default="all"
    )
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scan-chunk-size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_split_manifest(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Split manifest not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {str(Path(row["file_path"]).resolve()): row for row in rows}


def verify_split_paths(
    paths: list[Path], split: str, manifest: dict[str, dict[str, str]]
) -> list[int]:
    expected_counts: list[int] = []
    for path in paths:
        row = manifest.get(str(path.resolve()))
        if row is None:
            raise ValueError(f"Path is absent from split_manifest: {path}")
        if row["split"] != split:
            raise ValueError(f"{path.name} belongs to {row['split']}, not {split}")
        expected_counts.append(int(row["pre_qc_positive_count"]))
    return expected_counts


def build_one_split(
    split: str,
    *,
    splits_dir: Path,
    output_dir: Path,
    scan_chunk_size: int,
    overwrite: bool,
    manifest: dict[str, dict[str, str]],
) -> None:
    file_list = splits_dir / f"{split}_files.txt"
    index_path = output_dir / f"{split}.npy"
    metadata_path = output_dir / f"{split}.json"
    existing = [path for path in (index_path, metadata_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Outputs already exist ("
            + ", ".join(path.name for path in existing)
            + "); use --overwrite"
        )
    paths = load_path_list(file_list)
    expected_counts = verify_split_paths(paths, split, manifest)
    records, file_entries, z, chunk_count = build_stage1_index_records(
        paths, scan_chunk_size=scan_chunk_size
    )
    for entry, expected in zip(file_entries, expected_counts):
        if entry["positive_qc_count"] != expected:
            raise AssertionError(
                f"{entry['file_name']} positive_qc={entry['positive_qc_count']} "
                f"!= manifest={expected}"
            )
    atomic_save_npy(index_path, records)
    index_sha256 = sha256_file(index_path)
    metadata: dict[str, Any] = {
        "format_version": STAGE1_INDEX_FORMAT_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "dpr_positive_qc_intensity",
        "split": split,
        "selection_mask": "pre_positive_qc",
        "input_variables": list(STAGE1_INPUT_VARIABLES),
        "label_variable": "pre_dpr",
        "label_transform": "log1p",
        "file_list": str(file_list.resolve()),
        "file_list_sha256": sha256_file(file_list),
        "split_manifest": str((splits_dir / "split_manifest.csv").resolve()),
        "split_manifest_sha256": sha256_file(splits_dir / "split_manifest.csv"),
        "scan_chunk_size": scan_chunk_size,
        "processed_chunk_count": chunk_count,
        "file_count": len(paths),
        "sample_count": int(records.size),
        "positive_qc_count": int(sum(entry["positive_qc_count"] for entry in file_entries)),
        "excluded_missing_input_count": int(
            sum(entry["excluded_missing_input_count"] for entry in file_entries)
        ),
        "heights_km": [float(value) for value in z],
        "index_file": index_path.name,
        "index_sha256": index_sha256,
        "index_dtype": STAGE1_INDEX_DTYPE.descr,
        "index_itemsize": STAGE1_INDEX_DTYPE.itemsize,
        "files": file_entries,
    }
    atomic_save_json(metadata_path, metadata)
    print(
        f"{split}: files={len(paths)}, samples={records.size:,}, "
        f"excluded_missing_inputs={metadata['excluded_missing_input_count']:,}, "
        f"index={index_path}"
    )


def main() -> None:
    args = parse_args()
    if args.scan_chunk_size <= 0:
        raise ValueError("--scan-chunk-size must be positive")
    manifest_path = args.splits_dir / "split_manifest.csv"
    manifest = load_split_manifest(manifest_path)
    selected = SPLIT_NAMES if args.split == "all" else (args.split,)
    for split in selected:
        build_one_split(
            split,
            splits_dir=args.splits_dir,
            output_dir=args.output_dir,
            scan_chunk_size=args.scan_chunk_size,
            overwrite=args.overwrite,
            manifest=manifest,
        )


if __name__ == "__main__":
    main()

