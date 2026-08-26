#!/usr/bin/env python3
"""Build fixed-core patch indices for stage-one 3D U-Net experiments."""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from precipitation_inversion.data.dataset import load_path_list, sha256_file  # noqa: E402
from precipitation_inversion.data.patch_dataset import (  # noqa: E402
    PATCH_INDEX_DTYPE,
    PATCH_INDEX_FORMAT_VERSION,
    PATCH_INPUT_CHANNELS,
    PATCH_INPUT_VARIABLE,
    PATCH_LABEL_VARIABLE,
    build_stage1_patch_index_records,
    ceil_to_multiple,
    save_patch_index,
)


DEFAULT_SPLITS_DIR = PROJECT_ROOT / "metadata" / "splits"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "metadata" / "stage1_patch_indices"
SPLIT_NAMES = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build non-overlapping-core indices for stage-one 3D patches."
    )
    parser.add_argument("--split", choices=(*SPLIT_NAMES, "all"), default="all")
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--core-size", type=int, default=32)
    parser.add_argument("--halo-size", type=int, default=16)
    parser.add_argument(
        "--horizontal-multiple",
        "--spatial-multiple",
        dest="horizontal_multiple",
        type=int,
        default=16,
        help="Pad nscan/nray to this multiple; z is never padded.",
    )
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
    expected_positive_counts: list[int] = []
    for path in paths:
        row = manifest.get(str(path.resolve()))
        if row is None:
            raise ValueError(f"Path is absent from split_manifest: {path}")
        if row["split"] != split:
            raise ValueError(f"{path.name} belongs to {row['split']}, not {split}")
        expected_positive_counts.append(int(row["pre_qc_positive_count"]))
    return expected_positive_counts


def build_one_split(
    split: str,
    *,
    splits_dir: Path,
    output_dir: Path,
    core_size: int,
    halo_size: int,
    horizontal_multiple: int,
    overwrite: bool,
    manifest: dict[str, dict[str, str]],
) -> None:
    if core_size <= 0 or halo_size < 0 or horizontal_multiple <= 0:
        raise ValueError(
            "core-size/horizontal-multiple must be positive and halo non-negative"
        )
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
    records, files, z, nray, processed_file_count = build_stage1_patch_index_records(
        paths, core_size=core_size
    )
    for entry, expected in zip(files, expected_counts):
        if int(entry["positive_count"]) != expected:
            raise AssertionError(
                f"{entry['file_name']} patch positives={entry['positive_count']} "
                f"!= manifest={expected}"
            )

    input_size = core_size + 2 * halo_size
    padded_shape = [
        ceil_to_multiple(input_size, horizontal_multiple),
        ceil_to_multiple(nray, horizontal_multiple),
        len(z),
    ]
    metadata: dict[str, Any] = {
        "format_version": PATCH_INDEX_FORMAT_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "dpr_positive_qc_3d_core_with_halo",
        "split": split,
        "input_variable": PATCH_INPUT_VARIABLE,
        "input_channels": list(PATCH_INPUT_CHANNELS),
        "label_variable": PATCH_LABEL_VARIABLE,
        "label_transform": "log1p",
        "loss_mask": "pre_positive_qc_in_core",
        "output_mask": "dpr_reflectivity_valid_above_cfb_in_core",
        "core_size": core_size,
        "halo_size": halo_size,
        "input_size": input_size,
        "horizontal_multiple": horizontal_multiple,
        "height_padding": 0,
        "unpadded_patch_shape": [input_size, nray, len(z)],
        "padded_patch_shape": padded_shape,
        "nray": nray,
        "z_size": len(z),
        "heights_km": [float(value) for value in z],
        "file_list": str(file_list.resolve()),
        "file_list_sha256": sha256_file(file_list),
        "split_manifest": str((splits_dir / "split_manifest.csv").resolve()),
        "split_manifest_sha256": sha256_file(splits_dir / "split_manifest.csv"),
        "file_count": processed_file_count,
        "patch_count": int(records.size),
        "positive_patch_count": int(np.count_nonzero(records["positive_count"])),
        "input_count": int(records["input_count"].sum()),
        "positive_count": int(records["positive_count"].sum()),
        "index_dtype": PATCH_INDEX_DTYPE.descr,
        "index_itemsize": PATCH_INDEX_DTYPE.itemsize,
        "files": files,
    }
    save_patch_index(index_path, metadata_path, records, metadata)
    print(
        f"{split}: files={len(files)}, patches={len(records):,}, "
        f"positive_patches={metadata['positive_patch_count']:,}, "
        f"shape={tuple(padded_shape)}, index={index_path}"
    )


def main() -> None:
    args = parse_args()
    manifest = load_split_manifest(args.splits_dir / "split_manifest.csv")
    selected = SPLIT_NAMES if args.split == "all" else (args.split,)
    for split in selected:
        build_one_split(
            split,
            splits_dir=args.splits_dir,
            output_dir=args.output_dir,
            core_size=args.core_size,
            halo_size=args.halo_size,
            horizontal_multiple=args.horizontal_multiple,
            overwrite=args.overwrite,
            manifest=manifest,
        )


if __name__ == "__main__":
    main()
