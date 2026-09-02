#!/usr/bin/env python3
"""Build fixed-core patch indices for stage-two GR-to-DPR experiments."""

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
from precipitation_inversion.data.patch_dataset import ceil_to_multiple  # noqa: E402
from precipitation_inversion.data.stage2_patch_dataset import (  # noqa: E402
    STAGE2_INPUT_CHANNELS,
    STAGE2_INPUT_VARIABLE,
    STAGE2_PATCH_INDEX_DTYPE,
    STAGE2_PATCH_INDEX_FORMAT_VERSION,
    STAGE2_TARGET_VARIABLE,
    build_stage2_patch_index_records,
    save_stage2_patch_index,
)


DEFAULT_SPLITS_DIR = PROJECT_ROOT / "metadata" / "splits"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "metadata" / "stage2_patch_indices"
SPLIT_NAMES = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build non-overlapping-core stage-two 3-D patch indices."
    )
    parser.add_argument("--split", choices=(*SPLIT_NAMES, "all"), default="all")
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--core-size", type=int, default=32)
    parser.add_argument("--halo-size", type=int, default=16)
    parser.add_argument("--horizontal-multiple", type=int, default=16)
    parser.add_argument("--strong-dbz-threshold", type=float, default=35.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_split_manifest(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"split manifest not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"file_path", "split", "sample_id"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"split manifest must contain {sorted(required)}")
        return {
            str(Path(row["file_path"]).expanduser().resolve()): row for row in reader
        }


def verify_split_paths(
    paths: list[Path], split: str, manifest: dict[str, dict[str, str]]
) -> None:
    for path in paths:
        row = manifest.get(str(path.resolve()))
        if row is None:
            raise ValueError(f"path absent from split manifest: {path}")
        if row["split"] != split:
            raise ValueError(f"{path.name} belongs to {row['split']}, not {split}")


def build_one_split(
    split: str,
    *,
    splits_dir: Path,
    output_dir: Path,
    core_size: int,
    halo_size: int,
    horizontal_multiple: int,
    strong_dbz_threshold: float,
    overwrite: bool,
    manifest: dict[str, dict[str, str]],
) -> dict[str, Any]:
    if core_size <= 0 or halo_size < 0 or horizontal_multiple <= 0:
        raise ValueError(
            "core-size/horizontal-multiple must be positive and halo non-negative"
        )
    if not np.isfinite(strong_dbz_threshold):
        raise ValueError("strong-dbz-threshold must be finite")
    file_list = splits_dir / f"{split}_files.txt"
    index_path = output_dir / f"{split}.npy"
    metadata_path = output_dir / f"{split}.json"
    existing = [path for path in (index_path, metadata_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "outputs already exist ("
            + ", ".join(path.name for path in existing)
            + "); use --overwrite"
        )
    paths = load_path_list(file_list)
    verify_split_paths(paths, split, manifest)
    records, files, z, nray, file_count = build_stage2_patch_index_records(
        paths,
        core_size=core_size,
        strong_dbz_threshold=strong_dbz_threshold,
    )
    input_size = core_size + 2 * halo_size
    padded_shape = [
        ceil_to_multiple(input_size, horizontal_multiple),
        ceil_to_multiple(nray, horizontal_multiple),
        len(z),
    ]
    count_names = STAGE2_PATCH_INDEX_DTYPE.names[3:]
    assert count_names is not None
    metadata: dict[str, Any] = {
        "format_version": STAGE2_PATCH_INDEX_FORMAT_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": 2,
        "task": "sparse_gr_to_dense_dpr_reflectivity_core_with_halo",
        "split": split,
        "input_variable": STAGE2_INPUT_VARIABLE,
        "input_channels": list(STAGE2_INPUT_CHANNELS),
        "target_variable": STAGE2_TARGET_VARIABLE,
        "target_transform": "per_height_standardization",
        "support_target": "dpr_value",
        "support_loss_mask": "occupancy_domain_in_core",
        "regression_loss_mask": "dpr_value_in_core",
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
        "strong_dbz_threshold": float(strong_dbz_threshold),
        "file_list": str(file_list.resolve()),
        "file_list_sha256": sha256_file(file_list),
        "split_manifest": str((splits_dir / "split_manifest.csv").resolve()),
        "split_manifest_sha256": sha256_file(splits_dir / "split_manifest.csv"),
        "file_count": file_count,
        "patch_count": int(records.size),
        "target_patch_count": int(np.count_nonzero(records["dpr_count"])),
        "strong_patch_count": int(np.count_nonzero(records["strong_dpr_count"])),
        "index_dtype": STAGE2_PATCH_INDEX_DTYPE.descr,
        "index_itemsize": STAGE2_PATCH_INDEX_DTYPE.itemsize,
        "files": files,
    }
    for name in count_names:
        metadata[name] = int(records[name].sum(dtype=np.uint64))
    save_stage2_patch_index(index_path, metadata_path, records, metadata)
    print(
        f"{split}: files={file_count}, patches={len(records):,}, "
        f"target_patches={metadata['target_patch_count']:,}, "
        f"shape={tuple(padded_shape)}, index={index_path}"
    )
    return metadata


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
            strong_dbz_threshold=args.strong_dbz_threshold,
            overwrite=args.overwrite,
            manifest=manifest,
        )


if __name__ == "__main__":
    main()
