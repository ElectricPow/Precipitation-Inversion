#!/usr/bin/env python3
"""Create leakage-safe train/validation/test lists from dataset_manifest.csv.

The default strategy keeps every date in exactly one split and searches for a
70/15/15 partition that balances coverage, positive rain, heavy-rain tails, and
precipitation types.  Source NetCDF files and the input manifest are read-only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from precipitation_inversion.data.splits import (  # noqa: E402
    DEFAULT_BALANCE_WEIGHTS,
    assert_valid_split,
    balanced_group_split,
    chronological_group_split,
)


DEFAULT_MANIFEST = PROJECT_ROOT / "metadata" / "manifests" / "dataset_manifest.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "metadata" / "splits"
OUTPUT_SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create grouped train/validation/test file lists from a manifest."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--strategy",
        choices=("balanced", "chronological"),
        default="balanced",
        help="Balanced random search or consecutive chronological groups.",
    )
    parser.add_argument(
        "--group-field",
        default="date",
        help="Manifest field kept wholly within one split (default: date).",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument(
        "--trials",
        type=int,
        default=10_000,
        help="Candidate balanced partitions to evaluate (default: 10000).",
    )
    parser.add_argument(
        "--path-mode",
        choices=("absolute", "name"),
        default="absolute",
        help="Write absolute paths or basenames to split text files.",
    )
    parser.add_argument(
        "--allow-warnings",
        action="store_true",
        help="Include manifest rows whose audit status is 'warning'.",
    )
    parser.add_argument(
        "--skip-file-check",
        action="store_true",
        help="Do not verify that every manifest file_path currently exists.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing split files in --output-dir.",
    )
    return parser.parse_args()


def load_manifest(path: Path, *, allow_warnings: bool) -> tuple[list[dict[str, str]], list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {path}")
        records = list(reader)
        fieldnames = list(reader.fieldnames)
    if not records:
        raise ValueError(f"Manifest contains no records: {path}")

    unsupported = sorted(
        {
            record.get("status", "")
            for record in records
            if record.get("status", "") not in {"ok", "warning"}
        }
    )
    if unsupported:
        raise ValueError(f"Manifest contains unsupported status values: {unsupported}")
    warnings = [record["file_name"] for record in records if record.get("status") == "warning"]
    if warnings and not allow_warnings:
        raise ValueError(
            f"Manifest contains {len(warnings)} warning row(s); inspect the audit or "
            "use --allow-warnings"
        )
    return records, fieldnames


def verify_source_files(records: Iterable[Mapping[str, str]]) -> None:
    missing = [record["file_path"] for record in records if not Path(record["file_path"]).is_file()]
    if missing:
        preview = ", ".join(missing[:3])
        raise FileNotFoundError(
            f"{len(missing)} source NetCDF file(s) are missing; first entries: {preview}"
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def number(record: Mapping[str, str], field: str) -> float:
    value = record.get(field, "")
    return float(value) if value not in {"", None} else 0.0


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = [output_dir / f"{split}_files.txt" for split in OUTPUT_SPLITS]
    targets.extend(
        [output_dir / "split_manifest.csv", output_dir / "split_summary.json"]
    )
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Split outputs already exist ("
            + ", ".join(path.name for path in existing)
            + "); use --overwrite to replace them"
        )


def atomic_text_write(path: Path, text: str) -> None:
    partial = path.with_name(path.name + ".partial")
    partial.write_text(text, encoding="utf-8")
    partial.replace(path)


def atomic_csv_write(
    path: Path, fieldnames: list[str], rows: Iterable[Mapping[str, Any]]
) -> None:
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    partial.replace(path)


def atomic_json_write(path: Path, value: Mapping[str, Any]) -> None:
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    partial.replace(path)


def records_by_split(
    records: list[dict[str, str]], assignments: Mapping[str, str]
) -> dict[str, list[dict[str, str]]]:
    grouped = {split: [] for split in OUTPUT_SPLITS}
    for record in records:
        split = assignments[record["sample_id"]]
        grouped[split].append(record)
    for split_records in grouped.values():
        split_records.sort(
            key=lambda record: (
                record.get("date", ""),
                record.get("start_time", ""),
                record.get("file_name", ""),
            )
        )
    return grouped


def split_statistics(
    records: list[dict[str, str]],
    *,
    group_field: str,
    all_records: list[dict[str, str]],
) -> dict[str, Any]:
    feature_totals = {
        field: sum(number(record, field) for record in all_records)
        for field in DEFAULT_BALANCE_WEIGHTS
        if field != "file_count"
    }
    groups = sorted({record[group_field] for record in records})
    totals = {
        "file_count": len(records),
        **{
            field: int(sum(number(record, field) for record in records))
            for field in DEFAULT_BALANCE_WEIGHTS
            if field != "file_count"
        },
    }
    pre_valid = int(sum(number(record, "pre_valid_count") for record in records))
    pre_zero = int(sum(number(record, "pre_zero_count") for record in records))
    gr_dpr_overlap = int(
        sum(number(record, "gr_dpr_overlap_count") for record in records)
    )
    dpr_valid = int(sum(number(record, "dpr_dbz_valid_count") for record in records))
    feature_shares = {
        "file_count": len(records) / len(all_records),
        **{
            field: (totals[field] / total if total else None)
            for field, total in feature_totals.items()
        },
    }
    return {
        "group_count": len(groups),
        "groups": groups,
        "first_group": groups[0],
        "last_group": groups[-1],
        "totals": totals,
        "feature_shares": feature_shares,
        "pre_zero_ratio_among_valid": pre_zero / pre_valid if pre_valid else None,
        "gr_dpr_overlap_ratio_of_dpr": (
            gr_dpr_overlap / dpr_valid if dpr_valid else None
        ),
    }


def build_summary(
    *,
    manifest_path: Path,
    records: list[dict[str, str]],
    grouped_records: Mapping[str, list[dict[str, str]]],
    result,
    requested_ratios: Mapping[str, float],
    path_mode: str,
) -> dict[str, Any]:
    group_sets = {
        split: {record[result.group_field] for record in split_records}
        for split, split_records in grouped_records.items()
    }
    no_group_overlap = all(
        group_sets[first].isdisjoint(group_sets[second])
        for index, first in enumerate(OUTPUT_SPLITS)
        for second in OUTPUT_SPLITS[index + 1 :]
    )
    normalized = dict(zip(result.split_names, result.ratios))
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(manifest_path.resolve()),
        "source_manifest_sha256": sha256_file(manifest_path),
        "strategy": result.strategy,
        "group_field": result.group_field,
        "seed": result.seed,
        "trials": result.trials,
        "path_mode": path_mode,
        "requested_ratios": dict(requested_ratios),
        "normalized_ratios": normalized,
        "balance_weights": dict(DEFAULT_BALANCE_WEIGHTS),
        "objective_score": result.score,
        "record_count": len(records),
        "group_count": len(result.group_assignments),
        "checks": {
            "all_records_assigned_once": len(result.record_assignments) == len(records),
            "no_group_overlap": no_group_overlap,
            "all_splits_nonempty": all(grouped_records[split] for split in OUTPUT_SPLITS),
        },
        "splits": {
            split: {
                "target_ratio": normalized[split],
                **split_statistics(
                    split_records,
                    group_field=result.group_field,
                    all_records=records,
                ),
            }
            for split, split_records in grouped_records.items()
        },
    }


def main() -> None:
    args = parse_args()
    requested_ratios = {
        "train": args.train_ratio,
        "val": args.val_ratio,
        "test": args.test_ratio,
    }
    records, manifest_fields = load_manifest(
        args.manifest, allow_warnings=args.allow_warnings
    )
    if args.group_field not in manifest_fields:
        raise KeyError(f"Manifest has no group field {args.group_field!r}")
    required_fields = {
        "sample_id",
        "file_name",
        "file_path",
        args.group_field,
        *[field for field in DEFAULT_BALANCE_WEIGHTS if field != "file_count"],
    }
    missing_fields = sorted(required_fields.difference(manifest_fields))
    if missing_fields:
        raise KeyError("Manifest is missing required fields: " + ", ".join(missing_fields))
    if not args.skip_file_check:
        verify_source_files(records)

    if args.strategy == "balanced":
        result = balanced_group_split(
            records,
            ratios=requested_ratios,
            group_field=args.group_field,
            balance_weights=DEFAULT_BALANCE_WEIGHTS,
            seed=args.seed,
            trials=args.trials,
        )
    else:
        result = chronological_group_split(
            records,
            ratios=requested_ratios,
            group_field=args.group_field,
            balance_weights=DEFAULT_BALANCE_WEIGHTS,
        )
    assert_valid_split(records, result)
    grouped_records = records_by_split(records, result.record_assignments)
    prepare_output_dir(args.output_dir, args.overwrite)

    for split, split_records in grouped_records.items():
        values = [
            record["file_path"] if args.path_mode == "absolute" else record["file_name"]
            for record in split_records
        ]
        atomic_text_write(
            args.output_dir / f"{split}_files.txt", "\n".join(values) + "\n"
        )

    split_manifest_rows = [
        {**record, "split": result.record_assignments[record["sample_id"]]}
        for record in records
    ]
    split_manifest_rows.sort(key=lambda row: (row["split"], row["date"], row["file_name"]))
    atomic_csv_write(
        args.output_dir / "split_manifest.csv",
        [*manifest_fields, "split"],
        split_manifest_rows,
    )
    summary = build_summary(
        manifest_path=args.manifest,
        records=records,
        grouped_records=grouped_records,
        result=result,
        requested_ratios=requested_ratios,
        path_mode=args.path_mode,
    )
    atomic_json_write(args.output_dir / "split_summary.json", summary)

    print(f"Strategy: {result.strategy}")
    print(f"Group field: {result.group_field}")
    print(f"Objective score: {result.score:.8f}")
    for split in OUTPUT_SPLITS:
        stats = summary["splits"][split]
        print(
            f"{split:>5}: files={stats['totals']['file_count']:3d}, "
            f"groups={stats['group_count']:3d}, "
            f"positive_share={stats['feature_shares']['pre_positive_count']:.3%}, "
            f">20_share={stats['feature_shares']['pre_gt_20_count']:.3%}, "
            f"convective_share={stats['feature_shares']['convective_profile_count']:.3%}"
        )
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()

