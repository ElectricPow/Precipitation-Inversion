#!/usr/bin/env python3
"""Run the validation-only S2-R0 local finite-shift support audit.

This script never trains a model.  It searches finite integer shifts with the
true DPR support and therefore produces an oracle diagnostic, not an inference
feature.  Every candidate is evaluated inside the same Stage-2 occupancy label
domain and on the same fixed target cells.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from netCDF4 import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from precipitation_inversion.data.masks import to_float_array  # noqa: E402
from precipitation_inversion.data.stage2_masks import (  # noqa: E402
    build_stage2_spatial_masks,
)
from precipitation_inversion.metrics.stage2_local_shift import (  # noqa: E402
    LocalShiftOptions,
    aggregate_local_shift_audits,
    audit_orbit_local_shifts,
)
from precipitation_inversion.metrics.stage2_reflectivity import (  # noqa: E402
    finite_metrics_for_json,
)


FORMAT = "stage2_r0_local_shift_audit_v1"
SAMPLE_ID_HASH_CONTRACT = "sha256(compact-utf8-json-ordered-sample-ids)"
DEFAULT_MANIFEST = PROJECT_ROOT / "metadata/splits/split_manifest.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/stage2_r0_local_shift_audit"


@dataclass(frozen=True)
class OrbitTask:
    path: Path
    sample_id: str
    options: LocalShiftOptions


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ordered_sample_ids_sha256(sample_ids: Sequence[str]) -> str:
    payload = json.dumps(
        [str(value) for value in sample_ids],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--count",
        type=int,
        help="Validation-orbit limit for a smoke test; the result is non-formal.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--window-scan", type=int, default=64)
    parser.add_argument("--window-ray", type=int, default=49)
    parser.add_argument("--max-shift", type=int, default=2)
    parser.add_argument("--min-dpr-support", type=int, default=1)
    parser.add_argument("--min-gr-support", type=int, default=1)
    parser.add_argument(
        "--drop-partial-windows",
        action="store_true",
        help="Drop final incomplete target windows instead of retaining them.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_validation_tasks(
    manifest_path: Path,
    *,
    options: LocalShiftOptions,
    count: int | None,
) -> tuple[list[OrbitTask], int]:
    """Load deterministic validation-only tasks and the full val orbit count."""

    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if count is not None and count <= 0:
        raise ValueError("count must be positive")
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "file_path", "split"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"split manifest must contain columns {sorted(required)}"
            )
        rows = [dict(row) for row in reader if row["split"] == "val"]
    rows.sort(key=lambda row: row["sample_id"])
    expected_count = len(rows)
    if expected_count == 0:
        raise ValueError("split manifest contains no validation orbits")
    if count is not None:
        rows = rows[:count]

    tasks: list[OrbitTask] = []
    for row in rows:
        path = Path(row["file_path"]).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        tasks.append(OrbitTask(path=path, sample_id=row["sample_id"], options=options))
    return tasks, expected_count


def analyze_task(task: OrbitTask) -> dict[str, Any]:
    """Read one orbit and apply the exact Stage-2 mask contract."""

    with Dataset(task.path, "r") as dataset:
        required = {"z", "dbz_gr_sparse", "dbz_dpr", "pre_dpr"}
        missing = sorted(required.difference(dataset.variables))
        if missing:
            raise KeyError(f"{task.path.name} missing variables: {', '.join(missing)}")
        heights = to_float_array(dataset["z"][:])
        gr_raw = dataset["dbz_gr_sparse"][:]
        dpr_raw = dataset["dbz_dpr"][:]
        pre_raw = dataset["pre_dpr"][:]
    if gr_raw.shape != dpr_raw.shape or gr_raw.shape != pre_raw.shape:
        raise ValueError(f"{task.path.name} Stage-2 source/target shapes differ")
    if gr_raw.ndim != 3 or gr_raw.shape[-1] != heights.size:
        raise ValueError(f"{task.path.name} does not use (nscan,nray,z)")

    masks = build_stage2_spatial_masks(gr_raw, dpr_raw, pre_dpr=pre_raw)
    occupancy = masks["occupancy_domain"]
    # Direct support means a physical (non-sentinel, non-native-missing) dBZ
    # value.  Both supports are restricted to the same trustworthy Stage-2
    # occurrence-label domain before any candidate shift is evaluated.
    gr_support = occupancy & masks["gr_value"]
    dpr_support = occupancy & masks["dpr_value"]
    return audit_orbit_local_shifts(
        gr_support,
        dpr_support,
        occupancy,
        heights,
        sample_id=task.sample_id,
        file_name=task.path.name,
        options=task.options,
    )


def run_tasks(tasks: Sequence[OrbitTask], *, workers: int) -> list[dict[str, Any]]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    if workers == 1:
        return [analyze_task(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(analyze_task, tasks))


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(
            finite_metrics_for_json(value),
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
    os.replace(temporary, path)


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(finite_metrics_for_json(list(rows)))
    temporary.replace(path)


def prepare_output_directory(output_dir: Path, *, overwrite: bool) -> None:
    protected = (
        "summary.json",
        "local_window_height.csv",
        "per_orbit_height.csv",
        "per_orbit_height_shift.csv",
        "per_orbit.csv",
        "per_orbit_shift.csv",
        "best_shift_histogram.csv",
        "all_validation_shift.csv",
        "report.md",
    )
    existing = [output_dir / name for name in protected if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "local-shift audit outputs already exist; pass --overwrite: "
            + ", ".join(path.name for path in existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def _format(value: Any, digits: int = 5) -> str:
    if not isinstance(value, (int, float)) or not np.isfinite(float(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def build_report(summary: Mapping[str, Any]) -> str:
    metrics = summary["metrics"]
    gain = metrics["window_csi_gain_distribution"]
    return "\n".join(
        [
            "# S2-R0 局部有限位移审计",
            "",
            f"- 验证轨道：{summary['selected_file_count']}/"
            f"{summary['expected_validation_file_count']}",
            f"- 正式完整验证：`{summary['formal_result']}`",
            f"- 有效窗口×高度：{metrics['valid_window_height_count']}/"
            f"{metrics['planned_window_height_count']}",
            f"- exact (0,0) pooled CSI："
            f"{_format(metrics['exact_support_csi'])}",
            f"- 全验证单一位移 oracle CSI："
            f"{_format(metrics['one_shift_all_validation_oracle_support_csi'])}",
            f"- 逐轨单一位移 oracle CSI："
            f"{_format(metrics['per_orbit_single_shift_oracle_support_csi'])}",
            f"- 逐轨逐高度 oracle CSI："
            f"{_format(metrics['per_orbit_height_oracle_support_csi'])}",
            f"- 局部窗口逐高度 oracle CSI："
            f"{_format(metrics['local_window_height_oracle_support_csi'])}",
            f"- 局部窗口CSI增益中位数 / P90："
            f"{_format(gain['median'])} / {_format(gain['p90'])}",
            f"- 同轨出现相反局部位移："
            f"{metrics['opposing_local_shift_orbit_count']}/"
            f"{metrics['orbit_count']}",
            f"- 存在抵消证据的轨道："
            f"{metrics['cancellation_evidence_orbit_count']}/"
            f"{metrics['orbit_count']}",
            "",
            "## 边界",
            "",
            "所有best shift均使用真实DPR标签事后搜索，只是有限位移oracle上限，"
            "不得作为模型输入、训练标签或部署时配准量。候选shift始终使用相同固定"
            "目标occupancy domain；没有跨轨聚合后再按高度搜索局部位移。",
            "",
        ]
    )


def write_outputs(
    output_dir: Path,
    aggregate: Mapping[str, Any],
    *,
    manifest_path: Path,
    tasks: Sequence[OrbitTask],
    expected_count: int,
) -> dict[str, Any]:
    complete = len(tasks) == expected_count
    options = tasks[0].options
    ordered_sample_ids = [task.sample_id for task in tasks]
    summary = {
        "format": FORMAT,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "S2-R0-LocalFiniteShiftAudit",
        "split": "val",
        "test_set_accessed": False,
        "formal_result": complete,
        "selected_file_count": len(tasks),
        "expected_validation_file_count": expected_count,
        "split_manifest": str(manifest_path.resolve()),
        "split_manifest_sha256": _sha256_file(manifest_path.resolve()),
        "sample_ids": ordered_sample_ids,
        "sample_ids_sha256": _ordered_sample_ids_sha256(ordered_sample_ids),
        "sample_id_hash_contract": SAMPLE_ID_HASH_CONTRACT,
        "configuration": {
            "window_scan": options.window_scan,
            "window_ray": options.window_ray,
            "max_shift": options.max_shift,
            "min_dpr_support": options.min_dpr_support,
            "min_gr_support": options.min_gr_support,
            "include_partial_windows": options.include_partial_windows,
            "shift_direction": (
                "positive shift compares GR(i,j,z) with "
                "DPR(i+scan_shift,j+ray_shift,z)"
            ),
            "fixed_domain": (
                "target occupancy_domain is fixed across all shifts; direct GR/DPR "
                "support is physical dBZ value AND occupancy_domain"
            ),
            "window_contract": (
                "non-overlapping target windows; source may cross an internal window "
                "boundary; only the global max_shift border is excluded"
            ),
        },
        "metrics": aggregate["summary"],
        "interpretation_boundary": (
            "DPR labels choose every oracle shift after the fact. Results quantify a "
            "finite local displacement upper bound and cannot be used as input."
        ),
    }
    _atomic_json(output_dir / "summary.json", summary)
    _atomic_csv(output_dir / "local_window_height.csv", aggregate["window_height_rows"])
    _atomic_csv(output_dir / "per_orbit_height.csv", aggregate["per_height_rows"])
    _atomic_csv(
        output_dir / "per_orbit_height_shift.csv",
        aggregate["orbit_height_shift_rows"],
    )
    _atomic_csv(output_dir / "per_orbit.csv", aggregate["orbit_rows"])
    _atomic_csv(output_dir / "per_orbit_shift.csv", aggregate["orbit_shift_rows"])
    _atomic_csv(output_dir / "best_shift_histogram.csv", aggregate["histogram_rows"])
    _atomic_csv(
        output_dir / "all_validation_shift.csv",
        aggregate["all_validation_shift_rows"],
    )
    temporary = output_dir / "report.md.partial"
    temporary.write_text(build_report(summary), encoding="utf-8")
    temporary.replace(output_dir / "report.md")
    return summary


def main() -> None:
    args = parse_args()
    options = LocalShiftOptions(
        window_scan=args.window_scan,
        window_ray=args.window_ray,
        max_shift=args.max_shift,
        min_dpr_support=args.min_dpr_support,
        min_gr_support=args.min_gr_support,
        include_partial_windows=not args.drop_partial_windows,
    )
    manifest_path = args.split_manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    tasks, expected_count = load_validation_tasks(
        manifest_path, options=options, count=args.count
    )
    prepare_output_directory(output_dir, overwrite=args.overwrite)
    audits = run_tasks(tasks, workers=args.workers)
    aggregate = aggregate_local_shift_audits(audits)
    summary = write_outputs(
        output_dir,
        aggregate,
        manifest_path=manifest_path,
        tasks=tasks,
        expected_count=expected_count,
    )
    print(
        "S2-R0 local finite-shift audit: "
        f"val={summary['selected_file_count']}/"
        f"{summary['expected_validation_file_count']} "
        f"formal={summary['formal_result']} -> {output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
