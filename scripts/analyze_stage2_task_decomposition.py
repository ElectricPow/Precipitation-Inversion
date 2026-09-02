#!/usr/bin/env python3
"""Assemble the validation-only S2-R0 task decomposition audit.

This script performs no neural-network training.  It converts an existing
complete Stage-2 validation evaluation into an explicit error budget, joins
the existing finite-shift GR/DPR alignment oracle, and optionally attaches the
region-by-region frozen-Stage-1 oracle-replacement audit produced by
``evaluate_stage2_stage1_cascade.py --r0-decomposition-oracles``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for value in (PROJECT_ROOT, SRC_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from precipitation_inversion.metrics.stage2_reflectivity import (  # noqa: E402
    finite_metrics_for_json,
)


R0_STATIC_AUDIT_FORMAT = "stage2_r0_static_decomposition_v1"
R0_ORACLE_AUDIT_FORMAT = "stage2_r0_decomposition_oracle_v1"
R0_DIAGNOSTICS_FORMAT = "stage2_r0_decomposition_diagnostics_v1"
R0_LOCAL_SHIFT_FORMAT = "stage2_r0_local_shift_audit_v1"
R0_SAMPLE_ID_HASH_CONTRACT = "sha256(compact-utf8-json-ordered-sample-ids)"
REQUIRED_REGION_NAMES = (
    "all_domain",
    "q11_direct_overlap",
    "q01_direct_missing",
    "q10_gr_only",
    "q00_neither",
    "dpr_gap_proxy",
    "dpr_outside_proxy",
    "dpr_dbz_ge35",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage2-evaluation-dir",
        type=Path,
        required=True,
        help="Directory containing metrics.json and per_region.csv from complete val.",
    )
    parser.add_argument(
        "--patch-index",
        type=Path,
        default=PROJECT_ROOT / "metadata/stage2_patch_indices/val.json",
    )
    parser.add_argument(
        "--alignment-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/stage2_alignment_audit",
    )
    parser.add_argument(
        "--oracle-audit",
        type=Path,
        help="Optional r0_decomposition_oracles.json from the frozen cascade run.",
    )
    parser.add_argument(
        "--local-shift-audit",
        type=Path,
        help="Optional summary.json from analyze_stage2_local_shift_audit.py.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Permit a smoke-test subset; never use its results for model decisions.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV has no data rows: {path}")
    return rows


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


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _verified_file_record(
    record: Mapping[str, Any],
    *,
    path_key: str,
    sha256_key: str,
    name: str,
) -> tuple[Path, str]:
    raw_path = record.get(path_key)
    recorded_hash = record.get(sha256_key)
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{name}.{path_key} must be a non-empty path string")
    if (
        not isinstance(recorded_hash, str)
        or len(recorded_hash) != 64
        or any(character not in "0123456789abcdef" for character in recorded_hash)
    ):
        raise ValueError(f"{name}.{sha256_key} must be a lowercase SHA256")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{name} file not found: {path}")
    actual_hash = _sha256_file(path)
    if actual_hash != recorded_hash:
        raise ValueError(f"{name} SHA256 differs from the recorded provenance")
    return path, actual_hash


def _number(row: Mapping[str, Any], name: str, *, integer: bool = False) -> int | float:
    if name not in row or row[name] in (None, ""):
        return 0 if integer else math.nan
    value = float(row[name])
    if not math.isfinite(value):
        return 0 if integer else math.nan
    return int(round(value)) if integer else value


def validate_complete_validation_evaluation(
    metrics: Mapping[str, Any],
    patch_index: Mapping[str, Any],
    *,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    """Reject test-set or truncated inputs before scientific interpretation."""

    if metrics.get("stage") != 2:
        raise ValueError("R0 requires a Stage-2 metrics.json")
    if metrics.get("split") != "val":
        raise ValueError("S2-R0 may only consume split=val")
    if patch_index.get("stage") != 2 or patch_index.get("split") != "val":
        raise ValueError("R0 patch index must be Stage-2 split=val")
    evaluated_files = int(metrics.get("file_count", -1))
    expected_files = int(patch_index.get("file_count", len(patch_index.get("files", ()))))
    complete = evaluated_files == expected_files and expected_files > 0
    if not complete and not allow_incomplete:
        raise ValueError(
            f"Stage-2 evaluation covers {evaluated_files}/{expected_files} validation "
            "orbits; pass --allow-incomplete only for a smoke test"
        )
    return {
        "split": "val",
        "evaluated_file_count": evaluated_files,
        "expected_file_count": expected_files,
        "complete_validation": complete,
        "checkpoint": metrics.get("checkpoint"),
        "checkpoint_epoch": metrics.get("checkpoint_epoch"),
        "support_threshold": metrics.get("support_threshold"),
    }


def validate_r0_local_shift_audit(
    payload: Mapping[str, Any],
    patch_index: Mapping[str, Any],
    *,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    """Verify that a local-shift oracle covers the intended validation orbits."""

    if payload.get("format") != R0_LOCAL_SHIFT_FORMAT:
        raise ValueError("local-shift audit has an unsupported format")
    if payload.get("split") != "val" or payload.get("test_set_accessed") is not False:
        raise ValueError("local-shift audit must be validation-only")
    selected = payload.get("selected_file_count")
    expected = payload.get("expected_validation_file_count")
    formal = payload.get("formal_result")
    if (
        isinstance(selected, bool)
        or not isinstance(selected, int)
        or selected < 0
        or isinstance(expected, bool)
        or not isinstance(expected, int)
        or expected <= 0
        or not isinstance(formal, bool)
    ):
        raise ValueError("local-shift coverage metadata is invalid")
    patch_files = patch_index.get("files")
    if not isinstance(patch_files, list) or not patch_files:
        raise ValueError("Stage-2 patch index must contain validation files")
    patch_ids = [
        str(_mapping(entry, name="patch-index file").get("sample_id"))
        for entry in patch_files
    ]
    if any(value in {"", "None"} for value in patch_ids):
        raise ValueError("Stage-2 patch index contains an invalid sample_id")
    patch_expected = int(patch_index.get("file_count", len(patch_ids)))
    if expected != patch_expected or patch_expected != len(patch_ids):
        raise ValueError("local-shift expected count differs from Stage-2 val index")
    raw_ids = payload.get("sample_ids")
    if not isinstance(raw_ids, list) or any(
        not isinstance(value, str) or not value for value in raw_ids
    ):
        raise ValueError("local-shift sample_ids must be an ordered string array")
    if selected != len(raw_ids) or raw_ids != patch_ids[:selected]:
        raise ValueError("local-shift ordered sample IDs differ from Stage-2 val index")
    if payload.get("sample_id_hash_contract") != R0_SAMPLE_ID_HASH_CONTRACT:
        raise ValueError("local-shift sample-ID hash contract is unsupported")
    sample_hash = _ordered_sample_ids_sha256(raw_ids)
    if payload.get("sample_ids_sha256") != sample_hash:
        raise ValueError("local-shift sample_ids_sha256 is inconsistent")
    manifest_raw = payload.get("split_manifest")
    manifest_hash = payload.get("split_manifest_sha256")
    if not isinstance(manifest_raw, str) or not manifest_raw:
        raise ValueError("local-shift split_manifest path is missing")
    manifest_path = Path(manifest_raw).expanduser().resolve()
    if not manifest_path.is_file() or manifest_hash != _sha256_file(manifest_path):
        raise ValueError("local-shift split manifest provenance is invalid")
    complete = selected == expected
    if formal != complete:
        raise ValueError("local-shift formal_result is inconsistent with coverage")
    if not complete and not allow_incomplete:
        raise ValueError(
            f"local-shift audit covers {selected}/{expected} validation orbits; "
            "pass --allow-incomplete only for a smoke test"
        )
    metrics = _mapping(payload.get("metrics"), name="local-shift metrics")
    required_metrics = (
        "exact_support_csi",
        "one_shift_all_validation_oracle_support_csi",
        "per_orbit_height_oracle_support_csi",
        "local_window_height_oracle_support_csi",
        "orbit_count",
        "opposing_local_shift_orbit_count",
        "cancellation_evidence_orbit_count",
    )
    if any(name not in metrics for name in required_metrics):
        raise ValueError("local-shift metrics are incomplete")
    if metrics["orbit_count"] != selected:
        raise ValueError("local-shift metric orbit_count differs from coverage")
    return {
        "format": R0_LOCAL_SHIFT_FORMAT,
        "evaluated_file_count": selected,
        "expected_file_count": expected,
        "complete_validation": complete,
        "accepted_as_formal": bool(complete and formal and not allow_incomplete),
        "sample_ids_sha256": sample_hash,
    }


def validate_r0_oracle_audit(
    oracle: Mapping[str, Any],
    stage2_metrics: Mapping[str, Any],
    patch_index: Mapping[str, Any],
    *,
    patch_index_path: Path | None = None,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    """Validate a self-contained R0 oracle payload before it is attached.

    A filename is not evidence of scientific provenance.  This validator
    checks the JSON schema, ordered orbit identity, on-disk checkpoint hashes,
    validation-selected threshold, and the exact Stage-2 checkpoint used by
    the static reflectivity evaluation.  A prefix smoke subset is accepted
    only with ``allow_incomplete=True`` and is always returned as non-formal.
    """

    format_name = oracle.get("format")
    if format_name == R0_DIAGNOSTICS_FORMAT:
        raise ValueError(
            "oracle audit points to the diagnostics schema, not the oracle schema"
        )
    if format_name != R0_ORACLE_AUDIT_FORMAT:
        raise ValueError("oracle audit has an unsupported format")
    provenance = _mapping(oracle.get("provenance"), name="oracle.provenance")
    required = {
        "split",
        "file_count",
        "expected_file_count",
        "formal_complete_validation",
        "sample_ids",
        "sample_ids_sha256",
        "sample_id_hash_contract",
        "stage1",
        "stage2_runs",
    }
    missing = sorted(required.difference(provenance))
    if missing:
        raise ValueError("oracle provenance is missing fields: " + ", ".join(missing))
    if provenance["split"] != "val":
        raise ValueError("R0 oracle provenance must use split=val")

    def strict_count(name: str) -> int:
        value = provenance[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"oracle provenance {name} must be a non-negative integer")
        return value

    file_count = strict_count("file_count")
    declared_expected = strict_count("expected_file_count")
    declared_formal = provenance["formal_complete_validation"]
    if not isinstance(declared_formal, bool):
        raise ValueError("oracle formal_complete_validation must be boolean")

    raw_sample_ids = provenance["sample_ids"]
    if isinstance(raw_sample_ids, (str, bytes)) or not isinstance(
        raw_sample_ids, list
    ):
        raise ValueError("oracle sample_ids must be an ordered JSON array")
    if any(not isinstance(value, str) or not value for value in raw_sample_ids):
        raise ValueError("oracle sample_ids must contain non-empty strings")
    sample_ids = list(raw_sample_ids)
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("oracle sample_ids must be unique")
    if file_count != len(sample_ids):
        raise ValueError("oracle file_count differs from sample_ids length")
    if provenance["sample_id_hash_contract"] != R0_SAMPLE_ID_HASH_CONTRACT:
        raise ValueError("oracle sample-ID hash contract is unsupported")
    expected_sample_hash = _ordered_sample_ids_sha256(sample_ids)
    if provenance["sample_ids_sha256"] != expected_sample_hash:
        raise ValueError("oracle sample_ids_sha256 is inconsistent")

    patch_files = patch_index.get("files")
    if not isinstance(patch_files, list) or not patch_files:
        raise ValueError("Stage-2 patch index must contain a non-empty files array")
    expected_sample_ids: list[str] = []
    for position, entry in enumerate(patch_files):
        if not isinstance(entry, Mapping) or not isinstance(entry.get("sample_id"), str):
            raise ValueError(f"patch index file {position} has no sample_id")
        expected_sample_ids.append(str(entry["sample_id"]))
    expected_file_count = int(
        patch_index.get("file_count", len(expected_sample_ids))
    )
    if expected_file_count != len(expected_sample_ids) or declared_expected != expected_file_count:
        raise ValueError("oracle expected_file_count differs from the validation index")
    if file_count > expected_file_count:
        raise ValueError("oracle contains more orbits than the validation index")
    # The cascade evaluator's --max-files contract selects a deterministic
    # prefix.  Requiring exact order prevents equal-count foreign splits from
    # masquerading as this validation set.
    if sample_ids != expected_sample_ids[:file_count]:
        raise ValueError("oracle ordered sample IDs differ from the validation index")
    complete = file_count == expected_file_count
    if declared_formal != complete:
        raise ValueError(
            "oracle formal_complete_validation is inconsistent with orbit coverage"
        )
    if not complete and not allow_incomplete:
        raise ValueError(
            f"oracle audit covers {file_count}/{expected_file_count} validation "
            "orbits; pass --allow-incomplete only for a smoke test"
        )

    stage1 = _mapping(provenance["stage1"], name="oracle.provenance.stage1")
    stage1_path, stage1_hash = _verified_file_record(
        stage1,
        path_key="checkpoint_path",
        sha256_key="checkpoint_sha256",
        name="oracle Stage-1 checkpoint",
    )
    if isinstance(stage1.get("checkpoint_epoch"), bool) or not isinstance(
        stage1.get("checkpoint_epoch"), int
    ):
        raise ValueError("oracle Stage-1 checkpoint_epoch must be an integer")
    _verified_file_record(
        stage1,
        path_key="index_path",
        sha256_key="index_sha256",
        name="oracle Stage-1 index",
    )

    static_checkpoint_raw = stage2_metrics.get("checkpoint")
    if not isinstance(static_checkpoint_raw, str) or not static_checkpoint_raw:
        raise ValueError("static Stage-2 metrics has no checkpoint path")
    static_checkpoint = Path(static_checkpoint_raw).expanduser().resolve()
    if not static_checkpoint.is_file():
        raise FileNotFoundError(f"static Stage-2 checkpoint not found: {static_checkpoint}")
    static_checkpoint_hash = _sha256_file(static_checkpoint)
    static_threshold = stage2_metrics.get("support_threshold")
    if isinstance(static_threshold, bool) or not isinstance(
        static_threshold, (int, float)
    ) or not math.isfinite(float(static_threshold)):
        raise ValueError("static Stage-2 support_threshold is invalid")

    expected_patch_hash = (
        _sha256_file(patch_index_path.expanduser().resolve())
        if patch_index_path is not None
        else None
    )
    stage2_runs = _mapping(
        provenance["stage2_runs"], name="oracle.provenance.stage2_runs"
    )
    audit_runs = _mapping(oracle.get("runs"), name="oracle.runs")
    if not stage2_runs or set(stage2_runs) != set(audit_runs):
        raise ValueError("oracle provenance Stage-2 runs differ from oracle metric runs")

    matching_runs: list[str] = []
    for slug, raw_record in stage2_runs.items():
        record = _mapping(raw_record, name=f"oracle Stage-2 run {slug}")
        checkpoint_path, checkpoint_hash = _verified_file_record(
            record,
            path_key="checkpoint_path",
            sha256_key="checkpoint_sha256",
            name=f"oracle Stage-2 run {slug} checkpoint",
        )
        if isinstance(record.get("checkpoint_epoch"), bool) or not isinstance(
            record.get("checkpoint_epoch"), int
        ):
            raise ValueError(f"oracle Stage-2 run {slug} epoch must be an integer")
        run_threshold = record.get("support_threshold")
        if isinstance(run_threshold, bool) or not isinstance(
            run_threshold, (int, float)
        ) or not math.isfinite(float(run_threshold)):
            raise ValueError(f"oracle Stage-2 run {slug} threshold is invalid")
        threshold_path, _ = _verified_file_record(
            record,
            path_key="threshold_file_path",
            sha256_key="threshold_file_sha256",
            name=f"oracle Stage-2 run {slug} threshold file",
        )
        threshold_payload = _load_json(threshold_path)
        if threshold_payload.get("selected_on_split") != "val" or not math.isclose(
            float(threshold_payload.get("threshold", math.nan)),
            float(run_threshold),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(f"oracle Stage-2 run {slug} threshold provenance differs")
        _, index_hash = _verified_file_record(
            record,
            path_key="index_path",
            sha256_key="index_sha256",
            name=f"oracle Stage-2 run {slug} index",
        )
        if expected_patch_hash is not None and index_hash != expected_patch_hash:
            raise ValueError(f"oracle Stage-2 run {slug} uses a different patch index")
        run_metrics = _mapping(audit_runs[slug], name=f"oracle.runs.{slug}")
        if run_metrics.get("label") != record.get("label"):
            raise ValueError(f"oracle Stage-2 run {slug} label differs from provenance")
        if checkpoint_path == static_checkpoint and checkpoint_hash == static_checkpoint_hash:
            matching_runs.append(str(slug))

    if len(matching_runs) != 1:
        raise ValueError(
            "oracle must contain exactly one Stage-2 run matching the static checkpoint"
        )
    matched_slug = matching_runs[0]
    matched = _mapping(stage2_runs[matched_slug], name=f"oracle Stage-2 run {matched_slug}")
    if not math.isclose(
        float(matched["support_threshold"]),
        float(static_threshold),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("oracle support threshold differs from static Stage-2 metrics")
    static_epoch = stage2_metrics.get("checkpoint_epoch")
    if static_epoch is not None and int(static_epoch) != int(matched["checkpoint_epoch"]):
        raise ValueError("oracle checkpoint epoch differs from static Stage-2 metrics")

    return {
        "format": R0_ORACLE_AUDIT_FORMAT,
        "split": "val",
        "evaluated_file_count": file_count,
        "expected_file_count": expected_file_count,
        "complete_validation": complete,
        "declared_formal_complete_validation": declared_formal,
        "accepted_as_formal": bool(complete and declared_formal and not allow_incomplete),
        "sample_ids_sha256": expected_sample_hash,
        "stage1_checkpoint": str(stage1_path),
        "stage1_checkpoint_sha256": stage1_hash,
        "matched_stage2_run": matched_slug,
        "stage2_checkpoint": str(static_checkpoint),
        "stage2_checkpoint_sha256": static_checkpoint_hash,
        "support_threshold": float(static_threshold),
    }


def build_stage2_region_error_budget(
    raw_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compute target/SSE/FN/FP contributions from ``per_region.csv``."""

    by_name = {str(row.get("region")): row for row in raw_rows}
    missing = [name for name in REQUIRED_REGION_NAMES if name not in by_name]
    if missing:
        raise ValueError(f"per_region.csv is missing required regions: {missing}")

    all_row = by_name["all_domain"]
    total_target = int(
        _number(all_row, "reflectivity_on_target_support_count", integer=True)
    )
    total_rmse = float(_number(all_row, "reflectivity_on_target_support_rmse_dbz"))
    total_sse = total_target * total_rmse * total_rmse
    total_fn = int(_number(all_row, "support_false_negative", integer=True))
    total_fp = int(_number(all_row, "support_false_positive", integer=True))
    if total_target <= 0 or not math.isfinite(total_sse) or total_sse <= 0.0:
        raise ValueError("all_domain has no valid reflectivity error budget")

    semantics = {
        "all_domain": ("global", "complete trustworthy support-label domain"),
        "q11_direct_overlap": ("observability", "GR and DPR both have dBZ"),
        "q01_direct_missing": ("observability", "DPR has dBZ and direct GR is missing"),
        "dpr_gap_proxy": ("observability", "Q01 reachable by GR interpolation proxy"),
        "dpr_outside_proxy": ("observability", "Q01 outside interpolation proxy"),
        "q10_gr_only": ("negative_support", "GR has dBZ but DPR has no echo"),
        "q00_neither": ("negative_support", "neither GR nor DPR has echo"),
        "dpr_dbz_ge35": ("intensity", "true DPR dBZ >= 35"),
    }
    rows: list[dict[str, Any]] = []
    for region in REQUIRED_REGION_NAMES:
        source = by_name[region]
        count = int(
            _number(source, "reflectivity_on_target_support_count", integer=True)
        )
        rmse = float(_number(source, "reflectivity_on_target_support_rmse_dbz"))
        sse = count * rmse * rmse if count and math.isfinite(rmse) else 0.0
        fn = int(_number(source, "support_false_negative", integer=True))
        fp = int(_number(source, "support_false_positive", integer=True))
        axis, description = semantics[region]
        rows.append(
            {
                "region": region,
                "axis": axis,
                "description": description,
                "target_count": count,
                "target_fraction": count / total_target,
                "mae_dbz": _number(
                    source, "reflectivity_on_target_support_mae_dbz"
                ),
                "rmse_dbz": rmse,
                "bias_dbz": _number(
                    source, "reflectivity_on_target_support_bias_dbz"
                ),
                "pearson_r": _number(
                    source, "reflectivity_on_target_support_pearson_r"
                ),
                "squared_error_sum": sse,
                "squared_error_fraction": sse / total_sse,
                "support_false_negative": fn,
                "false_negative_fraction": fn / total_fn if total_fn else math.nan,
                "support_false_positive": fp,
                "false_positive_fraction": fp / total_fp if total_fp else math.nan,
                "support_recall": _number(source, "support_recall"),
                "support_false_alarm_ratio": _number(
                    source, "support_false_alarm_ratio"
                ),
            }
        )

    counts = {row["region"]: row for row in rows}
    if counts["q11_direct_overlap"]["target_count"] + counts[
        "q01_direct_missing"
    ]["target_count"] != total_target:
        raise ValueError("Q11+Q01 target counts do not equal all DPR targets")
    if counts["dpr_gap_proxy"]["target_count"] + counts[
        "dpr_outside_proxy"
    ]["target_count"] != counts["q01_direct_missing"]["target_count"]:
        raise ValueError("gap+outside target counts do not equal Q01")
    if (
        counts["q11_direct_overlap"]["support_false_negative"]
        + counts["dpr_gap_proxy"]["support_false_negative"]
        + counts["dpr_outside_proxy"]["support_false_negative"]
        != total_fn
    ):
        raise ValueError("Q11+gap+outside false negatives do not equal global FN")
    if (
        counts["q10_gr_only"]["support_false_positive"]
        + counts["q00_neither"]["support_false_positive"]
        != total_fp
    ):
        raise ValueError("Q10+Q00 false positives do not equal global FP")
    return rows, {
        "target_count": total_target,
        "squared_error_sum": total_sse,
        "support_false_negative": total_fn,
        "support_false_positive": total_fp,
        "partition_checks": {
            "q11_plus_q01_equals_all_target": True,
            "gap_plus_outside_equals_q01": True,
            "q11_gap_outside_fn_equals_all_fn": True,
            "q10_plus_q00_fp_equals_all_fp": True,
        },
        "non_additive_axis_warning": (
            "dpr_dbz_ge35 overlaps observability regions and must not be summed with them"
        ),
    }


def build_alignment_oracle_summary(
    best_rows: Sequence[Mapping[str, Any]],
    shift_rows: Sequence[Mapping[str, Any]],
    *,
    group: str = "val",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compare each height's best finite shift with the exact zero shift."""

    selected_best = [row for row in best_rows if row.get("group") == group]
    if not selected_best:
        raise ValueError(f"alignment best-shift table has no group={group}")
    zero_by_height = {
        int(_number(row, "height_index", integer=True)): row
        for row in shift_rows
        if row.get("group") == group
        and int(_number(row, "scan_shift", integer=True)) == 0
        and int(_number(row, "ray_shift", integer=True)) == 0
    }
    rows: list[dict[str, Any]] = []
    for best in sorted(
        selected_best, key=lambda row: int(_number(row, "height_index", integer=True))
    ):
        height_index = int(_number(best, "height_index", integer=True))
        if height_index not in zero_by_height:
            raise ValueError(f"alignment table lacks zero shift at height {height_index}")
        zero = zero_by_height[height_index]
        best_csi = float(_number(best, "best_support_csi"))
        zero_csi = float(_number(zero, "support_csi"))
        scan_shift = int(_number(best, "best_scan_shift", integer=True))
        ray_shift = int(_number(best, "best_ray_shift", integer=True))
        rows.append(
            {
                "height_index": height_index,
                "height_km": _number(best, "height_km"),
                "best_scan_shift": scan_shift,
                "best_ray_shift": ray_shift,
                "best_is_zero_shift": scan_shift == 0 and ray_shift == 0,
                "zero_shift_support_csi": zero_csi,
                "best_shift_support_csi": best_csi,
                "support_csi_gain": best_csi - zero_csi,
                "zero_shift_pearson_r": _number(zero, "pearson_r"),
                "pearson_at_best_support_shift": _number(
                    best, "pearson_at_best_support_shift"
                ),
                "overlap_count_at_best_shift": _number(
                    best, "overlap_count", integer=True
                ),
            }
        )
    gains = [float(row["support_csi_gain"]) for row in rows]
    nonzero = [row for row in rows if not row["best_is_zero_shift"]]
    return rows, {
        "group": group,
        "height_count": len(rows),
        "zero_best_height_count": len(rows) - len(nonzero),
        "nonzero_best_height_count": len(nonzero),
        "mean_support_csi_gain": sum(gains) / len(gains),
        "max_support_csi_gain": max(gains),
        "interpretation_boundary": (
            "This is an oracle finite global shift per height, not a deployable "
            "registration model and not proof that displacement causes all mismatch."
        ),
    }


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
    fields: list[str] = []
    for row in rows:
        for name in row:
            if name not in fields:
                fields.append(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(finite_metrics_for_json(list(rows)))
    temporary.replace(path)


def _format(value: Any, digits: int = 4) -> str:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def build_markdown_report(
    completeness: Mapping[str, Any],
    budget_rows: Sequence[Mapping[str, Any]],
    alignment: Mapping[str, Any],
    *,
    oracle_attached: bool,
    local_shift_payload: Mapping[str, Any] | None = None,
) -> str:
    by_name = {str(row["region"]): row for row in budget_rows}
    display = (
        "q11_direct_overlap",
        "dpr_gap_proxy",
        "dpr_outside_proxy",
        "dpr_dbz_ge35",
    )
    lines = [
        "# S2-R0 Stage 2任务分解审计",
        "",
        f"- 验证轨道：{completeness['evaluated_file_count']}/"
        f"{completeness['expected_file_count']}；完整性："
        f"`{completeness['complete_validation']}`",
        f"- checkpoint：`{completeness.get('checkpoint')}`",
        f"- support阈值：`{completeness.get('support_threshold')}`",
        "",
        "## 区域误差预算",
        "",
        "| 区域 | DPR目标占比 | dBZ RMSE | dBZ Bias | dBZ r | SSE贡献 | support FN贡献 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in display:
        row = by_name[name]
        lines.append(
            f"| `{name}` | {_format(100*row['target_fraction'], 2)}% | "
            f"{_format(row['rmse_dbz'])} | {_format(row['bias_dbz'])} | "
            f"{_format(row['pearson_r'])} | "
            f"{_format(100*row['squared_error_fraction'], 2)}% | "
            f"{_format(100*row['false_negative_fraction'], 2)}% |"
        )
    lines.extend(
        [
            "",
            "`>=35 dBZ`与可观测性区域重叠，SSE贡献不能相加。",
            "",
            "## 局部有限位移oracle",
            "",
        ]
    )
    if local_shift_payload is None:
        lines.append("尚未附加同口径局部位移审计；旧的全局结果不能排除局部相反位移抵消。")
    else:
        local = _mapping(local_shift_payload["metrics"], name="local-shift metrics")
        lines.extend(
            [
                f"- exact `(0,0)` pooled CSI：{_format(local['exact_support_csi'], 6)}",
                "- 全验证单一位移oracle CSI："
                f"{_format(local['one_shift_all_validation_oracle_support_csi'], 6)}",
                "- 逐轨逐高度oracle CSI："
                f"{_format(local['per_orbit_height_oracle_support_csi'], 6)}",
                "- 局部窗口逐高度oracle CSI："
                f"{_format(local['local_window_height_oracle_support_csi'], 6)}",
                "- 出现相反局部位移的轨道："
                f"{local['opposing_local_shift_orbit_count']}/"
                f"{local['orbit_count']}",
                "- 存在抵消证据的轨道："
                f"{local['cancellation_evidence_orbit_count']}/"
                f"{local['orbit_count']}",
            ]
        )
    lines.extend(
        [
            "",
            "局部best shift使用真实DPR事后选择，只表示有限搜索半径内的理论上限，不能作为部署输入。",
            "",
            "### 历史全验证聚合逐高度位移（仅作对照）",
            "",
            f"- 逐高度最佳位移仍为零的层数："
            f"{alignment['zero_best_height_count']}/{alignment['height_count']}",
            f"- 非零最佳位移层数：{alignment['nonzero_best_height_count']}",
            f"- 平均CSI oracle增益：{_format(alignment['mean_support_csi_gain'], 6)}",
            f"- 最大CSI oracle增益：{_format(alignment['max_support_csi_gain'], 6)}",
            "",
            "该旧口径会把不同轨道或局部的相反位移聚合抵消，不能单独据此关闭配准路线。",
            "",
            "## 冻结Stage 1区域oracle替换",
            "",
            (
                "已附加`r0_decomposition_oracles.json`，可据最终降水闭合比例确定R1优先级。"
                if oracle_attached
                else "尚未附加；运行完整验证集区域oracle替换后重新执行本脚本。"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def plot_r0_summary(output_dir: Path, summary: Mapping[str, Any]) -> list[str]:
    """Write compact R0 decision plots and return paths relative to output."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    region_order = (
        "q11_direct_overlap",
        "dpr_gap_proxy",
        "dpr_outside_proxy",
        "dpr_dbz_ge35",
    )
    region_by_name = {
        str(row["region"]): row for row in summary.get("regions", ())
    }
    if all(name in region_by_name for name in region_order):
        x = np.arange(len(region_order), dtype=float)
        width = 0.24
        fig, axis = plt.subplots(figsize=(10.0, 5.2), constrained_layout=True)
        axis.bar(
            x - width,
            [100.0 * region_by_name[name]["target_fraction"] for name in region_order],
            width,
            label="DPR target voxels",
        )
        axis.bar(
            x,
            [
                100.0 * region_by_name[name]["squared_error_fraction"]
                for name in region_order
            ],
            width,
            label="dBZ squared error",
        )
        axis.bar(
            x + width,
            [
                100.0 * region_by_name[name]["false_negative_fraction"]
                for name in region_order
            ],
            width,
            label="support false negatives",
        )
        axis.set_xticks(x, ("Q11", "gap", "outside", ">=35 dBZ"))
        axis.set_ylabel("Share (%)")
        axis.set_title("S2-R0 regional error budget (overlapping intensity axis)")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False)
        path = plot_dir / "region_error_budget.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(str(path.relative_to(output_dir)))

    local_payload = summary.get("local_shift_oracle")
    if isinstance(local_payload, Mapping) and isinstance(
        local_payload.get("metrics"), Mapping
    ):
        metrics = local_payload["metrics"]
        names = (
            "Exact",
            "One global",
            "Per orbit",
            "Orbit/height",
            "Local/height",
        )
        keys = (
            "exact_support_csi",
            "one_shift_all_validation_oracle_support_csi",
            "per_orbit_single_shift_oracle_support_csi",
            "per_orbit_height_oracle_support_csi",
            "local_window_height_oracle_support_csi",
        )
        values = [float(metrics[key]) for key in keys]
        fig, axis = plt.subplots(figsize=(9.5, 5.0), constrained_layout=True)
        bars = axis.bar(names, values, color=("#777777", "#4c78a8", "#72b7b2", "#f58518", "#e45756"))
        axis.set_ylabel("Support CSI")
        axis.set_title("Finite-shift oracle hierarchy (DPR-selected, non-deployable)")
        lower = min(values)
        upper = max(values)
        margin = max(0.005, (upper - lower) * 0.35)
        axis.set_ylim(max(0.0, lower - margin), min(1.0, upper + margin))
        axis.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                value,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        path = plot_dir / "local_shift_oracle.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(str(path.relative_to(output_dir)))

    oracle = summary.get("oracle_replacement")
    if isinstance(oracle, Mapping) and isinstance(oracle.get("runs"), Mapping):
        for run_slug, run in oracle["runs"].items():
            if not isinstance(run, Mapping):
                continue
            components: list[tuple[str, Mapping[str, Any]]] = []
            for key, title in (("value_oracle", "Value"), ("support_oracle", "Support")):
                component = run.get(key)
                if isinstance(component, Mapping):
                    components.append((title, component))
            if not components:
                continue
            fig, axes = plt.subplots(
                1,
                len(components),
                figsize=(7.0 * len(components), 5.0),
                constrained_layout=True,
                squeeze=False,
            )
            for axis, (title, component) in zip(axes[0], components):
                metric = component["metrics"]["reliable_positive_rain"]["rmse"]
                regions = list(component["regions"])
                closure = [
                    metric["regions"][name][
                        "fraction_of_baseline_to_reference_gap_closed"
                    ]
                    for name in regions
                ]
                values = [
                    float(value) if isinstance(value, (int, float)) else np.nan
                    for value in closure
                ]
                axis.bar(np.arange(len(regions)), values, color="#4c78a8")
                axis.axhline(0.0, color="black", linewidth=0.8)
                axis.set_xticks(np.arange(len(regions)), regions, rotation=35, ha="right")
                axis.set_ylabel("Fraction of RMSE gap closed")
                axis.set_title(f"{run_slug}: regional {title.lower()} oracle")
                axis.grid(axis="y", alpha=0.25)
            path = plot_dir / f"oracle_gap_closure_{run_slug}.png"
            fig.savefig(path, dpi=180)
            plt.close(fig)
            written.append(str(path.relative_to(output_dir)))
    return written


def main() -> None:
    args = parse_args()
    evaluation_dir = args.stage2_evaluation_dir.expanduser().resolve()
    metrics_path = evaluation_dir / "metrics.json"
    per_region_path = evaluation_dir / "per_region.csv"
    patch_index_path = args.patch_index.expanduser().resolve()
    alignment_dir = args.alignment_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    protected = output_dir / "summary.json"
    if protected.exists() and not args.overwrite:
        raise FileExistsError("R0 audit output exists; pass --overwrite")

    metrics = _load_json(metrics_path)
    patch_index = _load_json(patch_index_path)
    completeness = validate_complete_validation_evaluation(
        metrics, patch_index, allow_incomplete=args.allow_incomplete
    )
    budget_rows, budget_summary = build_stage2_region_error_budget(
        _read_csv(per_region_path)
    )
    alignment_rows, alignment_summary = build_alignment_oracle_summary(
        _read_csv(alignment_dir / "best_shift_by_height.csv"),
        _read_csv(alignment_dir / "shift_metrics.csv"),
        group="val",
    )
    oracle_payload = None
    oracle_validation = None
    if args.oracle_audit is not None:
        oracle_payload = _load_json(args.oracle_audit.expanduser().resolve())
        oracle_validation = validate_r0_oracle_audit(
            oracle_payload,
            metrics,
            patch_index,
            patch_index_path=patch_index_path,
            allow_incomplete=args.allow_incomplete,
        )
    local_shift_payload = None
    local_shift_validation = None
    if args.local_shift_audit is not None:
        local_shift_payload = _load_json(
            args.local_shift_audit.expanduser().resolve()
        )
        local_shift_validation = validate_r0_local_shift_audit(
            local_shift_payload,
            patch_index,
            allow_incomplete=args.allow_incomplete,
        )

    static_formal_result = bool(
        completeness["complete_validation"] and not args.allow_incomplete
    )
    # The complete R0 experiment requires all three evidence blocks.  Static
    # error budgeting remains independently formal even before the expensive
    # frozen-cascade oracle has been run.
    formal_result = bool(
        static_formal_result
        and oracle_validation is not None
        and oracle_validation["accepted_as_formal"]
        and local_shift_validation is not None
        and local_shift_validation["accepted_as_formal"]
    )

    summary = {
        "format": R0_STATIC_AUDIT_FORMAT,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "S2-R0-DecompositionAudit",
        "formal_result": formal_result,
        "static_formal_result": static_formal_result,
        "completeness": completeness,
        "sources": {
            "stage2_metrics": str(metrics_path),
            "per_region": str(per_region_path),
            "patch_index": str(patch_index_path),
            "alignment_directory": str(alignment_dir),
            "oracle_audit": (
                str(args.oracle_audit.expanduser().resolve())
                if args.oracle_audit is not None
                else None
            ),
            "local_shift_audit": (
                str(args.local_shift_audit.expanduser().resolve())
                if args.local_shift_audit is not None
                else None
            ),
        },
        "error_budget": budget_summary,
        "regions": budget_rows,
        "alignment_oracle": alignment_summary,
        "local_shift_attached": local_shift_payload is not None,
        "local_shift_validation": local_shift_validation,
        "local_shift_oracle": local_shift_payload,
        "oracle_replacement_attached": oracle_payload is not None,
        "oracle_replacement_validation": oracle_validation,
        "oracle_replacement": oracle_payload,
        "interpretation": {
            "observability": (
                "observed/near/far is a Stage-2 task partition, not evidence that "
                "the project should wait for a better GR dataset"
            ),
            "direct_overlap_r": (
                "conditional on both sensors having values; not a full-domain ceiling"
            ),
            "test_set_accessed": False,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary["plots"] = plot_r0_summary(output_dir, summary)
    _atomic_json(output_dir / "summary.json", summary)
    _atomic_csv(output_dir / "region_error_budget.csv", budget_rows)
    _atomic_csv(output_dir / "alignment_oracle_by_height.csv", alignment_rows)
    report = build_markdown_report(
        completeness,
        budget_rows,
        alignment_summary,
        oracle_attached=oracle_payload is not None,
        local_shift_payload=local_shift_payload,
    )
    temporary = output_dir / "report.md.partial"
    temporary.write_text(report, encoding="utf-8")
    temporary.replace(output_dir / "report.md")
    print(
        f"S2-R0 static audit: complete_val={completeness['complete_validation']} "
        f"-> {output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
