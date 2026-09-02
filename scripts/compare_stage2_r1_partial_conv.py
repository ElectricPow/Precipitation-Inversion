#!/usr/bin/env python3
"""Apply the pre-registered R1-P acceptance gates against the R1-O control."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = PROJECT_ROOT / "outputs" / "stage2_rethink" / "r1_o_dpr_sparse_value"
DEFAULT_CANDIDATE = PROJECT_ROOT / "outputs" / "stage2_rethink" / "r1_p_partial_conv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _float(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _rows_by_key(
    rows: Sequence[Mapping[str, Any]], key: str
) -> dict[str, Mapping[str, Any]]:
    result = {str(row[key]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate {key!r} rows")
    return result


def _completion_cascade_row(
    rows: Sequence[Mapping[str, Any]], label: str
) -> Mapping[str, Any]:
    selected = [row for row in rows if str(row.get("mode")) != "dpr_oracle"]
    if len(selected) != 1:
        raise ValueError(f"{label} cascade must contain one completion mode")
    return selected[0]


def _relative_reduction(baseline: float, candidate: float) -> float:
    if baseline <= 0.0:
        raise ValueError("baseline error must be positive")
    return (baseline - candidate) / baseline


def build_gate_comparison(
    baseline_regions: Sequence[Mapping[str, Any]],
    candidate_regions: Sequence[Mapping[str, Any]],
    baseline_cascade: Sequence[Mapping[str, Any]],
    candidate_cascade: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate the fixed gates without selecting thresholds on candidate data."""

    baseline = _rows_by_key(baseline_regions, "region")
    candidate = _rows_by_key(candidate_regions, "region")
    required = (
        "dpr_outside_proxy", "dpr_gap_proxy", "dpr_unanchored", "dpr_dbz_ge35"
    )
    missing = [name for name in required if name not in baseline or name not in candidate]
    if missing:
        raise KeyError(f"missing comparison regions: {missing}")

    def region_values(name: str) -> dict[str, float]:
        return {
            "baseline_rmse_dbz": _float(baseline[name]["rmse_dbz"], f"{name} baseline RMSE"),
            "candidate_rmse_dbz": _float(candidate[name]["rmse_dbz"], f"{name} candidate RMSE"),
            "baseline_bias_dbz": _float(baseline[name]["bias_dbz"], f"{name} baseline bias"),
            "candidate_bias_dbz": _float(candidate[name]["bias_dbz"], f"{name} candidate bias"),
            "baseline_pearson_r": _float(baseline[name]["pearson_r"], f"{name} baseline r"),
            "candidate_pearson_r": _float(candidate[name]["pearson_r"], f"{name} candidate r"),
        }

    outside = region_values("dpr_outside_proxy")
    gap = region_values("dpr_gap_proxy")
    unanchored = region_values("dpr_unanchored")
    strong = region_values("dpr_dbz_ge35")
    outside_reduction = _relative_reduction(
        outside["baseline_rmse_dbz"], outside["candidate_rmse_dbz"]
    )
    outside_r_gain = outside["candidate_pearson_r"] - outside["baseline_pearson_r"]
    gap_change = (
        gap["candidate_rmse_dbz"] - gap["baseline_rmse_dbz"]
    ) / gap["baseline_rmse_dbz"]
    strong_reduction = _relative_reduction(
        strong["baseline_rmse_dbz"], strong["candidate_rmse_dbz"]
    )

    baseline_cascade_row = _completion_cascade_row(baseline_cascade, "baseline")
    candidate_cascade_row = _completion_cascade_row(candidate_cascade, "candidate")
    baseline_rain_r = _float(
        baseline_cascade_row["positive_pearson_r"], "baseline cascade r"
    )
    candidate_rain_r = _float(
        candidate_cascade_row["positive_pearson_r"], "candidate cascade r"
    )
    baseline_drdz_r = _float(
        baseline_cascade_row["drdz_pearson_r"], "baseline dR/dz r"
    )
    candidate_drdz_r = _float(
        candidate_cascade_row["drdz_pearson_r"], "candidate dR/dz r"
    )

    gates = {
        "outside_primary": {
            "criterion": "RMSE reduction >=3% OR Pearson r gain >=0.02",
            "passed": outside_reduction >= 0.03 or outside_r_gain >= 0.02,
            "rmse_reduction_fraction": outside_reduction,
            "pearson_r_gain": outside_r_gain,
        },
        "unanchored_not_anchor_only": {
            "criterion": "unanchored RMSE and Pearson r both improve",
            "passed": (
                unanchored["candidate_rmse_dbz"] < unanchored["baseline_rmse_dbz"]
                and unanchored["candidate_pearson_r"] > unanchored["baseline_pearson_r"]
            ),
            "rmse_reduction_fraction": _relative_reduction(
                unanchored["baseline_rmse_dbz"], unanchored["candidate_rmse_dbz"]
            ),
            "pearson_r_gain": (
                unanchored["candidate_pearson_r"] - unanchored["baseline_pearson_r"]
            ),
        },
        "gap_non_regression": {
            "criterion": "gap RMSE degradation <=2%",
            "passed": gap_change <= 0.02,
            "rmse_change_fraction": gap_change,
        },
        "strong_ge35_rmse": {
            "criterion": ">=35 dBZ RMSE reduction >=5%",
            "passed": strong_reduction >= 0.05,
            "rmse_reduction_fraction": strong_reduction,
        },
        "strong_ge35_bias": {
            "criterion": "absolute >=35 dBZ bias moves toward zero",
            "passed": abs(strong["candidate_bias_dbz"]) < abs(strong["baseline_bias_dbz"]),
            "baseline_abs_bias_dbz": abs(strong["baseline_bias_dbz"]),
            "candidate_abs_bias_dbz": abs(strong["candidate_bias_dbz"]),
        },
        "frozen_stage1_cascade": {
            "criterion": "final-rain Pearson r gain >=0.02",
            "passed": candidate_rain_r - baseline_rain_r >= 0.02,
            "baseline_pearson_r": baseline_rain_r,
            "candidate_pearson_r": candidate_rain_r,
            "pearson_r_gain": candidate_rain_r - baseline_rain_r,
        },
        "drdz_non_regression": {
            "criterion": "dR/dz Pearson r does not decrease",
            "passed": candidate_drdz_r >= baseline_drdz_r,
            "baseline_pearson_r": baseline_drdz_r,
            "candidate_pearson_r": candidate_drdz_r,
            "pearson_r_gain": candidate_drdz_r - baseline_drdz_r,
        },
    }
    return {
        "format": "stage2_r1_p_partial_conv_gate_comparison_v1",
        "selection_policy": "all thresholds were fixed before candidate validation",
        "regions": {
            "outside": outside,
            "gap": gap,
            "unanchored": unanchored,
            "dbz_ge35": strong,
        },
        "gates": gates,
        "all_gates_passed": all(bool(item["passed"]) for item in gates.values()),
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _require_formal(directory: Path, label: str) -> None:
    for relative in (
        Path("analysis/full_validation/metrics.json"),
        Path("analysis/frozen_stage1_cascade/metrics.json"),
    ):
        value = _read_json(directory / relative)
        if not bool(value.get("formal_validation_result")):
            raise ValueError(f"{label} is not a complete formal validation result: {relative}")
        if value.get("split") != "val" or bool(value.get("test_set_accessed")):
            raise ValueError(f"{label} comparison must use validation only")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
        suffix=".tmp", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def _markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# S2-R1-P-PartialConv 固定验收门槛",
        "",
        "所有阈值均在候选模型验证前固定；本报告只使用完整 validation。",
        "",
        "| 门槛 | 判定标准 | 是否通过 |",
        "|---|---|---:|",
    ]
    for name, item in result["gates"].items():
        lines.append(
            f"| `{name}` | {item['criterion']} | {'通过' if item['passed'] else '未通过'} |"
        )
    lines.extend(
        (
            "",
            f"总体：**{'全部通过' if result['all_gates_passed'] else '未全部通过'}**。",
            "",
            "若outside主门槛未通过，则按预注册路线停止继续堆叠稀疏卷积变体；"
            "若通过，再结合非锚点、强回波、冻结串联和dR/dz决定是否进入R2。",
            "",
        )
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    baseline = args.baseline_dir.expanduser().resolve()
    candidate = args.candidate_dir.expanduser().resolve()
    output = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else candidate / "analysis" / "comparison_to_r1_o"
    )
    if not args.overwrite and (output / "comparison.json").exists():
        raise FileExistsError("comparison exists; pass --overwrite")
    _require_formal(baseline, "R1-O baseline")
    _require_formal(candidate, "R1-P candidate")
    result = build_gate_comparison(
        _read_csv(baseline / "analysis" / "full_validation" / "per_region.csv"),
        _read_csv(candidate / "analysis" / "full_validation" / "per_region.csv"),
        _read_csv(baseline / "analysis" / "frozen_stage1_cascade" / "comparison.csv"),
        _read_csv(candidate / "analysis" / "frozen_stage1_cascade" / "comparison.csv"),
    )
    result = {
        **result,
        "baseline_directory": str(baseline),
        "candidate_directory": str(candidate),
        "split": "val",
        "test_set_accessed": False,
    }
    _atomic_json(output / "comparison.json", result)
    (output / "comparison.md").write_text(_markdown(result), encoding="utf-8")
    print(
        f"R1-P gates: {'PASS' if result['all_gates_passed'] else 'NOT ALL PASSED'} -> {output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
