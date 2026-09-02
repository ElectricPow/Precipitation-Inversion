#!/usr/bin/env python3
"""Compare physical dR/dz metrics from multiple stage-one evaluations.

Every input must be a complete validation result produced by
``evaluate_stage1_unet3d.py``. The script refuses to rank runs whose reliable
adjacent-height support differs, preventing weak-CFB labels or incomplete
patch subsets from silently changing the comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "ablations" / "stage1_drdz_comparison"
STANDARD_METRICS = ("mae", "rmse", "bias", "r2", "pearson_r")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=METRICS_JSON",
        help="Repeat for each run, for example E0=outputs/.../metrics.json.",
    )
    parser.add_argument("--baseline", default="E0")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def _number(value: Any) -> float:
    if value is None:
        return math.nan
    result = float(value)
    return result if math.isfinite(result) else math.nan


def _load_run(specification: str) -> tuple[str, Path, dict[str, Any]]:
    if "=" not in specification:
        raise ValueError(f"run must use LABEL=PATH syntax: {specification!r}")
    label, raw_path = specification.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError("run label cannot be empty")
    path = Path(raw_path).expanduser()
    path = path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    patch = value.get("patch_evaluation", {})
    coverage = patch.get("coverage", {}) if isinstance(patch, Mapping) else {}
    if value.get("split") != "val":
        raise ValueError(f"{label} is not a validation evaluation")
    if coverage.get("complete_patch_support") is not True:
        raise ValueError(
            f"{label} is a legacy or incomplete evaluation and does not cover "
            "the complete patch split. Backfill all runs first with "
            "`python scripts/backfill_stage1_drdz.py --device cuda:0`."
        )
    metrics = patch.get("metrics", {}) if isinstance(patch, Mapping) else {}
    drdz = metrics.get("physical_drdz") if isinstance(metrics, Mapping) else None
    if not isinstance(drdz, dict):
        raise ValueError(
            f"{label} predates the physical dR/dz evaluator and has no "
            "physical_drdz metrics. Backfill all runs first with "
            "`python scripts/backfill_stage1_drdz.py --device cuda:0`."
        )
    return label, path, drdz


def _assert_same_support(runs: Mapping[str, Mapping[str, Any]]) -> None:
    iterator = iter(runs.items())
    reference_label, reference = next(iterator)
    reference_all = reference["all"]
    reference_fingerprint = reference.get("support", {}).get("sha256")
    reference_height = reference["by_midpoint_height_km"]
    reference_files = reference.get("filewise", {}).get("per_file", {})
    for label, current in iterator:
        if current.get("definitions") != reference.get("definitions"):
            raise ValueError(
                f"dR/dz protocol differs between {reference_label} and {label}"
            )
        current_all = current["all"]
        current_fingerprint = current.get("support", {}).get("sha256")
        # A missing fingerprint may be accepted only when it is missing from
        # every input (for backward-compatible synthetic/legacy reports).  Do
        # not silently compare one exact-support report with one count-only
        # report, because equal counts do not prove equal voxel locations.
        if current_fingerprint != reference_fingerprint:
            raise ValueError(
                f"exact reliable dR/dz pair support differs for {label}"
            )
        if int(current_all["count"]) != int(reference_all["count"]):
            raise ValueError(
                f"reliable dR/dz pair count differs: {reference_label}="
                f"{reference_all['count']}, {label}={current_all['count']}"
            )
        if not math.isclose(
            float(current_all["mean_abs_target"]),
            float(reference_all["mean_abs_target"]),
            rel_tol=1e-10,
            abs_tol=1e-10,
        ):
            raise ValueError(f"target dR/dz distribution differs for {label}")
        current_height = current["by_midpoint_height_km"]
        if tuple(current_height) != tuple(reference_height):
            raise ValueError(f"dR/dz height groups differ for {label}")
        for height in reference_height:
            if int(current_height[height]["count"]) != int(
                reference_height[height]["count"]
            ):
                raise ValueError(f"dR/dz support differs at {height} for {label}")
        current_files = current.get("filewise", {}).get("per_file", {})
        if tuple(current_files) != tuple(reference_files):
            raise ValueError(f"filewise dR/dz labels differ for {label}")
        for file_label in reference_files:
            if int(current_files[file_label]["count"]) != int(
                reference_files[file_label]["count"]
            ):
                raise ValueError(
                    f"filewise dR/dz support differs for {label}/{file_label}"
                )


def _paired_bootstrap(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    seed: int,
    replicates: int,
    confidence: float,
) -> dict[str, Any]:
    baseline_files = baseline["filewise"]["per_file"]
    candidate_files = candidate["filewise"]["per_file"]
    labels = [
        label
        for label, values in baseline_files.items()
        if int(values["count"]) > 0 and int(candidate_files[label]["count"]) > 0
    ]
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(labels), size=(replicates, len(labels)))
    alpha = (1.0 - confidence) / 2.0
    result: dict[str, Any] = {
        "definition": "candidate minus baseline; paired resampling unit is one orbit",
        "file_count": len(labels),
        "seed": seed,
        "replicates": replicates,
        "confidence": confidence,
        "metrics": {},
    }
    for name in STANDARD_METRICS:
        difference = np.asarray(
            [
                _number(candidate_files[label].get(name))
                - _number(baseline_files[label].get(name))
                for label in labels
            ],
            dtype=float,
        )
        finite = np.isfinite(difference)
        point = float(difference[finite].mean()) if finite.any() else math.nan
        sampled_values = difference[sampled]
        sampled_finite = np.isfinite(sampled_values)
        denominator = sampled_finite.sum(axis=1)
        replicate_mean = np.divide(
            np.where(sampled_finite, sampled_values, 0.0).sum(axis=1),
            denominator,
            out=np.full(replicates, np.nan),
            where=denominator > 0,
        )
        valid = replicate_mean[np.isfinite(replicate_mean)]
        low, high = (
            np.quantile(valid, (alpha, 1.0 - alpha))
            if valid.size
            else (math.nan, math.nan)
        )
        result["metrics"][name] = {
            "mean_delta": point,
            "ci_low": float(low),
            "ci_high": float(high),
            "valid_replicates": int(valid.size),
        }
    return result


def _height_values(groups: Mapping[str, Any]) -> np.ndarray:
    values = []
    for label in groups:
        if not (label.startswith("z_") and label.endswith("_km")):
            raise ValueError(f"invalid midpoint-height label: {label}")
        text = label[2:-3]
        negative = text.startswith("m")
        magnitude = text[1:] if negative else text
        value = float(magnitude.replace("p", "."))
        values.append(-value if negative else value)
    return np.asarray(values)


def _write_outputs(
    runs: Mapping[str, Mapping[str, Any]],
    paths: Mapping[str, Path],
    *,
    baseline_label: str,
    output_dir: Path,
    seed: int,
    replicates: int,
    confidence: float,
    dpi: int,
    scope_description: str = "同一完整验证集",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = list(runs)
    fields = (
        "run",
        "count",
        "mae",
        "rmse",
        "bias",
        "r2",
        "pearson_r",
        "mean_abs_prediction",
        "mean_abs_target",
        "mean_abs_gradient_ratio",
        "sign_agreement_fraction",
    )
    with (output_dir / "metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for label in labels:
            overall = runs[label]["all"]
            writer.writerow(
                {"run": label, **{name: overall.get(name) for name in fields[1:]}}
            )

    paired = {
        label: _paired_bootstrap(
            runs[baseline_label],
            runs[label],
            seed=seed,
            replicates=replicates,
            confidence=confidence,
        )
        for label in labels
        if label != baseline_label
    }
    with (output_dir / "paired_vs_baseline.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields_paired = (
            "run",
            "metric",
            "mean_delta",
            "ci_low",
            "ci_high",
            "valid_replicates",
        )
        writer = csv.DictWriter(handle, fieldnames=fields_paired)
        writer.writeheader()
        for label, result in paired.items():
            for name, values in result["metrics"].items():
                writer.writerow({"run": label, "metric": name, **values})

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    panels = (
        ("pearson_r", "Pearson r", "#2a9d8f", None),
        ("mae", "MAE", "#f4a261", "mm h$^{-1}$ km$^{-1}$"),
        ("rmse", "RMSE", "#e76f51", "mm h$^{-1}$ km$^{-1}$"),
        ("bias", "Bias", "#457b9d", "mm h$^{-1}$ km$^{-1}$"),
        ("mean_abs_gradient_ratio", "|dR/dz| amplitude ratio", "#6a4c93", None),
        ("sign_agreement_fraction", "Sign agreement", "#8ab17d", None),
    )
    positions = np.arange(len(labels))
    for axis, (key, title, color, unit) in zip(axes.flat, panels):
        values = [_number(runs[label]["all"].get(key)) for label in labels]
        axis.bar(positions, values, color=color)
        axis.set_xticks(positions, labels)
        axis.set_title(title)
        if unit:
            axis.set_ylabel(unit)
        if key == "bias":
            axis.axhline(0.0, color="black", linewidth=0.7)
        if key == "mean_abs_gradient_ratio":
            axis.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
        if key == "pearson_r":
            axis.set_ylim(-1.0, 1.0)
        elif key == "sign_agreement_fraction":
            axis.set_ylim(0.0, 1.0)
        axis.grid(axis="y", alpha=0.25)
    # Keep plot text ASCII because the shared training environment does not
    # guarantee a CJK font. The exact Chinese scope remains in summary.md/json.
    fig.suptitle("Stage-1 physical dR/dz comparison")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_dir / "comparison.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    reference_groups = runs[baseline_label]["by_midpoint_height_km"]
    height = _height_values(reference_groups)
    fig, axes = plt.subplots(1, 3, figsize=(17, 7), sharey=True)
    for label in labels:
        groups = runs[label]["by_midpoint_height_km"]
        axes[0].plot(
            [_number(groups[key].get("rmse")) for key in groups], height, label=label
        )
        axes[1].plot(
            [_number(groups[key].get("bias")) for key in groups], height, label=label
        )
        axes[2].plot(
            [_number(groups[key].get("pearson_r")) for key in groups],
            height,
            label=label,
        )
    axes[0].set_xlabel("RMSE (mm h$^{-1}$ km$^{-1}$)")
    axes[1].set_xlabel("Bias (mm h$^{-1}$ km$^{-1}$)")
    axes[2].set_xlabel("Pearson r")
    axes[0].set_ylabel("Gradient midpoint height (km)")
    axes[1].axvline(0.0, color="black", linewidth=0.7)
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle("Physical dR/dz quality by height")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_dir / "comparison_by_height.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    best_r = max(labels, key=lambda label: _number(runs[label]["all"]["pearson_r"]))
    best_rmse = min(labels, key=lambda label: _number(runs[label]["all"]["rmse"]))
    best_amplitude = min(
        labels,
        key=lambda label: abs(
            _number(runs[label]["all"]["mean_abs_gradient_ratio"]) - 1.0
        ),
    )
    summary = {
        "baseline": baseline_label,
        "scope": scope_description,
        "support_pair_count": int(runs[baseline_label]["all"]["count"]),
        "support_sha256": runs[baseline_label].get("support", {}).get("sha256"),
        "sources": {label: str(paths[label]) for label in labels},
        "best": {
            "pearson_r": best_r,
            "rmse": best_rmse,
            "amplitude_ratio_closest_to_one": best_amplitude,
        },
        "paired_bootstrap_vs_baseline": paired,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    rows = []
    for label in labels:
        values = runs[label]["all"]
        rows.append(
            f"| {label} | {int(values['count']):,} | "
            f"{_number(values['mae']):.4f} | {_number(values['rmse']):.4f} | "
            f"{_number(values['bias']):+.4f} | "
            f"{_number(values['pearson_r']):.4f} | "
            f"{_number(values['mean_abs_gradient_ratio']):.4f} | "
            f"{_number(values['sign_agreement_fraction']):.4f} |"
        )
    (output_dir / "summary.md").write_text(
        "# Stage-1物理dR/dz统一比较\n\n"
        f"范围：{scope_description}。所有模型使用同一可靠相邻层pair mask和"
        "同一向上差分定义。\n\n"
        "| 模型 | N | MAE | RMSE | Bias | Pearson r | 幅值比 | 符号一致率 |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|\n"
        + "\n".join(rows)
        + "\n\n"
        f"- 最高相关性：`{best_r}`；\n"
        f"- 最低RMSE：`{best_rmse}`；\n"
        f"- 平均绝对梯度幅值比最接近1：`{best_amplitude}`；\n"
        "- `paired_vs_baseline.csv` 使用整轨配对bootstrap，正差表示候选减基线；\n"
        "- 幅值比小于1提示垂直结构被平滑，大于1提示可能过度振荡。\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    if args.bootstrap_seed < 0:
        raise ValueError("bootstrap-seed must be non-negative")
    if args.bootstrap_replicates <= 0:
        raise ValueError("bootstrap-replicates must be positive")
    if not 0.0 < args.confidence < 1.0:
        raise ValueError("confidence must lie between zero and one")
    if args.dpi <= 0:
        raise ValueError("dpi must be positive")
    runs: dict[str, dict[str, Any]] = {}
    paths: dict[str, Path] = {}
    for specification in args.run:
        label, path, metrics = _load_run(specification)
        if label in runs:
            raise ValueError(f"duplicate run label: {label}")
        runs[label] = metrics
        paths[label] = path
    if args.baseline not in runs:
        raise ValueError(f"baseline {args.baseline!r} is not among run labels")
    if len(runs) < 2:
        raise ValueError("at least two runs are required")
    _assert_same_support(runs)
    result = _write_outputs(
        runs,
        paths,
        baseline_label=args.baseline,
        output_dir=args.output_dir.expanduser().resolve(),
        seed=args.bootstrap_seed,
        replicates=args.bootstrap_replicates,
        confidence=args.confidence,
        dpi=args.dpi,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
