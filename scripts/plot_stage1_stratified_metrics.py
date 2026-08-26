#!/usr/bin/env python3
"""Plot full-validation stratified metrics produced by checkpoint evaluation."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation_json", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def _metric(value: Any) -> float:
    if value is None:
        return float("nan")
    result = float(value)
    return result if math.isfinite(result) else float("nan")


def load_stratified_metrics(path: Path) -> tuple[dict[str, Any], str]:
    """Return stratified metrics from patch or full-orbit evaluation JSON."""

    source = path.expanduser().resolve()
    value = json.loads(source.read_text(encoding="utf-8"))
    candidates = (
        ("patch_evaluation", value.get("patch_evaluation", {})),
        ("full_orbit_evaluation", value.get("full_orbit_evaluation", {})),
    )
    for label, section in candidates:
        metrics = section.get("metrics", {}) if isinstance(section, Mapping) else {}
        result = metrics.get("stratified") if isinstance(metrics, Mapping) else None
        if isinstance(result, dict):
            return result, label
    raise ValueError(f"no stratified metrics found in {source}")


def _decode_number(text: str) -> float:
    negative = text.startswith("m")
    magnitude = text[1:] if negative else text
    value = float(magnitude.replace("p", "."))
    return -value if negative else value


def _height_axis(labels: list[str]) -> tuple[np.ndarray, list[str]]:
    """Decode exact-level keys; fall back to ordered categorical positions."""

    exact: list[float] = []
    for label in labels:
        if not (label.startswith("z_") and label.endswith("_km")):
            return np.arange(len(labels), dtype=float), labels
        exact.append(_decode_number(label[2:-3]))
    return np.asarray(exact, dtype=float), [f"{value:g}" for value in exact]


def _write_group_csv(groups: Mapping[str, Any], path: Path) -> None:
    fields = ("group", "count", "mae", "rmse", "bias", "r2", "pearson_r")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for label, values in groups.items():
            writer.writerow({"group": label, **{key: values.get(key) for key in fields[1:]}})


def _write_drdz_overall_csv(values: Mapping[str, Any], path: Path) -> None:
    fields = (
        "count",
        "mae",
        "rmse",
        "bias",
        "r2",
        "pearson_r",
        "mean_abs_prediction",
        "mean_abs_target",
        "mean_abs_gradient_ratio",
        "sign_epsilon",
        "sign_evaluated_count",
        "sign_agreement_fraction",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({name: values.get(name) for name in fields})


def _plot_height(groups: Mapping[str, Any], destination: Path, dpi: int) -> None:
    labels = list(groups)
    height, tick_labels = _height_axis(labels)
    count = np.asarray([_metric(groups[key].get("count")) for key in labels])
    rmse = np.asarray([_metric(groups[key].get("rmse")) for key in labels])
    mae = np.asarray([_metric(groups[key].get("mae")) for key in labels])
    bias = np.asarray([_metric(groups[key].get("bias")) for key in labels])
    correlation = np.asarray([_metric(groups[key].get("pearson_r")) for key in labels])

    fig, axes = plt.subplots(1, 3, figsize=(16, 7), sharey=True)
    axes[0].plot(count, height, color="#264653")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Reliable voxel count (log scale)")
    axes[0].set_ylabel("Height (km)" if labels and labels[0].startswith("z_") else "Height group")
    axes[0].set_title("Evaluation support")
    axes[1].plot(rmse, height, label="RMSE", color="#e76f51")
    axes[1].plot(mae, height, label="MAE", color="#f4a261")
    axes[1].plot(bias, height, label="Bias", color="#457b9d")
    axes[1].axvline(0, color="black", linewidth=0.7)
    axes[1].set_xlabel("Error (mm/h)")
    axes[1].set_title("Error by height")
    axes[1].legend()
    axes[2].plot(correlation, height, color="#2a9d8f")
    axes[2].set_xlim(-0.05, 1.0)
    axes[2].set_xlabel("Pearson r")
    axes[2].set_title("Correlation by height")
    if not (labels and labels[0].startswith("z_")):
        for ax in axes:
            ax.set_yticks(height, tick_labels, fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle("Full-validation metrics by absolute height")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(destination, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_drdz_height(
    groups: Mapping[str, Any], destination: Path, dpi: int
) -> None:
    """Plot signed adjacent-level gradient quality at 59 height midpoints."""

    labels = list(groups)
    height, _ = _height_axis(labels)
    count = np.asarray([_metric(groups[key].get("count")) for key in labels])
    rmse = np.asarray([_metric(groups[key].get("rmse")) for key in labels])
    mae = np.asarray([_metric(groups[key].get("mae")) for key in labels])
    bias = np.asarray([_metric(groups[key].get("bias")) for key in labels])
    correlation = np.asarray(
        [_metric(groups[key].get("pearson_r")) for key in labels]
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 7), sharey=True)
    axes[0].plot(count, height, color="#264653")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Reliable adjacent-height pair count (log)")
    axes[0].set_ylabel("Gradient midpoint height (km)")
    axes[0].set_title("dR/dz support")
    axes[1].plot(rmse, height, label="RMSE", color="#e76f51")
    axes[1].plot(mae, height, label="MAE", color="#f4a261")
    axes[1].plot(bias, height, label="Bias", color="#457b9d")
    axes[1].axvline(0.0, color="black", linewidth=0.7)
    axes[1].set_xlabel("mm h$^{-1}$ km$^{-1}$")
    axes[1].set_title("Physical gradient error")
    axes[1].legend()
    axes[2].plot(correlation, height, color="#2a9d8f")
    axes[2].axvline(0.0, color="black", linewidth=0.7)
    axes[2].set_xlim(-1.0, 1.0)
    axes[2].set_xlabel("Pearson r")
    axes[2].set_title("Signed-gradient correlation")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.suptitle("Physical vertical rain-rate gradient by height")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(destination, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_drdz_groups(
    views: list[tuple[str, Mapping[str, Any]]],
    destination: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(len(views), 2, figsize=(17, 5 * len(views)))
    axes = np.atleast_2d(axes)
    for row, (title, groups) in enumerate(views):
        labels = list(groups)
        positions = np.arange(len(labels))
        count = np.asarray([_metric(groups[key].get("count")) for key in labels])
        rmse = np.asarray([_metric(groups[key].get("rmse")) for key in labels])
        bias = np.asarray([_metric(groups[key].get("bias")) for key in labels])
        correlation = np.asarray(
            [_metric(groups[key].get("pearson_r")) for key in labels]
        )
        axes[row, 0].bar(positions, count, color="#457b9d")
        axes[row, 0].set_yscale("symlog", linthresh=1)
        axes[row, 0].set_ylabel("Reliable gradient-pair count")
        axes[row, 0].set_title(f"{title}: support")
        axes[row, 1].plot(positions, rmse, "o-", label="RMSE", color="#e76f51")
        axes[row, 1].plot(positions, bias, "o-", label="Bias", color="#457b9d")
        twin = axes[row, 1].twinx()
        twin.plot(
            positions,
            correlation,
            "s--",
            label="Pearson r",
            color="#2a9d8f",
        )
        twin.set_ylim(-1.0, 1.0)
        twin.set_ylabel("Pearson r")
        axes[row, 1].axhline(0.0, color="black", linewidth=0.7)
        axes[row, 1].set_ylabel("mm h$^{-1}$ km$^{-1}$")
        axes[row, 1].set_title(f"{title}: signed-gradient quality")
        lines, line_labels = axes[row, 1].get_legend_handles_labels()
        twin_lines, twin_labels = twin.get_legend_handles_labels()
        axes[row, 1].legend(lines + twin_lines, line_labels + twin_labels)
        for axis in axes[row]:
            axis.set_xticks(positions, labels, rotation=25, ha="right")
            axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Physical dR/dz grouped diagnostics")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(destination, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_height_intensity(
    crossed: Mapping[str, Any], destination: Path, dpi: int
) -> None:
    height_labels = list(crossed)
    intensity_labels = list(next(iter(crossed.values()))) if crossed else []
    count = np.asarray(
        [[_metric(crossed[h][rain].get("count")) for rain in intensity_labels] for h in height_labels]
    )
    rmse = np.asarray(
        [[_metric(crossed[h][rain].get("rmse")) for rain in intensity_labels] for h in height_labels]
    )
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), sharey=True)
    images = (
        axes[0].imshow(np.log10(count + 1.0), aspect="auto", origin="lower", cmap="viridis"),
        axes[1].imshow(np.ma.masked_invalid(rmse), aspect="auto", origin="lower", cmap="magma"),
    )
    axes[0].set_title("log10(N + 1)")
    axes[1].set_title("RMSE (mm/h)")
    stride = max(1, len(height_labels) // 15)
    ticks = np.arange(0, len(height_labels), stride)
    for ax in axes:
        ax.set_xticks(range(len(intensity_labels)), intensity_labels, rotation=35, ha="right")
        ax.set_yticks(ticks, [height_labels[index] for index in ticks], fontsize=8)
        ax.set_xlabel("Target intensity group")
    axes[0].set_ylabel("Absolute height group")
    fig.colorbar(images[0], ax=axes[0], shrink=0.75)
    fig.colorbar(images[1], ax=axes[1], shrink=0.75)
    fig.suptitle("Full-validation height × rain-intensity diagnostics")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(destination, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_group_comparison(
    cfb: Mapping[str, Any], precipitation_type: Mapping[str, Any], destination: Path, dpi: int
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(17, 10))
    for row, (title, groups) in enumerate(
        (("Signed distance to CFB", cfb), ("DPR precipitation type", precipitation_type))
    ):
        labels = list(groups)
        positions = np.arange(len(labels))
        count = [_metric(groups[key].get("count")) for key in labels]
        rmse = [_metric(groups[key].get("rmse")) for key in labels]
        bias = [_metric(groups[key].get("bias")) for key in labels]
        axes[row, 0].bar(positions, count, color="#457b9d")
        axes[row, 0].set_yscale("symlog", linthresh=1)
        axes[row, 0].set_ylabel("Voxel count")
        axes[row, 0].set_title(f"{title}: support")
        width = 0.38
        axes[row, 1].bar(positions - width / 2, rmse, width, label="RMSE", color="#e76f51")
        axes[row, 1].bar(positions + width / 2, bias, width, label="Bias", color="#2a9d8f")
        axes[row, 1].axhline(0, color="black", linewidth=0.7)
        axes[row, 1].set_ylabel("mm/h")
        axes[row, 1].set_title(f"{title}: error")
        axes[row, 1].legend()
        for ax in axes[row]:
            ax.set_xticks(positions, labels, rotation=25, ha="right")
            ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Full-validation CFB and precipitation-type diagnostics")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(destination, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _below_cfb_groups(diagnostic: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten the independent below-CFB rain metrics into plot/CSV groups."""

    rain = diagnostic.get("rain", {})
    if not isinstance(rain, Mapping):
        return {}
    all_values = rain.get("all")
    bins = rain.get("target_bins_mm_h", {})
    if not isinstance(all_values, Mapping) or not isinstance(bins, Mapping):
        return {}
    return {"all": dict(all_values), **{str(key): value for key, value in bins.items()}}


def _plot_below_cfb_diagnostic(
    diagnostic: Mapping[str, Any], destination: Path, dpi: int
) -> None:
    groups = _below_cfb_groups(diagnostic)
    labels = list(groups)
    positions = np.arange(len(labels))
    count = np.asarray([_metric(groups[key].get("count")) for key in labels])
    rmse = np.asarray([_metric(groups[key].get("rmse")) for key in labels])
    mae = np.asarray([_metric(groups[key].get("mae")) for key in labels])
    bias = np.asarray([_metric(groups[key].get("bias")) for key in labels])
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].bar(positions, count, color="#457b9d")
    axes[0].set_yscale("symlog", linthresh=1)
    axes[0].set_ylabel("Native-positive voxel count")
    axes[0].set_title("Diagnostic support below CFB")
    width = 0.25
    axes[1].bar(positions - width, rmse, width, label="RMSE", color="#e76f51")
    axes[1].bar(positions, mae, width, label="MAE", color="#f4a261")
    axes[1].bar(positions + width, bias, width, label="Bias", color="#2a9d8f")
    axes[1].axhline(0.0, color="black", linewidth=0.7)
    axes[1].set_ylabel("mm/h")
    axes[1].set_title("Raw model output vs native positive DPR label")
    axes[1].legend()
    for axis in axes:
        axis.set_xticks(positions, labels, rotation=30, ha="right")
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle(
        "Independent below-CFB diagnostic (excluded from model selection)"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(destination, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _write_filewise_csv(filewise: Mapping[str, Any], path: Path) -> None:
    per_file = filewise.get("per_file", {})
    if not isinstance(per_file, Mapping):
        return
    _write_group_csv(per_file, path)


def _write_macro_bootstrap_csv(filewise: Mapping[str, Any], path: Path) -> None:
    macro = filewise.get("macro_average", {})
    bootstrap = filewise.get("bootstrap", {})
    intervals = (
        bootstrap.get("confidence_interval", {})
        if isinstance(bootstrap, Mapping)
        else {}
    )
    fields = ("metric", "macro_average", "ci_low", "ci_high", "valid_replicates")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        if not isinstance(macro, Mapping):
            return
        for name, value in macro.items():
            interval = intervals.get(name, {}) if isinstance(intervals, Mapping) else {}
            writer.writerow(
                {
                    "metric": name,
                    "macro_average": value,
                    "ci_low": interval.get("low"),
                    "ci_high": interval.get("high"),
                    "valid_replicates": interval.get("valid_replicates"),
                }
            )


def _plot_filewise_macro(
    filewise: Mapping[str, Any], destination: Path, dpi: int
) -> None:
    per_file = filewise.get("per_file", {})
    macro = filewise.get("macro_average", {})
    bootstrap = filewise.get("bootstrap", {})
    intervals = (
        bootstrap.get("confidence_interval", {})
        if isinstance(bootstrap, Mapping)
        else {}
    )
    records = [
        (str(label), values)
        for label, values in per_file.items()
        if isinstance(values, Mapping) and int(values.get("count", 0)) > 0
    ]
    records.sort(key=lambda item: _metric(item[1].get("rmse")))
    rmse = np.asarray([_metric(values.get("rmse")) for _, values in records])
    bias = np.asarray([_metric(values.get("bias")) for _, values in records])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    positions = np.arange(len(records))
    axes[0].plot(positions, rmse, marker="o", markersize=3, label="RMSE")
    axes[0].plot(positions, bias, marker=".", markersize=3, label="Bias")
    axes[0].axhline(0.0, color="black", linewidth=0.7)
    axes[0].set_xlabel("Non-empty file/orbit (sorted by RMSE)")
    axes[0].set_ylabel("mm/h")
    axes[0].set_title("Per-file reliable-support errors")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    names = ("mae", "rmse", "bias")
    values = np.asarray([_metric(macro.get(name)) for name in names])
    lower = []
    upper = []
    for name, value in zip(names, values):
        interval = intervals.get(name, {}) if isinstance(intervals, Mapping) else {}
        low = _metric(interval.get("low"))
        high = _metric(interval.get("high"))
        lower.append(max(0.0, value - low) if np.isfinite(low) else 0.0)
        upper.append(max(0.0, high - value) if np.isfinite(high) else 0.0)
    axes[1].errorbar(
        np.arange(len(names)),
        values,
        yerr=np.asarray([lower, upper]),
        fmt="o",
        capsize=5,
        color="#e76f51",
    )
    axes[1].axhline(0.0, color="black", linewidth=0.7)
    axes[1].set_xticks(np.arange(len(names)), [name.upper() for name in names])
    axes[1].set_ylabel("Macro average (mm/h)")
    axes[1].set_title(
        f"Whole-file bootstrap CI (seed={bootstrap.get('seed', 'unknown')})"
    )
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("File/orbit macro evaluation")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(destination, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def generate_stratified_analysis(
    evaluation_json: Path, *, output_dir: Path | None = None, dpi: int = 150
) -> dict[str, Any]:
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    source = evaluation_json.expanduser().resolve()
    metrics, section = load_stratified_metrics(source)
    destination = (
        output_dir.expanduser().resolve()
        if output_dir is not None
        else source.parent / "stratified"
    )
    destination.mkdir(parents=True, exist_ok=True)
    by_height = metrics["by_height_km"]
    by_cfb = metrics["by_cfb_distance_km"]
    crossed = metrics["by_height_and_intensity_mm_h"]
    by_type = metrics["by_precipitation_type"]
    _plot_height(by_height, destination / "metrics_by_height.png", dpi)
    _plot_height_intensity(crossed, destination / "height_intensity_heatmaps.png", dpi)
    _plot_group_comparison(by_cfb, by_type, destination / "cfb_and_precipitation_type.png", dpi)
    _write_group_csv(by_height, destination / "metrics_by_height.csv")
    _write_group_csv(by_cfb, destination / "metrics_by_cfb_distance.csv")
    _write_group_csv(by_type, destination / "metrics_by_precipitation_type.csv")
    payload = json.loads(source.read_text(encoding="utf-8"))
    evaluation = payload.get(section, {})
    evaluation_metrics = (
        evaluation.get("metrics", {}) if isinstance(evaluation, Mapping) else {}
    )
    physical_drdz = (
        evaluation_metrics.get("physical_drdz")
        if isinstance(evaluation_metrics, Mapping)
        else None
    )
    if isinstance(physical_drdz, Mapping):
        drdz_all = physical_drdz.get("all", {})
        drdz_height = physical_drdz.get("by_midpoint_height_km", {})
        drdz_cfb = physical_drdz.get("by_midpoint_cfb_distance_km", {})
        drdz_intensity = physical_drdz.get(
            "by_pair_mean_target_intensity_mm_h", {}
        )
        drdz_type = physical_drdz.get("by_precipitation_type", {})
        if not all(
            isinstance(value, Mapping)
            for value in (
                drdz_all,
                drdz_height,
                drdz_cfb,
                drdz_intensity,
                drdz_type,
            )
        ):
            raise ValueError("physical_drdz has an invalid grouped-metric structure")
        _write_drdz_overall_csv(drdz_all, destination / "drdz_overall.csv")
        _write_group_csv(drdz_height, destination / "drdz_by_height.csv")
        _write_group_csv(
            drdz_cfb, destination / "drdz_by_cfb_distance.csv"
        )
        _write_group_csv(
            drdz_intensity, destination / "drdz_by_target_intensity.csv"
        )
        _write_group_csv(
            drdz_type, destination / "drdz_by_precipitation_type.csv"
        )
        _plot_drdz_height(
            drdz_height, destination / "drdz_by_height.png", dpi
        )
        _plot_drdz_groups(
            [
                ("Midpoint distance to CFB", drdz_cfb),
                ("Adjacent-level target rain intensity", drdz_intensity),
                ("DPR precipitation type", drdz_type),
            ],
            destination / "drdz_grouped.png",
            dpi,
        )
        drdz_filewise = physical_drdz.get("filewise")
        if isinstance(drdz_filewise, Mapping):
            _write_filewise_csv(
                drdz_filewise, destination / "drdz_metrics_by_file.csv"
            )
            _write_macro_bootstrap_csv(
                drdz_filewise,
                destination / "drdz_filewise_macro_bootstrap.csv",
            )
        (destination / "drdz_summary.md").write_text(
            "# 物理垂直降水率梯度 dR/dz\n\n"
            "- 定义：相邻原生高度层的 "
            "`(R[k+1]-R[k])/(z[k+1]-z[k])`，高度向上为正；\n"
            "- 单位：`mm h^-1 km^-1`；输入60层，输出59个高度中点；\n"
            "- 主评价只使用两个端点都在可靠mask内的相邻层对，不跨缺测、"
            "padding或CFB弱监督区域；\n"
            "- 当前可靠mask是连续正降水条件口径，不包含有效零雨、雨顶或"
            "雨底发生边界，不能直接等同于PPT的“所有样本”口径；\n"
            f"- 有效梯度对：`{int(drdz_all.get('count', 0)):,}`；\n"
            f"- MAE：`{_metric(drdz_all.get('mae')):.6f}`，RMSE："
            f"`{_metric(drdz_all.get('rmse')):.6f}`，Bias："
            f"`{_metric(drdz_all.get('bias')):.6f}`；\n"
            f"- Pearson r：`{_metric(drdz_all.get('pearson_r')):.6f}`；\n"
            "- 平均绝对梯度幅值比（预测/标签）："
            f"`{_metric(drdz_all.get('mean_abs_gradient_ratio')):.6f}`；\n"
            "- 目标梯度超过阈值时的符号一致率："
            f"`{_metric(drdz_all.get('sign_agreement_fraction')):.6f}`。\n\n"
            "幅值比明显小于1表示垂直变化被平滑，明显大于1表示可能过度振荡；"
            "相关性、误差、幅值比和分高度结果需要联合判断。\n",
            encoding="utf-8",
        )
    diagnostics = (
        evaluation_metrics.get("diagnostics", {})
        if isinstance(evaluation_metrics, Mapping)
        else {}
    )
    below_cfb = (
        diagnostics.get("below_cfb_native_positive")
        if isinstance(diagnostics, Mapping)
        else None
    )
    if isinstance(below_cfb, Mapping) and _below_cfb_groups(below_cfb):
        _plot_below_cfb_diagnostic(
            below_cfb, destination / "below_cfb_native_positive.png", dpi
        )
        _write_group_csv(
            _below_cfb_groups(below_cfb),
            destination / "below_cfb_native_positive.csv",
        )
    filewise = (
        evaluation_metrics.get("filewise")
        if isinstance(evaluation_metrics, Mapping)
        else None
    )
    if isinstance(filewise, Mapping):
        _plot_filewise_macro(
            filewise, destination / "filewise_macro_bootstrap.png", dpi
        )
        _write_filewise_csv(filewise, destination / "metrics_by_file.csv")
        _write_macro_bootstrap_csv(
            filewise, destination / "filewise_macro_bootstrap.csv"
        )
    summary = {
        "source": str(source),
        "evaluation_section": section,
        "height_group_count": len(by_height),
        "cfb_distance_group_count": len(by_cfb),
        "precipitation_type_group_count": len(by_type),
        "has_below_cfb_native_positive_diagnostic": isinstance(
            below_cfb, Mapping
        ),
        "has_filewise_macro_bootstrap": isinstance(filewise, Mapping),
        "has_physical_drdz": isinstance(physical_drdz, Mapping),
    }
    (destination / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (destination / "summary.md").write_text(
        "# 完整验证集分层指标\n\n"
        "- `metrics_by_height.png/csv`：逐高度样本数与误差；\n"
        "- `height_intensity_heatmaps.png`：高度×目标强度的样本数与RMSE；\n"
        "- `cfb_and_precipitation_type.png`：相对CFB距离及层云/对流分组；\n"
        "- CFB以下不进入主评价；`below_cfb_native_positive.*` 单独展示"
        "原生正降水标签诊断，不参与模型选择；\n"
        "- `filewise_macro_bootstrap.*` 与 `metrics_by_file.csv`：逐轨宏平均"
        "及以整轨为抽样单位的可复现bootstrap区间；\n"
        "- `drdz_summary.md`、`drdz_*.csv/png`：统一物理dR/dz总体、逐高度、"
        "相对CFB、强度、类型及逐轨诊断。\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    result = generate_stratified_analysis(
        args.evaluation_json, output_dir=args.output_dir, dpi=args.dpi
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
