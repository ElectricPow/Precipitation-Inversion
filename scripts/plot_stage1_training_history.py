#!/usr/bin/env python3
"""Visualize every epoch recorded by stage-one training.

The script is intentionally independent of CUDA and can be rerun while a job
is still in progress.  All figures, a flat epoch CSV, JSON summary, and a short
Markdown report are written below ``OUTPUT_DIR/analysis/training_history``.
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


BIN_ORDER = ("lt_1", "1_to_5", "5_to_10", "10_to_30", "ge_30")
BIN_LABELS = {
    "lt_1": "<1",
    "1_to_5": "1–5",
    "5_to_10": "5–10",
    "10_to_30": "10–30",
    "ge_30": "≥30",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="Training output directory.")
    parser.add_argument("--analysis-dir", type=Path)
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def _finite(value: Any) -> float:
    if value is None:
        return float("nan")
    result = float(value)
    return result if math.isfinite(result) else float("nan")


def load_history(path: Path) -> list[dict[str, Any]]:
    """Load, validate, sort, and de-duplicate JSONL epoch records."""

    if not path.is_file():
        raise FileNotFoundError(f"training history not found: {path}")
    by_epoch: dict[int, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON at {path}:{line_number}") from error
        if not isinstance(row, dict) or "epoch" not in row:
            raise ValueError(f"history row {line_number} has no epoch")
        by_epoch[int(row["epoch"])] = row
    if not by_epoch:
        raise ValueError(f"training history is empty: {path}")
    return [by_epoch[epoch] for epoch in sorted(by_epoch)]


def _get(row: Mapping[str, Any], path: str) -> float:
    value: Any = row
    for key in path.split("."):
        if not isinstance(value, Mapping) or key not in value:
            return float("nan")
        value = value[key]
    return _finite(value)


def _series(rows: list[dict[str, Any]], path: str) -> np.ndarray:
    return np.asarray([_get(row, path) for row in rows], dtype=np.float64)


def flatten_mapping(value: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            result.update(flatten_mapping(item, name))
        elif not isinstance(item, (list, tuple)):
            result[name] = item
    return result


def _best_position(rows: list[dict[str, Any]]) -> int:
    values = _series(rows, "val.metrics.rain.all.rmse")
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("no finite validation rain RMSE is present")
    return int(np.nanargmin(values))


def _plot_line(ax, epochs, rows, train_path, val_path, ylabel, *, zero=False):
    ax.plot(epochs, _series(rows, train_path), label="train", linewidth=1.7)
    ax.plot(epochs, _series(rows, val_path), label="validation", linewidth=1.7)
    if zero:
        ax.axhline(0.0, color="black", linewidth=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend()


def plot_overview(
    rows: list[dict[str, Any]], config: Mapping[str, Any], destination: Path, dpi: int
) -> None:
    epochs = np.asarray([int(row["epoch"]) for row in rows])
    best = _best_position(rows)
    best_epoch = int(epochs[best])
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))

    _plot_line(
        axes[0, 0], epochs, rows, "train.loss", "val.loss", "Configured objective"
    )
    axes[0, 0].set_title("Optimization objective")
    _plot_line(
        axes[0, 1], epochs, rows,
        "train.metrics.rain.all.mae", "val.metrics.rain.all.mae", "MAE (mm/h)",
    )
    axes[0, 1].set_title("Physical-space MAE")
    _plot_line(
        axes[0, 2], epochs, rows,
        "train.metrics.rain.all.rmse", "val.metrics.rain.all.rmse", "RMSE (mm/h)",
    )
    axes[0, 2].axvline(best_epoch, color="#e63946", linestyle="--", label=f"best={best_epoch}")
    axes[0, 2].legend()
    axes[0, 2].set_title("Physical-space RMSE")

    _plot_line(
        axes[1, 0], epochs, rows,
        "train.metrics.rain.all.bias", "val.metrics.rain.all.bias", "Bias (mm/h)", zero=True,
    )
    axes[1, 0].set_title("Physical-space bias (prediction − target)")
    _plot_line(
        axes[1, 1], epochs, rows,
        "train.metrics.rain.all.r2", "val.metrics.rain.all.r2", "R²", zero=True,
    )
    axes[1, 1].set_title("Physical-space R²")
    _plot_line(
        axes[1, 2], epochs, rows,
        "train.metrics.rain.all.pearson_r", "val.metrics.rain.all.pearson_r", "Pearson r", zero=True,
    )
    axes[1, 2].set_title("Physical-space correlation")

    _plot_line(
        axes[2, 0], epochs, rows,
        "train.metrics.log.rmse", "val.metrics.log.rmse", "RMSE in log1p space",
    )
    axes[2, 0].set_title("Log-space RMSE")
    axes[2, 1].plot(epochs, _series(rows, "learning_rate"), color="#6a4c93")
    axes[2, 1].set_yscale("log")
    axes[2, 1].set_xlabel("Epoch")
    axes[2, 1].set_ylabel("Learning rate")
    axes[2, 1].set_title("Cosine learning-rate schedule")
    axes[2, 1].grid(alpha=0.25)
    axes[2, 2].plot(epochs, _series(rows, "train.duration_seconds"), label="train")
    axes[2, 2].plot(epochs, _series(rows, "val.duration_seconds"), label="validation")
    axes[2, 2].set_xlabel("Epoch")
    axes[2, 2].set_ylabel("Seconds")
    axes[2, 2].set_title("Epoch duration")
    axes[2, 2].grid(alpha=0.25)
    axes[2, 2].legend()

    model = config.get("model", {})
    data = config.get("data", {})
    optimizer = config.get("optimizer", {})
    fig.suptitle(
        "Stage-1 height-preserving 3D U-Net training history\n"
        f"epochs={len(rows)}, per-GPU batch={data.get('batch_size')}, "
        f"base channels={model.get('base_channels')}, optimizer={optimizer.get('name')}, "
        f"initial LR={optimizer.get('learning_rate')}, best epoch={best_epoch}",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(destination, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_loss_components(
    rows: list[dict[str, Any]], destination: Path, dpi: int
) -> bool:
    """Plot I, raw G, weighted G, and contribution fraction when available."""

    epochs = np.asarray([int(row["epoch"]) for row in rows])
    paths = (
        ("primary_log_smooth_l1", "Primary I: log1p Smooth-L1"),
        ("physical_drdz_smooth_l1", "Raw G: physical dR/dz Smooth-L1"),
        ("weighted_physical_drdz", "Weighted G: lambda_G x raw G"),
        ("weighted_physical_drdz_fraction", "Weighted G / total objective"),
    )
    available = any(
        np.isfinite(_series(rows, f"train.loss_components.{key}")).any()
        or np.isfinite(_series(rows, f"val.loss_components.{key}")).any()
        for key, _ in paths
    )
    if not available:
        return False
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for axis, (key, title) in zip(axes.flat, paths):
        axis.plot(
            epochs,
            _series(rows, f"train.loss_components.{key}"),
            label="train",
        )
        axis.plot(
            epochs,
            _series(rows, f"val.loss_components.{key}"),
            label="validation",
        )
        axis.set_xlabel("Epoch")
        axis.set_title(title)
        axis.grid(alpha=0.25)
        axis.legend()
    axes[1, 1].set_ylabel("Fraction")
    fig.suptitle("Stage-1 configured loss components", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(destination, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_stage3_three_task_loss_components(
    rows: list[dict[str, Any]], destination: Path, dpi: int
) -> bool:
    """Plot C2/D0 physical anchors and rain contribution separately."""

    probe = _series(rows, "train.loss_components.stage2_anchor.support")
    if not np.isfinite(probe).any():
        return False
    epochs = np.asarray([int(row["epoch"]) for row in rows])
    paths = (
        ("stage2_anchor.support", "Stage-2 support anchor"),
        (
            "stage2_anchor.reflectivity_standardized_dbz",
            "Stage-2 W1.25 dBZ anchor",
        ),
        ("rain_task.primary_log_smooth_l1", "Final-rain primary I"),
        ("rain_task.physical_drdz_smooth_l1", "Final-rain physical G"),
        ("weighted_stage2_anchor", "Weighted Stage-2 physical objective"),
        ("weighted_stage2_fraction", "Stage-2 physical / total objective"),
        ("weighted_rain_task", "Weighted final-rain objective"),
        ("weighted_rain_fraction", "Final rain / total objective"),
    )
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    for axis, (path, title) in zip(axes.flat, paths):
        axis.plot(
            epochs,
            _series(rows, f"train.loss_components.{path}"),
            label="train",
        )
        axis.plot(
            epochs,
            _series(rows, f"val.loss_components.{path}"),
            label="validation",
        )
        axis.set_xlabel("Epoch")
        axis.set_title(title)
        axis.grid(alpha=0.25)
        axis.legend()
    axes[1, 1].set_ylabel("Fraction")
    axes[1, 3].set_ylabel("Fraction")
    fig.suptitle(
        "Stage-3 support/dBZ/rain objective components", fontsize=15
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(destination, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_intensity_bins(rows: list[dict[str, Any]], destination: Path, dpi: int) -> None:
    epochs = np.asarray([int(row["epoch"]) for row in rows])
    best_epoch = int(epochs[_best_position(rows)])
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    for bin_name in BIN_ORDER:
        label = f"{BIN_LABELS[bin_name]} mm/h"
        prefix = f"val.metrics.rain.target_bins_mm_h.{bin_name}"
        axes[0, 0].plot(epochs, _series(rows, f"{prefix}.mae"), label=label)
        axes[0, 1].plot(epochs, _series(rows, f"{prefix}.rmse"), label=label)
        axes[1, 0].plot(epochs, _series(rows, f"{prefix}.bias"), label=label)
        axes[1, 1].plot(epochs, _series(rows, f"{prefix}.pearson_r"), label=label)
    for ax, title, ylabel in zip(
        axes.flat,
        ("Validation MAE", "Validation RMSE", "Validation bias", "Validation correlation"),
        ("MAE (mm/h)", "RMSE (mm/h)", "Bias (mm/h)", "Pearson r"),
    ):
        ax.axvline(best_epoch, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[1, 0].axhline(0, color="black", linewidth=0.7)
    fig.suptitle("Validation behavior by target precipitation intensity", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(destination, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_type_task(rows: list[dict[str, Any]], destination: Path, dpi: int) -> bool:
    """Plot auxiliary-loss share and profile classification quality."""

    if not np.isfinite(
        _series(rows, "val.metrics.precipitation_type.macro_f1")
    ).any():
        return False
    epochs = np.asarray([int(row["epoch"]) for row in rows])
    figure, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    paths = (
        ("loss_components.type_cross_entropy", "Type cross entropy"),
        ("loss_components.weighted_type_fraction", "Weighted type / total loss"),
        ("metrics.precipitation_type.macro_f1", "Macro-F1"),
        ("metrics.precipitation_type.balanced_accuracy", "Balanced accuracy"),
        ("metrics.precipitation_type.per_class.convective.recall", "Convective recall"),
        ("metrics.precipitation_type.accuracy", "Overall accuracy"),
    )
    for axis, (suffix, title) in zip(axes.flat, paths):
        axis.plot(epochs, _series(rows, f"train.{suffix}"), label="train")
        axis.plot(epochs, _series(rows, f"val.{suffix}"), label="validation")
        axis.set(title=title, xlabel="Epoch")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("DPR precipitation-type auxiliary task")
    figure.savefig(destination, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return True


def plot_generalization(rows: list[dict[str, Any]], destination: Path, dpi: int) -> None:
    epochs = np.asarray([int(row["epoch"]) for row in rows])
    train_loss = _series(rows, "train.loss")
    val_loss = _series(rows, "val.loss")
    train_rmse = _series(rows, "train.metrics.rain.all.rmse")
    val_rmse = _series(rows, "val.metrics.rain.all.rmse")
    best_epoch = int(epochs[_best_position(rows)])
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    axes[0].plot(epochs, val_loss - train_loss, color="#e76f51")
    axes[0].set_ylabel("Validation − train loss")
    axes[0].set_title("Loss generalization gap")
    axes[1].plot(epochs, val_rmse - train_rmse, color="#e76f51")
    axes[1].set_ylabel("Validation − train RMSE (mm/h)")
    axes[1].set_title("Physical RMSE gap")
    ratio = np.divide(val_rmse, train_rmse, out=np.full_like(val_rmse, np.nan), where=train_rmse > 0)
    axes[2].plot(epochs, ratio, color="#e76f51")
    axes[2].axhline(1.0, color="black", linewidth=0.8)
    axes[2].set_ylabel("Validation / train RMSE")
    axes[2].set_title("Relative generalization gap")
    for ax in axes:
        ax.axvline(best_epoch, color="#457b9d", linestyle="--", label=f"best epoch {best_epoch}")
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("Overfitting and model-selection diagnostics", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(destination, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_epoch_csv(rows: list[dict[str, Any]], path: Path) -> None:
    flattened = [flatten_mapping(row) for row in rows]
    fieldnames = sorted({key for row in flattened for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flattened)


def write_summary(
    rows: list[dict[str, Any]], config: Mapping[str, Any], analysis_dir: Path
) -> dict[str, Any]:
    position = _best_position(rows)
    best = rows[position]
    final = rows[-1]
    summary = {
        "recorded_epoch_count": len(rows),
        "first_epoch": int(rows[0]["epoch"]),
        "last_epoch": int(final["epoch"]),
        "best_epoch_by_validation_rain_rmse": int(best["epoch"]),
        "best_validation_rain": best["val"]["metrics"]["rain"]["all"],
        "best_training_rain": best["train"]["metrics"]["rain"]["all"],
        "final_validation_rain": final["val"]["metrics"]["rain"]["all"],
        "final_training_rain": final["train"]["metrics"]["rain"]["all"],
        "best_validation_loss": best["val"]["loss"],
        "final_validation_loss": final["val"]["loss"],
        "best_validation_loss_components": best["val"].get(
            "loss_components", {}
        ),
        "final_validation_loss_components": final["val"].get(
            "loss_components", {}
        ),
        "best_validation_precipitation_type": best["val"].get("metrics", {}).get(
            "precipitation_type"
        ),
        "final_global_step": final["global_step"],
        "configuration": config,
    }
    (analysis_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    model = config.get("model", {})
    data = config.get("data", {})
    training = config.get("training", {})
    optimizer = config.get("optimizer", {})
    loss_config = config.get("loss", {})
    gradient_config = loss_config.get("physical_gradient", {})
    type_config = config.get("type_task", {})
    rain = summary["best_validation_rain"]
    markdown = f"""# Stage-1训练曲线摘要

- 已记录epoch：{len(rows)}（{rows[0]['epoch']}–{final['epoch']}）
- 最优checkpoint对应epoch：{best['epoch']}
- 最优验证RMSE：{rain['rmse']:.6f} mm/h
- 最优验证MAE：{rain['mae']:.6f} mm/h
- 最优验证Bias：{rain['bias']:.6f} mm/h
- 最优验证Pearson r：{rain['pearson_r']:.6f}
- 最优验证R²：{rain['r2']:.6f}

## 关键配置

- 输入/输出通道：{model.get('in_channels')} / {model.get('out_channels')}
- 基础通道及倍数：{model.get('base_channels')} / {model.get('channel_multipliers')}
- 每GPU batch size：{data.get('batch_size')}
- 梯度累积：{training.get('accumulation_steps')}
- 优化器：{optimizer.get('name')}
- 初始学习率：{optimizer.get('learning_rate')}
- 权重衰减：{optimizer.get('weight_decay')}
- AMP：{training.get('amp')}
- 主损失：{loss_config.get('name')}（beta={loss_config.get('beta')}）
- 物理G损失：enabled={gradient_config.get('enabled', False)}，lambda={gradient_config.get('weight', 0.0)}，beta={gradient_config.get('beta', 1.0)}
- 类型辅助任务：enabled={type_config.get('enabled', False)}，head={type_config.get('head', {}).get('kind')}，lambda={type_config.get('loss_weight', 0.0)}，rain_feedback={type_config.get('rain_feedback', False)}

详细逐epoch数值见 `epoch_metrics.csv`，原始完整配置及摘要见 `summary.json`。
"""
    (analysis_dir / "summary.md").write_text(markdown, encoding="utf-8")
    return summary


def generate_training_analysis(
    output_dir: Path, *, analysis_dir: Path | None = None, dpi: int = 160
) -> dict[str, Any]:
    output = output_dir.expanduser().resolve()
    destination = (
        analysis_dir.expanduser().resolve()
        if analysis_dir is not None
        else output / "analysis" / "training_history"
    )
    destination.mkdir(parents=True, exist_ok=True)
    rows = load_history(output / "metrics.jsonl")
    config_path = output / "resolved_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("resolved_config.json root must be a mapping")
    plot_overview(rows, config, destination / "training_overview.png", dpi)
    plot_loss_components(rows, destination / "loss_components.png", dpi)
    plot_stage3_three_task_loss_components(
        rows, destination / "stage3_multitask_loss_components.png", dpi
    )
    plot_type_task(rows, destination / "type_task_history.png", dpi)
    plot_intensity_bins(rows, destination / "validation_intensity_bins.png", dpi)
    plot_generalization(rows, destination / "generalization_gap.png", dpi)
    write_epoch_csv(rows, destination / "epoch_metrics.csv")
    summary = write_summary(rows, config, destination)
    print(
        f"Training analysis saved to {destination}; "
        f"best epoch={summary['best_epoch_by_validation_rain_rmse']}",
        flush=True,
    )
    return summary


def main() -> None:
    args = parse_args()
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")
    generate_training_analysis(
        args.output_dir, analysis_dir=args.analysis_dir, dpi=args.dpi
    )


if __name__ == "__main__":
    main()
