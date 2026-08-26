#!/usr/bin/env python3
"""Compare best-checkpoint rain fields with DPR labels on fixed test orbits."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm, Normalize  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from precipitation_inversion.data.nc_reader import read_nc_sample  # noqa: E402
from precipitation_inversion.data.patch_dataset import (  # noqa: E402
    Stage1PatchDataset,
    stage1_patch_dataset_kwargs,
)
from precipitation_inversion.inference.sliding_window import predict_full_orbit  # noqa: E402
from precipitation_inversion.metrics.regression import PrecipitationRegressionMetrics  # noqa: E402
from precipitation_inversion.models.unet3d import Stage1UNet3D  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--sample-count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--height-km", type=float, default=2.0)
    parser.add_argument("--max-scatter-points", type=int, default=200_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif value == "cuda":
        value = "cuda:0"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def build_model(config: Mapping[str, Any]) -> Stage1UNet3D:
    values = config["model"]
    return Stage1UNet3D(
        in_channels=int(values["in_channels"]),
        out_channels=int(values["out_channels"]),
        base_channels=int(values["base_channels"]),
        channel_multipliers=tuple(values["channel_multipliers"]),
        max_groups=int(values["max_groups"]),
        bottleneck_dropout=float(values["bottleneck_dropout"]),
    )


def select_file_ids(
    file_count: int,
    sample_count: int,
    seed: int,
    *,
    eligible_ids: list[int] | None = None,
) -> list[int]:
    """Choose reproducible eligible test orbits without replacement."""

    if file_count <= 0 or sample_count <= 0:
        raise ValueError("file_count and sample_count must be positive")
    candidates = list(range(file_count)) if eligible_ids is None else list(eligible_ids)
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError("eligible_ids must be non-empty and unique")
    if any(value < 0 or value >= file_count for value in candidates):
        raise ValueError("eligible file id lies outside the dataset")
    count = min(len(candidates), sample_count)
    rng = np.random.default_rng(seed)
    return sorted(int(value) for value in rng.choice(candidates, count, replace=False))


def paired_metrics(target: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    selected = mask & np.isfinite(target) & np.isfinite(prediction)
    if not selected.any():
        return {"count": 0, "mae": None, "rmse": None, "bias": None, "pearson_r": None}
    reference = target[selected].astype(np.float64)
    estimate = prediction[selected].astype(np.float64)
    difference = estimate - reference
    correlation = float("nan")
    if reference.size > 1 and reference.std() > 0 and estimate.std() > 0:
        correlation = float(np.corrcoef(reference, estimate)[0, 1])
    return {
        "count": int(reference.size),
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "bias": float(np.mean(difference)),
        "pearson_r": correlation if math.isfinite(correlation) else None,
    }


def cumulative_distance_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Cumulative great-circle distance along one across-track A–B line."""

    latitude = np.radians(np.asarray(lat, dtype=np.float64))
    longitude = np.radians(np.asarray(lon, dtype=np.float64))
    result = np.zeros(latitude.size, dtype=np.float64)
    if latitude.size <= 1:
        return result
    dlat = np.diff(latitude)
    dlon = np.diff(longitude)
    haversine = np.sin(dlat / 2) ** 2 + np.cos(latitude[:-1]) * np.cos(latitude[1:]) * np.sin(dlon / 2) ** 2
    result[1:] = np.cumsum(2 * 6371.0 * np.arcsin(np.sqrt(np.clip(haversine, 0, 1))))
    return result


def vertical_statistics(
    target: np.ndarray, prediction: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z_size = target.shape[2]
    rmse = np.full(z_size, np.nan)
    bias = np.full(z_size, np.nan)
    correlation = np.full(z_size, np.nan)
    for level in range(z_size):
        values = paired_metrics(target[:, :, level], prediction[:, :, level], mask[:, :, level])
        if values["count"]:
            rmse[level] = float(values["rmse"])
            bias[level] = float(values["bias"])
            correlation[level] = (
                np.nan if values["pearson_r"] is None else float(values["pearson_r"])
            )
    return rmse, bias, correlation


def _rain_norm(target: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> LogNorm:
    values = np.concatenate((target[mask & (target > 0)], prediction[mask & (prediction > 0)]))
    upper = max(1.0, float(np.nanpercentile(values, 99.5))) if values.size else 1.0
    return LogNorm(vmin=0.05, vmax=upper)


def _display_metric(value: Any, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    numeric = float(value)
    return f"{numeric:.{digits}f}" if math.isfinite(numeric) else "N/A"


def _scatter_swath(ax, lon, lat, values, title, *, cmap, norm):
    valid = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(values)
    artist = ax.scatter(
        lon[valid], lat[valid], c=values[valid], s=8, cmap=cmap, norm=norm,
        edgecolors="none", rasterized=True,
    )
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    return artist


def _subsample_pair(x: np.ndarray, y: np.ndarray, maximum: int, rng) -> tuple[np.ndarray, np.ndarray]:
    if x.size <= maximum:
        return x, y
    indices = rng.choice(x.size, maximum, replace=False)
    return x[indices], y[indices]


def plot_sample_diagnostics(
    *,
    target: np.ndarray,
    prediction: np.ndarray,
    evaluation_mask: np.ndarray,
    positive_mask: np.ndarray,
    z: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    sample_id: str,
    height_km: float,
    max_scatter_points: int,
    rng: np.random.Generator,
    destination: Path,
    dpi: int,
) -> dict[str, Any]:
    """Create map, A–B section, distribution, and correlation diagnostics."""

    height_index = int(np.argmin(np.abs(z - height_km)))
    score = np.where(positive_mask, target, 0.0).sum(axis=(1, 2))
    profile_scan = int(np.argmax(score))
    rain_norm = _rain_norm(target, prediction, evaluation_mask)
    target_plot = np.where(evaluation_mask, target, np.nan)
    prediction_plot = np.where(evaluation_mask, prediction, np.nan)
    difference = np.where(evaluation_mask, prediction - target, np.nan)
    finite_difference = np.abs(difference[np.isfinite(difference)])
    difference_limit = (
        max(1.0, float(np.percentile(finite_difference, 99)))
        if finite_difference.size
        else 1.0
    )

    fig, axes = plt.subplots(3, 3, figsize=(20, 17))
    layer_target = target_plot[:, :, height_index]
    layer_prediction = prediction_plot[:, :, height_index]
    layer_difference = difference[:, :, height_index]
    target_artist = _scatter_swath(
        axes[0, 0], lon, lat, layer_target,
        f"DPR target at {z[height_index]:.2f} km", cmap="turbo", norm=rain_norm,
    )
    prediction_artist = _scatter_swath(
        axes[0, 1], lon, lat, layer_prediction,
        f"3D U-Net prediction at {z[height_index]:.2f} km", cmap="turbo", norm=rain_norm,
    )
    error_artist = _scatter_swath(
        axes[0, 2], lon, lat, layer_difference,
        "Prediction − DPR", cmap="coolwarm", norm=Normalize(-difference_limit, difference_limit),
    )
    for ax in axes[0, :]:
        valid_line = np.isfinite(lat[profile_scan]) & np.isfinite(lon[profile_scan])
        ax.plot(lon[profile_scan, valid_line], lat[profile_scan, valid_line], color="white", linewidth=1.2)
        ax.text(0.02, 0.98, "A", transform=ax.transAxes, va="top", color="white", weight="bold")
        ax.text(0.95, 0.98, "B", transform=ax.transAxes, va="top", color="white", weight="bold")
    fig.colorbar(target_artist, ax=axes[0, 0], shrink=0.78, label="Rain rate (mm/h)")
    fig.colorbar(prediction_artist, ax=axes[0, 1], shrink=0.78, label="Rain rate (mm/h)")
    fig.colorbar(error_artist, ax=axes[0, 2], shrink=0.78, label="Error (mm/h)")

    line_valid = np.isfinite(lat[profile_scan]) & np.isfinite(lon[profile_scan])
    distance = cumulative_distance_km(lat[profile_scan, line_valid], lon[profile_scan, line_valid])
    sections = (
        (target_plot[profile_scan, line_valid, :].T, "DPR target A–B section", "turbo", rain_norm),
        (prediction_plot[profile_scan, line_valid, :].T, "Prediction A–B section", "turbo", rain_norm),
        (difference[profile_scan, line_valid, :].T, "A–B error section", "coolwarm", Normalize(-difference_limit, difference_limit)),
    )
    for ax, (values, title, cmap, norm) in zip(axes[1, :], sections):
        artist = ax.pcolormesh(distance, z, np.ma.masked_invalid(values), shading="auto", cmap=cmap, norm=norm)
        ax.set_xlabel("Distance A–B (km)")
        ax.set_ylabel("Height (km)")
        ax.set_title(f"{title}; scan={profile_scan}")
        fig.colorbar(artist, ax=ax, shrink=0.78, label="mm/h")

    all_metrics = paired_metrics(target, prediction, evaluation_mask)
    positive_metrics = paired_metrics(target, prediction, positive_mask)
    x = target[positive_mask]
    y = prediction[positive_mask]
    x, y = _subsample_pair(x, y, max_scatter_points, rng)
    axes[2, 0].hexbin(np.log1p(x), np.log1p(y), gridsize=65, bins="log", mincnt=1, cmap="viridis")
    if x.size:
        limit = max(float(np.log1p(x).max()), float(np.log1p(y).max()), 1e-6)
        axes[2, 0].plot([0, limit], [0, limit], "r--", linewidth=1)
        axes[2, 0].set_xlim(0, limit)
        axes[2, 0].set_ylim(0, limit)
    axes[2, 0].set_xlabel("DPR log1p(rain rate)")
    axes[2, 0].set_ylabel("Prediction log1p(rain rate)")
    axes[2, 0].set_title(
        "Positive-rain correlation\n"
        f"n={positive_metrics['count']:,}, "
        f"RMSE={_display_metric(positive_metrics['rmse'])}, "
        f"r={_display_metric(positive_metrics['pearson_r'])}"
    )

    target_positive = target[positive_mask]
    predicted_positive = prediction[positive_mask]
    upper = (
        max(float(np.nanmax(target_positive)), float(np.nanmax(predicted_positive)), 1.0)
        if target_positive.size
        else 1.0
    )
    bins = np.geomspace(0.01, upper, 70)
    axes[2, 1].hist(target_positive[target_positive > 0], bins=bins, histtype="step", linewidth=2, label="DPR target")
    axes[2, 1].hist(predicted_positive[predicted_positive > 0], bins=bins, histtype="step", linewidth=2, label="prediction")
    axes[2, 1].set_xscale("log")
    axes[2, 1].set_yscale("log")
    axes[2, 1].set_xlabel("Rain rate (mm/h)")
    axes[2, 1].set_ylabel("Voxel count")
    axes[2, 1].set_title("Positive-rain distribution")
    axes[2, 1].legend()

    rmse, bias, correlation = vertical_statistics(target, prediction, positive_mask)
    axes[2, 2].plot(rmse, z, label="RMSE", color="#e76f51")
    axes[2, 2].plot(bias, z, label="bias", color="#457b9d")
    axes[2, 2].axvline(0, color="black", linewidth=0.7)
    axes[2, 2].set_xlabel("Error (mm/h)")
    axes[2, 2].set_ylabel("Height (km)")
    axes[2, 2].set_title("Positive-rain error by height")
    correlation_axis = axes[2, 2].twiny()
    correlation_axis.plot(correlation, z, label="Pearson r", color="#2a9d8f")
    correlation_axis.set_xlabel("Pearson r")
    correlation_axis.set_xlim(-0.1, 1.0)
    axes[2, 2].legend(loc="lower left")
    correlation_axis.legend(loc="lower right")
    axes[2, 2].grid(alpha=0.25)

    fig.suptitle(
        f"Stage-1 test-orbit diagnostic: {sample_id}\n"
        f"all valid: MAE={_display_metric(all_metrics['mae'])}, "
        f"RMSE={_display_metric(all_metrics['rmse'])}, "
        f"bias={_display_metric(all_metrics['bias'])}; positive target: "
        f"MAE={_display_metric(positive_metrics['mae'])}, "
        f"RMSE={_display_metric(positive_metrics['rmse'])}",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(destination, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return {
        "height_index": height_index,
        "height_km": float(z[height_index]),
        "ab_scan_index": profile_scan,
        "all_valid_metrics": all_metrics,
        "positive_target_metrics": positive_metrics,
        "vertical_rmse_positive": rmse.tolist(),
        "vertical_bias_positive": bias.tolist(),
        "vertical_pearson_r_positive": correlation.tolist(),
    }


def plot_aggregate(
    records: list[dict[str, Any]], target: np.ndarray, prediction: np.ndarray,
    mask: np.ndarray, z: np.ndarray, destination: Path, max_points: int,
    rng: np.random.Generator, dpi: int,
) -> None:
    selected_target = target[mask]
    selected_prediction = prediction[mask]
    x, y = _subsample_pair(selected_target, selected_prediction, max_points, rng)
    rmse, bias, correlation = vertical_statistics(target, prediction, mask)
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    axes[0, 0].hexbin(np.log1p(x), np.log1p(y), gridsize=70, bins="log", mincnt=1, cmap="viridis")
    limit = max(float(np.log1p(x).max()), float(np.log1p(y).max()), 1e-6)
    axes[0, 0].plot([0, limit], [0, limit], "r--")
    axes[0, 0].set_xlabel("DPR log1p(rain rate)")
    axes[0, 0].set_ylabel("Prediction log1p(rain rate)")
    axes[0, 0].set_title("Combined positive-rain correlation")

    upper = max(float(selected_target.max()), float(selected_prediction.max()), 1.0)
    bins = np.geomspace(0.01, upper, 75)
    axes[0, 1].hist(selected_target[selected_target > 0], bins=bins, histtype="step", linewidth=2, label="DPR")
    axes[0, 1].hist(selected_prediction[selected_prediction > 0], bins=bins, histtype="step", linewidth=2, label="prediction")
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xlabel("Rain rate (mm/h)")
    axes[0, 1].set_ylabel("Voxel count")
    axes[0, 1].set_title("Combined positive-rain distributions")
    axes[0, 1].legend()

    labels = [f"sample {record['selection_position']}" for record in records]
    rmse_values = [record["positive_target_metrics"]["rmse"] for record in records]
    mae_values = [record["positive_target_metrics"]["mae"] for record in records]
    positions = np.arange(len(records))
    width = 0.38
    axes[1, 0].bar(positions - width / 2, rmse_values, width, label="RMSE")
    axes[1, 0].bar(positions + width / 2, mae_values, width, label="MAE")
    axes[1, 0].set_xticks(positions, labels, rotation=20)
    axes[1, 0].set_ylabel("mm/h")
    axes[1, 0].set_title("Per-orbit positive-rain errors")
    axes[1, 0].legend()

    axes[1, 1].plot(rmse, z, label="RMSE", color="#e76f51")
    axes[1, 1].plot(bias, z, label="bias", color="#457b9d")
    axes[1, 1].set_xlabel("Error (mm/h)")
    axes[1, 1].set_ylabel("Height (km)")
    axes[1, 1].set_title("Combined error and correlation by height")
    correlation_axis = axes[1, 1].twiny()
    correlation_axis.plot(correlation, z, label="Pearson r", color="#2a9d8f")
    correlation_axis.set_xlabel("Pearson r")
    correlation_axis.set_xlim(-0.1, 1)
    axes[1, 1].legend(loc="lower left")
    correlation_axis.legend(loc="lower right")
    axes[1, 1].grid(alpha=0.25)
    fig.suptitle("Fixed test-orbit aggregate diagnostics", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(destination, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def generate_prediction_analysis(
    checkpoint_path: Path,
    *,
    output_dir: Path | None = None,
    sample_count: int = 6,
    seed: int = 2026,
    height_km: float = 2.0,
    max_scatter_points: int = 200_000,
    device_name: str = "auto",
    dpi: int = 150,
) -> dict[str, Any]:
    checkpoint_file = checkpoint_path.expanduser().resolve()
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("checkpoint does not contain a configuration mapping")
    if sample_count <= 0 or max_scatter_points <= 0 or dpi <= 0:
        raise ValueError("sample_count, max_scatter_points, and dpi must be positive")
    destination = (
        output_dir.expanduser().resolve()
        if output_dir is not None
        else checkpoint_file.parent / "analysis" / "test_predictions"
    )
    destination.mkdir(parents=True, exist_ok=True)
    device = resolve_device(device_name)
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    data_config = config["data"]
    dataset_options = stage1_patch_dataset_kwargs(data_config, config["loss"])
    dataset = Stage1PatchDataset(
        project_path(data_config["test_index"]),
        project_path(data_config["normalization"]),
        positive_only=False,
        cache_size=int(data_config["cache_size"]),
        **dataset_options,
    )
    if len(dataset.feature_names) != int(config["model"]["in_channels"]):
        raise ValueError(
            "checkpoint model input channels do not match Dataset features: "
            f"{config['model']['in_channels']} != {dataset.feature_names}"
        )
    eligible_ids = [
        file_id
        for file_id, entry in enumerate(dataset.source_files)
        if int(entry.get("positive_count", 0)) > 0
    ]
    file_ids = select_file_ids(
        len(dataset.source_files), sample_count, seed, eligible_ids=eligible_ids
    )
    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    aggregate_targets: list[np.ndarray] = []
    aggregate_predictions: list[np.ndarray] = []
    aggregate_masks: list[np.ndarray] = []
    aggregate_metric = PrecipitationRegressionMetrics(tuple(config["loss"]["thresholds_mm_h"]))
    reference_z: np.ndarray | None = None

    for selection_position, file_id in enumerate(file_ids):
        entry = dataset.source_files[file_id]
        sample_id = str(entry["sample_id"])
        sample_directory = destination / f"sample_{selection_position:02d}"
        sample_directory.mkdir(parents=True, exist_ok=True)
        print(f"[{selection_position + 1}/{len(file_ids)}] predicting {sample_id}", flush=True)
        prediction = predict_full_orbit(
            model,
            dataset,
            file_id,
            device=device,
            batch_size=int(data_config["batch_size"]),
            num_workers=0,
            use_amp=bool(config["training"]["amp"]),
        )
        sample = read_nc_sample(
            entry["file_path"],
            variables=("z", "lat", "lon", "dbz_dpr", "pre_dpr", "cfb"),
            dtype=np.float32,
            build_masks=True,
        )
        target = sample.variables["pre_dpr"]
        evaluation_mask = (
            sample.masks["pre_valid_qc"]
            & sample.masks["dpr_reflectivity_valid"]
            & np.isfinite(target)
        )
        positive_mask = evaluation_mask & (target > 0)
        z = sample.variables["z"]
        reference_z = z
        diagnostics = plot_sample_diagnostics(
            target=target,
            prediction=prediction,
            evaluation_mask=evaluation_mask,
            positive_mask=positive_mask,
            z=z,
            lat=sample.variables["lat"],
            lon=sample.variables["lon"],
            sample_id=sample_id,
            height_km=height_km,
            max_scatter_points=max_scatter_points,
            rng=rng,
            destination=sample_directory / "diagnostics.png",
            dpi=dpi,
        )
        np.savez_compressed(
            sample_directory / "prediction_and_target.npz",
            prediction_rain_mm_h=prediction.astype(np.float32),
            target_rain_mm_h=target.astype(np.float32),
            evaluation_mask=evaluation_mask,
            positive_target_mask=positive_mask,
            heights_km=z.astype(np.float32),
        )
        record = {
            "selection_position": selection_position,
            "file_id": file_id,
            "sample_id": sample_id,
            "file_path": str(entry["file_path"]),
            "original_shape": list(target.shape),
            **diagnostics,
        }
        (sample_directory / "metrics.json").write_text(
            json.dumps(_json_safe(record), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        records.append(record)
        aggregate_targets.append(target)
        aggregate_predictions.append(prediction)
        aggregate_masks.append(positive_mask)
        aggregate_metric.update_rain(prediction, target, positive_mask)

    assert reference_z is not None
    combined_target = np.concatenate(aggregate_targets, axis=0)
    combined_prediction = np.concatenate(aggregate_predictions, axis=0)
    combined_mask = np.concatenate(aggregate_masks, axis=0)
    plot_aggregate(
        records,
        combined_target,
        combined_prediction,
        combined_mask,
        reference_z,
        destination / "aggregate_diagnostics.png",
        max_scatter_points,
        rng,
        dpi,
    )
    summary = {
        "checkpoint": str(checkpoint_file),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "selection": {
            "split": "test",
            "method": (
                "positive_count>0 eligible orbits; numpy.default_rng choice "
                "without replacement, then sorted"
            ),
            "seed": seed,
            "requested_count": sample_count,
            "eligible_orbit_count": len(eligible_ids),
            "selected_file_ids": file_ids,
        },
        "evaluation_mask": "pre_valid_qc AND dpr_reflectivity_valid",
        "aggregate_positive_target_metrics": aggregate_metric.compute()["rain"],
        "samples": records,
    }
    safe_summary = _json_safe(summary)
    (destination / "summary.json").write_text(
        json.dumps(safe_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    aggregate_all = safe_summary["aggregate_positive_target_metrics"]["all"]
    (destination / "summary.md").write_text(
        "# 固定测试轨道预测可视化摘要\n\n"
        f"- checkpoint epoch：{checkpoint['epoch']}\n"
        f"- 随机种子：{seed}\n"
        f"- 轨道数量：{len(file_ids)}\n"
        f"- 正降水体素数：{aggregate_all['count']:,}\n"
        f"- MAE：{aggregate_all['mae']:.6f} mm/h\n"
        f"- RMSE：{aggregate_all['rmse']:.6f} mm/h\n"
        f"- Bias：{aggregate_all['bias']:.6f} mm/h\n"
        f"- Pearson r：{aggregate_all['pearson_r']:.6f}\n\n"
        "每个 `sample_XX/diagnostics.png` 包含固定高度平面、A–B剖面、"
        "正降水分布、线性相关性和分高度误差。\n",
        encoding="utf-8",
    )
    print(f"Prediction analysis saved to {destination}", flush=True)
    return safe_summary


def main() -> None:
    args = parse_args()
    generate_prediction_analysis(
        args.checkpoint,
        output_dir=args.output_dir,
        sample_count=args.sample_count,
        seed=args.seed,
        height_km=args.height_km,
        max_scatter_points=args.max_scatter_points,
        device_name=args.device,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
