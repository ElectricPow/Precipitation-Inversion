#!/usr/bin/env python3
"""Visualize complete Stage-2 test-orbit support and DPR-dBZ predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for value in (PROJECT_ROOT, SRC_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from precipitation_inversion.data.stage2_patch_dataset import Stage2PatchDataset  # noqa: E402
from precipitation_inversion.inference.stage2_sliding_window import (  # noqa: E402
    predict_stage2_full_orbit,
    reconstruct_stage2_targets,
)
from precipitation_inversion.metrics.stage2_reflectivity import (  # noqa: E402
    Stage2ReflectivityMetrics,
    finite_metrics_for_json,
)
from precipitation_inversion.training.engine import load_checkpoint  # noqa: E402
from scripts.evaluate_stage2_unet3d import load_threshold_file  # noqa: E402
from scripts.train_stage2_unet3d import build_model, project_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--threshold-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--height-km", type=float, default=2.0)
    parser.add_argument("--max-scatter-points", type=int, default=200_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def select_file_ids(
    file_count: int,
    sample_count: int,
    seed: int,
    *,
    eligible_ids: Sequence[int] | None = None,
) -> list[int]:
    if file_count <= 0 or sample_count <= 0:
        raise ValueError("file_count and sample_count must be positive")
    candidates = (
        np.arange(file_count, dtype=np.int64)
        if eligible_ids is None
        else np.asarray(sorted(set(int(value) for value in eligible_ids)), dtype=np.int64)
    )
    if candidates.size == 0 or np.any((candidates < 0) | (candidates >= file_count)):
        raise ValueError("eligible file IDs are empty or out of range")
    count = min(sample_count, candidates.size)
    selected = np.random.default_rng(seed).choice(candidates, size=count, replace=False)
    return sorted(int(value) for value in selected)


def _safe_dbz(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.where(mask, values, np.nan)


def _vertical_metrics(
    probability: np.ndarray,
    prediction_dbz: np.ndarray,
    target_support: np.ndarray,
    target_dbz: np.ndarray,
    domain: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    recall = np.full(probability.shape[-1], np.nan)
    csi = np.full_like(recall, np.nan)
    mae = np.full_like(recall, np.nan)
    predicted = probability >= threshold
    for level in range(probability.shape[-1]):
        selected = domain[..., level]
        truth = target_support[..., level]
        estimate = predicted[..., level]
        tp = np.count_nonzero(selected & truth & estimate)
        fp = np.count_nonzero(selected & ~truth & estimate)
        fn = np.count_nonzero(selected & truth & ~estimate)
        if tp + fn:
            recall[level] = tp / (tp + fn)
        if tp + fp + fn:
            csi[level] = tp / (tp + fp + fn)
        target_selected = selected & truth
        if np.any(target_selected):
            mae[level] = np.mean(
                np.abs(
                    prediction_dbz[..., level][target_selected]
                    - target_dbz[..., level][target_selected]
                )
            )
    return recall, csi, mae


def plot_stage2_orbit_diagnostics(
    *,
    support_probability: np.ndarray,
    prediction_dbz: np.ndarray,
    target_support: np.ndarray,
    target_dbz: np.ndarray,
    support_domain: np.ndarray,
    gr_value_mask: np.ndarray,
    heights_km: np.ndarray,
    threshold: float,
    sample_id: str,
    height_km: float,
    max_scatter_points: int,
    rng: np.random.Generator,
    destination: Path,
    dpi: int = 150,
) -> dict[str, Any]:
    """Create maps, an A-B vertical section, distributions and correlations."""

    arrays = (
        support_probability,
        prediction_dbz,
        target_support,
        target_dbz,
        support_domain,
        gr_value_mask,
    )
    shape = np.asarray(support_probability).shape
    if len(shape) != 3 or any(np.asarray(value).shape != shape for value in arrays):
        raise ValueError("all Stage-2 diagnostic fields must share (nscan,nray,z)")
    if np.asarray(target_support).dtype != np.bool_ or np.asarray(support_domain).dtype != np.bool_:
        raise TypeError("target support and domain must be boolean")
    z = np.asarray(heights_km, dtype=np.float64)
    if z.shape != (shape[-1],) or not np.all(np.isfinite(z)):
        raise ValueError("heights_km must match the final axis")
    if max_scatter_points <= 0 or dpi <= 0:
        raise ValueError("plot limits must be positive")
    level = int(np.argmin(np.abs(z - height_km)))
    predicted_support = support_probability >= threshold
    # A-B is the physical ray column containing the most target-support voxels.
    ray_scores = target_support.sum(axis=(0, 2))
    ray_index = int(np.argmax(ray_scores))
    target_selected = support_domain & target_support
    predicted_selected = support_domain & predicted_support
    metric = Stage2ReflectivityMetrics(fss_radii=(1, 2, 4), dense_prediction=True)
    metric.update(
        prediction_dbz,
        predicted_support,
        target_dbz,
        target_support,
        support_domain,
    )
    computed = metric.compute()
    recall, csi, mae = _vertical_metrics(
        support_probability,
        prediction_dbz,
        target_support,
        target_dbz,
        support_domain,
        threshold,
    )

    figure, axes = plt.subplots(3, 4, figsize=(20, 14), constrained_layout=True)
    maps = (
        (gr_value_mask[..., level], "GR physical support", "gray", 0, 1),
        (target_support[..., level], "DPR target support", "gray", 0, 1),
        (support_probability[..., level], "Predicted support probability", "viridis", 0, 1),
        (predicted_support[..., level], f"Predicted support @ {threshold:.2f}", "gray", 0, 1),
    )
    for axis, (values, title, cmap, lower, upper) in zip(axes[0], maps):
        image = axis.imshow(values.T, origin="lower", aspect="auto", cmap=cmap, vmin=lower, vmax=upper)
        axis.set_title(f"{title}\n{z[level]:.3f} km")
        axis.set_xlabel("scan")
        axis.set_ylabel("ray")
        figure.colorbar(image, ax=axis, shrink=0.75)

    target_map = _safe_dbz(target_dbz[..., level], target_support[..., level])
    prediction_map = _safe_dbz(prediction_dbz[..., level], predicted_support[..., level])
    error_map = _safe_dbz(
        prediction_dbz[..., level] - target_dbz[..., level],
        target_support[..., level],
    )
    for axis, values, title, cmap, lower, upper in (
        (axes[1, 0], target_map, "Target DPR dBZ", "turbo", 0, 50),
        (axes[1, 1], prediction_map, "Predicted DPR dBZ", "turbo", 0, 50),
        (axes[1, 2], error_map, "dBZ error (prediction-target)", "coolwarm", -15, 15),
    ):
        image = axis.imshow(values.T, origin="lower", aspect="auto", cmap=cmap, vmin=lower, vmax=upper)
        axis.set_title(title)
        axis.set_xlabel("scan")
        axis.set_ylabel("ray")
        figure.colorbar(image, ax=axis, shrink=0.75)

    selected_indices = np.flatnonzero(target_selected)
    if selected_indices.size > max_scatter_points:
        selected_indices = rng.choice(selected_indices, max_scatter_points, replace=False)
    target_values = target_dbz.ravel()[selected_indices]
    prediction_values = prediction_dbz.ravel()[selected_indices]
    axes[1, 3].scatter(target_values, prediction_values, s=2, alpha=0.18)
    if target_values.size:
        lower = float(min(target_values.min(), prediction_values.min()))
        upper = float(max(target_values.max(), prediction_values.max()))
        axes[1, 3].plot([lower, upper], [lower, upper], "k--", linewidth=1)
    axes[1, 3].set_title("All target-support dBZ correlation")
    axes[1, 3].set_xlabel("Target dBZ")
    axes[1, 3].set_ylabel("Prediction dBZ")

    target_section = _safe_dbz(
        target_dbz[:, ray_index, :], target_support[:, ray_index, :]
    )
    prediction_section = _safe_dbz(
        prediction_dbz[:, ray_index, :], predicted_support[:, ray_index, :]
    )
    extent = (0, shape[0], float(z[0]), float(z[-1]))
    for axis, values, title in (
        (axes[2, 0], target_section, "A-B target vertical section"),
        (axes[2, 1], prediction_section, "A-B predicted vertical section"),
    ):
        image = axis.imshow(
            values.T,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="turbo",
            vmin=0,
            vmax=50,
        )
        axis.set_title(f"{title} (ray={ray_index})")
        axis.set_xlabel("scan")
        axis.set_ylabel("height (km)")
        figure.colorbar(image, ax=axis, shrink=0.75)

    axes[2, 2].plot(recall, z, label="Recall")
    axes[2, 2].plot(csi, z, label="CSI")
    mae_axis = axes[2, 2].twiny()
    mae_axis.plot(mae, z, color="tab:red", label="dBZ MAE")
    axes[2, 2].set_xlim(0, 1)
    axes[2, 2].set_xlabel("support score")
    mae_axis.set_xlabel("MAE (dBZ)")
    axes[2, 2].set_ylabel("height (km)")
    axes[2, 2].legend(loc="lower right")
    mae_axis.legend(loc="upper right")
    axes[2, 2].set_title("Vertical support/dBZ metrics")

    positive_probability = support_probability[support_domain & target_support]
    negative_probability = support_probability[support_domain & ~target_support]
    axes[2, 3].hist(negative_probability, bins=40, range=(0, 1), density=True, alpha=0.6, label="DPR negative")
    axes[2, 3].hist(positive_probability, bins=40, range=(0, 1), density=True, alpha=0.6, label="DPR positive")
    axes[2, 3].axvline(threshold, color="black", linestyle="--", label=f"threshold={threshold:.2f}")
    axes[2, 3].set_title("Support probability distributions")
    axes[2, 3].set_xlabel("probability")
    axes[2, 3].legend()

    figure.suptitle(f"Stage-2 orbit diagnostics: {sample_id}", fontsize=15)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=dpi)
    plt.close(figure)
    return {
        "sample_id": sample_id,
        "height_index": level,
        "height_km": float(z[level]),
        "ab_ray_index": ray_index,
        "support_threshold": threshold,
        "target_support_voxels": int(target_selected.sum()),
        "predicted_support_voxels": int(predicted_selected.sum()),
        "metrics": finite_metrics_for_json(computed),
        "vertical_recall": finite_metrics_for_json(recall.tolist()),
        "vertical_csi": finite_metrics_for_json(csi.tolist()),
        "vertical_mae_dbz": finite_metrics_for_json(mae.tolist()),
    }


def main() -> None:
    args = parse_args()
    if args.sample_count <= 0 or args.max_scatter_points <= 0 or args.dpi <= 0:
        raise ValueError("sample-count, max-scatter-points, and dpi must be positive")
    threshold = load_threshold_file(args.threshold_file)
    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("checkpoint has no Stage-2 configuration")
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif device_name == "cuda":
        device_name = "cuda:0"
    device = torch.device(device_name)
    model = build_model(config).to(device)
    load_checkpoint(checkpoint_path, model, map_location=device)
    data = config["data"]
    dataset = Stage2PatchDataset(
        project_path(data["test_index"]),
        project_path(data["normalization"]),
        cache_size=int(data.get("cache_size", 1)),
        input_channels=data.get("input_channels"),
    )
    if len(dataset.feature_names) != int(config["model"]["in_channels"]):
        raise ValueError("checkpoint model.in_channels differs from Dataset channels")
    eligible = [
        index
        for index, entry in enumerate(dataset.files)
        if int(entry.get("dpr_count", 0)) > 0
    ]
    selected = select_file_ids(
        len(dataset.files), args.sample_count, args.seed, eligible_ids=eligible
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for position, file_id in enumerate(selected, start=1):
        entry = dataset.files[file_id]
        prediction = predict_stage2_full_orbit(
            model, dataset, file_id, device=device, batch_size=1, num_workers=0
        )
        targets = reconstruct_stage2_targets(dataset, file_id)
        destination = output_dir / f"{position:02d}_{entry['sample_id']}.png"
        result = plot_stage2_orbit_diagnostics(
            support_probability=prediction.support_probability,
            prediction_dbz=prediction.reflectivity_dbz,
            target_support=targets["target_support"],
            target_dbz=targets["target_dbz"],
            support_domain=targets["support_loss_mask"],
            gr_value_mask=targets["gr_value_mask"],
            heights_km=dataset.z,
            threshold=threshold,
            sample_id=entry["sample_id"],
            height_km=args.height_km,
            max_scatter_points=args.max_scatter_points,
            rng=np.random.default_rng(args.seed + file_id),
            destination=destination,
            dpi=args.dpi,
        )
        result["file_id"] = file_id
        result["file_name"] = entry["file_name"]
        result["figure"] = destination.name
        results.append(result)
        print(f"[visualize {position}/{len(selected)}] {entry['file_name']}", flush=True)
    payload = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "threshold_file": str(args.threshold_file.expanduser().resolve()),
        "support_threshold": threshold,
        "selection_seed": args.seed,
        "selected_file_ids": selected,
        "samples": results,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Stage-2固定测试轨道预测图",
        "",
        f"- 支持域阈值：`{threshold:.3f}`（只由验证集选择）",
        f"- 固定随机种子：`{args.seed}`",
        "- 每张图包含GR/DPR/预测支持域、dBZ平面、误差、相关性、A-B垂直剖面和逐高度指标。",
        "",
    ]
    lines.extend(
        f"- [{item['sample_id']}]({item['figure']})" for item in results
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
