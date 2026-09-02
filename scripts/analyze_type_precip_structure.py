#!/usr/bin/env python3
"""Audit DPR precipitation-type balance and ordered 3-D echo morphology.

This is a data task, not a GPU-training task. It uses the exact stage-one
patch/core contract and writes split counts, CFADs, height profiles, horizontal
texture, echo-top/bright-band proxies, centroid-trajectory statistics, and
representative scan-z/ray-z sections.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from precipitation_inversion.data.patch_dataset import (  # noqa: E402
    Stage1PatchDataset,
    stage1_patch_dataset_kwargs,
)
from precipitation_inversion.data.precipitation_type import (  # noqa: E402
    TYPE_NAMES,
    inverse_sqrt_class_weights,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "stage1_ablation_e0_n_i_intensity.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "type_precip_structure_audit",
    )
    parser.add_argument(
        "--max-patches-per-split",
        type=int,
        help="Debug limit; omit for the complete train/val/test audit.",
    )
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    text = path.expanduser().resolve().read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        value = json.loads(text)
    else:
        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError("configuration root must be a mapping")
    return value


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _normalization_arrays(path: Path) -> tuple[np.ndarray, np.ndarray]:
    value = json.loads(path.read_text(encoding="utf-8"))
    stats = value["variables"]["dbz_dpr"]
    mean = np.asarray([np.nan if item is None else item for item in stats["mean"]])
    std = np.asarray([np.nan if item is None else item for item in stats["std"]])
    return mean.astype(np.float32), std.astype(np.float32)


def _texture_sum_count(
    dbz: np.ndarray, valid: np.ndarray, profile_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Mean absolute adjacent horizontal dBZ difference at every height."""

    z_size = dbz.shape[-1]
    total = np.zeros(z_size, dtype=np.float64)
    count = np.zeros(z_size, dtype=np.int64)
    for axis in (0, 1):
        left = [slice(None)] * 3
        right = [slice(None)] * 3
        left[axis] = slice(None, -1)
        right[axis] = slice(1, None)
        pair = (
            valid[tuple(left)]
            & valid[tuple(right)]
            & profile_mask[tuple(left[:2])][..., None]
            & profile_mask[tuple(right[:2])][..., None]
        )
        difference = np.abs(dbz[tuple(left)] - dbz[tuple(right)])
        total += np.where(pair, difference, 0.0).sum(axis=(0, 1))
        count += pair.sum(axis=(0, 1))
    return total, count


def _centroid_track(dbz: np.ndarray, selected: np.ndarray, *, threshold: float = 35.0) -> dict[str, float] | None:
    """Summarize a patch/class strong-echo centroid trajectory over height."""

    tracks: list[tuple[int, float, float]] = []
    for level in range(dbz.shape[-1]):
        mask = selected[..., level] & (dbz[..., level] >= threshold)
        if int(mask.sum()) < 3:
            continue
        scan, ray = np.nonzero(mask)
        tracks.append((level, float(scan.mean()), float(ray.mean())))
    if len(tracks) < 3:
        return None
    xyz = np.asarray(tracks, dtype=np.float64)
    sections = np.array_split(xyz, 3)
    if any(section.size == 0 for section in sections):
        return None
    low, middle, high = [section[:, 1:].mean(axis=0) for section in sections]
    displacement = float(np.linalg.norm(high - low))
    curvature = float(np.linalg.norm(middle - 0.5 * (low + high)))
    return {
        "height_level_count": float(len(tracks)),
        "low_high_displacement_pixels": displacement,
        "midpoint_curvature_pixels": curvature,
    }


def _nan_summary(values: list[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return {"count": 0, "mean": None, "median": None, "p90": None}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
    }


def audit_split(
    dataset: Stage1PatchDataset,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    max_patches: int | None,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[int, dict[str, np.ndarray]]]:
    z = dataset.z.astype(np.float64)
    dbz_edges = np.arange(-10.0, 66.0, 2.0)
    class_count = len(TYPE_NAMES)
    profile_counts = np.zeros(class_count, dtype=np.int64)
    cfad = np.zeros((class_count, z.size, dbz_edges.size - 1), dtype=np.int64)
    profile_sum = np.zeros((class_count, z.size), dtype=np.float64)
    profile_voxels = np.zeros((class_count, z.size), dtype=np.int64)
    texture_sum = np.zeros((class_count, z.size), dtype=np.float64)
    texture_count = np.zeros((class_count, z.size), dtype=np.int64)
    echo_tops: list[list[float]] = [[] for _ in TYPE_NAMES]
    centroid_features: list[list[dict[str, float]]] = [[] for _ in TYPE_NAMES]
    examples: dict[int, dict[str, np.ndarray]] = {}
    example_scores = np.zeros(class_count, dtype=np.int64)
    limit = len(dataset) if max_patches is None else min(len(dataset), max_patches)

    for index in range(limit):
        item = dataset[index]
        inputs = item["inputs"].numpy()
        standardized = inputs[0]
        valid = inputs[1].astype(bool)
        dbz = standardized * std.reshape(1, 1, -1) + mean.reshape(1, 1, -1)
        valid &= np.isfinite(dbz)
        target = item["type_target"].numpy()
        supervised = item["type_loss_mask"].numpy()
        for class_index in range(class_count):
            profiles = supervised & (target == class_index)
            count = int(profiles.sum())
            profile_counts[class_index] += count
            if not count:
                continue
            selected = valid & profiles[..., None]
            profile_sum[class_index] += np.where(selected, dbz, 0.0).sum(axis=(0, 1))
            profile_voxels[class_index] += selected.sum(axis=(0, 1))
            for level in range(z.size):
                values = dbz[..., level][selected[..., level]]
                if values.size:
                    cfad[class_index, level] += np.histogram(values, bins=dbz_edges)[0]
            local_sum, local_count = _texture_sum_count(dbz, valid, profiles)
            texture_sum[class_index] += local_sum
            texture_count[class_index] += local_count
            for scan, ray in np.argwhere(profiles):
                levels = np.flatnonzero(valid[scan, ray] & (dbz[scan, ray] >= 18.0))
                if levels.size:
                    echo_tops[class_index].append(float(z[levels[-1]]))
            trajectory = _centroid_track(dbz, selected)
            if trajectory is not None:
                centroid_features[class_index].append(trajectory)
            if count > example_scores[class_index]:
                example_scores[class_index] = count
                examples[class_index] = {
                    "dbz": dbz.copy(),
                    "valid": valid.copy(),
                    "profiles": profiles.copy(),
                    "patch_index": np.asarray(index),
                }

    mean_profile = np.divide(
        profile_sum,
        profile_voxels,
        out=np.full_like(profile_sum, np.nan),
        where=profile_voxels > 0,
    )
    texture = np.divide(
        texture_sum,
        texture_count,
        out=np.full_like(texture_sum, np.nan),
        where=texture_count > 0,
    )
    type_values: dict[str, Any] = {}
    for class_index, name in enumerate(TYPE_NAMES):
        trajectory = centroid_features[class_index]
        # Bright-band proxy: strongest 2--6 km mean-profile enhancement above
        # the line joining endpoints. It is a diagnostic, not a DPR algorithm.
        band = np.flatnonzero((z >= 2.0) & (z <= 6.0) & np.isfinite(mean_profile[class_index]))
        bright_band = None
        if band.size >= 3:
            baseline = np.linspace(
                mean_profile[class_index, band[0]],
                mean_profile[class_index, band[-1]],
                band.size,
            )
            bright_band = float(np.max(mean_profile[class_index, band] - baseline))
        type_values[name] = {
            "profile_count": int(profile_counts[class_index]),
            "profile_fraction": float(profile_counts[class_index] / profile_counts.sum()) if profile_counts.sum() else None,
            "echo_top_18dbz_km": _nan_summary(echo_tops[class_index]),
            "bright_band_proxy_db": bright_band,
            "strong_echo_low_high_displacement_pixels": _nan_summary(
                [value["low_high_displacement_pixels"] for value in trajectory]
            ),
            "strong_echo_midpoint_curvature_pixels": _nan_summary(
                [value["midpoint_curvature_pixels"] for value in trajectory]
            ),
        }
    summary = {
        "patch_count_processed": limit,
        "patch_count_total": len(dataset),
        "complete": limit == len(dataset),
        "profile_count": int(profile_counts.sum()),
        "types": type_values,
    }
    arrays = {
        "z_km": z,
        "dbz_edges": dbz_edges,
        "cfad": cfad,
        "mean_dbz_profile": mean_profile,
        "horizontal_texture_db": texture,
        "profile_counts": profile_counts,
    }
    return summary, arrays, examples


def plot_audit(
    split: str,
    arrays: dict[str, np.ndarray],
    examples: dict[int, dict[str, np.ndarray]],
    output_dir: Path,
    *,
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = arrays["z_km"]
    centers = 0.5 * (arrays["dbz_edges"][:-1] + arrays["dbz_edges"][1:])
    figure, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for index, name in enumerate(TYPE_NAMES):
        values = arrays["cfad"][index].astype(float)
        denominator = values.sum(axis=1, keepdims=True)
        values = np.divide(values, denominator, out=np.zeros_like(values), where=denominator > 0)
        image = axes[0, index].pcolormesh(centers, z, values, shading="auto", cmap="viridis")
        axes[0, index].set(title=f"{name} CFAD", xlabel="DPR dBZ", ylabel="height (km)")
        figure.colorbar(image, ax=axes[0, index], label="height-normalized frequency")
        axes[1, index].plot(arrays["mean_dbz_profile"][index], z, label="mean dBZ")
        twin = axes[1, index].twiny()
        twin.plot(arrays["horizontal_texture_db"][index], z, color="tab:orange", label="horizontal texture")
        axes[1, index].set(xlabel="mean dBZ", ylabel="height (km)")
        twin.set_xlabel("mean adjacent |ΔdBZ|")
        axes[1, index].grid(alpha=0.25)
    figure.suptitle(f"{split}: typePrecip ordered-height morphology")
    figure.savefig(output_dir / f"{split}_cfad_profiles.png", dpi=dpi)
    plt.close(figure)

    for class_index, example in examples.items():
        dbz = np.where(example["valid"], example["dbz"], np.nan)
        profiles = example["profiles"]
        coordinates = np.argwhere(profiles)
        if not coordinates.size:
            continue
        center_scan, center_ray = np.rint(coordinates.mean(axis=0)).astype(int)
        figure, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
        safe_dbz = np.where(np.isfinite(dbz), dbz, -999.0)
        projection = safe_dbz.max(axis=-1)
        projection[projection <= -900.0] = np.nan
        axes[0].imshow(projection.T, origin="lower", aspect="auto", cmap="turbo", vmin=0, vmax=50)
        axes[0].contour(profiles.T.astype(float), levels=[0.5], colors="white", linewidths=0.6)
        axes[0].set(title="max-Z map + class outline", xlabel="scan", ylabel="ray")
        axes[1].imshow(dbz[:, center_ray, :].T, origin="lower", aspect="auto", cmap="turbo", vmin=0, vmax=50, extent=[0, dbz.shape[0], z[0], z[-1]])
        axes[1].set(title=f"scan-z at ray={center_ray}", xlabel="scan", ylabel="height (km)")
        axes[2].imshow(dbz[center_scan, :, :].T, origin="lower", aspect="auto", cmap="turbo", vmin=0, vmax=50, extent=[0, dbz.shape[1], z[0], z[-1]])
        axes[2].set(title=f"ray-z at scan={center_scan}", xlabel="ray", ylabel="height (km)")
        figure.suptitle(f"{split} {TYPE_NAMES[class_index]} representative patch {int(example['patch_index'])}")
        figure.savefig(output_dir / f"{split}_example_{TYPE_NAMES[class_index]}.png", dpi=dpi)
        plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.max_patches_per_split is not None and args.max_patches_per_split <= 0:
        raise ValueError("max-patches-per-split must be positive")
    config = load_config(args.config)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data = config["data"]
    loss = config["loss"]
    normalization_path = project_path(data["normalization"])
    mean, std = _normalization_arrays(normalization_path)
    summary: dict[str, Any] = {
        "definition": {
            "class_codes": {"1": "stratiform", "2": "convective", "3": "other"},
            "support": "non-overlapping core + valid type code + at least one native DPR echo",
            "input_leakage": False,
            "centroid_caveat": "patch-level aggregate; multiple cells may share one trajectory",
        },
        "splits": {},
    }
    for split in ("train", "val", "test"):
        dataset = Stage1PatchDataset(
            project_path(data[f"{split}_index"]),
            normalization_path,
            positive_only=False,
            cache_size=int(data.get("cache_size", 1)),
            **stage1_patch_dataset_kwargs(data, loss),
        )
        split_summary, arrays, examples = audit_split(
            dataset, mean, std, max_patches=args.max_patches_per_split
        )
        summary["splits"][split] = split_summary
        np.savez_compressed(output_dir / f"{split}_audit_arrays.npz", **arrays)
        plot_audit(split, arrays, examples, output_dir, dpi=args.dpi)
        print(f"[audit] {split}: {split_summary['profile_count']} profiles", flush=True)
    train_counts = [
        summary["splits"]["train"]["types"][name]["profile_count"]
        for name in TYPE_NAMES
    ]
    try:
        weights = inverse_sqrt_class_weights(train_counts).tolist()
    except ValueError:
        weights = None
    summary["train_only_inverse_sqrt_class_weights"] = weights
    summary["complete_all_splits"] = all(
        value["complete"] for value in summary["splits"].values()
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "README.md").write_text(
        "# typePrecip 与三维形态审计\n\n"
        "`summary.json` 给出三类样本量、回波顶、亮带代理及强回波质心轨迹统计；"
        "`*_cfad_profiles.png` 比较 CFAD、平均垂直廓线和水平纹理；"
        "`*_example_*.png` 展示代表性水平投影及 scan-z/ray-z 剖面。\n\n"
        "注意：`>`/`S` 的经典描述原本对应垂直速度廓线；当前数据没有垂直速度，"
        "这里只审计 DPR 反射率可观察到的代理形态。\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
