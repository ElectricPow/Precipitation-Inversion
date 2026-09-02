#!/usr/bin/env python3
"""Evaluate a Stage-2 checkpoint on complete reconstructed val/test orbits."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for value in (PROJECT_ROOT, SRC_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from precipitation_inversion.data.stage2_patch_dataset import Stage2PatchDataset  # noqa: E402
from precipitation_inversion.inference.stage2_sliding_window import (  # noqa: E402
    SupportThresholdSweep,
    predict_stage2_full_orbit,
    reconstruct_stage2_targets,
)
from precipitation_inversion.metrics.stage2_reflectivity import (  # noqa: E402
    Stage2ReflectivityMetrics,
    finite_metrics_for_json,
)
from precipitation_inversion.training.engine import load_checkpoint  # noqa: E402
from precipitation_inversion.models.stage3_direct import (  # noqa: E402
    build_stage3_d0_model,
)
from scripts.train_stage2_unet3d import build_model, project_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--support-threshold", type=float)
    parser.add_argument("--threshold-file", type=Path)
    parser.add_argument("--select-threshold", action="store_true")
    parser.add_argument("--threshold-output", type=Path)
    parser.add_argument("--save-orbits", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif value == "cuda":
        value = "cuda:0"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def load_threshold_file(path: Path) -> float:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    threshold = float(value["threshold"])
    if not math.isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise ValueError("threshold file contains an invalid threshold")
    if value.get("selected_on_split") != "val":
        raise ValueError("support threshold must have been selected on split=val")
    return threshold


def resolve_support_threshold(
    *,
    explicit: float | None,
    threshold_file: Path | None,
    configured: float,
) -> float:
    if explicit is not None and threshold_file is not None:
        raise ValueError("use either --support-threshold or --threshold-file")
    result = (
        load_threshold_file(threshold_file)
        if threshold_file is not None
        else configured if explicit is None else explicit
    )
    threshold = float(result)
    if not math.isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise ValueError("support threshold must lie in (0,1)")
    return threshold


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
        raise ValueError(f"cannot write empty CSV {path.name}")
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


def _flatten(metrics: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for section, values in metrics.items():
        if section == "fss":
            for radius, radius_values in values.items():
                for name, value in radius_values.items():
                    if name not in {"radius", "window_size"}:
                        output[f"fss_r{radius}_{name}"] = value
        else:
            for name, value in values.items():
                output[f"{section}_{name}"] = value
    return output


def _region_masks(
    targets: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    domain = targets["support_loss_mask"]
    target_support = targets["target_support"]
    dbz = targets["target_dbz"]
    regions = {
        "all_domain": domain,
        "q11_direct_overlap": domain & targets["overlap_mask"],
        "q01_direct_missing": domain & targets["dpr_only_mask"],
        "q10_gr_only": domain & targets["gr_only_mask"],
        "q00_neither": domain & targets["neither_mask"],
        "dpr_gap_proxy": domain & targets["gap_proxy_mask"],
        "dpr_outside_proxy": domain & targets["outside_proxy_mask"],
        "dpr_below_cfb": domain & targets["below_cfb_target_mask"],
        "dpr_above_cfb": domain & target_support & ~targets["below_cfb_target_mask"],
    }
    edges = (-np.inf, 15.0, 25.0, 35.0, np.inf)
    labels = ("lt15", "15to25", "25to35", "ge35")
    for name, lower, upper in zip(labels, edges, edges[1:]):
        regions[f"dpr_dbz_{name}"] = (
            domain & target_support & (dbz >= lower) & (dbz < upper)
        )
    return regions


def select_threshold_on_validation(
    model: torch.nn.Module,
    dataset: Stage2PatchDataset,
    file_ids: Sequence[int],
    *,
    candidates: Sequence[float],
    objective: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    use_amp: bool,
) -> dict[str, Any]:
    sweep = SupportThresholdSweep(candidates, objective=objective)
    for position, file_id in enumerate(file_ids, start=1):
        prediction = predict_stage2_full_orbit(
            model,
            dataset,
            file_id,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            use_amp=use_amp,
        )
        targets = reconstruct_stage2_targets(dataset, file_id)
        sweep.update(
            prediction.support_probability,
            targets["target_support"],
            targets["support_loss_mask"],
        )
        print(
            f"[threshold {position}/{len(file_ids)}] {dataset.files[file_id]['file_name']}",
            flush=True,
        )
    return sweep.compute().to_dict()


def evaluate_complete_orbits(
    model: torch.nn.Module,
    dataset: Stage2PatchDataset,
    file_ids: Sequence[int],
    *,
    support_threshold: float,
    fss_radii: tuple[int, ...],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    use_amp: bool,
    save_orbits: int,
    orbit_directory: Path,
) -> dict[str, Any]:
    aggregate = Stage2ReflectivityMetrics(
        fss_radii=fss_radii, dense_prediction=True
    )
    height_metrics = [
        Stage2ReflectivityMetrics(fss_radii=(), dense_prediction=True)
        for _ in range(dataset.z.size)
    ]
    region_metrics: dict[str, Stage2ReflectivityMetrics] = {}
    per_file: list[dict[str, Any]] = []
    for position, file_id in enumerate(file_ids, start=1):
        prediction = predict_stage2_full_orbit(
            model,
            dataset,
            file_id,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            use_amp=use_amp,
        )
        targets = reconstruct_stage2_targets(dataset, file_id)
        predicted_support = prediction.support_probability >= support_threshold
        metric = Stage2ReflectivityMetrics(
            fss_radii=fss_radii, dense_prediction=True
        )
        metric.update(
            prediction.reflectivity_dbz,
            predicted_support,
            targets["target_dbz"],
            targets["target_support"],
            targets["support_loss_mask"],
        )
        aggregate.merge(metric)
        computed = metric.compute()
        entry = dataset.files[file_id]
        per_file.append(
            {
                "file_id": file_id,
                "sample_id": entry["sample_id"],
                "file_name": entry["file_name"],
                **_flatten(computed),
            }
        )
        for level in range(dataset.z.size):
            section = (..., level)
            height_metrics[level].update(
                prediction.reflectivity_dbz[section],
                predicted_support[section],
                targets["target_dbz"][section],
                targets["target_support"][section],
                targets["support_loss_mask"][section],
            )
        for region, domain in _region_masks(targets).items():
            if region not in region_metrics:
                region_metrics[region] = Stage2ReflectivityMetrics(
                    fss_radii=(), dense_prediction=True
                )
            region_metrics[region].update(
                prediction.reflectivity_dbz,
                predicted_support,
                targets["target_dbz"],
                targets["target_support"],
                domain,
            )
        if position <= save_orbits:
            orbit_directory.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                orbit_directory / f"{entry['sample_id']}.npz",
                support_probability=prediction.support_probability,
                predicted_support=predicted_support,
                predicted_dbz=prediction.reflectivity_dbz,
                target_support=targets["target_support"],
                target_dbz=targets["target_dbz"],
                support_loss_mask=targets["support_loss_mask"],
                heights_km=dataset.z,
            )
        print(f"[evaluate {position}/{len(file_ids)}] {entry['file_name']}", flush=True)
    return {
        "aggregate": aggregate.compute(),
        "per_file": per_file,
        "per_height": [
            {
                "height_index": level,
                "height_km": float(dataset.z[level]),
                **_flatten(metric.compute()),
            }
            for level, metric in enumerate(height_metrics)
        ],
        "per_region": [
            {"region": name, **_flatten(metric.compute())}
            for name, metric in sorted(region_metrics.items())
        ],
    }


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("checkpoint does not contain a Stage-2 configuration")
    device = _resolve_device(args.device)
    if checkpoint.get("stage3_format") == "stage3_d0_direct_multihead_v1":
        scope = str(checkpoint.get("stage3_trainable_scope", "rain_head_only"))
        model = build_stage3_d0_model(config["model"], trainable_scope=scope).to(device)
    else:
        model = build_model(config).to(device)
    load_checkpoint(checkpoint_path, model, map_location=device)
    data = config["data"]
    dataset = Stage2PatchDataset(
        project_path(data[f"{args.split}_index"]),
        project_path(data["normalization"]),
        cache_size=int(data.get("cache_size", 1)),
        input_channels=data.get("input_channels"),
    )
    if len(dataset.feature_names) != int(config["model"]["in_channels"]):
        raise ValueError("checkpoint model.in_channels differs from Dataset channels")
    file_ids = list(range(len(dataset.files)))
    if args.max_files is not None:
        if args.max_files <= 0:
            raise ValueError("max_files must be positive")
        file_ids = file_ids[: args.max_files]
    if not file_ids:
        raise ValueError("no Stage-2 files selected")
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    output_dir = args.output_dir.expanduser().resolve()
    expected = (output_dir / "metrics.json", output_dir / "per_file.csv")
    if not args.overwrite and any(path.exists() for path in expected):
        raise FileExistsError("evaluation output exists; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation = config.get("evaluation", {})
    use_amp = device.type == "cuda"

    selection: dict[str, Any] | None = None
    if args.select_threshold:
        if args.split != "val":
            raise ValueError("support threshold may only be selected on split=val")
        if args.support_threshold is not None or args.threshold_file is not None:
            raise ValueError("threshold selection cannot use a preselected threshold")
        selection = select_threshold_on_validation(
            model,
            dataset,
            file_ids,
            candidates=evaluation["threshold_candidates"],
            objective=str(evaluation.get("threshold_objective", "csi")),
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_amp=use_amp,
        )
        threshold = float(selection["threshold"])
        threshold_payload = {
            "format_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "selected_on_split": "val",
            "selected_file_count": len(file_ids),
            **selection,
        }
        threshold_output = (
            args.threshold_output.expanduser().resolve()
            if args.threshold_output is not None
            else output_dir / "support_threshold.json"
        )
        _atomic_json(threshold_output, threshold_payload)
        _atomic_csv(output_dir / "threshold_curve.csv", selection["candidates"])
    else:
        threshold = resolve_support_threshold(
            explicit=args.support_threshold,
            threshold_file=args.threshold_file,
            configured=float(evaluation.get("training_support_threshold", 0.5)),
        )

    save_orbits = (
        int(evaluation.get("save_orbits", 0))
        if args.save_orbits is None
        else args.save_orbits
    )
    if save_orbits < 0:
        raise ValueError("save_orbits must be non-negative")
    result = evaluate_complete_orbits(
        model,
        dataset,
        file_ids,
        support_threshold=threshold,
        fss_radii=tuple(int(value) for value in evaluation.get("fss_radii", (1, 2, 4))),
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_amp=use_amp,
        save_orbits=save_orbits,
        orbit_directory=output_dir / "orbits",
    )
    summary = {
        "format_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": 2,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "split": args.split,
        "file_count": len(file_ids),
        "support_threshold": threshold,
        "threshold_source": (
            "validation_selection"
            if selection is not None
            else "threshold_file" if args.threshold_file is not None else "explicit_or_config"
        ),
        "metric_semantics": {
            "support": "inside M_support only",
            "reflectivity_on_target_support": "all M_dbz points; threshold independent",
            "reflectivity_on_common_support": "M_dbz and predicted support; threshold dependent",
            "fss": "complete-orbit horizontal neighborhoods, height never mixed",
        },
        "metrics": result["aggregate"],
    }
    _atomic_json(output_dir / "metrics.json", summary)
    _atomic_csv(output_dir / "per_file.csv", result["per_file"])
    _atomic_csv(output_dir / "per_height.csv", result["per_height"])
    _atomic_csv(output_dir / "per_region.csv", result["per_region"])
    support = result["aggregate"]["support"]
    dbz = result["aggregate"]["reflectivity_on_target_support"]
    print(
        f"Stage-2 {args.split}: threshold={threshold:.3f}, CSI={support['csi']:.4f}, "
        f"Recall={support['recall']:.4f}, target-MAE={dbz['mae_dbz']:.4f} dBZ -> {output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
