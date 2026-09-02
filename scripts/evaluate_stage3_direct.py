#!/usr/bin/env python3
"""Evaluate a D0 GR-only direct multi-head checkpoint on complete orbits."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

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
    predict_stage3_d0_full_orbit,
    reconstruct_stage2_targets,
)
from precipitation_inversion.inference.stage2_stage1_cascade import Stage1CascadePrediction  # noqa: E402
from precipitation_inversion.metrics.stage2_reflectivity import finite_metrics_for_json  # noqa: E402
from precipitation_inversion.models.stage3_direct import build_stage3_d0_model  # noqa: E402
from precipitation_inversion.training.engine import load_checkpoint  # noqa: E402
from scripts.evaluate_stage2_stage1_cascade import (  # noqa: E402
    CASCADE_ORBIT_FORMAT,
    CascadeMetricBundle,
    _atomic_csv,
    _atomic_json,
    _load_json,
    _load_source_fields,
    _mode_row,
    _selected_orbit_ids,
    _validate_index_alignment,
    _write_orbit_bundles,
    project_path,
    resolve_device,
)


STAGE3_D0_CHECKPOINT_FORMAT = "stage3_d0_direct_multihead_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--save-orbits", type=int, default=6)
    parser.add_argument("--selection-seed", type=int, default=2026)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--support-threshold", type=float)
    parser.add_argument("--threshold-file", type=Path)
    parser.add_argument("--select-threshold", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_d0(
    checkpoint_path: Path, device: torch.device
) -> tuple[dict[str, Any], Mapping[str, Any], torch.nn.Module]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("stage3_format") != STAGE3_D0_CHECKPOINT_FORMAT:
        raise ValueError("checkpoint is not an S3-D0 DirectMultiHead checkpoint")
    config = payload.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("D0 checkpoint has no embedded Stage-2 configuration")
    scope = str(payload.get("stage3_trainable_scope", "rain_head_only"))
    model = build_stage3_d0_model(config["model"], trainable_scope=scope).to(device)
    load_checkpoint(checkpoint_path, model, map_location=device)
    model.eval()
    return payload, config, model


def _threshold_from_json(path: Path) -> float:
    value = _load_json(path)
    if value.get("selected_on_split") != "val":
        raise ValueError("D0 support threshold must be selected on validation")
    threshold = float(value["threshold"])
    if not 0.0 < threshold < 1.0:
        raise ValueError("invalid support threshold")
    return threshold


def _select_threshold(
    model: torch.nn.Module,
    dataset: Stage2PatchDataset,
    file_ids: list[int],
    config: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    evaluation = config.get("evaluation", {})
    sweep = SupportThresholdSweep(
        evaluation.get("threshold_candidates", (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)),
        objective=str(evaluation.get("threshold_objective", "csi")),
    )
    for position, file_id in enumerate(file_ids, start=1):
        prediction = predict_stage3_d0_full_orbit(
            model, dataset, file_id, device=None, batch_size=args.batch_size,
            num_workers=args.num_workers, use_amp=True,
        )
        target = reconstruct_stage2_targets(dataset, file_id)
        sweep.update(
            prediction.support_probability,
            target["target_support"],
            target["support_loss_mask"],
        )
        print(f"[D0 threshold {position}/{len(file_ids)}]", flush=True)
    return sweep.compute().to_dict()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers non-negative")
    if args.max_files is not None and args.max_files <= 0:
        raise ValueError("max-files must be positive")
    if args.save_orbits < 0 or args.bootstrap_replicates <= 0:
        raise ValueError("save-orbits and bootstrap-replicates are invalid")
    if not 0.0 < args.bootstrap_confidence < 1.0:
        raise ValueError("bootstrap-confidence must lie in (0,1)")
    if sum((args.select_threshold, args.support_threshold is not None, args.threshold_file is not None)) != 1:
        raise ValueError("choose exactly one of threshold selection/file/explicit value")
    if args.select_threshold and args.split != "val":
        raise ValueError("threshold selection is validation-only")

    output_dir = args.output_dir.expanduser().resolve()
    if not args.overwrite and (output_dir / "metrics.json").exists():
        raise FileExistsError("D0 evaluation exists; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint.expanduser().resolve()
    device = resolve_device(args.device)
    payload, config, model = _load_d0(checkpoint_path, device)
    data = config["data"]
    dataset = Stage2PatchDataset(
        project_path(data[f"{args.split}_index"]),
        project_path(data["normalization"]),
        cache_size=int(data.get("cache_size", 1)),
        input_channels=data.get("input_channels"),
    )
    sources = payload.get("stage3_sources")
    if not isinstance(sources, Mapping):
        raise ValueError("D0 checkpoint has no source metadata")
    stage1_payload = torch.load(sources["stage1_checkpoint"], map_location="cpu", weights_only=False)
    stage1_config = stage1_payload["config"]
    stage1_index = _load_json(project_path(stage1_config["data"][f"{args.split}_index"]))
    _validate_index_alignment(stage1_index, dataset.index_metadata, label="D0")
    stage1_files = stage1_index["files"]
    file_ids = list(range(len(stage1_files)))
    if args.max_files is not None:
        file_ids = file_ids[: args.max_files]
    if not file_ids:
        raise ValueError("no complete orbit selected")

    if args.select_threshold:
        selection = _select_threshold(model, dataset, file_ids, config, args)
        threshold = float(selection["threshold"])
        threshold_payload = {
            "format_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": int(payload["epoch"]),
            "selected_on_split": "val",
            "selected_file_count": len(file_ids),
            **selection,
        }
        _atomic_json(output_dir / "support_threshold.json", threshold_payload)
        _atomic_csv(output_dir / "threshold_curve.csv", selection["candidates"])
    elif args.threshold_file is not None:
        threshold = _threshold_from_json(args.threshold_file.expanduser().resolve())
    else:
        threshold = float(args.support_threshold)
        if not 0.0 < threshold < 1.0:
            raise ValueError("support threshold must lie in (0,1)")

    heights = np.asarray(stage1_index["heights_km"], dtype=np.float32)
    thresholds = tuple(float(v) for v in stage1_config["loss"]["thresholds_mm_h"])
    stratified = stage1_config.get("evaluation", {}).get("stratified", {})
    cfb_edges = tuple(float(v) for v in stratified.get("cfb_distance_edges_km", (-1, 0, 0.5, 2)))
    gradient = stage1_config.get("evaluation", {}).get("physical_drdz", {})
    sign_epsilon = float(gradient.get("sign_epsilon_mm_h_km", 0.1))
    labels = [str(stage1_files[index]["sample_id"]) for index in file_ids]
    modes = {
        "d0_oracle_support": {
            "display_name": "D0 rain + true DPR support (diagnostic)",
            "input_kind": "direct_gr_rain_true_dpr_support",
            "reflectivity_source": "not used by the direct rain head",
            "support_source": "true DPR support",
            "deployable": False,
        },
        "d0_predicted_support": {
            "display_name": "D0 rain + predicted support",
            "input_kind": "direct_gr_rain_predicted_support",
            "reflectivity_source": "not used by the direct rain head",
            "support_source": f"D0 probability >= validation threshold {threshold:g}",
            "deployable": True,
        },
    }
    bundles = {
        slug: CascadeMetricBundle.create(
            heights, labels, thresholds_mm_h=thresholds,
            cfb_distance_edges_km=cfb_edges, sign_epsilon=sign_epsilon,
            bootstrap_seed=args.bootstrap_seed,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_confidence=args.bootstrap_confidence,
        )
        for slug in modes
    }
    selected_ids = _selected_orbit_ids(stage1_files, file_ids, args.save_orbits, args.selection_seed)
    saved: dict[int, dict[str, np.ndarray]] = {file_id: {} for file_id in selected_ids}
    support_rows: list[dict[str, Any]] = []
    for position, file_id in enumerate(file_ids, start=1):
        entry = stage1_files[file_id]
        source = _load_source_fields(Path(entry["file_path"]))
        prediction = predict_stage3_d0_full_orbit(
            model, dataset, file_id, device=device, batch_size=args.batch_size,
            num_workers=args.num_workers, use_amp=bool(config.get("training", {}).get("amp", True)),
        )
        predicted_support = prediction.support_probability >= threshold
        rain_routes = {
            "d0_oracle_support": (prediction.rain_rate, source["dpr_valid"]),
            "d0_predicted_support": (prediction.rain_rate, predicted_support),
        }
        for slug, (raw_rain, support) in rain_routes.items():
            # Dense rain head remains differentiable in training; the explicit
            # support decision is applied only here, after complete-orbit stitching.
            rain = np.where(support, raw_rain, 0.0).astype(np.float32)
            bundles[slug].update(
                rain, source["target_rain"],
                reliable_positive_mask=source["reliable_positive_mask"],
                qc_label_mask=source["qc_label_mask"], heights_km=source["z"],
                cfb_distance_km=source["cfb_distance_km"],
                precipitation_type=source["precipitation_type"],
                file_id=file_ids.index(file_id),
            )
            support_rows.append({
                "mode": slug, "file_id": file_id, "sample_id": entry["sample_id"],
                "input_support_count": int(support.sum()),
                "output_support_count": int(support.sum()),
                "true_dpr_support_count": int(source["dpr_valid"].sum()),
                "reliable_positive_count": int(source["reliable_positive_mask"].sum()),
                "qc_label_count": int(source["qc_label_mask"].sum()),
            })
            if file_id in saved:
                fields = saved[file_id]
                if not fields:
                    fields.update({
                        "target_rain_mm_h": source["target_rain"].astype(np.float32),
                        "reliable_positive_mask": source["reliable_positive_mask"],
                        "qc_label_mask": source["qc_label_mask"],
                        "heights_km": source["z"].astype(np.float32),
                        "lat": source["lat"].astype(np.float32),
                        "lon": source["lon"].astype(np.float32),
                        "cfb": source["cfb"].astype(np.float32),
                        "precipitation_type": source["precipitation_type"].astype(np.float32),
                        "true_dpr_support": source["dpr_valid"].astype(bool),
                        "d0_rain_ungated_mm_h": prediction.rain_rate.astype(np.float32),
                        "d0_support_probability": prediction.support_probability.astype(np.float32),
                        "d0_reflectivity_dbz": prediction.reflectivity_dbz.astype(np.float32),
                    })
                fields[f"rain__{slug}"] = rain
                fields[f"input_support__{slug}"] = support.astype(bool)
                fields[f"output_support__{slug}"] = support.astype(bool)
        print(f"[D0 evaluate {position}/{len(file_ids)}] {entry['file_name']}", flush=True)

    computed = {slug: bundle.compute() for slug, bundle in bundles.items()}
    rows = [_mode_row(slug, modes[slug], computed[slug]) for slug in modes]
    orbit_records = _write_orbit_bundles(output_dir, selected_ids, stage1_files, saved, modes)
    summary = {
        "format_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "stage3_d0_direct_multihead_complete_orbit",
        "split": args.split,
        "file_count": len(file_ids),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(payload["epoch"]),
        "support_threshold": threshold,
        "model_input_contract": {
            "channels": list(data["input_channels"]),
            "tensor_shape": f"(B,{len(data['input_channels'])},64,64,{heights.size})",
            "satellite_variables_used_as_input": False,
        },
        "modes": modes,
        "metrics": finite_metrics_for_json(computed),
        "saved_orbit_selection": {"seed": args.selection_seed, "orbits": orbit_records},
    }
    manifest = {
        "format": CASCADE_ORBIT_FORMAT,
        "evaluation_summary": "metrics.json",
        "split": args.split,
        "modes": [{"slug": slug, **values} for slug, values in modes.items()],
        "orbits": orbit_records,
    }
    _atomic_json(output_dir / "metrics.json", summary)
    _atomic_json(output_dir / "orbit_manifest.json", manifest)
    _atomic_csv(output_dir / "comparison.csv", rows)
    _atomic_csv(output_dir / "support_per_file.csv", support_rows)
    if args.visualize and args.save_orbits > 0:
        command = [
            sys.executable, str(PROJECT_ROOT / "scripts" / "visualize_stage2_stage1_cascade.py"),
            "--input-dir", str(output_dir), "--output-dir", str(output_dir / "visualizations"),
            "--modes", *modes.keys(), "--overwrite",
        ]
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    print(f"D0 evaluation complete -> {output_dir}", flush=True)
    for row in rows:
        print(f"  {row['mode']}: RMSE={row['positive_rmse']:.4f}, r={row['positive_pearson_r']:.4f}", flush=True)


if __name__ == "__main__":
    main()
