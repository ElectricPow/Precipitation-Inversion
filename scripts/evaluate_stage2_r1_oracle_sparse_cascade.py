#!/usr/bin/env python3
"""Cascade an R1 completion model through frozen Stage 1 with true support."""

from __future__ import annotations

import argparse
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

from precipitation_inversion.data.stage2_patch_dataset import (  # noqa: E402
    Stage2PatchDataset,
)
from precipitation_inversion.data.transforms import PerLevelStandardizer  # noqa: E402
from precipitation_inversion.inference.stage2_completion_sliding_window import (  # noqa: E402
    predict_stage2_completion_full_orbit,
)
from precipitation_inversion.inference.stage2_stage1_cascade import (  # noqa: E402
    predict_stage1_from_reflectivity_orbit,
)
from precipitation_inversion.training.engine import load_checkpoint  # noqa: E402
from scripts.evaluate_stage1_unet3d import build_model as build_stage1_model  # noqa: E402
from scripts.evaluate_stage2_stage1_cascade import (  # noqa: E402
    CascadeMetricBundle,
    _atomic_csv,
    _atomic_json,
    _atomic_npz,
    _load_json,
    _load_source_fields,
    _mode_row,
    _selected_orbit_ids,
    _validate_index_alignment,
    project_path,
    resolve_device,
)
from scripts.train_stage2_r1_oracle_sparse_value import (  # noqa: E402
    R1_DISPLAY_NAMES,
    build_model as build_r1_model,
    stage2_completion_contract,
    validate_r1_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--stage2-checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stage1-batch-size", type=int, default=1)
    parser.add_argument("--stage2-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--save-orbits", type=int, default=6)
    parser.add_argument("--selection-seed", type=int, default=2026)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage1_batch_size <= 0 or args.stage2_batch_size <= 0:
        raise ValueError("Stage-1/2 batch sizes must be positive")
    if args.num_workers < 0 or args.bootstrap_replicates <= 0:
        raise ValueError("num-workers must be non-negative and bootstrap positive")
    if not 0.0 < args.bootstrap_confidence < 1.0:
        raise ValueError("bootstrap-confidence must lie in (0,1)")
    if args.max_files is not None and args.max_files <= 0:
        raise ValueError("max-files must be positive")
    output_dir = args.output_dir.expanduser().resolve()
    if not args.overwrite and (output_dir / "metrics.json").exists():
        raise FileExistsError("R1 completion cascade exists; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    stage1_path = args.stage1_checkpoint.expanduser().resolve()
    stage1_payload = torch.load(stage1_path, map_location="cpu", weights_only=False)
    stage1_config = stage1_payload.get("config")
    if not isinstance(stage1_config, Mapping):
        raise ValueError("Stage-1 checkpoint has no configuration")
    stage1_model = build_stage1_model(stage1_config).to(device)
    stage1_model.load_state_dict(stage1_payload["model"])
    stage1_model.eval()
    stage1_data = stage1_config["data"]
    stage1_index_path = project_path(stage1_data[f"{args.split}_index"])
    stage1_index = _load_json(stage1_index_path)
    stage1_files = stage1_index["files"]

    stage2_path = args.stage2_checkpoint.expanduser().resolve()
    stage2_payload = torch.load(stage2_path, map_location="cpu", weights_only=False)
    stage2_config = stage2_payload.get("config")
    if not isinstance(stage2_config, Mapping):
        raise ValueError("R1 completion checkpoint has no configuration")
    channels = validate_r1_config(stage2_config)
    task, architecture, checkpoint_format = stage2_completion_contract(stage2_config)
    if stage2_payload.get("stage2_format") != checkpoint_format:
        raise ValueError(
            f"Stage-2 checkpoint format does not match {R1_DISPLAY_NAMES[task]}"
        )
    stage2_data = stage2_config["data"]
    stage2_index_path = project_path(stage2_data[f"{args.split}_index"])
    dataset = Stage2PatchDataset(
        stage2_index_path, project_path(stage2_data["normalization"]),
        cache_size=int(stage2_data.get("cache_size", 1)), input_channels=channels,
    )
    _validate_index_alignment(
        stage1_index, dataset.index_metadata, label=R1_DISPLAY_NAMES[task]
    )
    r1_model = build_r1_model(stage2_config).to(device)
    load_checkpoint(stage2_path, r1_model, map_location=device)
    r1_model.eval()

    file_ids = list(range(len(stage1_files)))
    if args.max_files is not None:
        file_ids = file_ids[:args.max_files]
    if not file_ids:
        raise ValueError("no complete cascade orbits selected")
    selected_ids = _selected_orbit_ids(
        stage1_files, file_ids, args.save_orbits, args.selection_seed
    )
    normalization = _load_json(project_path(stage1_data["normalization"]))
    stage1_standardizer = PerLevelStandardizer.from_dict(
        normalization["variables"]["dbz_dpr"]
    )
    heights = np.asarray(stage1_index["heights_km"], dtype=np.float32)
    if not np.allclose(dataset.z, heights, rtol=0.0, atol=1e-6):
        raise ValueError("R1 completion and Stage-1 height grids differ")
    expected_stage1_channels = 4 if stage1_data.get("cfb_input_mode") == "signed_distance" else 3
    if int(stage1_config["model"]["in_channels"]) != expected_stage1_channels:
        raise ValueError("Stage-1 channel count differs from frozen cascade contract")
    thresholds = tuple(float(value) for value in stage1_config["loss"]["thresholds_mm_h"])
    stratified = stage1_config.get("evaluation", {}).get("stratified", {})
    cfb_edges = tuple(float(value) for value in stratified.get(
        "cfb_distance_edges_km", (-1, 0, 0.5, 2)
    ))
    gradient = stage1_config.get("evaluation", {}).get("physical_drdz", {})
    sign_epsilon = float(gradient.get("sign_epsilon_mm_h_km", 0.1))
    file_labels = [str(stage1_files[index]["sample_id"]) for index in file_ids]
    completion_slug = (
        "r1_o_oracle_support"
        if task == "stage2_r1_o_dpr_sparse_value"
        else "r1_p_partial_conv_oracle_support"
    )
    completion_dbz_field = (
        "r1_o_dbz"
        if task == "stage2_r1_o_dpr_sparse_value"
        else "r1_p_partial_conv_dbz"
    )
    mode_metadata = {
        "dpr_oracle": {
            "display_name": "True DPR dBZ + true DPR support",
            "input_kind": "true_dpr_true_support",
            "deployable": False,
        },
        completion_slug: {
            "display_name": f"{R1_DISPLAY_NAMES[task]} completed dBZ + true DPR support",
            "input_kind": "oracle_sparse_dpr_value_true_support",
            "deployable": False,
        },
    }
    bundles = {
        slug: CascadeMetricBundle.create(
            heights, file_labels, thresholds_mm_h=thresholds,
            cfb_distance_edges_km=cfb_edges, sign_epsilon=sign_epsilon,
            bootstrap_seed=args.bootstrap_seed,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_confidence=args.bootstrap_confidence,
        )
        for slug in mode_metadata
    }
    inference_options = {
        "heights_km": heights,
        "standardizer": stage1_standardizer,
        "core_size": int(stage1_index["core_size"]),
        "halo_size": int(stage1_index["halo_size"]),
        "horizontal_multiple": int(stage1_index["horizontal_multiple"]),
        "cfb_input_mode": str(stage1_data.get("cfb_input_mode", "baseline")),
        "cfb_distance_scale_km": float(stage1_data.get("cfb_distance_scale_km", 2.0)),
        "weak_cfb_layer_weights": tuple(stage1_data.get("weak_cfb_layer_weights", ())),
        "device": device,
        "batch_size": args.stage1_batch_size,
        "use_amp": bool(stage1_config.get("training", {}).get("amp", True)),
    }
    saved: dict[int, dict[str, np.ndarray]] = {}
    for position, file_id in enumerate(file_ids, start=1):
        entry = stage1_files[file_id]
        source = _load_source_fields(Path(entry["file_path"]))
        prediction = predict_stage2_completion_full_orbit(
            r1_model, dataset, file_id, device=device,
            batch_size=args.stage2_batch_size, num_workers=args.num_workers,
            use_amp=bool(stage2_config.get("training", {}).get("amp", True)),
        )
        common = {
            **inference_options,
            "cfb_clutter": source["cfb_clutter"],
            "cfb_index": source["cfb"],
        }
        cascades = {
            "dpr_oracle": predict_stage1_from_reflectivity_orbit(
                stage1_model, source["dbz_dpr"], source["dpr_valid"], **common
            ),
            completion_slug: predict_stage1_from_reflectivity_orbit(
                stage1_model, prediction.reflectivity_dbz,
                source["dpr_valid"], **common
            ),
        }
        for slug, cascade in cascades.items():
            bundles[slug].update(
                cascade.rain_rate_mm_h, source["target_rain"],
                reliable_positive_mask=source["reliable_positive_mask"],
                qc_label_mask=source["qc_label_mask"],
                heights_km=source["z"],
                cfb_distance_km=source["cfb_distance_km"],
                precipitation_type=source["precipitation_type"],
                file_id=file_ids.index(file_id),
            )
        if file_id in selected_ids:
            saved[file_id] = {
                "target_rain_mm_h": source["target_rain"].astype(np.float32),
                "true_dpr_dbz": source["dbz_dpr"].astype(np.float32),
                "true_dpr_support": source["dpr_valid"].astype(bool),
                completion_dbz_field: prediction.reflectivity_dbz.astype(np.float32),
                "rain__dpr_oracle": cascades["dpr_oracle"].rain_rate_mm_h.astype(np.float32),
                f"rain__{completion_slug}": cascades[completion_slug].rain_rate_mm_h.astype(np.float32),
                "reliable_positive_mask": source["reliable_positive_mask"].astype(bool),
                "qc_label_mask": source["qc_label_mask"].astype(bool),
                "heights_km": heights,
                "lat": source["lat"].astype(np.float32),
                "lon": source["lon"].astype(np.float32),
            }
        print(
            f"[{R1_DISPLAY_NAMES[task]} cascade {position}/{len(file_ids)}] "
            f"{entry['file_name']}",
            flush=True,
        )

    computed = {slug: bundle.compute() for slug, bundle in bundles.items()}
    rows = [_mode_row(slug, mode_metadata[slug], computed[slug]) for slug in mode_metadata]
    orbit_rows: list[dict[str, Any]] = []
    for position, file_id in enumerate(selected_ids):
        entry = stage1_files[file_id]
        relative = Path("orbits") / f"{position:02d}_{entry['sample_id']}.npz"
        _atomic_npz(output_dir / relative, saved[file_id])
        orbit_rows.append({
            "file_id": file_id, "sample_id": entry["sample_id"],
            "file": str(relative),
        })
    complete = len(file_ids) == len(stage1_files)
    summary = {
        "format": (
            "stage2_r1_o_frozen_stage1_cascade_v1"
            if task == "stage2_r1_o_dpr_sparse_value"
            else "stage2_r1_p_partial_conv_frozen_stage1_cascade_v1"
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": f"{R1_DISPLAY_NAMES[task]} frozen Stage-1 cascade",
        "task": task,
        "architecture": architecture,
        "deployable": False,
        "split": args.split,
        "file_count": len(file_ids),
        "expected_file_count": len(stage1_files),
        "complete_split": complete,
        "formal_validation_result": bool(args.split == "val" and complete),
        "test_set_accessed": args.split == "test",
        "stage1_checkpoint": str(stage1_path),
        "stage1_checkpoint_epoch": int(stage1_payload["epoch"]),
        "stage2_checkpoint": str(stage2_path),
        "stage2_checkpoint_epoch": int(stage2_payload["epoch"]),
        "support_contract": "true DPR support for both modes; isolates dBZ completion",
        "modes": mode_metadata,
        "metrics": computed,
        "saved_orbits": orbit_rows,
    }
    _atomic_json(output_dir / "metrics.json", summary)
    _atomic_csv(output_dir / "comparison.csv", rows)
    print(
        f"{R1_DISPLAY_NAMES[task]} frozen cascade: {len(file_ids)} files -> {output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
