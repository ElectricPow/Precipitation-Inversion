#!/usr/bin/env python3
"""Evaluate an R1 value-completion control on reconstructed full orbits."""

from __future__ import annotations

import argparse
import json
import sys
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

from precipitation_inversion.data.stage2_patch_dataset import (  # noqa: E402
    STAGE2_R1_ORACLE_SPARSE_VALUE_INPUT_CHANNELS,
    Stage2PatchDataset,
)
from precipitation_inversion.inference.stage2_completion_sliding_window import (  # noqa: E402
    predict_stage2_completion_full_orbit,
)
from precipitation_inversion.inference.stage2_sliding_window import (  # noqa: E402
    reconstruct_stage2_targets,
)
from precipitation_inversion.metrics.stage2_decomposition import (  # noqa: E402
    Stage2DecompositionDiagnostics,
)
from precipitation_inversion.metrics.stage2_reflectivity import (  # noqa: E402
    ReflectivityRegressionAccumulator,
    Stage2ReflectivityMetrics,
)
from precipitation_inversion.training.engine import load_checkpoint  # noqa: E402
from scripts.evaluate_stage2_unet3d import (  # noqa: E402
    _atomic_csv,
    _atomic_json,
    _flatten,
    _region_masks,
    _resolve_device,
)
from scripts.train_stage2_r1_oracle_sparse_value import (  # noqa: E402
    R1_DISPLAY_NAMES,
    build_model,
    project_path,
    stage2_completion_contract,
    validate_r1_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--save-orbits", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _regression_row(
    accumulator: ReflectivityRegressionAccumulator,
) -> dict[str, Any]:
    return dict(accumulator.compute())


def evaluate_complete_orbits(
    model: torch.nn.Module,
    dataset: Stage2PatchDataset,
    file_ids: Sequence[int],
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    use_amp: bool,
    thresholds_dbz: Sequence[float],
    fss_radii: Sequence[int],
    save_orbits: int,
    orbit_directory: Path,
    progress_label: str = "R1 completion",
) -> dict[str, Any]:
    aggregate = Stage2ReflectivityMetrics(fss_radii=(), dense_prediction=True)
    per_height = [ReflectivityRegressionAccumulator() for _ in dataset.z]
    by_region: dict[str, ReflectivityRegressionAccumulator] = {}
    diagnostics = Stage2DecompositionDiagnostics.create(
        dataset.z,
        thresholds_dbz=thresholds_dbz,
        fss_radii=fss_radii,
    )
    per_file: list[dict[str, Any]] = []

    for position, file_id in enumerate(file_ids, start=1):
        prediction = predict_stage2_completion_full_orbit(
            model, dataset, file_id, device=device, batch_size=batch_size,
            num_workers=num_workers, use_amp=use_amp,
        )
        targets = reconstruct_stage2_targets(dataset, file_id)
        target_support = targets["target_support"]
        # Oracle support is deliberately used on both sides.  Thus every
        # reported difference below is conditional dBZ completion error, not
        # support-classification error.
        metric = Stage2ReflectivityMetrics(fss_radii=(), dense_prediction=True)
        metric.update(
            prediction.reflectivity_dbz, target_support,
            targets["target_dbz"], target_support, target_support,
        )
        aggregate.merge(metric)
        entry = dataset.files[file_id]
        per_file.append({
            "file_id": file_id,
            "sample_id": entry["sample_id"],
            "file_name": entry["file_name"],
            "anchor_count": int(np.count_nonzero(targets["dpr_sparse_anchor_mask"])),
            "target_count": int(np.count_nonzero(target_support)),
            **_flatten(metric.compute()),
        })
        for level, accumulator in enumerate(per_height):
            section = (..., level)
            accumulator.update(
                prediction.reflectivity_dbz[section],
                targets["target_dbz"][section],
                target_support[section],
            )

        regions = _region_masks(targets)
        regions.update({
            "dpr_sparse_anchor": target_support & targets["dpr_sparse_anchor_mask"],
            "dpr_unanchored": target_support & ~targets["dpr_sparse_anchor_mask"],
        })
        for name, domain in regions.items():
            accumulator = by_region.setdefault(name, ReflectivityRegressionAccumulator())
            accumulator.update(
                prediction.reflectivity_dbz, targets["target_dbz"],
                domain & target_support,
            )
        diagnostics.update(
            target_support.astype(np.float32),
            prediction.reflectivity_dbz,
            target_support,
            targets["target_dbz"],
            target_support,
            target_support,
        )
        if position <= save_orbits:
            orbit_directory.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                orbit_directory / f"{entry['sample_id']}.npz",
                predicted_dbz=prediction.reflectivity_dbz,
                target_dbz=targets["target_dbz"],
                target_support=target_support,
                dpr_sparse_anchor_mask=targets["dpr_sparse_anchor_mask"],
                gap_proxy_mask=targets["gap_proxy_mask"],
                outside_proxy_mask=targets["outside_proxy_mask"],
                heights_km=dataset.z,
            )
        print(
            f"[{progress_label} {position}/{len(file_ids)}] {entry['file_name']}",
            flush=True,
        )

    physical = diagnostics.compute()
    return {
        "aggregate": aggregate.compute(),
        "per_file": per_file,
        "per_height": [
            {"height_index": level, "height_km": float(dataset.z[level]),
             **_regression_row(accumulator)}
            for level, accumulator in enumerate(per_height)
        ],
        "per_region": [
            {"region": name, **_regression_row(accumulator)}
            for name, accumulator in sorted(by_region.items())
        ],
        "physical_diagnostics": physical,
        "cfad": diagnostics.cfad.rows(),
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers non-negative")
    if args.max_files is not None and args.max_files <= 0:
        raise ValueError("max-files must be positive")
    checkpoint_path = args.checkpoint.expanduser().resolve()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = payload.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("R1 completion checkpoint does not contain a configuration")
    channels = validate_r1_config(config)
    task, architecture, checkpoint_format = stage2_completion_contract(config)
    if payload.get("stage2_format") != checkpoint_format:
        raise ValueError(
            f"checkpoint format does not match {R1_DISPLAY_NAMES[task]}: "
            f"expected {checkpoint_format!r}"
        )
    if channels != STAGE2_R1_ORACLE_SPARSE_VALUE_INPUT_CHANNELS:
        raise RuntimeError("R1 completion input contract changed unexpectedly")
    device = _resolve_device(args.device)
    model = build_model(config).to(device)
    load_checkpoint(checkpoint_path, model, map_location=device)
    model.eval()
    data = config["data"]
    dataset = Stage2PatchDataset(
        project_path(data[f"{args.split}_index"]),
        project_path(data["normalization"]),
        cache_size=int(data.get("cache_size", 1)),
        input_channels=channels,
    )
    file_ids = list(range(len(dataset.files)))
    if args.max_files is not None:
        file_ids = file_ids[:args.max_files]
    if not file_ids:
        raise ValueError("no R1 completion files selected")
    output_dir = args.output_dir.expanduser().resolve()
    if not args.overwrite and (output_dir / "metrics.json").exists():
        raise FileExistsError("R1 completion evaluation exists; pass --overwrite")
    evaluation = config.get("evaluation", {})
    save_orbits = int(evaluation.get("save_orbits", 0) if args.save_orbits is None else args.save_orbits)
    if save_orbits < 0:
        raise ValueError("save-orbits must be non-negative")
    result = evaluate_complete_orbits(
        model, dataset, file_ids, device=device, batch_size=args.batch_size,
        num_workers=args.num_workers, use_amp=device.type == "cuda",
        thresholds_dbz=tuple(evaluation.get("dbz_thresholds", (15, 25, 35))),
        fss_radii=tuple(evaluation.get("fss_radii", (0, 1, 2, 4))),
        save_orbits=save_orbits, orbit_directory=output_dir / "orbits",
        progress_label=R1_DISPLAY_NAMES[task],
    )
    complete = len(file_ids) == len(dataset.files)
    summary = {
        "format": (
            "stage2_r1_o_dpr_sparse_value_evaluation_v1"
            if task == "stage2_r1_o_dpr_sparse_value"
            else "stage2_r1_p_partial_conv_evaluation_v1"
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": 2,
        "experiment": R1_DISPLAY_NAMES[task],
        "task": task,
        "architecture": architecture,
        "deployable": False,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(payload["epoch"]),
        "split": args.split,
        "file_count": len(file_ids),
        "expected_file_count": len(dataset.files),
        "complete_split": complete,
        "formal_validation_result": bool(args.split == "val" and complete),
        "test_set_accessed": args.split == "test",
        "input_channels": list(channels),
        "metric_semantics": {
            "support": "true DPR support is supplied as oracle; support skill is not a learned result",
            "reflectivity": "conditional dBZ error on every true DPR support voxel",
            "dpr_sparse_anchor": "gr_value_mask AND dpr_valid at the same voxel",
            "dpr_unanchored": "dpr_valid AND NOT dpr_sparse_anchor",
            "echo_top_base": "not a learned quantity because true support is fixed",
        },
        "metrics": result["aggregate"],
        "physical_diagnostics": result["physical_diagnostics"],
    }
    _atomic_json(output_dir / "metrics.json", summary)
    _atomic_csv(output_dir / "per_file.csv", result["per_file"])
    _atomic_csv(output_dir / "per_height.csv", result["per_height"])
    _atomic_csv(output_dir / "per_region.csv", result["per_region"])
    _atomic_csv(output_dir / "cfad.csv", result["cfad"])
    dbz = result["aggregate"]["reflectivity_on_target_support"]
    print(
        f"{R1_DISPLAY_NAMES[task]} {args.split}: MAE={dbz['mae_dbz']:.4f} dBZ, "
        f"r={dbz['pearson_r']:.4f}, files={len(file_ids)} -> {output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
