#!/usr/bin/env python3
"""Evaluate a stage-one checkpoint on masked patches and optional full orbits."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from precipitation_inversion.data.nc_reader import read_nc_sample  # noqa: E402
from precipitation_inversion.data.patch_dataset import Stage1PatchDataset  # noqa: E402
from precipitation_inversion.inference.sliding_window import (  # noqa: E402
    predict_full_orbit,
)
from precipitation_inversion.losses.masked_losses import MaskedSmoothL1Loss  # noqa: E402
from precipitation_inversion.metrics.regression import (  # noqa: E402
    PrecipitationRegressionMetrics,
)
from precipitation_inversion.models.unet3d import Stage1UNet3D  # noqa: E402
from precipitation_inversion.training.engine import (  # noqa: E402
    evaluate_one_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--config", type=Path, help="Override checkpoint config.")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-batches", type=int)
    parser.add_argument(
        "--full-orbits",
        type=int,
        help="Also reconstruct and evaluate the first N complete source orbits.",
    )
    parser.add_argument("--orbit-output-dir", type=Path)
    parser.add_argument("--output", type=Path)
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


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


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


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif value == "cuda":
        value = "cuda:0"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format version")
    config = (
        load_config(args.config)
        if args.config is not None
        else checkpoint.get("config")
    )
    if not isinstance(config, dict):
        raise ValueError("checkpoint has no config; pass --config explicitly")
    device = resolve_device(args.device)
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    data_config = config["data"]
    index_key = f"{args.split}_index"
    dataset = Stage1PatchDataset(
        project_path(data_config[index_key]),
        project_path(data_config["normalization"]),
        positive_only=True,
        cache_size=int(data_config["cache_size"]),
    )
    workers = int(data_config["num_workers"])
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(data_config["batch_size"]),
        shuffle=False,
        num_workers=workers,
        pin_memory=bool(data_config["pin_memory"]),
        persistent_workers=bool(data_config["persistent_workers"]) and workers > 0,
    )
    loss_config = config["loss"]
    criterion = MaskedSmoothL1Loss(beta=float(loss_config["beta"]))
    thresholds = tuple(float(value) for value in loss_config["thresholds_mm_h"])
    amp_enabled = bool(config["training"]["amp"]) and device.type == "cuda"
    patch_result = evaluate_one_epoch(
        model,
        loader,
        criterion,
        device,
        use_amp=amp_enabled,
        thresholds_mm_h=thresholds,
        max_batches=args.max_batches,
    )
    result: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "split": args.split,
        "patch_evaluation": patch_result.to_dict(),
    }

    configured_orbits = int(config["evaluation"].get("save_orbits", 0))
    full_orbit_count = (
        configured_orbits if args.full_orbits is None else args.full_orbits
    )
    if full_orbit_count < 0:
        raise ValueError("full-orbits must be non-negative")
    if full_orbit_count:
        full_dataset = Stage1PatchDataset(
            project_path(data_config[index_key]),
            project_path(data_config["normalization"]),
            positive_only=False,
            cache_size=int(data_config["cache_size"]),
        )
        full_metrics = PrecipitationRegressionMetrics(thresholds)
        orbit_directory = project_path(
            args.orbit_output_dir
            if args.orbit_output_dir is not None
            else Path(config["experiment"]["output_dir"]) / f"{args.split}_orbits"
        )
        orbit_directory.mkdir(parents=True, exist_ok=True)
        processed = min(full_orbit_count, len(full_dataset.source_files))
        for file_id in range(processed):
            # Each patch output=(1,64,64,60); core cropping reconstructs
            # prediction_rain=(source_nscan,49,60) in physical mm/h.
            prediction_rain = predict_full_orbit(
                model,
                full_dataset,
                file_id,
                device=device,
                batch_size=int(data_config["batch_size"]),
                num_workers=workers,
                use_amp=amp_enabled,
            )
            entry = full_dataset.source_files[file_id]
            sample = read_nc_sample(
                entry["file_path"],
                variables=("z", "pre_dpr", "cfb"),
                dtype=np.float32,
                build_masks=True,
            )
            target_rain = sample.variables["pre_dpr"]
            positive_mask = sample.masks["pre_positive_qc"]
            full_metrics.update_rain(prediction_rain, target_rain, positive_mask)
            np.savez_compressed(
                orbit_directory / f"{entry['sample_id']}.npz",
                prediction_rain_mm_h=prediction_rain.astype(np.float32),
                target_rain_mm_h=target_rain.astype(np.float32),
                evaluation_mask=positive_mask,
            )
        result["full_orbit_evaluation"] = {
            "orbit_count": processed,
            "metrics": full_metrics.compute(),
            "output_dir": str(orbit_directory),
        }

    safe_result = json_safe(result)
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else checkpoint_path.parent / f"evaluation_{args.split}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(safe_result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(safe_result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
