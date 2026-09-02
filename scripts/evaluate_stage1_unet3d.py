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
from precipitation_inversion.data.patch_dataset import (  # noqa: E402
    Stage1PatchDataset,
    stage1_patch_dataset_kwargs,
)
from precipitation_inversion.inference.sliding_window import (  # noqa: E402
    predict_full_orbit,
)
from precipitation_inversion.losses.masked_losses import (  # noqa: E402
    build_stage1_loss,
)
from precipitation_inversion.losses.masked_classification import (  # noqa: E402
    MaskedCrossEntropyLoss,
)
from precipitation_inversion.metrics.regression import (  # noqa: E402
    FilewisePrecipitationMetrics,
    PhysicalRainGradientMetrics,
    PrecipitationRegressionMetrics,
    StratifiedPrecipitationMetrics,
)
from precipitation_inversion.models.unet3d import Stage1UNet3D  # noqa: E402
from precipitation_inversion.models.multitask_unet3d import (  # noqa: E402
    Stage1MultiTaskUNet3D,
)
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
        "--num-workers",
        type=int,
        help="Override data.num_workers (use 0 for restricted/CPU diagnostics).",
    )
    parser.add_argument(
        "--stratified",
        action="store_true",
        help="Force absolute-height/CFB/intensity/type diagnostics.",
    )
    parser.add_argument(
        "--no-physical-drdz",
        action="store_true",
        help="Disable the default physical vertical rain-gradient diagnostic.",
    )
    parser.add_argument(
        "--full-orbits",
        type=int,
        help="Also reconstruct and evaluate the first N complete source orbits.",
    )
    parser.add_argument("--orbit-output-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=2026,
        help="Reproducible whole-file bootstrap seed (default: 2026).",
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=2000,
        help="Number of whole-file bootstrap resamples (default: 2000).",
    )
    parser.add_argument(
        "--bootstrap-confidence",
        type=float,
        default=0.95,
        help="Two-sided macro-metric confidence level (default: 0.95).",
    )
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
    common = dict(
        in_channels=int(values["in_channels"]),
        out_channels=int(values["out_channels"]),
        base_channels=int(values["base_channels"]),
        channel_multipliers=tuple(values["channel_multipliers"]),
        max_groups=int(values["max_groups"]),
        bottleneck_dropout=float(values["bottleneck_dropout"]),
    )
    type_task = config.get("type_task", {})
    if bool(type_task.get("enabled", False)):
        head = type_task.get("head", {})
        return Stage1MultiTaskUNet3D(
            **common,
            type_head_kind=str(head.get("kind", "ordered_3d")),
            type_head_config=head,
        )
    return Stage1UNet3D(**common)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif value == "cuda":
        value = "cuda:0"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def build_stratified_metrics(
    config: Mapping[str, Any], heights_km: np.ndarray, *, force: bool = False
) -> StratifiedPrecipitationMetrics | None:
    """Build optional full-support diagnostics from evaluation configuration."""

    values = config.get("evaluation", {}).get("stratified", {})
    if not force and not bool(values.get("enabled", False)):
        return None
    thresholds = tuple(float(value) for value in config["loss"]["thresholds_mm_h"])
    options = {
        "cfb_distance_edges_km": tuple(
            float(value)
            for value in values.get(
                "cfb_distance_edges_km", (-1.0, 0.0, 0.5, 2.0)
            )
        ),
        "intensity_thresholds_mm_h": thresholds,
    }
    height_edges = values.get("height_bin_edges_km")
    if height_edges is None:
        return StratifiedPrecipitationMetrics(
            np.asarray(heights_km, dtype=float).tolist(), **options
        )
    return StratifiedPrecipitationMetrics(
        height_bin_edges_km=tuple(float(value) for value in height_edges),
        **options,
    )


def build_physical_gradient_metrics(
    config: Mapping[str, Any],
    heights_km: np.ndarray,
    *,
    file_labels: list[str] | None = None,
    disabled: bool = False,
    bootstrap_seed: int = 2026,
    bootstrap_replicates: int = 2000,
    confidence_level: float = 0.95,
) -> PhysicalRainGradientMetrics | None:
    """Build the shared physical dR/dz implementation used by every model."""

    values = config.get("evaluation", {}).get("physical_drdz", {})
    if disabled or not bool(values.get("enabled", True)):
        return None
    stratified = config.get("evaluation", {}).get("stratified", {})
    cfb_edges = values.get(
        "cfb_distance_edges_km",
        stratified.get("cfb_distance_edges_km", (-1.0, 0.0, 0.5, 2.0)),
    )
    return PhysicalRainGradientMetrics(
        np.asarray(heights_km, dtype=float).tolist(),
        cfb_distance_edges_km=tuple(float(value) for value in cfb_edges),
        intensity_thresholds_mm_h=tuple(
            float(value) for value in config["loss"]["thresholds_mm_h"]
        ),
        sign_epsilon_mm_h_km=float(
            values.get("sign_epsilon_mm_h_km", 0.1)
        ),
        file_labels=file_labels,
        bootstrap_seed=bootstrap_seed,
        bootstrap_replicates=bootstrap_replicates,
        confidence_level=confidence_level,
    )


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
    loss_config = config["loss"]
    dataset_options = stage1_patch_dataset_kwargs(data_config, loss_config)
    index_key = f"{args.split}_index"
    dataset = Stage1PatchDataset(
        project_path(data_config[index_key]),
        project_path(data_config["normalization"]),
        # Complete split coverage is required for below-CFB native-positive
        # diagnostics and file-macro statistics. Primary reliable metrics keep
        # their historical positive mask, so adding dry/below-only patches does
        # not change model selection support.
        positive_only=False,
        cache_size=int(data_config["cache_size"]),
        **dataset_options,
    )
    if len(dataset.feature_names) != int(config["model"]["in_channels"]):
        raise ValueError(
            "checkpoint model input channels do not match Dataset features: "
            f"{config['model']['in_channels']} != {dataset.feature_names}"
        )
    workers = (
        int(data_config["num_workers"])
        if args.num_workers is None
        else args.num_workers
    )
    if workers < 0:
        raise ValueError("num-workers must be non-negative")
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(data_config["batch_size"]),
        shuffle=False,
        num_workers=workers,
        pin_memory=bool(data_config["pin_memory"]),
        persistent_workers=bool(data_config["persistent_workers"]) and workers > 0,
    )
    # Reconstruct the checkpoint's complete objective, including an optional
    # physical G term, instead of reporting a legacy primary-only val.loss.
    criterion = build_stage1_loss(loss_config)
    type_task = config.get("type_task", {})
    if bool(type_task.get("enabled", False)):
        class_weights = type_task.get("resolved_class_weights")
        if class_weights is None:
            configured_weights = type_task.get("class_weights")
            if not isinstance(configured_weights, list):
                raise ValueError(
                    "multitask checkpoint lacks resolved train-only class weights"
                )
            class_weights = configured_weights
        type_criterion = MaskedCrossEntropyLoss(class_weights).to(device)
        type_loss_weight = float(type_task.get("loss_weight", 0.01))
    else:
        type_criterion = None
        type_loss_weight = 0.0
    thresholds = tuple(float(value) for value in loss_config["thresholds_mm_h"])
    amp_enabled = bool(config["training"]["amp"]) and device.type == "cuda"
    filewise_metrics = FilewisePrecipitationMetrics(
        [str(entry["sample_id"]) for entry in dataset.source_files],
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_replicates=args.bootstrap_replicates,
        confidence_level=args.bootstrap_confidence,
    )
    file_labels = [str(entry["sample_id"]) for entry in dataset.source_files]
    physical_gradient_metrics = build_physical_gradient_metrics(
        config,
        dataset.z,
        file_labels=file_labels,
        disabled=args.no_physical_drdz,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_replicates=args.bootstrap_replicates,
        confidence_level=args.bootstrap_confidence,
    )
    patch_result = evaluate_one_epoch(
        model,
        loader,
        criterion,
        device,
        use_amp=amp_enabled,
        thresholds_mm_h=thresholds,
        max_batches=args.max_batches,
        stratified_metrics=build_stratified_metrics(
            config, dataset.z, force=args.stratified
        ),
        filewise_metrics=filewise_metrics,
        physical_gradient_metrics=physical_gradient_metrics,
        type_criterion=type_criterion,
        type_loss_weight=type_loss_weight,
    )
    result: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "split": args.split,
        "patch_evaluation": {
            **patch_result.to_dict(),
            "coverage": {
                "complete_patch_support": args.max_batches is None,
                "max_batches": args.max_batches,
                "sampling_unit": "unique non-overlapping output-core patch",
            },
        },
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
            **dataset_options,
        )
        full_metrics = PrecipitationRegressionMetrics(thresholds)
        full_stratified = build_stratified_metrics(
            config, full_dataset.z, force=args.stratified
        )
        orbit_directory = project_path(
            args.orbit_output_dir
            if args.orbit_output_dir is not None
            else Path(config["experiment"]["output_dir"]) / f"{args.split}_orbits"
        )
        orbit_directory.mkdir(parents=True, exist_ok=True)
        processed = min(full_orbit_count, len(full_dataset.source_files))
        full_gradient_metrics = build_physical_gradient_metrics(
            config,
            full_dataset.z,
            file_labels=[
                str(full_dataset.source_files[index]["sample_id"])
                for index in range(processed)
            ],
            disabled=args.no_physical_drdz,
            bootstrap_seed=args.bootstrap_seed,
            bootstrap_replicates=args.bootstrap_replicates,
            confidence_level=args.bootstrap_confidence,
        )
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
                variables=("z", "dbz_dpr", "pre_dpr", "cfb", "typePrecip"),
                dtype=np.float32,
                build_masks=True,
            )
            target_rain = sample.variables["pre_dpr"]
            positive_mask = (
                sample.masks["pre_positive_qc"]
                & sample.masks["dpr_reflectivity_valid"]
            )
            full_metrics.update_rain(prediction_rain, target_rain, positive_mask)
            z = sample.variables["z"]
            cfb = sample.variables["cfb"]
            # Do not cast NaN/missing CFB values directly to integers: NumPy
            # would create a meaningless sentinel. Invalid profiles use 0 only
            # for safe indexing and are masked out again after broadcasting.
            finite_cfb = np.isfinite(cfb)
            cfb_index = np.where(finite_cfb, cfb, 0.0).astype(np.int64)
            valid_cfb = (
                finite_cfb
                & (cfb == cfb_index)
                & (cfb_index >= 0)
                & (cfb_index < z.size)
            )
            safe_index = np.where(valid_cfb, cfb_index, 0)
            boundary = z[safe_index]
            cfb_distance = z.reshape(1, 1, -1) - boundary[..., np.newaxis]
            cfb_distance = np.where(
                valid_cfb[..., np.newaxis], cfb_distance, np.nan
            )
            if full_gradient_metrics is not None:
                full_gradient_metrics.update_rain(
                    prediction_rain,
                    target_rain,
                    positive_mask,
                    height_km=z,
                    cfb_distance_km=cfb_distance,
                    precipitation_type=sample.variables["typePrecip"],
                    file_id=file_id,
                )
            if full_stratified is not None:
                full_stratified.update_rain(
                    prediction_rain,
                    target_rain,
                    positive_mask,
                    height_km=z,
                    cfb_distance_km=cfb_distance,
                    precipitation_type=sample.variables["typePrecip"],
                )
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
        if full_stratified is not None:
            result["full_orbit_evaluation"]["metrics"][
                "stratified"
            ] = full_stratified.compute()
        if full_gradient_metrics is not None:
            result["full_orbit_evaluation"]["metrics"][
                "physical_drdz"
            ] = full_gradient_metrics.compute()

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
