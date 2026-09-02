#!/usr/bin/env python3
"""Train the Stage-3 C1-O cascade: freeze Stage 2, adapt Stage 1."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.distributed as distributed


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for value in (PROJECT_ROOT, SRC_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from precipitation_inversion.data.dataset import sha256_file  # noqa: E402
from precipitation_inversion.data.patch_dataset import (  # noqa: E402
    stage1_patch_dataset_kwargs,
)
from precipitation_inversion.data.samplers import FileBlockBatchSampler  # noqa: E402
from precipitation_inversion.data.stage3_patch_dataset import (  # noqa: E402
    Stage3C1PatchDataset,
)
from precipitation_inversion.losses.stage3_losses import (  # noqa: E402
    build_stage3_c1_loss,
)
from precipitation_inversion.metrics.regression import (  # noqa: E402
    StratifiedPrecipitationMetrics,
)
from precipitation_inversion.models.stage3_cascade import (  # noqa: E402
    Stage3C1OracleCascade,
    assert_c1_freeze_contract,
)
from precipitation_inversion.training.engine import (  # noqa: E402
    load_checkpoint,
    save_checkpoint,
    seed_everything,
    should_save_periodic_checkpoint,
    validate_checkpoint_every,
    validate_training_output_directory,
)
from precipitation_inversion.training.stage3_engine import (  # noqa: E402
    evaluate_stage3_c1_one_epoch,
    train_stage3_c1_one_epoch,
)
from scripts.train_stage1_unet3d import (  # noqa: E402
    _initialize_distributed,
    _json_safe,
    _resolve_device,
    build_model as build_stage1_model,
    load_config,
    project_path,
)
from scripts.train_stage2_unet3d import (  # noqa: E402
    build_model as build_stage2_model,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "stage3_c1_o_freeze_s2_train_s1.yaml"
STAGE3_CHECKPOINT_FORMAT = "stage3_c1_oracle_stage1_only_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--ddp-backend", choices=("auto", "nccl", "gloo"))
    parser.add_argument("--ddp-timeout-seconds", type=int)
    parser.add_argument(
        "--smoke-test", action="store_true", help="Run two train/val batches on one epoch."
    )
    parser.add_argument("--skip-postprocessing", action="store_true")
    return parser.parse_args()


def _load_checkpoint_payload(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict) or value.get("format_version") != 1:
        raise ValueError(f"unsupported source checkpoint: {path}")
    if not isinstance(value.get("config"), Mapping):
        raise ValueError(f"source checkpoint has no embedded configuration: {path}")
    return value


def _normalization_vectors(path: Path, variable: str) -> tuple[np.ndarray, np.ndarray]:
    value = json.loads(path.read_text(encoding="utf-8"))
    statistics = value.get("variables", {}).get(variable)
    if not isinstance(statistics, Mapping):
        raise KeyError(f"normalization {path} has no variable {variable!r}")

    def vector(name: str) -> np.ndarray:
        result = np.asarray(
            [np.nan if item is None else item for item in statistics[name]],
            dtype=np.float32,
        )
        if result.ndim != 1 or not np.all(np.isfinite(result)):
            raise ValueError(f"{path}:{variable}.{name} must be a fitted finite vector")
        return result

    mean, std = vector("mean"), vector("std")
    if mean.shape != std.shape or np.any(std <= 0.0):
        raise ValueError(f"{path}:{variable} mean/std contract is invalid")
    return mean, std


def _source_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    sources = config["sources"]
    stage1_path = project_path(sources["stage1_checkpoint"])
    stage2_path = project_path(sources["stage2_checkpoint"])
    threshold_path = project_path(sources["stage2_threshold_file"])
    if not threshold_path.is_file():
        raise FileNotFoundError(threshold_path)
    return {
        "stage1_checkpoint": str(stage1_path),
        "stage1_sha256": sha256_file(stage1_path),
        "stage2_checkpoint": str(stage2_path),
        "stage2_sha256": sha256_file(stage2_path),
        "stage2_threshold_file": str(threshold_path),
        "stage2_threshold_sha256": sha256_file(threshold_path),
    }


def build_c1_components(
    config: dict[str, Any], device: torch.device
) -> tuple[
    Stage3C1OracleCascade,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """Load both source checkpoints and build the frozen/learnable cascade."""

    source = _source_contract(config)
    stage1_payload = _load_checkpoint_payload(Path(source["stage1_checkpoint"]))
    stage2_payload = _load_checkpoint_payload(Path(source["stage2_checkpoint"]))
    stage1_config = dict(stage1_payload["config"])
    stage2_config = dict(stage2_payload["config"])
    if int(stage1_config["model"]["in_channels"]) != 3:
        raise ValueError("C1-O source Stage 1 must use three input channels")
    if not bool(stage1_config.get("type_task", {}).get("enabled", False)):
        raise ValueError("C1-O preregistration expects the sealed T3D Stage-1 source")
    build_stage3_c1_loss(stage1_config["loss"])

    stage1_data = stage1_config["data"]
    stage2_data = stage2_config["data"]
    stage1_norm = project_path(stage1_data["normalization"])
    stage2_norm = project_path(stage2_data["normalization"])
    stage1_mean, stage1_std = _normalization_vectors(stage1_norm, "dbz_dpr")
    stage2_mean, stage2_std = _normalization_vectors(stage2_norm, "dbz_dpr")
    if stage1_mean.shape != stage2_mean.shape:
        raise ValueError("Stage-1/Stage-2 normalization height counts differ")

    stage1_model = build_stage1_model(stage1_config)
    stage2_model = build_stage2_model(stage2_config)
    stage1_model.load_state_dict(stage1_payload["model"], strict=True)
    stage2_model.load_state_dict(stage2_payload["model"], strict=True)
    model = Stage3C1OracleCascade(
        stage2_model,
        stage1_model,
        stage2_dbz_mean=stage2_mean,
        stage2_dbz_std=stage2_std,
        stage1_dbz_mean=stage1_mean,
        stage1_dbz_std=stage1_std,
        stage2_in_channels=int(stage2_config["model"]["in_channels"]),
    ).to(device)
    assert_c1_freeze_contract(model)
    source.update(
        {
            "stage1_source_epoch": int(stage1_payload.get("epoch", -1)),
            "stage2_source_epoch": int(stage2_payload.get("epoch", -1)),
            "stage1_normalization": str(stage1_norm),
            "stage1_normalization_sha256": sha256_file(stage1_norm),
            "stage2_normalization": str(stage2_norm),
            "stage2_normalization_sha256": sha256_file(stage2_norm),
        }
    )
    return model, stage1_config, stage2_config, source


def build_datasets(
    config: Mapping[str, Any],
    stage1_config: Mapping[str, Any],
    stage2_config: Mapping[str, Any],
) -> tuple[Stage3C1PatchDataset, Stage3C1PatchDataset]:
    stage1_data = stage1_config["data"]
    stage2_data = stage2_config["data"]
    stage3_data = config["data"]
    options = stage1_patch_dataset_kwargs(stage1_data, stage1_config["loss"])
    common = {
        "stage1_normalization_stats": project_path(stage1_data["normalization"]),
        "stage2_normalization_stats": project_path(stage2_data["normalization"]),
        "stage2_input_channels": tuple(stage2_data["input_channels"]),
        "cache_size": int(stage3_data["cache_size"]),
        "stage1_options": options,
    }
    train = Stage3C1PatchDataset(
        stage1_index_metadata=project_path(stage1_data["train_index"]),
        stage2_index_metadata=project_path(stage2_data["train_index"]),
        positive_only=bool(stage1_data["train_positive_only"]),
        **common,
    )
    validation = Stage3C1PatchDataset(
        stage1_index_metadata=project_path(stage1_data["val_index"]),
        stage2_index_metadata=project_path(stage2_data["val_index"]),
        positive_only=bool(stage1_data["val_positive_only"]),
        **common,
    )
    expected = int(stage2_config["model"]["in_channels"]) + 2
    if len(train.feature_names) != expected:
        raise ValueError("packed Dataset channel count differs from cascade contract")
    if train.feature_names != validation.feature_names:
        raise ValueError("training and validation packed channels differ")
    return train, validation


def build_loader(
    dataset: Stage3C1PatchDataset,
    config: Mapping[str, Any],
    *,
    training: bool,
    seed: int,
) -> tuple[torch.utils.data.DataLoader, FileBlockBatchSampler]:
    values = config["data"]
    batch_size = int(values["batch_size"])
    sampler = FileBlockBatchSampler(
        dataset,
        batch_size=batch_size,
        block_size=int(values["block_size"]),
        shuffle=training,
        drop_last=training,
        seed=seed,
        even_batches=training,
    )
    workers = int(values["num_workers"])
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=workers,
        pin_memory=bool(values["pin_memory"]),
        persistent_workers=bool(values["persistent_workers"]) and workers > 0,
    )
    return loader, sampler


def _stage1_from_cascade(model: torch.nn.Module) -> torch.nn.Module:
    candidate = model.module if hasattr(model, "module") else model
    if not isinstance(candidate, Stage3C1OracleCascade):
        raise TypeError("unexpected Stage-3 model type")
    return candidate.stage1_model


def _checkpoint_extra(
    config: Mapping[str, Any], source: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "stage3_format": STAGE3_CHECKPOINT_FORMAT,
        "stage3_config": dict(config),
        "stage3_sources": dict(source),
        "stage3_trainable_component": "stage1",
        "stage3_support_interface": "true_dpr_support_oracle",
    }


def _postprocessing_environment() -> dict[str, str]:
    result = dict(os.environ)
    for name in (
        "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
        "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE",
    ):
        result.pop(name, None)
    return result


def run_postprocessing(
    output_dir: Path, config: Mapping[str, Any], *, skip: bool
) -> None:
    values = config.get("postprocessing", {})
    if skip or not bool(values.get("enabled", True)):
        print("[postprocessing] skipped", flush=True)
        return
    commands = [
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "plot_stage1_training_history.py"),
            str(output_dir),
            "--dpi", str(int(values.get("dpi", 150))),
        ],
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "evaluate_stage3_cascade.py"),
            "--checkpoint", str(output_dir / "best.pt"),
            "--split", "val",
            "--output-dir", str(output_dir / "analysis" / "full_validation_cascade"),
            "--device", str(values.get("device", "auto")),
            "--save-orbits", str(int(values.get("save_orbits", 6))),
            "--visualize",
            "--overwrite",
        ],
    ]
    environment = _postprocessing_environment()
    for command in commands:
        print("[postprocessing] running: " + " ".join(command), flush=True)
        subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if config.get("experiment", {}).get("method") != "S3-C1-O-S1Adapt":
        raise ValueError("this trainer only implements S3-C1-O-S1Adapt")
    training = config["training"]
    checkpoint_every = validate_checkpoint_every(training.get("checkpoint_every", 10))
    training["checkpoint_every"] = checkpoint_every
    output_dir = validate_training_output_directory(
        project_path(args.output_dir or config["experiment"]["output_dir"]),
        resume=args.resume,
    )

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    device = _resolve_device(args.device, local_rank, world_size)
    distributed_values = config.get("runtime", {}).get("distributed", {})
    backend = str(args.ddp_backend or distributed_values.get("backend", "auto"))
    timeout = int(
        args.ddp_timeout_seconds
        if args.ddp_timeout_seconds is not None
        else distributed_values.get("timeout_seconds", 600)
    )
    active_backend = _initialize_distributed(
        device, world_size, backend=backend, timeout_seconds=timeout
    )
    is_main = rank == 0
    if args.smoke_test:
        config["data"]["num_workers"] = 0
        config["data"]["persistent_workers"] = False
        config["data"]["pin_memory"] = device.type == "cuda"

    seed = int(config["experiment"]["seed"])
    seed_everything(seed + rank, deterministic=bool(config["runtime"]["deterministic"]))
    model, stage1_config, stage2_config, source = build_c1_components(config, device)
    train_dataset, val_dataset = build_datasets(config, stage1_config, stage2_config)
    train_loader, train_sampler = build_loader(
        train_dataset, config, training=True, seed=seed
    )
    val_loader, val_sampler = build_loader(
        val_dataset, config, training=False, seed=seed
    )
    config["resolved_contract"] = {
        "packed_input_channels": train_dataset.feature_names,
        "stage1_input_channels": [
            "predicted_dpr_dbz_standardized_stage1",
            "true_dpr_support_oracle",
            "height_scaled",
        ],
        "packed_shape": ["B", len(train_dataset.feature_names), 64, 64, 60],
        "stage1_shape": ["B", 3, 64, 64, 60],
        "loss": "I + 0.02G; no type loss",
        "sources": source,
    }
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "resolved_config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "method": "S3-C1-O-S1Adapt",
                    "world_size": world_size,
                    "backend": active_backend,
                    "device": str(device),
                    "train_patches": len(train_dataset),
                    "val_patches": len(val_dataset),
                    "packed_channels": train_dataset.feature_names,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            output_device=device.index if device.type == "cuda" else None,
            broadcast_buffers=bool(distributed_values.get("broadcast_buffers", False)),
            find_unused_parameters=bool(distributed_values.get("find_unused_parameters", False)),
            gradient_as_bucket_view=bool(distributed_values.get("gradient_as_bucket_view", True)),
            static_graph=bool(distributed_values.get("static_graph", True)),
            bucket_cap_mb=float(distributed_values.get("bucket_cap_mb", 25.0)),
        )
    assert_c1_freeze_contract(model)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(config["optimizer"]["learning_rate"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )
    epochs = 1 if args.smoke_test else int(training["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=float(config["scheduler"]["eta_min"])
    )
    criterion = build_stage3_c1_loss(stage1_config["loss"])
    amp_enabled = bool(training["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler(
        "cuda", enabled=amp_enabled,
        init_scale=float(training.get("amp_initial_scale", 1024.0)),
    )
    thresholds = tuple(float(value) for value in stage1_config["loss"]["thresholds_mm_h"])

    start_epoch = 0
    global_step = 0
    best_rmse = math.inf
    bad_epochs = 0
    extra = _checkpoint_extra(config, source)
    if args.resume is not None:
        resumed = load_checkpoint(
            args.resume,
            _stage1_from_cascade(model),
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            map_location=device,
        )
        if resumed.get("stage3_format") != STAGE3_CHECKPOINT_FORMAT:
            raise ValueError("--resume is not a C1-O Stage-3 checkpoint")
        previous_sources = resumed.get("stage3_sources", {})
        if previous_sources.get("stage2_sha256") != source["stage2_sha256"]:
            raise ValueError("resume checkpoint uses a different frozen Stage-2 model")
        start_epoch = int(resumed["epoch"]) + 1
        global_step = int(resumed["global_step"])
        prior_metrics = resumed.get("metrics") or {}
        best_rmse = float(prior_metrics.get("best_rain_rmse", math.inf))
        bad_epochs = int(prior_metrics.get("early_stopping_bad_epochs", 0))

    early = training["early_stopping"]
    patience = int(early["patience"])
    min_delta = float(early["min_delta"])
    if str(early["monitor"]).lower() != "val_rain_rmse":
        raise ValueError("C1-O early stopping must monitor val_rain_rmse")
    max_batches = 2 if args.smoke_test else config["evaluation"].get("max_batches")
    stratified = StratifiedPrecipitationMetrics(
        val_dataset.z.tolist(),
        cfb_distance_edges_km=tuple(
            float(value) for value in config["evaluation"]["cfb_distance_edges_km"]
        ),
        intensity_thresholds_mm_h=thresholds,
    )
    completed = False
    try:
        for epoch in range(start_epoch, epochs):
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)
            train_result = train_stage3_c1_one_epoch(
                model, train_loader, optimizer, criterion, device,
                scaler=scaler, use_amp=amp_enabled,
                grad_clip_norm=float(training["grad_clip_norm"]),
                accumulation_steps=int(training["accumulation_steps"]),
                thresholds_mm_h=thresholds, max_batches=max_batches,
            )
            val_result = evaluate_stage3_c1_one_epoch(
                model, val_loader, criterion, device,
                use_amp=amp_enabled, thresholds_mm_h=thresholds,
                max_batches=max_batches, stratified_metrics=stratified,
            )
            if train_result.optimizer_steps:
                scheduler.step()
            global_step += train_result.optimizer_steps
            val_rmse = float(val_result.metrics["rain"]["all"]["rmse"])
            improved = math.isfinite(val_rmse) and val_rmse < best_rmse - min_delta
            if improved:
                best_rmse, bad_epochs = val_rmse, 0
            else:
                bad_epochs += 1
            should_stop = bool(early["enabled"]) and bad_epochs >= patience
            record = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "epoch": epoch,
                "global_step": global_step,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train": train_result.to_dict(),
                "val": val_result.to_dict(),
                "best_rain_rmse": best_rmse,
                "early_stopping_bad_epochs": bad_epochs,
                "early_stopping_triggered": should_stop,
            }
            if is_main:
                safe = _json_safe(record)
                print(json.dumps(safe, ensure_ascii=False), flush=True)
                with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(safe, ensure_ascii=False) + "\n")
                checkpoint_metrics = {
                    "best_rain_rmse": best_rmse,
                    "early_stopping_bad_epochs": bad_epochs,
                    "train": train_result.to_dict(),
                    "val": val_result.to_dict(),
                }
                save_targets = [("last.pt", True), ("best.pt", improved)]
                if should_save_periodic_checkpoint(epoch, checkpoint_every):
                    save_targets.append((f"epoch_{epoch:04d}.pt", True))
                for name, selected in save_targets:
                    if selected:
                        # C1 checkpoints contain only Stage-1 model weights;
                        # the frozen Stage-2 path/hash/config live in metadata.
                        save_checkpoint(
                            output_dir / name,
                            _stage1_from_cascade(model),
                            epoch=epoch, global_step=global_step,
                            optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                            config=stage1_config, metrics=checkpoint_metrics,
                            extra=extra,
                        )
            if world_size > 1:
                distributed.barrier(
                    device_ids=[device.index] if device.type == "cuda" else None
                )
            if should_stop:
                if is_main:
                    print(f"[early-stopping] epoch={epoch}, best_rmse={best_rmse:.6f}")
                break
        completed = True
    finally:
        if distributed.is_available() and distributed.is_initialized():
            distributed.destroy_process_group()

    if completed and is_main:
        del model, optimizer, scheduler, scaler
        if device.type == "cuda":
            torch.cuda.empty_cache()
        try:
            run_postprocessing(output_dir, config, skip=args.skip_postprocessing)
        except Exception as error:
            failure = output_dir / "analysis" / "POSTPROCESSING_FAILED.txt"
            failure.parent.mkdir(parents=True, exist_ok=True)
            failure.write_text(f"{type(error).__name__}: {error}\n", encoding="utf-8")
            print(f"[postprocessing] ERROR: {error}", file=sys.stderr, flush=True)
            if bool(config.get("postprocessing", {}).get("fail_on_error", False)):
                raise


if __name__ == "__main__":
    main()
