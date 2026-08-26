#!/usr/bin/env python3
"""Train the height-preserving stage-one 3D U-Net on patch indices.

Launch one GPU with this script directly. On a shared multi-GPU server, use
``scripts/launch_stage1_ddp.sh`` so physical GPU IDs are selected explicitly.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.distributed as distributed


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from precipitation_inversion.data.patch_dataset import (  # noqa: E402
    Stage1PatchDataset,
    stage1_patch_dataset_kwargs,
)
from precipitation_inversion.data.samplers import FileBlockBatchSampler  # noqa: E402
from precipitation_inversion.losses.masked_losses import MaskedSmoothL1Loss  # noqa: E402
from precipitation_inversion.metrics.regression import (  # noqa: E402
    StratifiedPrecipitationMetrics,
)
from precipitation_inversion.models.unet3d import Stage1UNet3D  # noqa: E402
from precipitation_inversion.training.engine import (  # noqa: E402
    evaluate_one_epoch,
    load_checkpoint,
    save_checkpoint,
    seed_everything,
    train_one_epoch,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "stage1_unet3d.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", help="auto, cpu, cuda, or cuda:N", default="auto")
    parser.add_argument(
        "--ddp-backend",
        choices=("auto", "nccl", "gloo"),
        help="Override runtime.distributed.backend from the configuration.",
    )
    parser.add_argument(
        "--ddp-timeout-seconds",
        type=int,
        help="Override the process-group timeout; useful for fast diagnostics.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one epoch with two train and two validation batches.",
    )
    parser.add_argument(
        "--skip-postprocessing",
        action="store_true",
        help="Do not create training-history and fixed-test visualizations.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    """Load YAML; JSON-compatible YAML works without an optional dependency."""

    source = path.expanduser().resolve()
    text = source.read_text(encoding="utf-8")
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


def _json_safe(value: Any) -> Any:
    """Replace NaN/Inf before writing strict JSON logs."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _resolve_device(requested: str, local_rank: int, world_size: int) -> torch.device:
    # Under torchrun, CUDA_VISIBLE_DEVICES defines the physical cards and
    # LOCAL_RANK selects one visible card per process. For example, physical
    # cards 5 and 6 become local cuda:0 and cuda:1.
    if requested == "auto":
        requested = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    elif requested == "cuda":
        requested = f"cuda:{local_rank}"
    elif world_size > 1 and requested.startswith("cuda:"):
        raise ValueError(
            "do not pass an explicit cuda:N to a multi-process run; use "
            "--device auto (or cuda) so every LOCAL_RANK gets a different GPU"
        )
    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        torch.cuda.set_device(device)
    return device


def _initialize_distributed(
    device: torch.device,
    world_size: int,
    *,
    backend: str,
    timeout_seconds: int,
) -> str | None:
    if world_size <= 1:
        return None
    if timeout_seconds <= 0:
        raise ValueError("distributed timeout_seconds must be positive")
    resolved_backend = "nccl" if backend == "auto" and device.type == "cuda" else backend
    if resolved_backend == "auto":
        resolved_backend = "gloo"
    if resolved_backend == "nccl" and device.type != "cuda":
        raise ValueError("the NCCL backend requires CUDA devices")
    distributed.init_process_group(
        backend=resolved_backend,
        init_method="env://",
        timeout=timedelta(seconds=timeout_seconds),
        # PyTorch can form NCCL communicators eagerly and knows which device to
        # use for barriers, avoiding an ambiguous-device runtime warning.
        device_id=device if device.type == "cuda" else None,
    )
    return resolved_backend


def build_model(config: Mapping[str, Any]) -> Stage1UNet3D:
    model_config = config["model"]
    return Stage1UNet3D(
        in_channels=int(model_config["in_channels"]),
        out_channels=int(model_config["out_channels"]),
        base_channels=int(model_config["base_channels"]),
        channel_multipliers=tuple(model_config["channel_multipliers"]),
        max_groups=int(model_config["max_groups"]),
        bottleneck_dropout=float(model_config["bottleneck_dropout"]),
    )


def _postprocessing_environment() -> dict[str, str]:
    """Remove torchrun rank variables before launching a single child process."""

    environment = dict(os.environ)
    for name in (
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "GROUP_RANK",
        "ROLE_RANK",
        "ROLE_WORLD_SIZE",
    ):
        environment.pop(name, None)
    return environment


def run_postprocessing(
    output_dir: Path, config: Mapping[str, Any], *, skip: bool = False
) -> None:
    """Run history, full-validation/dRdz, and test diagnostics once."""

    values = config.get("postprocessing", {})
    if skip or not bool(values.get("enabled", True)):
        print("[postprocessing] skipped", flush=True)
        return
    best_checkpoint = output_dir / "best.pt"
    if not best_checkpoint.is_file():
        raise FileNotFoundError(
            f"cannot visualize test predictions without {best_checkpoint}"
        )
    commands = [
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "plot_stage1_training_history.py"),
            str(output_dir),
            "--dpi",
            str(int(values.get("dpi", 150))),
        ],
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "evaluate_stage1_unet3d.py"),
            str(best_checkpoint),
            "--split",
            "val",
            "--stratified",
            "--device",
            str(values.get("device", "auto")),
            "--output",
            str(output_dir / "analysis" / "full_validation" / "metrics.json"),
        ],
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "plot_stage1_stratified_metrics.py"),
            str(output_dir / "analysis" / "full_validation" / "metrics.json"),
            "--output-dir",
            str(output_dir / "analysis" / "full_validation" / "stratified"),
            "--dpi",
            str(int(values.get("dpi", 150))),
        ],
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "visualize_stage1_test_predictions.py"),
            str(best_checkpoint),
            "--output-dir",
            str(output_dir / "analysis" / "test_predictions"),
            "--sample-count",
            str(int(values.get("test_sample_count", 6))),
            "--seed",
            str(int(values.get("selection_seed", 2026))),
            "--height-km",
            str(float(values.get("height_km", 2.0))),
            "--max-scatter-points",
            str(int(values.get("max_scatter_points", 200_000))),
            "--device",
            str(values.get("device", "auto")),
            "--dpi",
            str(int(values.get("dpi", 150))),
        ],
    ]
    environment = _postprocessing_environment()
    for command in commands:
        print(f"[postprocessing] running: {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)
    analysis_directory = output_dir / "analysis"
    analysis_directory.mkdir(parents=True, exist_ok=True)
    (analysis_directory / "README.md").write_text(
        "# Stage-1训练后自动分析\n\n"
        "- [训练历史、强度分箱和泛化间隙](training_history/summary.md)\n"
        "- [完整验证集分高度、相对CFB、强度及降水类型分析]"
        "(full_validation/stratified/summary.md)\n"
        "- [物理垂直降水率梯度dR/dz分析]"
        "(full_validation/stratified/drdz_summary.md)\n"
        "- [固定测试轨道预测与DPR标签对比](test_predictions/summary.md)\n\n"
        "详细PNG、CSV、JSON和逐轨NPZ位于对应子目录。\n",
        encoding="utf-8",
    )


def build_loader(
    dataset: Stage1PatchDataset,
    config: Mapping[str, Any],
    *,
    training: bool,
    seed: int,
) -> tuple[torch.utils.data.DataLoader, FileBlockBatchSampler]:
    data_config = config["data"]
    batch_size = int(data_config["batch_size"])
    sampler = FileBlockBatchSampler(
        dataset,
        batch_size=batch_size,
        block_size=int(data_config["block_size"]),
        shuffle=training,
        drop_last=training,
        seed=seed,
        # Training requires equal-length shards for DDP backward. Validation
        # uses the unwrapped, already-synchronized module for forward passes,
        # so uneven disjoint shards safely retain every patch exactly once.
        even_batches=training,
    )
    workers = int(data_config["num_workers"])
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=workers,
        pin_memory=bool(data_config["pin_memory"]),
        persistent_workers=bool(data_config["persistent_workers"]) and workers > 0,
    )
    return loader, sampler


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size <= 0 or not 0 <= rank < world_size or local_rank < 0:
        raise ValueError(
            f"invalid distributed environment: RANK={rank}, LOCAL_RANK={local_rank}, "
            f"WORLD_SIZE={world_size}"
        )
    device = _resolve_device(args.device, local_rank, world_size)
    distributed_config = config.get("runtime", {}).get("distributed", {})
    requested_backend = str(
        args.ddp_backend or distributed_config.get("backend", "auto")
    ).lower()
    if requested_backend not in ("auto", "nccl", "gloo"):
        raise ValueError("distributed backend must be auto, nccl, or gloo")
    timeout_seconds = int(
        args.ddp_timeout_seconds
        if args.ddp_timeout_seconds is not None
        else distributed_config.get("timeout_seconds", 300)
    )
    if rank == 0 and world_size > 1:
        print(
            f"[stage1-ddp] initializing {world_size} processes; "
            f"timeout={timeout_seconds}s. Do not interrupt on transient c10d "
            "hostname warnings.",
            flush=True,
        )
    active_backend = _initialize_distributed(
        device,
        world_size,
        backend=requested_backend,
        timeout_seconds=timeout_seconds,
    )
    is_main = rank == 0

    experiment = config["experiment"]
    data_config = config["data"]
    training_config = config["training"]
    loss_config = config["loss"]
    seed = int(experiment["seed"])
    seed_everything(
        seed + rank,
        deterministic=bool(config["runtime"]["deterministic"]),
    )
    output_dir = project_path(
        args.output_dir if args.output_dir is not None else experiment["output_dir"]
    )
    if is_main:
        device_name = (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
        )
        print(
            json.dumps(
                {
                    "distributed": world_size > 1,
                    "world_size": world_size,
                    "backend": active_backend,
                    "device_type": device.type,
                    "rank0_device": str(device),
                    "rank0_device_name": device_name,
                    "master_addr": os.environ.get("MASTER_ADDR"),
                    "master_port": os.environ.get("MASTER_PORT"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "resolved_config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    dataset_options = stage1_patch_dataset_kwargs(data_config, loss_config)
    train_dataset = Stage1PatchDataset(
        project_path(data_config["train_index"]),
        project_path(data_config["normalization"]),
        positive_only=bool(data_config["train_positive_only"]),
        cache_size=int(data_config["cache_size"]),
        **dataset_options,
    )
    val_dataset = Stage1PatchDataset(
        project_path(data_config["val_index"]),
        project_path(data_config["normalization"]),
        positive_only=bool(data_config["val_positive_only"]),
        cache_size=int(data_config["cache_size"]),
        **dataset_options,
    )
    expected_channels = int(config["model"]["in_channels"])
    if len(train_dataset.feature_names) != expected_channels:
        raise ValueError(
            "model.in_channels does not match the configured Dataset: "
            f"model={expected_channels}, dataset={len(train_dataset.feature_names)} "
            f"({train_dataset.feature_names})"
        )
    if val_dataset.feature_names != train_dataset.feature_names:
        raise ValueError("training and validation input channels differ")
    train_loader, train_sampler = build_loader(
        train_dataset, config, training=True, seed=seed
    )
    val_loader, val_sampler = build_loader(
        val_dataset, config, training=False, seed=seed
    )

    model: torch.nn.Module = build_model(config).to(device)
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            output_device=device.index if device.type == "cuda" else None,
            broadcast_buffers=bool(distributed_config.get("broadcast_buffers", False)),
            find_unused_parameters=bool(
                distributed_config.get("find_unused_parameters", False)
            ),
            gradient_as_bucket_view=bool(
                distributed_config.get("gradient_as_bucket_view", True)
            ),
            static_graph=bool(distributed_config.get("static_graph", True)),
            bucket_cap_mb=float(distributed_config.get("bucket_cap_mb", 25.0)),
        )
    optimizer_config = config["optimizer"]
    if str(optimizer_config["name"]).lower() != "adamw":
        raise ValueError("only AdamW is currently supported")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_config["learning_rate"]),
        weight_decay=float(optimizer_config["weight_decay"]),
    )
    epochs = 1 if args.smoke_test else int(training_config["epochs"])
    scheduler_config = config["scheduler"]
    scheduler_name = str(scheduler_config["name"]).lower()
    scheduler = (
        None
        if scheduler_name == "none"
        else torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=float(scheduler_config["eta_min"]),
        )
    )
    if scheduler_name not in ("none", "cosine"):
        raise ValueError("scheduler name must be 'none' or 'cosine'")
    if str(loss_config["name"]).lower() != "masked_smooth_l1":
        raise ValueError("only masked_smooth_l1 is currently supported")
    criterion = MaskedSmoothL1Loss(beta=float(loss_config["beta"]))
    amp_enabled = bool(training_config["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
        init_scale=float(training_config.get("amp_initial_scale", 1024.0)),
    )
    thresholds = tuple(float(value) for value in loss_config["thresholds_mm_h"])
    stratified_config = config.get("evaluation", {}).get("stratified", {})
    stratified_metrics: StratifiedPrecipitationMetrics | None = None
    if bool(stratified_config.get("enabled", False)):
        height_edges = stratified_config.get("height_bin_edges_km")
        common_stratified_options = {
            "cfb_distance_edges_km": tuple(
                float(value)
                for value in stratified_config.get(
                    "cfb_distance_edges_km", (-1.0, 0.0, 0.5, 2.0)
                )
            ),
            "intensity_thresholds_mm_h": thresholds,
        }
        stratified_metrics = (
            StratifiedPrecipitationMetrics(
                val_dataset.z.tolist(), **common_stratified_options
            )
            if height_edges is None
            else StratifiedPrecipitationMetrics(
                height_bin_edges_km=tuple(float(value) for value in height_edges),
                **common_stratified_options,
            )
        )

    start_epoch = 0
    global_step = 0
    best_rmse = math.inf
    early_stopping = training_config.get("early_stopping", {})
    early_stopping_enabled = bool(early_stopping.get("enabled", False))
    early_stopping_patience = int(early_stopping.get("patience", 12))
    early_stopping_min_delta = float(early_stopping.get("min_delta", 0.0))
    early_stopping_monitor = str(
        early_stopping.get("monitor", "val_rain_rmse")
    ).lower()
    if early_stopping_enabled and early_stopping_patience <= 0:
        raise ValueError("early-stopping patience must be positive")
    if early_stopping_min_delta < 0 or not math.isfinite(early_stopping_min_delta):
        raise ValueError("early-stopping min_delta must be finite and non-negative")
    if early_stopping_monitor != "val_rain_rmse":
        raise ValueError("only early-stopping monitor 'val_rain_rmse' is supported")
    early_stopping_bad_epochs = 0
    if args.resume is not None:
        checkpoint = load_checkpoint(
            args.resume,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            map_location=device,
        )
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        previous = checkpoint.get("metrics") or {}
        best_rmse = float(previous.get("best_rain_rmse", math.inf))
        early_stopping_bad_epochs = int(
            previous.get("early_stopping_bad_epochs", 0)
        )

    max_batches = 2 if args.smoke_test else config["evaluation"]["max_batches"]
    training_completed = False
    try:
        for epoch in range(start_epoch, epochs):
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)
            train_result = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                scaler=scaler,
                use_amp=amp_enabled,
                grad_clip_norm=float(training_config["grad_clip_norm"]),
                accumulation_steps=int(training_config["accumulation_steps"]),
                thresholds_mm_h=thresholds,
                max_batches=max_batches,
            )
            val_result = evaluate_one_epoch(
                model,
                val_loader,
                criterion,
                device,
                use_amp=amp_enabled,
                thresholds_mm_h=thresholds,
                max_batches=max_batches,
                stratified_metrics=stratified_metrics,
            )
            if scheduler is not None and train_result.optimizer_steps > 0:
                scheduler.step()
            global_step += train_result.optimizer_steps
            val_rmse = float(val_result.metrics["rain"]["all"]["rmse"])
            improved = (
                math.isfinite(val_rmse)
                and val_rmse < best_rmse - early_stopping_min_delta
            )
            if improved:
                best_rmse = val_rmse
                early_stopping_bad_epochs = 0
            else:
                early_stopping_bad_epochs += 1
            should_stop = (
                early_stopping_enabled
                and early_stopping_bad_epochs >= early_stopping_patience
            )

            if is_main:
                record = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "train": train_result.to_dict(),
                    "val": val_result.to_dict(),
                    "best_rain_rmse": best_rmse,
                    "early_stopping_bad_epochs": early_stopping_bad_epochs,
                    "early_stopping_triggered": should_stop,
                }
                safe_record = _json_safe(record)
                print(json.dumps(safe_record, ensure_ascii=False), flush=True)
                with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(safe_record, ensure_ascii=False) + "\n")
                checkpoint_metrics = {
                    "best_rain_rmse": best_rmse,
                    "early_stopping_bad_epochs": early_stopping_bad_epochs,
                    "train": train_result.to_dict(),
                    "val": val_result.to_dict(),
                }
                save_checkpoint(
                    output_dir / "last.pt",
                    model,
                    epoch=epoch,
                    global_step=global_step,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    config=config,
                    metrics=checkpoint_metrics,
                )
                if improved:
                    save_checkpoint(
                        output_dir / "best.pt",
                        model,
                        epoch=epoch,
                        global_step=global_step,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        config=config,
                        metrics=checkpoint_metrics,
                    )
                checkpoint_every = int(training_config["checkpoint_every"])
                if checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:
                    save_checkpoint(
                        output_dir / f"epoch_{epoch:04d}.pt",
                        model,
                        epoch=epoch,
                        global_step=global_step,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        config=config,
                        metrics=checkpoint_metrics,
                    )
            if world_size > 1:
                distributed.barrier(
                    device_ids=[device.index] if device.type == "cuda" else None
                )
            if should_stop:
                if is_main:
                    print(
                        "[early-stopping] no validation rain-RMSE improvement "
                        f"greater than {early_stopping_min_delta:g} for "
                        f"{early_stopping_bad_epochs} epochs; stopping at epoch "
                        f"{epoch} (best={best_rmse:.6f} mm/h).",
                        flush=True,
                    )
                break
        training_completed = True
    finally:
        if distributed.is_available() and distributed.is_initialized():
            distributed.destroy_process_group()

    # Every DDP rank has crossed the final barrier. Only rank 0 performs the
    # expensive single-process analysis, after releasing its training model.
    if training_completed and is_main:
        del model, optimizer, scheduler, scaler
        if device.type == "cuda":
            torch.cuda.empty_cache()
        try:
            run_postprocessing(output_dir, config, skip=args.skip_postprocessing)
        except Exception as error:
            print(f"[postprocessing] ERROR: {error}", file=sys.stderr, flush=True)
            failure = output_dir / "analysis" / "POSTPROCESSING_FAILED.txt"
            failure.parent.mkdir(parents=True, exist_ok=True)
            failure.write_text(
                "训练已经完成，但自动分析失败。请检查训练日志并手动重跑。\n"
                f"错误：{type(error).__name__}: {error}\n",
                encoding="utf-8",
            )
            if bool(config.get("postprocessing", {}).get("fail_on_error", False)):
                raise


if __name__ == "__main__":
    main()
