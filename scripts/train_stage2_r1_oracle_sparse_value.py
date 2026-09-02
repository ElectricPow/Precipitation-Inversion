#!/usr/bin/env python3
"""Train a non-deployable R1 DPR-sparse spatial-completion control."""

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
for value in (PROJECT_ROOT, SRC_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from precipitation_inversion.data.stage2_patch_dataset import (  # noqa: E402
    STAGE2_R1_ORACLE_SPARSE_VALUE_INPUT_CHANNELS,
    Stage2PatchDataset,
    stage2_patch_dataset_kwargs,
    validate_stage2_input_channels,
)
from precipitation_inversion.losses.stage2_completion_losses import (  # noqa: E402
    build_stage2_completion_loss,
)
from precipitation_inversion.models.stage2_completion_unet3d import (  # noqa: E402
    Stage2CompletionUNet3D,
)
from precipitation_inversion.models.stage2_partial_completion_unet3d import (  # noqa: E402
    Stage2PartialCompletionUNet3D,
)
from precipitation_inversion.training.engine import (  # noqa: E402
    load_checkpoint,
    save_checkpoint,
    seed_everything,
    should_save_periodic_checkpoint,
    validate_checkpoint_every,
    validate_training_output_directory,
)
from precipitation_inversion.training.stage2_completion_engine import (  # noqa: E402
    evaluate_stage2_completion_one_epoch,
    train_stage2_completion_one_epoch,
)
from scripts.train_stage2_unet3d import (  # noqa: E402
    _initialize_distributed,
    _json_safe,
    _postprocessing_environment,
    _resolve_device,
    build_loaders,
    load_config,
    project_path,
)

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "stage2_r1_o_dpr_sparse_value.yaml"

R1_O_TASK = "stage2_r1_o_dpr_sparse_value"
R1_P_TASK = "stage2_r1_p_partial_conv"
R1_O_ARCHITECTURE = "standard_unet3d"
R1_P_ARCHITECTURE = "partial_conv_unet3d"
R1_CHECKPOINT_FORMATS = {
    R1_O_TASK: "stage2_r1_o_dpr_sparse_value_v1",
    R1_P_TASK: "stage2_r1_p_partial_conv_v1",
}
R1_DISPLAY_NAMES = {
    R1_O_TASK: "S2-R1-O-DPRSparseValue",
    R1_P_TASK: "S2-R1-P-PartialConv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--ddp-backend", choices=("auto", "nccl", "gloo"))
    parser.add_argument("--ddp-timeout-seconds", type=int)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--skip-postprocessing", action="store_true")
    return parser.parse_args()


def stage2_completion_contract(config: Mapping[str, Any]) -> tuple[str, str, str]:
    """Return the sealed ``(task, architecture, checkpoint_format)`` contract."""

    task = str(config.get("experiment", {}).get("task", ""))
    if task not in R1_CHECKPOINT_FORMATS:
        raise ValueError(f"unsupported R1 completion task: {task!r}")
    architecture = str(
        config.get("model", {}).get("architecture", R1_O_ARCHITECTURE)
    )
    expected = R1_O_ARCHITECTURE if task == R1_O_TASK else R1_P_ARCHITECTURE
    if architecture != expected:
        raise ValueError(
            f"{R1_DISPLAY_NAMES[task]} requires model.architecture={expected!r}"
        )
    return task, architecture, R1_CHECKPOINT_FORMATS[task]


def build_model(config: Mapping[str, Any]) -> torch.nn.Module:
    values = config["model"]
    _task, architecture, _checkpoint_format = stage2_completion_contract(config)
    model_type = (
        Stage2CompletionUNet3D
        if architecture == R1_O_ARCHITECTURE
        else Stage2PartialCompletionUNet3D
    )
    return model_type(
        in_channels=int(values["in_channels"]),
        base_channels=int(values["base_channels"]),
        channel_multipliers=tuple(int(item) for item in values["channel_multipliers"]),
        max_groups=int(values["max_groups"]),
        bottleneck_dropout=float(values["bottleneck_dropout"]),
    )


def validate_r1_config(config: Mapping[str, Any]) -> tuple[str, ...]:
    experiment = config.get("experiment", {})
    task, _architecture, _checkpoint_format = stage2_completion_contract(config)
    if experiment.get("deployable") is not False:
        raise ValueError("R1 completion controls must explicitly declare deployable=false")
    channels = validate_stage2_input_channels(config["data"].get("input_channels"))
    if channels != STAGE2_R1_ORACLE_SPARSE_VALUE_INPUT_CHANNELS:
        raise ValueError(f"{R1_DISPLAY_NAMES[task]} requires the sealed oracle input")
    if int(config["model"]["in_channels"]) != len(channels):
        raise ValueError("R1 completion model.in_channels differs from input channels")
    build_stage2_completion_loss(config["loss"])
    return channels


def run_postprocessing(
    output_dir: Path,
    config: Mapping[str, Any],
    *,
    skip: bool,
) -> None:
    task, _architecture, _checkpoint_format = stage2_completion_contract(config)
    display_name = R1_DISPLAY_NAMES[task]
    values = config.get("postprocessing", {})
    if skip or not bool(values.get("enabled", True)):
        print(f"[{display_name}-postprocessing] skipped", flush=True)
        return
    checkpoint = output_dir / "best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    analysis = output_dir / "analysis"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "evaluate_stage2_r1_oracle_sparse_value.py"),
        str(checkpoint),
        "--split", "val",
        "--output-dir", str(analysis / "full_validation"),
        "--device", str(values.get("device", "auto")),
        "--overwrite",
    ]
    environment = _postprocessing_environment()
    print(f"[{display_name}-postprocessing] running: " + " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)

    stage1 = values.get("stage1_checkpoint")
    if stage1:
        cascade = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "evaluate_stage2_r1_oracle_sparse_cascade.py"),
            "--stage1-checkpoint", str(project_path(stage1)),
            "--stage2-checkpoint", str(checkpoint),
            "--split", "val",
            "--output-dir", str(analysis / "frozen_stage1_cascade"),
            "--device", str(values.get("device", "auto")),
            "--bootstrap-replicates", str(int(values.get("bootstrap_replicates", 2000))),
            "--selection-seed", str(int(values.get("selection_seed", 2026))),
            "--overwrite",
        ]
        print(f"[{display_name}-postprocessing] running: " + " ".join(cascade), flush=True)
        subprocess.run(cascade, cwd=PROJECT_ROOT, env=environment, check=True)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    channels = validate_r1_config(config)
    task, _architecture, checkpoint_format = stage2_completion_contract(config)
    display_name = R1_DISPLAY_NAMES[task]
    training = config["training"]
    checkpoint_every = validate_checkpoint_every(training.get("checkpoint_every", 10))
    training["checkpoint_every"] = checkpoint_every
    output_dir = project_path(
        args.output_dir if args.output_dir is not None else config["experiment"]["output_dir"]
    )
    output_dir = validate_training_output_directory(output_dir, resume=args.resume)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size <= 0 or not 0 <= rank < world_size or local_rank < 0:
        raise ValueError("invalid torchrun rank environment")
    device = _resolve_device(args.device, local_rank, world_size)
    distributed_values = config.get("runtime", {}).get("distributed", {})
    backend = str(args.ddp_backend or distributed_values.get("backend", "auto"))
    timeout = int(
        args.ddp_timeout_seconds
        if args.ddp_timeout_seconds is not None
        else distributed_values.get("timeout_seconds", 300)
    )
    active_backend = _initialize_distributed(
        device, world_size, backend=backend, timeout_seconds=timeout
    )
    is_main = rank == 0
    seed = int(config["experiment"]["seed"])
    seed_everything(seed + rank, deterministic=bool(config["runtime"]["deterministic"]))
    data = config["data"]
    if args.smoke_test:
        data["num_workers"] = 0
        data["persistent_workers"] = False
        data["pin_memory"] = device.type == "cuda"
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(json.dumps({
            "experiment": display_name,
            "deployable": False,
            "world_size": world_size,
            "backend": active_backend,
            "device": str(device),
            "input_channels": channels,
        }, ensure_ascii=False), flush=True)

    options = stage2_patch_dataset_kwargs(config["loss"])
    train_dataset = Stage2PatchDataset(
        project_path(data["train_index"]), project_path(data["normalization"]),
        cache_size=int(data["cache_size"]), input_channels=channels, **options,
    )
    val_dataset = Stage2PatchDataset(
        project_path(data["val_index"]), project_path(data["normalization"]),
        cache_size=int(data["cache_size"]), input_channels=channels, **options,
    )
    config["data"]["input_channels"] = list(channels)
    train_loader, train_sampler, val_loader, val_sampler = build_loaders(
        train_dataset, val_dataset, config, seed=seed
    )
    if is_main:
        config["data"]["resolved_train_stratum_counts"] = train_sampler.stratum_counts
        config["data"]["resolved_train_epoch_quotas"] = train_sampler.epoch_quotas

    model: torch.nn.Module = build_model(config).to(device)
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
    optimizer_values = config["optimizer"]
    if str(optimizer_values["name"]).lower() != "adamw":
        raise ValueError("only AdamW is supported")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_values["learning_rate"]),
        weight_decay=float(optimizer_values["weight_decay"]),
    )
    epochs = 1 if args.smoke_test else int(training["epochs"])
    scheduler_values = config["scheduler"]
    scheduler_name = str(scheduler_values["name"]).lower()
    if scheduler_name not in {"none", "cosine"}:
        raise ValueError("scheduler name must be none or cosine")
    scheduler = None if scheduler_name == "none" else torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=float(scheduler_values["eta_min"])
    )
    criterion = build_stage2_completion_loss(config["loss"])
    amp_enabled = bool(training["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler(
        "cuda", enabled=amp_enabled,
        init_scale=float(training.get("amp_initial_scale", 1024.0)),
    )
    dpr_mean = train_dataset.dpr_standardizer.mean.tolist()
    dpr_std = train_dataset.dpr_standardizer.std.tolist()

    start_epoch = global_step = bad_epochs = 0
    best_val_loss = math.inf
    early = training.get("early_stopping", {})
    if str(early.get("monitor", "val_loss")) != "val_loss":
        raise ValueError("R1 completion early stopping may only monitor val_loss")
    patience = int(early.get("patience", 15))
    min_delta = float(early.get("min_delta", 0.0))
    if patience <= 0 or min_delta < 0.0 or not math.isfinite(min_delta):
        raise ValueError("invalid R1 completion early-stopping settings")
    if args.resume is not None:
        checkpoint = load_checkpoint(
            args.resume, model, optimizer=optimizer, scheduler=scheduler,
            scaler=scaler, map_location=device,
        )
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        previous = checkpoint.get("metrics") or {}
        best_val_loss = float(previous.get("best_val_loss", math.inf))
        bad_epochs = int(previous.get("early_stopping_bad_epochs", 0))

    if is_main:
        (output_dir / "resolved_config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    max_batches = 2 if args.smoke_test else config["evaluation"].get("max_batches")
    completed = False
    try:
        for epoch in range(start_epoch, epochs):
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)
            train_result = train_stage2_completion_one_epoch(
                model, train_loader, optimizer, criterion, device,
                dpr_mean=dpr_mean, dpr_std=dpr_std, scaler=scaler,
                use_amp=amp_enabled, grad_clip_norm=float(training["grad_clip_norm"]),
                accumulation_steps=int(training["accumulation_steps"]),
                max_batches=max_batches,
            )
            val_result = evaluate_stage2_completion_one_epoch(
                model, val_loader, criterion, device,
                dpr_mean=dpr_mean, dpr_std=dpr_std, use_amp=amp_enabled,
                max_batches=max_batches,
            )
            if scheduler is not None and train_result.optimizer_steps > 0:
                scheduler.step()
            global_step += train_result.optimizer_steps
            improved = math.isfinite(val_result.loss) and val_result.loss < best_val_loss - min_delta
            if improved:
                best_val_loss = val_result.loss
                bad_epochs = 0
            else:
                bad_epochs += 1
            should_stop = bool(early.get("enabled", False)) and bad_epochs >= patience
            if is_main:
                record = _json_safe({
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "epoch": epoch, "global_step": global_step,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "train": train_result.to_dict(), "val": val_result.to_dict(),
                    "best_val_loss": best_val_loss,
                    "checkpoint_improved": improved,
                    "early_stopping_bad_epochs": bad_epochs,
                    "early_stopping_triggered": should_stop,
                })
                print(json.dumps(record, ensure_ascii=False), flush=True)
                with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                checkpoint_metrics = {
                    "best_val_loss": best_val_loss,
                    "early_stopping_bad_epochs": bad_epochs,
                    "train": train_result.to_dict(), "val": val_result.to_dict(),
                }
                save_checkpoint(
                    output_dir / "last.pt", model, epoch=epoch,
                    global_step=global_step, optimizer=optimizer,
                    scheduler=scheduler, scaler=scaler, config=config,
                    metrics=checkpoint_metrics,
                    extra={"stage2_format": checkpoint_format},
                )
                if improved:
                    save_checkpoint(
                        output_dir / "best.pt", model, epoch=epoch,
                        global_step=global_step, optimizer=optimizer,
                        scheduler=scheduler, scaler=scaler, config=config,
                        metrics=checkpoint_metrics,
                        extra={"stage2_format": checkpoint_format},
                    )
                if should_save_periodic_checkpoint(epoch, checkpoint_every):
                    save_checkpoint(
                        output_dir / f"epoch_{epoch:04d}.pt", model, epoch=epoch,
                        global_step=global_step, optimizer=optimizer,
                        scheduler=scheduler, scaler=scaler, config=config,
                        metrics=checkpoint_metrics,
                        extra={"stage2_format": checkpoint_format},
                    )
            if world_size > 1:
                distributed.barrier(
                    device_ids=[device.index] if device.type == "cuda" else None
                )
            if should_stop:
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
            failure.write_text(
                f"训练完成，但{display_name}自动分析失败：{type(error).__name__}: {error}\n",
                encoding="utf-8",
            )
            print(f"[{display_name}-postprocessing] ERROR: {error}", file=sys.stderr, flush=True)
            if bool(config.get("postprocessing", {}).get("fail_on_error", False)):
                raise


if __name__ == "__main__":
    main()
