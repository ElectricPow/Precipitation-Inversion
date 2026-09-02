#!/usr/bin/env python3
"""Train S3-D0 DirectMultiHead from the sealed Stage-2 W1.25 source."""

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

import torch
import torch.distributed as distributed


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for value in (PROJECT_ROOT, SRC_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from precipitation_inversion.data.patch_dataset import stage1_patch_dataset_kwargs  # noqa: E402
from precipitation_inversion.data.dataset import sha256_file  # noqa: E402
from precipitation_inversion.data.stage2_patch_dataset import stage2_patch_dataset_kwargs  # noqa: E402
from precipitation_inversion.data.stage3_patch_dataset import Stage3D0PatchDataset  # noqa: E402
from precipitation_inversion.losses.stage3_losses import build_stage3_d0_loss, validate_c1_oracle_loss  # noqa: E402
from precipitation_inversion.models.stage3_direct import (  # noqa: E402
    STAGE3_D0_DECODER_AND_HEADS,
    STAGE3_D0_RAIN_HEAD_ONLY,
    Stage3DirectMultiHeadUNet3D,
    assert_d0_trainable_contract,
    build_stage3_d0_model,
    load_stage2_source_into_d0,
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
    audit_stage3_d0_gradient_scale,
    evaluate_stage3_d0_one_epoch,
    train_stage3_d0_one_epoch,
)
from scripts.train_stage1_unet3d import (  # noqa: E402
    _initialize_distributed,
    _json_safe,
    _resolve_device,
    load_config,
    project_path,
)
from scripts.train_stage3_c2 import (  # noqa: E402
    BestC2Validation,
    _validate_w1p25_source,
    build_loaders,
)
from scripts.train_stage3_cascade import (  # noqa: E402
    _load_checkpoint_payload,
    _postprocessing_environment,
    _source_contract,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "stage3_d0_h_frozen_feature_probe.yaml"
STAGE3_D0_CHECKPOINT_FORMAT = "stage3_d0_direct_multihead_v1"
STAGE3_D0_ANCHOR_PRIMARY = "anchor_primary"
STAGE3_D0_RAIN_PRIMARY = "rain_primary"
STAGE3_D0_OBJECTIVE_MODES = frozenset(
    (STAGE3_D0_ANCHOR_PRIMARY, STAGE3_D0_RAIN_PRIMARY)
)


def _objective_mode(config: Mapping[str, Any]) -> str:
    """Resolve the loss hierarchy while preserving legacy D0 configs."""

    value = str(
        config.get("adaptation", {}).get("objective_mode", STAGE3_D0_ANCHOR_PRIMARY)
    ).strip().lower()
    if value not in STAGE3_D0_OBJECTIVE_MODES:
        raise ValueError("D0 objective_mode must be anchor_primary or rain_primary")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--ddp-backend", choices=("auto", "nccl", "gloo"))
    parser.add_argument("--ddp-timeout-seconds", type=int)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--skip-postprocessing", action="store_true")
    return parser.parse_args()


def build_d0_components(
    config: Mapping[str, Any], device: torch.device
) -> tuple[Stage3DirectMultiHeadUNet3D, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load Stage-2 or a hash-matched D0-H state under the requested scope."""

    source = _source_contract(config)
    stage1_payload = _load_checkpoint_payload(Path(source["stage1_checkpoint"]))
    stage2_payload = _load_checkpoint_payload(Path(source["stage2_checkpoint"]))
    stage1_config = dict(stage1_payload["config"])
    stage2_config = dict(stage2_payload["config"])
    if int(stage1_config["model"]["in_channels"]) != 3:
        raise ValueError("D0 rain labels must follow the sealed three-channel Stage 1")
    if not bool(stage1_config.get("type_task", {}).get("enabled", False)):
        raise ValueError("D0 expects the sealed T3D Stage-1 rain contract")
    validate_c1_oracle_loss(stage1_config["loss"])
    _validate_w1p25_source(stage2_config)
    scope = str(config["adaptation"]["trainable_scope"])
    model = build_stage3_d0_model(stage2_config["model"], trainable_scope=scope)
    initialization_value = config.get("adaptation", {}).get("initialization_checkpoint")
    if initialization_value:
        # RainPrimary starts from the completed D0-H rain probe rather than
        # discarding its learned 1x1x1 head. The source Stage-1/Stage-2 hashes
        # must still match the explicit sources in the new configuration.
        initialization_path = project_path(initialization_value)
        initialization_payload = _load_checkpoint_payload(initialization_path)
        if initialization_payload.get("stage3_format") != STAGE3_D0_CHECKPOINT_FORMAT:
            raise ValueError("D0 initialization checkpoint has an unsupported format")
        if initialization_payload.get("stage3_trainable_scope") != STAGE3_D0_RAIN_HEAD_ONLY:
            raise ValueError("RainPrimary initialization must be a completed D0-H probe")
        initialization_sources = initialization_payload.get("stage3_sources", {})
        for key in ("stage1_sha256", "stage2_sha256"):
            if initialization_sources.get(key) != source[key]:
                raise ValueError(f"D0 initialization checkpoint has a different {key}")
        model.load_state_dict(initialization_payload["model"], strict=True)
        source.update(
            {
                "initialization_checkpoint": str(initialization_path),
                "initialization_sha256": sha256_file(initialization_path),
                "initialization_epoch": int(initialization_payload.get("epoch", -1)),
                "initialization_scope": str(
                    initialization_payload.get("stage3_trainable_scope")
                ),
                "initialization": "complete D0-H state including its learned rain head",
            }
        )
    else:
        load_stage2_source_into_d0(model, stage2_payload["model"])
        source["initialization"] = (
            "all shared/support/dbz weights from Stage2 W1.25; rain head new"
        )
    model.to(device)
    assert_d0_trainable_contract(model)
    source.update(
        {
            "stage1_source_epoch": int(stage1_payload.get("epoch", -1)),
            "stage2_source_epoch": int(stage2_payload.get("epoch", -1)),
            "trainable_scope": scope,
        }
    )
    return model, stage1_config, stage2_config, source


def build_datasets(
    config: Mapping[str, Any],
    stage1_config: Mapping[str, Any],
    stage2_config: Mapping[str, Any],
) -> tuple[Stage3D0PatchDataset, Stage3D0PatchDataset]:
    stage1_data, stage2_data, data = (
        stage1_config["data"],
        stage2_config["data"],
        config["data"],
    )
    common = {
        "stage1_normalization_stats": project_path(stage1_data["normalization"]),
        "stage2_normalization_stats": project_path(stage2_data["normalization"]),
        "stage2_input_channels": tuple(stage2_data["input_channels"]),
        "cache_size": int(data["cache_size"]),
        "stage1_options": stage1_patch_dataset_kwargs(stage1_data, stage1_config["loss"]),
        "stage2_options": stage2_patch_dataset_kwargs(stage2_config["loss"]),
    }
    train = Stage3D0PatchDataset(
        stage1_index_metadata=project_path(stage1_data["train_index"]),
        stage2_index_metadata=project_path(stage2_data["train_index"]),
        **common,
    )
    validation = Stage3D0PatchDataset(
        stage1_index_metadata=project_path(stage1_data["val_index"]),
        stage2_index_metadata=project_path(stage2_data["val_index"]),
        **common,
    )
    expected = int(stage2_config["model"]["in_channels"])
    if len(train.feature_names) != expected:
        raise ValueError("D0 Dataset must contain exactly the GR-only Stage-2 channels")
    if train.feature_names != validation.feature_names:
        raise ValueError("D0 train/validation feature channels differ")
    return train, validation


def _gradient_contract(
    model: Stage3DirectMultiHeadUNet3D,
    train_loader: Any,
    stage1_config: Mapping[str, Any],
    stage2_config: Mapping[str, Any],
    config: Mapping[str, Any],
    device: torch.device,
    *,
    smoke_test: bool,
) -> dict[str, Any]:
    scope = model.trainable_scope
    objective_mode = _objective_mode(config)
    values = config["gradient_audit"]
    if scope == STAGE3_D0_RAIN_HEAD_ONLY:
        if objective_mode != STAGE3_D0_ANCHOR_PRIMARY:
            raise ValueError("a frozen D0-H probe cannot use rain_primary balancing")
        weight = float(values.get("fixed_rain_weight", 1.0))
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError("D0-H fixed_rain_weight must be finite and positive")
        return {
            "selection_scope": "preregistered_fixed_frozen_feature_probe",
            "selected_rain_weight": weight,
            "selected_stage2_weight": 1.0,
            "objective_mode": objective_mode,
            "reason": "no shared trainable parameter exists between frozen physical heads and rain head",
        }
    if scope != STAGE3_D0_DECODER_AND_HEADS:
        raise ValueError("unknown D0 trainable scope")
    audit_criterion = build_stage3_d0_loss(
        stage2_config["loss"], stage1_config["loss"], rain_weight=1.0
    ).to(device)
    if objective_mode == STAGE3_D0_RAIN_PRIMARY:
        ratio = float(values["target_weighted_stage2_to_rain_gradient_ratio"])
        minimum = float(values["min_stage2_weight"])
        maximum = float(values["max_stage2_weight"])
        scaled_task = "stage2"
    else:
        ratio = float(values["target_weighted_rain_to_anchor_gradient_ratio"])
        minimum = float(values["min_rain_weight"])
        maximum = float(values["max_rain_weight"])
        scaled_task = "rain"
    return audit_stage3_d0_gradient_scale(
        model,
        train_loader,
        audit_criterion,
        device,
        target_gradient_ratio=ratio,
        min_rain_weight=minimum,
        max_rain_weight=maximum,
        valid_batches_per_rank=1 if smoke_test else int(values["valid_batches_per_rank"]),
        max_candidate_batches_per_rank=2 if smoke_test else int(values["max_candidate_batches_per_rank"]),
        scaled_task=scaled_task,
    )


def _checkpoint_extra(
    config: Mapping[str, Any], sources: Mapping[str, Any], audit: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "stage3_format": STAGE3_D0_CHECKPOINT_FORMAT,
        "stage3_config": dict(config),
        "stage3_sources": dict(sources),
        "stage3_gradient_audit": dict(audit),
        "stage3_rain_weight": float(audit["selected_rain_weight"]),
        "stage3_physical_weight": float(audit.get("selected_stage2_weight", 1.0)),
        "stage3_objective_mode": _objective_mode(config),
        "stage3_trainable_component": "direct_gr_multihead",
        "stage3_trainable_scope": str(config["adaptation"]["trainable_scope"]),
        "stage3_support_interface": "independent_predicted_support_head",
    }


def run_postprocessing(output_dir: Path, config: Mapping[str, Any], *, skip: bool) -> None:
    values = config.get("postprocessing", {})
    if skip or not bool(values.get("enabled", True)):
        print("[stage3-d0-postprocessing] skipped", flush=True)
        return
    analysis = output_dir / "analysis" / "full_validation_direct"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "evaluate_stage3_direct.py"),
        "--checkpoint", str(output_dir / "best.pt"),
        "--split", "val",
        "--output-dir", str(analysis),
        "--device", str(values.get("device", "auto")),
        "--save-orbits", str(int(values.get("save_orbits", 6))),
        "--select-threshold", "--visualize", "--overwrite",
    ]
    plot = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "plot_stage1_training_history.py"),
        str(output_dir), "--dpi", str(int(values.get("dpi", 150))),
    ]
    environment = _postprocessing_environment()
    for current in (plot, command):
        print("[stage3-d0-postprocessing] running: " + " ".join(current), flush=True)
        subprocess.run(current, cwd=PROJECT_ROOT, env=environment, check=True)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    method = str(config.get("experiment", {}).get("method"))
    if method not in (
        "S3-D0-H-DirectMultiHead",
        "S3-D0-D-DirectMultiHead",
        "S3-D0-D-RainPrimary",
    ):
        raise ValueError("this trainer only implements registered S3-D0 methods")
    training = config["training"]
    checkpoint_every = validate_checkpoint_every(training.get("checkpoint_every", 10))
    training["checkpoint_every"] = checkpoint_every
    if bool(training.get("early_stopping", {}).get("enabled", False)):
        raise ValueError("formal D0 experiments must complete their cosine schedule")
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
    timeout = int(args.ddp_timeout_seconds or distributed_values.get("timeout_seconds", 600))
    active_backend = _initialize_distributed(device, world_size, backend=backend, timeout_seconds=timeout)
    is_main = rank == 0
    if args.smoke_test:
        config["data"].update(num_workers=0, persistent_workers=False, pin_memory=device.type == "cuda")

    # Identical initialization on every rank matters because D0 adds a new
    # rain head before DDP broadcasts parameters and before D0-D audits them.
    seed = int(config["experiment"]["seed"])
    seed_everything(seed, deterministic=bool(config["runtime"]["deterministic"]))
    model, stage1_config, stage2_config, sources = build_d0_components(config, device)
    train_dataset, val_dataset = build_datasets(config, stage1_config, stage2_config)
    train_loader, train_sampler, val_loader, val_sampler = build_loaders(
        train_dataset, val_dataset, config, stage2_config, seed=seed
    )

    resume_payload: Mapping[str, Any] | None = None
    if args.resume is not None:
        resume_payload = _load_checkpoint_payload(project_path(args.resume))
        if resume_payload.get("stage3_format") != STAGE3_D0_CHECKPOINT_FORMAT:
            raise ValueError("--resume is not a D0 checkpoint")
        if resume_payload.get("stage3_trainable_scope") != model.trainable_scope:
            raise ValueError("D0 resume scope differs from the current configuration")
        if resume_payload.get("stage3_objective_mode", STAGE3_D0_ANCHOR_PRIMARY) != _objective_mode(config):
            raise ValueError("D0 resume objective hierarchy differs from the configuration")
        audit = dict(resume_payload.get("stage3_gradient_audit") or {})
    else:
        audit = _gradient_contract(
            model, train_loader, stage1_config, stage2_config, config, device,
            smoke_test=args.smoke_test,
        )
    if "selected_rain_weight" not in audit:
        raise ValueError("D0 gradient contract has no rain weight")
    rain_weight = float(audit["selected_rain_weight"])
    stage2_weight = float(audit.get("selected_stage2_weight", 1.0))
    config["resolved_contract"] = {
        "input_channels": train_dataset.feature_names,
        "input_shape": ["B", len(train_dataset.feature_names), 64, 64, 60],
        "satellite_variables_are_labels_only": ["dbz_dpr", "pre_dpr", "DPR support"],
        "outputs": {
            "support_logits": ["B", 1, 64, 64, 60],
            "reflectivity_standardized": ["B", 1, 64, 64, 60],
            "rain_log1p": ["B", 1, 64, 64, 60],
        },
        "loss": (
            "(I + 0.02G) + lambda_phys*(L_support + L_dbz_W1.25)"
            if _objective_mode(config) == STAGE3_D0_RAIN_PRIMARY
            else "L_support + L_dbz_W1.25 + lambda_R*(I + 0.02G)"
        ),
        "selected_rain_weight": rain_weight,
        "selected_stage2_weight": stage2_weight,
        "objective_mode": _objective_mode(config),
        "trainable_scope": model.trainable_scope,
        "gradient_audit": audit,
        "sources": sources,
    }
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "gradient_scale_audit.json").write_text(
            json.dumps(_json_safe(audit), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (output_dir / "resolved_config.json").write_text(
            json.dumps(_json_safe(config), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps({
            "method": method, "world_size": world_size, "backend": active_backend,
            "device": str(device), "train_patches": len(train_dataset),
            "val_patches": len(val_dataset), "train_strata": train_sampler.stratum_counts,
            "train_quotas": train_sampler.epoch_quotas,
            "selected_rain_weight": rain_weight,
            "selected_stage2_weight": stage2_weight,
            "objective_mode": _objective_mode(config),
            "trainable_scope": model.trainable_scope,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        }, ensure_ascii=False), flush=True)
    if args.audit_only:
        if distributed.is_initialized():
            distributed.destroy_process_group()
        return

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
    assert_d0_trainable_contract(model)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(config["optimizer"]["learning_rate"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )
    epochs = 1 if args.smoke_test else int(training["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=float(config["scheduler"]["eta_min"])
    )
    criterion = build_stage3_d0_loss(
        stage2_config["loss"], stage1_config["loss"],
        rain_weight=rain_weight, stage2_weight=stage2_weight,
    ).to(device)
    amp_enabled = bool(training["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler(
        "cuda", enabled=amp_enabled, init_scale=float(training.get("amp_initial_scale", 1024.0))
    )
    dpr_mean = train_dataset.stage2_dataset.dpr_standardizer.mean.tolist()
    dpr_std = train_dataset.stage2_dataset.dpr_standardizer.std.tolist()
    support_threshold = float(config["evaluation"]["training_support_threshold"])
    thresholds = tuple(float(v) for v in stage1_config["loss"]["thresholds_mm_h"])
    start_epoch = global_step = 0
    best = BestC2Validation()
    if resume_payload is not None:
        checkpoint = load_checkpoint(
            args.resume, model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            map_location=device,
        )
        if checkpoint.get("stage3_sources", {}).get("stage2_sha256") != sources["stage2_sha256"]:
            raise ValueError("resume checkpoint uses a different Stage-2 source")
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        best = BestC2Validation.from_metrics(checkpoint.get("metrics"))

    max_batches = 2 if args.smoke_test else config["evaluation"].get("max_batches")
    extra = _checkpoint_extra(config, sources, audit)
    completed = False
    try:
        for epoch in range(start_epoch, epochs):
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)
            common = dict(
                dpr_mean=dpr_mean, dpr_std=dpr_std,
                support_threshold=support_threshold, thresholds_mm_h=thresholds,
                use_amp=amp_enabled, max_batches=max_batches,
            )
            train_result = train_stage3_d0_one_epoch(
                model, train_loader, optimizer, criterion, device,
                scaler=scaler, grad_clip_norm=float(training["grad_clip_norm"]),
                accumulation_steps=int(training["accumulation_steps"]), **common,
            )
            val_result = evaluate_stage3_d0_one_epoch(
                model, val_loader, criterion, device, **common
            )
            if train_result.optimizer_steps:
                scheduler.step()
            global_step += train_result.optimizer_steps
            improved = best.update(
                rain_rmse=float(val_result.metrics["rain"]["all"]["rmse"]),
                joint_loss=float(val_result.loss),
                dbz_loss=float(val_result.loss_components["stage2_anchor"]["reflectivity_standardized_dbz"]),
                support_loss=float(val_result.loss_components["stage2_anchor"]["support"]),
            )
            record = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "epoch": epoch, "global_step": global_step,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train": train_result.to_dict(), "val": val_result.to_dict(),
                **best.to_dict(), "checkpoint_improved": improved,
                "early_stopping_bad_epochs": 0, "early_stopping_triggered": False,
            }
            if is_main:
                safe = _json_safe(record)
                print(json.dumps(safe, ensure_ascii=False), flush=True)
                with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(safe, ensure_ascii=False) + "\n")
                checkpoint_metrics = {
                    **best.to_dict(), "train": train_result.to_dict(), "val": val_result.to_dict()
                }
                targets = [
                    ("last.pt", True), ("best.pt", improved["rain"]),
                    ("best_rain.pt", improved["rain"]), ("best_joint.pt", improved["joint"]),
                    ("best_dbz.pt", improved["dbz"]), ("best_support.pt", improved["support"]),
                ]
                if should_save_periodic_checkpoint(epoch, checkpoint_every):
                    targets.append((f"epoch_{epoch:04d}.pt", True))
                for name, selected in targets:
                    if selected:
                        # Full D0 state is required: unlike C2, rain_head is new.
                        save_checkpoint(
                            output_dir / name, model, epoch=epoch, global_step=global_step,
                            optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                            config=stage2_config, metrics=checkpoint_metrics, extra=extra,
                        )
            if world_size > 1:
                distributed.barrier(device_ids=[device.index] if device.type == "cuda" else None)
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
            print(f"[stage3-d0-postprocessing] ERROR: {error}", file=sys.stderr, flush=True)
            if bool(config.get("postprocessing", {}).get("fail_on_error", False)):
                raise


if __name__ == "__main__":
    main()
