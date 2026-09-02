#!/usr/bin/env python3
"""Train S3-C2-O: freeze Stage 1 and task-adapt a restricted Stage 2."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
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

from precipitation_inversion.data.patch_dataset import (  # noqa: E402
    stage1_patch_dataset_kwargs,
)
from precipitation_inversion.data.samplers import FileBlockBatchSampler  # noqa: E402
from precipitation_inversion.data.stage2_patch_dataset import (  # noqa: E402
    stage2_patch_dataset_kwargs,
)
from precipitation_inversion.data.stage2_samplers import (  # noqa: E402
    Stage2StratifiedBatchSampler,
)
from precipitation_inversion.data.stage3_patch_dataset import (  # noqa: E402
    Stage3C2PatchDataset,
)
from precipitation_inversion.losses.stage3_losses import (  # noqa: E402
    build_stage3_c2_loss,
    validate_c1_oracle_loss,
)
from precipitation_inversion.models.stage3_cascade import (  # noqa: E402
    STAGE3_C2_TRAINABLE_SCOPE,
    Stage3C2OracleCascade,
    assert_c2_freeze_contract,
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
    audit_stage3_c2_gradient_scale,
    evaluate_stage3_c2_one_epoch,
    train_stage3_c2_one_epoch,
)
from scripts.train_stage1_unet3d import (  # noqa: E402
    _initialize_distributed,
    _json_safe,
    _resolve_device,
    build_model as build_stage1_model,
    load_config,
    project_path,
)
from scripts.train_stage2_unet3d import build_model as build_stage2_model  # noqa: E402
from scripts.train_stage3_cascade import (  # noqa: E402
    _load_checkpoint_payload,
    _normalization_vectors,
    _postprocessing_environment,
    _source_contract,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "stage3_c2_o_freeze_s1_train_s2.yaml"
STAGE3_C2_CHECKPOINT_FORMAT = "stage3_c2_oracle_stage2_only_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--ddp-backend", choices=("auto", "nccl", "gloo"))
    parser.add_argument("--ddp-timeout-seconds", type=int)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use one audit batch and two train/validation batches for one epoch.",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Write the train-only gradient-scale audit and exit before training.",
    )
    parser.add_argument("--skip-postprocessing", action="store_true")
    return parser.parse_args()


def _validate_w1p25_source(stage2_config: Mapping[str, Any]) -> None:
    """Prevent C2 from silently changing its physical Stage-2 anchor."""

    reflectivity = stage2_config.get("loss", {}).get("reflectivity", {})
    edges = tuple(float(value) for value in reflectivity.get("intensity_bin_edges_dbz", ()))
    weights = tuple(float(value) for value in reflectivity.get("intensity_bin_weights", ()))
    if edges != (25.0, 35.0) or weights != (1.0, 1.1, 1.25):
        raise ValueError("C2-O source Stage 2 must use the exact W1.25 dBZ weights")
    channels = tuple(stage2_config.get("data", {}).get("input_channels", ()))
    expected = (
        "dbz_gr_sparse_standardized",
        "gr_value_mask",
        "gr_nearest_distance_scaled",
        "height_scaled",
    )
    if channels != expected or int(stage2_config["model"]["in_channels"]) != 4:
        raise ValueError("C2-O source must be the four-channel distance model")


def build_c2_components(
    config: Mapping[str, Any], device: torch.device
) -> tuple[Stage3C2OracleCascade, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load sealed T3D and W1.25 sources, then apply the exact freeze scope."""

    source = _source_contract(config)
    stage1_payload = _load_checkpoint_payload(Path(source["stage1_checkpoint"]))
    stage2_payload = _load_checkpoint_payload(Path(source["stage2_checkpoint"]))
    stage1_config = dict(stage1_payload["config"])
    stage2_config = dict(stage2_payload["config"])
    if int(stage1_config["model"]["in_channels"]) != 3:
        raise ValueError("C2-O source Stage 1 must use three input channels")
    if not bool(stage1_config.get("type_task", {}).get("enabled", False)):
        raise ValueError("C2-O expects the sealed T3D Stage-1 source")
    validate_c1_oracle_loss(stage1_config["loss"])
    _validate_w1p25_source(stage2_config)

    stage1_norm = project_path(stage1_config["data"]["normalization"])
    stage2_norm = project_path(stage2_config["data"]["normalization"])
    stage1_mean, stage1_std = _normalization_vectors(stage1_norm, "dbz_dpr")
    stage2_mean, stage2_std = _normalization_vectors(stage2_norm, "dbz_dpr")
    if stage1_mean.shape != stage2_mean.shape:
        raise ValueError("Stage-1/Stage-2 normalization height counts differ")

    stage1_model = build_stage1_model(stage1_config)
    stage2_model = build_stage2_model(stage2_config)
    stage1_model.load_state_dict(stage1_payload["model"], strict=True)
    stage2_model.load_state_dict(stage2_payload["model"], strict=True)
    scope = str(config["adaptation"]["trainable_scope"])
    model = Stage3C2OracleCascade(
        stage2_model,
        stage1_model,
        stage2_dbz_mean=stage2_mean,
        stage2_dbz_std=stage2_std,
        stage1_dbz_mean=stage1_mean,
        stage1_dbz_std=stage1_std,
        stage2_in_channels=int(stage2_config["model"]["in_channels"]),
        trainable_scope=scope,
    ).to(device)
    assert_c2_freeze_contract(model)
    source.update(
        {
            "stage1_source_epoch": int(stage1_payload.get("epoch", -1)),
            "stage2_source_epoch": int(stage2_payload.get("epoch", -1)),
            "stage1_normalization": str(stage1_norm),
            "stage2_normalization": str(stage2_norm),
            "trainable_scope": scope,
        }
    )
    return model, stage1_config, stage2_config, source


def build_datasets(
    config: Mapping[str, Any],
    stage1_config: Mapping[str, Any],
    stage2_config: Mapping[str, Any],
) -> tuple[Stage3C2PatchDataset, Stage3C2PatchDataset]:
    stage1_data = stage1_config["data"]
    stage2_data = stage2_config["data"]
    stage3_data = config["data"]
    common = {
        "stage1_normalization_stats": project_path(stage1_data["normalization"]),
        "stage2_normalization_stats": project_path(stage2_data["normalization"]),
        "stage2_input_channels": tuple(stage2_data["input_channels"]),
        "cache_size": int(stage3_data["cache_size"]),
        "stage1_options": stage1_patch_dataset_kwargs(
            stage1_data, stage1_config["loss"]
        ),
        "stage2_options": stage2_patch_dataset_kwargs(stage2_config["loss"]),
    }
    train = Stage3C2PatchDataset(
        stage1_index_metadata=project_path(stage1_data["train_index"]),
        stage2_index_metadata=project_path(stage2_data["train_index"]),
        **common,
    )
    validation = Stage3C2PatchDataset(
        stage1_index_metadata=project_path(stage1_data["val_index"]),
        stage2_index_metadata=project_path(stage2_data["val_index"]),
        **common,
    )
    expected = int(stage2_config["model"]["in_channels"]) + 2
    if len(train.feature_names) != expected:
        raise ValueError("packed Dataset channels differ from C2 cascade contract")
    if train.feature_names != validation.feature_names:
        raise ValueError("C2 train/validation packed channels differ")
    return train, validation


def build_loaders(
    train_dataset: Stage3C2PatchDataset,
    val_dataset: Stage3C2PatchDataset,
    config: Mapping[str, Any],
    stage2_config: Mapping[str, Any],
    *,
    seed: int,
) -> tuple[
    torch.utils.data.DataLoader,
    Stage2StratifiedBatchSampler,
    torch.utils.data.DataLoader,
    FileBlockBatchSampler,
]:
    data = config["data"]
    source_sampler = stage2_config["data"]["sampler"]
    batch_size = int(data["batch_size"])
    epoch_size = source_sampler.get("epoch_size")
    train_sampler = Stage2StratifiedBatchSampler(
        train_dataset,
        batch_size=batch_size,
        stratum_weights=source_sampler["stratum_weights"],
        fill_fraction_threshold=float(source_sampler["fill_fraction_threshold"]),
        epoch_size=None if epoch_size is None else int(epoch_size),
        seed=seed,
        shuffle=True,
        group_by_file=bool(source_sampler["group_by_file"]),
        drop_last=bool(source_sampler["drop_last"]),
        even_batches=True,
    )
    val_sampler = FileBlockBatchSampler(
        val_dataset,
        batch_size=batch_size,
        block_size=max(batch_size, batch_size * 16),
        shuffle=False,
        drop_last=False,
        seed=seed,
        even_batches=False,
    )
    workers = int(data["num_workers"])
    kwargs = {
        "num_workers": workers,
        "pin_memory": bool(data["pin_memory"]),
        "persistent_workers": bool(data["persistent_workers"]) and workers > 0,
    }
    return (
        torch.utils.data.DataLoader(
            train_dataset, batch_sampler=train_sampler, **kwargs
        ),
        train_sampler,
        torch.utils.data.DataLoader(val_dataset, batch_sampler=val_sampler, **kwargs),
        val_sampler,
    )


def _stage2_from_cascade(model: torch.nn.Module) -> torch.nn.Module:
    candidate = model.module if hasattr(model, "module") else model
    if not isinstance(candidate, Stage3C2OracleCascade):
        raise TypeError("unexpected Stage-3 C2 model type")
    return candidate.stage2_model


@dataclass
class BestC2Validation:
    rain_rmse: float = math.inf
    joint_loss: float = math.inf
    dbz_loss: float = math.inf
    support_loss: float = math.inf

    @classmethod
    def from_metrics(cls, values: Mapping[str, Any] | None) -> "BestC2Validation":
        source = values or {}
        return cls(
            rain_rmse=float(source.get("best_rain_rmse", math.inf)),
            joint_loss=float(source.get("best_joint_val_loss", math.inf)),
            dbz_loss=float(source.get("best_dbz_val_loss", math.inf)),
            support_loss=float(source.get("best_support_val_loss", math.inf)),
        )

    def update(
        self, *, rain_rmse: float, joint_loss: float, dbz_loss: float, support_loss: float
    ) -> dict[str, bool]:
        values = {
            "rain": rain_rmse,
            "joint": joint_loss,
            "dbz": dbz_loss,
            "support": support_loss,
        }
        attributes = {
            "rain": "rain_rmse",
            "joint": "joint_loss",
            "dbz": "dbz_loss",
            "support": "support_loss",
        }
        result: dict[str, bool] = {}
        for name, value in values.items():
            attribute = attributes[name]
            changed = math.isfinite(value) and value < float(getattr(self, attribute))
            result[name] = changed
            if changed:
                setattr(self, attribute, float(value))
        return result

    def to_dict(self) -> dict[str, float]:
        return {
            "best_rain_rmse": self.rain_rmse,
            "best_joint_val_loss": self.joint_loss,
            "best_dbz_val_loss": self.dbz_loss,
            "best_support_val_loss": self.support_loss,
        }


def _checkpoint_extra(
    config: Mapping[str, Any],
    sources: Mapping[str, Any],
    gradient_audit: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "stage3_format": STAGE3_C2_CHECKPOINT_FORMAT,
        "stage3_config": dict(config),
        "stage3_sources": dict(sources),
        "stage3_gradient_audit": dict(gradient_audit),
        "stage3_rain_weight": float(gradient_audit["selected_rain_weight"]),
        "stage3_trainable_component": "stage2",
        "stage3_trainable_scope": STAGE3_C2_TRAINABLE_SCOPE,
        "stage3_support_interface": "true_dpr_support_oracle",
    }


def run_postprocessing(
    output_dir: Path, config: Mapping[str, Any], *, skip: bool
) -> None:
    values = config.get("postprocessing", {})
    if skip or not bool(values.get("enabled", True)):
        print("[stage3-c2-postprocessing] skipped", flush=True)
        return
    stage2_validation = output_dir / "analysis" / "stage2_validation"
    threshold = stage2_validation / "support_threshold.json"
    cascade = output_dir / "analysis" / "full_validation_cascade"
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
            str(PROJECT_ROOT / "scripts" / "evaluate_stage2_unet3d.py"),
            str(output_dir / "best.pt"),
            "--split",
            "val",
            "--output-dir",
            str(stage2_validation),
            "--device",
            str(values.get("device", "auto")),
            "--select-threshold",
            "--threshold-output",
            str(threshold),
            "--save-orbits",
            "0",
            "--overwrite",
        ],
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "evaluate_stage3_cascade.py"),
            "--checkpoint",
            str(output_dir / "best.pt"),
            "--threshold-file",
            str(threshold),
            "--split",
            "val",
            "--output-dir",
            str(cascade),
            "--device",
            str(values.get("device", "auto")),
            "--save-orbits",
            str(int(values.get("save_orbits", 6))),
            "--visualize",
            "--overwrite",
        ],
    ]
    environment = _postprocessing_environment()
    for command in commands:
        print("[stage3-c2-postprocessing] running: " + " ".join(command), flush=True)
        subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if config.get("experiment", {}).get("method") != "S3-C2-O-S2TaskAware":
        raise ValueError("this trainer only implements S3-C2-O-S2TaskAware")
    training = config["training"]
    checkpoint_every = validate_checkpoint_every(training.get("checkpoint_every", 10))
    training["checkpoint_every"] = checkpoint_every
    if bool(training.get("early_stopping", {}).get("enabled", False)):
        raise ValueError("formal C2-O must complete its full cosine schedule")
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
    model, stage1_config, stage2_config, sources = build_c2_components(config, device)
    train_dataset, val_dataset = build_datasets(config, stage1_config, stage2_config)
    train_loader, train_sampler, val_loader, val_sampler = build_loaders(
        train_dataset, val_dataset, config, stage2_config, seed=seed
    )

    resume_payload: Mapping[str, Any] | None = None
    if args.resume is not None:
        resume_payload = _load_checkpoint_payload(project_path(args.resume))
        if resume_payload.get("stage3_format") != STAGE3_C2_CHECKPOINT_FORMAT:
            raise ValueError("--resume is not a C2-O Stage-3 checkpoint")
        gradient_audit = dict(resume_payload.get("stage3_gradient_audit") or {})
        if "selected_rain_weight" not in gradient_audit:
            raise ValueError("C2 resume checkpoint has no gradient-audit weight")
    else:
        audit_values = config["gradient_audit"]
        audit_criterion = build_stage3_c2_loss(
            stage2_config["loss"], stage1_config["loss"], rain_weight=1.0
        ).to(device)
        gradient_audit = audit_stage3_c2_gradient_scale(
            model,
            train_loader,
            audit_criterion,
            device,
            target_gradient_ratio=float(
                audit_values["target_weighted_rain_to_anchor_gradient_ratio"]
            ),
            min_rain_weight=float(audit_values["min_rain_weight"]),
            max_rain_weight=float(audit_values["max_rain_weight"]),
            valid_batches_per_rank=(
                1 if args.smoke_test else int(audit_values["valid_batches_per_rank"])
            ),
            max_candidate_batches_per_rank=(
                2
                if args.smoke_test
                else int(audit_values["max_candidate_batches_per_rank"])
            ),
        )
    rain_weight = float(gradient_audit["selected_rain_weight"])
    config["resolved_contract"] = {
        "packed_input_channels": train_dataset.feature_names,
        "packed_shape": ["B", len(train_dataset.feature_names), 64, 64, 60],
        "stage1_input_channels": [
            "predicted_dpr_dbz_standardized_stage1",
            "true_dpr_support_oracle",
            "height_scaled",
        ],
        "stage1_shape": ["B", 3, 64, 64, 60],
        "outputs": {
            "stage2_support_logits": ["B", 1, 64, 64, 60],
            "stage2_reflectivity": ["B", 1, 64, 64, 60],
            "rain_log1p": ["B", 1, 64, 64, 60],
        },
        "loss": "L_dbz_W1.25 + L_support + lambda_R*(I + 0.02G)",
        "selected_rain_weight": rain_weight,
        "gradient_audit": gradient_audit,
        "sources": sources,
    }
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "gradient_scale_audit.json").write_text(
            json.dumps(_json_safe(gradient_audit), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output_dir / "resolved_config.json").write_text(
            json.dumps(_json_safe(config), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "method": "S3-C2-O-S2TaskAware",
                    "world_size": world_size,
                    "backend": active_backend,
                    "device": str(device),
                    "train_patches": len(train_dataset),
                    "val_patches": len(val_dataset),
                    "train_strata": train_sampler.stratum_counts,
                    "train_quotas": train_sampler.epoch_quotas,
                    "selected_rain_weight": rain_weight,
                    "trainable_scope": STAGE3_C2_TRAINABLE_SCOPE,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    if args.audit_only:
        if distributed.is_available() and distributed.is_initialized():
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
    assert_c2_freeze_contract(model)
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
    criterion = build_stage3_c2_loss(
        stage2_config["loss"], stage1_config["loss"], rain_weight=rain_weight
    ).to(device)
    amp_enabled = bool(training["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
        init_scale=float(training.get("amp_initial_scale", 1024.0)),
    )
    dpr_mean = train_dataset.stage2_dataset.dpr_standardizer.mean.tolist()
    dpr_std = train_dataset.stage2_dataset.dpr_standardizer.std.tolist()
    support_threshold = float(config["evaluation"]["training_support_threshold"])
    thresholds = tuple(float(value) for value in stage1_config["loss"]["thresholds_mm_h"])

    start_epoch = global_step = 0
    best = BestC2Validation()
    if args.resume is not None:
        checkpoint = load_checkpoint(
            args.resume,
            _stage2_from_cascade(model),
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            map_location=device,
        )
        if checkpoint.get("stage3_sources", {}).get("stage1_sha256") != sources["stage1_sha256"]:
            raise ValueError("resume checkpoint uses a different frozen Stage 1")
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        best = BestC2Validation.from_metrics(checkpoint.get("metrics"))

    max_batches = 2 if args.smoke_test else config["evaluation"].get("max_batches")
    extra = _checkpoint_extra(config, sources, gradient_audit)
    completed = False
    try:
        for epoch in range(start_epoch, epochs):
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)
            train_result = train_stage3_c2_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                dpr_mean=dpr_mean,
                dpr_std=dpr_std,
                support_threshold=support_threshold,
                thresholds_mm_h=thresholds,
                scaler=scaler,
                use_amp=amp_enabled,
                grad_clip_norm=float(training["grad_clip_norm"]),
                accumulation_steps=int(training["accumulation_steps"]),
                max_batches=max_batches,
            )
            val_result = evaluate_stage3_c2_one_epoch(
                model,
                val_loader,
                criterion,
                device,
                dpr_mean=dpr_mean,
                dpr_std=dpr_std,
                support_threshold=support_threshold,
                thresholds_mm_h=thresholds,
                use_amp=amp_enabled,
                max_batches=max_batches,
            )
            if train_result.optimizer_steps:
                scheduler.step()
            global_step += train_result.optimizer_steps
            improved = best.update(
                rain_rmse=float(val_result.metrics["rain"]["all"]["rmse"]),
                joint_loss=float(val_result.loss),
                dbz_loss=float(
                    val_result.loss_components["stage2_anchor"][
                        "reflectivity_standardized_dbz"
                    ]
                ),
                support_loss=float(
                    val_result.loss_components["stage2_anchor"]["support"]
                ),
            )
            record = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "epoch": epoch,
                "global_step": global_step,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train": train_result.to_dict(),
                "val": val_result.to_dict(),
                **best.to_dict(),
                "checkpoint_improved": improved,
                "early_stopping_bad_epochs": 0,
                "early_stopping_triggered": False,
            }
            if is_main:
                safe = _json_safe(record)
                print(json.dumps(safe, ensure_ascii=False), flush=True)
                with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(safe, ensure_ascii=False) + "\n")
                checkpoint_metrics = {
                    **best.to_dict(),
                    "train": train_result.to_dict(),
                    "val": val_result.to_dict(),
                }
                save_targets = [("last.pt", True), ("best.pt", improved["rain"])]
                save_targets.extend(
                    (
                        ("best_rain.pt", improved["rain"]),
                        ("best_joint.pt", improved["joint"]),
                        ("best_dbz.pt", improved["dbz"]),
                        ("best_support.pt", improved["support"]),
                    )
                )
                if should_save_periodic_checkpoint(epoch, checkpoint_every):
                    save_targets.append((f"epoch_{epoch:04d}.pt", True))
                for name, selected in save_targets:
                    if selected:
                        # Only Stage-2 weights are stored. The frozen Stage-1
                        # path/hash and selected lambda_R remain in metadata.
                        save_checkpoint(
                            output_dir / name,
                            _stage2_from_cascade(model),
                            epoch=epoch,
                            global_step=global_step,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            scaler=scaler,
                            config=stage2_config,
                            metrics=checkpoint_metrics,
                            extra=extra,
                        )
            if world_size > 1:
                distributed.barrier(
                    device_ids=[device.index] if device.type == "cuda" else None
                )
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
            print(f"[stage3-c2-postprocessing] ERROR: {error}", file=sys.stderr, flush=True)
            if bool(config.get("postprocessing", {}).get("fail_on_error", False)):
                raise


if __name__ == "__main__":
    main()
