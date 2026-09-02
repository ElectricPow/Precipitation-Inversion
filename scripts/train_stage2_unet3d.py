#!/usr/bin/env python3
"""Train the Stage-2 sparse-GR to dense-DPR dual-head 3D U-Net."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.distributed as distributed


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from precipitation_inversion.data.samplers import FileBlockBatchSampler  # noqa: E402
from precipitation_inversion.data.stage2_patch_dataset import (  # noqa: E402
    Stage2PatchDataset,
    stage2_patch_dataset_kwargs,
    validate_stage2_input_channels,
)
from precipitation_inversion.data.stage2_samplers import (  # noqa: E402
    Stage2StratifiedBatchSampler,
)
from precipitation_inversion.losses.stage2_losses import build_stage2_loss  # noqa: E402
from precipitation_inversion.models.stage2_unet3d import Stage2UNet3D  # noqa: E402
from precipitation_inversion.training.engine import (  # noqa: E402
    load_checkpoint,
    save_checkpoint,
    seed_everything,
    should_save_periodic_checkpoint,
    validate_checkpoint_every,
    validate_training_output_directory,
)
from precipitation_inversion.training.stage2_engine import (  # noqa: E402
    evaluate_stage2_one_epoch,
    train_stage2_one_epoch,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "stage2_unet3d.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument(
        "--resume",
        type=Path,
        help="Restore model, optimizer, scheduler, scaler, epoch and early-stop state.",
    )
    initialization.add_argument(
        "--initialize-from",
        type=Path,
        help=(
            "Load model weights only, then start a fresh optimizer/scheduler at "
            "fine-tuning epoch 0. This is intentionally different from --resume."
        ),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--ddp-backend", choices=("auto", "nccl", "gloo"))
    parser.add_argument("--ddp-timeout-seconds", type=int)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one epoch with two train and two validation batches.",
    )
    parser.add_argument("--skip-postprocessing", action="store_true")
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


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


@dataclass
class BestValidationLosses:
    """Track independent Stage-2 validation optima.

    ``joint`` drives early stopping and remains the backward-compatible
    ``best.pt`` criterion. ``support`` and ``reflectivity`` only select their
    own diagnostic checkpoints; neither silently changes the early-stop clock.
    """

    joint: float = math.inf
    support: float = math.inf
    reflectivity: float = math.inf

    @classmethod
    def from_metrics(cls, metrics: Mapping[str, Any] | None) -> "BestValidationLosses":
        values = metrics or {}
        return cls(
            joint=float(
                values.get("best_joint_val_loss", values.get("best_val_loss", math.inf))
            ),
            support=float(values.get("best_support_val_loss", math.inf)),
            reflectivity=float(values.get("best_reflectivity_val_loss", math.inf)),
        )

    def update(
        self,
        *,
        joint: float,
        support: float,
        reflectivity: float,
        joint_min_delta: float,
        task_min_delta: float,
    ) -> dict[str, bool]:
        candidates = {
            "joint": (float(joint), float(joint_min_delta)),
            "support": (float(support), float(task_min_delta)),
            "reflectivity": (float(reflectivity), float(task_min_delta)),
        }
        improved: dict[str, bool] = {}
        for name, (value, delta) in candidates.items():
            previous = float(getattr(self, name))
            changed = math.isfinite(value) and value < previous - delta
            improved[name] = changed
            if changed:
                setattr(self, name, value)
        return improved

    def to_dict(self) -> dict[str, float]:
        return {
            "best_val_loss": self.joint,
            "best_joint_val_loss": self.joint,
            "best_support_val_loss": self.support,
            "best_reflectivity_val_loss": self.reflectivity,
        }


def initialize_model_weights(
    checkpoint_path: str | Path,
    model: torch.nn.Module,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load only model weights and return auditable source metadata.

    No optimizer, scheduler or AMP scaler object is passed to
    :func:`load_checkpoint`, so their source states cannot leak into a fresh
    low-learning-rate fine-tuning run.
    """

    source = project_path(checkpoint_path)
    checkpoint = load_checkpoint(source, model, map_location=map_location)
    return {
        "mode": "weights_only",
        "checkpoint": str(source),
        "source_epoch": int(checkpoint.get("epoch", -1)),
        "source_global_step": int(checkpoint.get("global_step", 0)),
    }


def _resolve_device(requested: str, local_rank: int, world_size: int) -> torch.device:
    if requested == "auto":
        requested = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    elif requested == "cuda":
        requested = f"cuda:{local_rank}"
    elif world_size > 1 and requested.startswith("cuda:"):
        raise ValueError(
            "multi-process training requires --device auto or cuda; select physical "
            "GPUs through CUDA_VISIBLE_DEVICES"
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
        raise ValueError("distributed timeout must be positive")
    resolved = "nccl" if backend == "auto" and device.type == "cuda" else backend
    if resolved == "auto":
        resolved = "gloo"
    if resolved == "nccl" and device.type != "cuda":
        raise ValueError("NCCL requires CUDA")
    distributed.init_process_group(
        backend=resolved,
        init_method="env://",
        timeout=timedelta(seconds=timeout_seconds),
        device_id=device if device.type == "cuda" else None,
    )
    return resolved


def build_model(config: Mapping[str, Any]) -> Stage2UNet3D:
    values = config["model"]
    return Stage2UNet3D(
        in_channels=int(values["in_channels"]),
        base_channels=int(values["base_channels"]),
        channel_multipliers=tuple(int(item) for item in values["channel_multipliers"]),
        max_groups=int(values["max_groups"]),
        bottleneck_dropout=float(values["bottleneck_dropout"]),
        support_prior_probability=(
            None
            if values.get("support_prior_probability") is None
            else float(values["support_prior_probability"])
        ),
    )


def build_loaders(
    train_dataset: Stage2PatchDataset,
    val_dataset: Stage2PatchDataset,
    config: Mapping[str, Any],
    *,
    seed: int,
) -> tuple[
    torch.utils.data.DataLoader,
    Stage2StratifiedBatchSampler,
    torch.utils.data.DataLoader,
    FileBlockBatchSampler,
]:
    data = config["data"]
    sampler_values = data["sampler"]
    batch_size = int(data["batch_size"])
    epoch_size = sampler_values.get("epoch_size")
    train_sampler = Stage2StratifiedBatchSampler(
        train_dataset,
        batch_size=batch_size,
        stratum_weights=sampler_values["stratum_weights"],
        fill_fraction_threshold=float(sampler_values["fill_fraction_threshold"]),
        epoch_size=None if epoch_size is None else int(epoch_size),
        seed=seed,
        shuffle=True,
        group_by_file=bool(sampler_values["group_by_file"]),
        drop_last=bool(sampler_values["drop_last"]),
        even_batches=True,
    )
    # Validation uses every patch once. Unequal DDP shards are safe because the
    # Stage-2 evaluation engine calls the already-synchronized unwrapped model.
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
    loader_kwargs = {
        "num_workers": workers,
        "pin_memory": bool(data["pin_memory"]),
        "persistent_workers": bool(data["persistent_workers"]) and workers > 0,
    }
    return (
        torch.utils.data.DataLoader(
            train_dataset, batch_sampler=train_sampler, **loader_kwargs
        ),
        train_sampler,
        torch.utils.data.DataLoader(
            val_dataset, batch_sampler=val_sampler, **loader_kwargs
        ),
        val_sampler,
    )


def _postprocessing_environment() -> dict[str, str]:
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


def _nested_metric(values: Mapping[str, Any], *keys: str) -> Any:
    current: Any = values
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def write_validation_candidate_comparison(
    destination: Path,
    candidates: Mapping[str, Path],
) -> dict[str, Any]:
    """Collect full-orbit val metrics for joint/support/dBZ checkpoints.

    Candidate evaluation directories are produced independently, including an
    independently selected validation threshold. The summary deliberately does
    not name a winner: model selection is deferred until the physical trade-off
    between topology and dBZ has been inspected, and no test labels are used.
    """

    rows: list[dict[str, Any]] = []
    for role, checkpoint in candidates.items():
        metrics_path = destination / role / "metrics.json"
        threshold_path = destination / role / "support_threshold.json"
        if not metrics_path.is_file() or not threshold_path.is_file():
            raise FileNotFoundError(
                f"missing full-validation result for {role}: {metrics_path}"
            )
        metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        threshold_payload = json.loads(threshold_path.read_text(encoding="utf-8"))
        metrics = metrics_payload.get("metrics", {})
        row = {
            "role": role,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_epoch": metrics_payload.get("checkpoint_epoch"),
            "support_threshold": threshold_payload.get("threshold"),
            "support_csi": _nested_metric(metrics, "support", "csi"),
            "support_precision": _nested_metric(metrics, "support", "precision"),
            "support_recall": _nested_metric(metrics, "support", "recall"),
            "support_f1": _nested_metric(metrics, "support", "f1"),
            "fss_r1": _nested_metric(metrics, "fss", "1", "fss"),
            "fss_r2": _nested_metric(metrics, "fss", "2", "fss"),
            "fss_r4": _nested_metric(metrics, "fss", "4", "fss"),
            "target_mae_dbz": _nested_metric(
                metrics, "reflectivity_on_target_support", "mae_dbz"
            ),
            "target_rmse_dbz": _nested_metric(
                metrics, "reflectivity_on_target_support", "rmse_dbz"
            ),
            "target_bias_dbz": _nested_metric(
                metrics, "reflectivity_on_target_support", "bias_dbz"
            ),
            "target_pearson_r": _nested_metric(
                metrics, "reflectivity_on_target_support", "pearson_r"
            ),
            "target_ccc": _nested_metric(
                metrics, "reflectivity_on_target_support", "ccc"
            ),
            "common_mae_dbz": _nested_metric(
                metrics, "reflectivity_on_common_support", "mae_dbz"
            ),
            "common_pearson_r": _nested_metric(
                metrics, "reflectivity_on_common_support", "pearson_r"
            ),
        }
        rows.append(_json_safe(row))

    destination.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "stage": 2,
        "split": "val",
        "selection_status": "pending_analysis",
        "selection_rule": (
            "compare support CSI/FSS and all-target dBZ metrics on complete val "
            "orbits; test is intentionally not used for candidate selection"
        ),
        "candidates": rows,
    }
    (destination / "comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    if rows:
        with (destination / "comparison.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return payload


def run_postprocessing(
    output_dir: Path,
    config: Mapping[str, Any],
    *,
    skip: bool = False,
) -> None:
    """Run leakage-safe full-orbit checkpoint analysis.

    Normal training evaluates ``best.pt`` on val/test as before. Controlled
    fine-tuning can set ``compare_task_checkpoints=true`` and
    ``evaluate_test=false``: all three task-specific checkpoints are then
    compared only on complete validation orbits before a human selects the
    next experiment.
    """

    values = config.get("postprocessing", {})
    if skip or not bool(values.get("enabled", True)):
        print("[stage2-postprocessing] skipped", flush=True)
        return
    analysis = output_dir / "analysis"
    compare_candidates = bool(values.get("compare_task_checkpoints", False))
    evaluate_test = bool(values.get("evaluate_test", True))
    visualize_test = bool(values.get("visualize_test", evaluate_test))
    if visualize_test and not evaluate_test:
        raise ValueError("postprocessing.visualize_test requires evaluate_test=true")

    if compare_candidates:
        candidates = {
            "joint": output_dir / "best_joint.pt",
            "support": output_dir / "best_support.pt",
            "reflectivity": output_dir / "best_dbz.pt",
        }
        missing = [str(path) for path in candidates.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "missing task-specific Stage-2 checkpoints: " + ", ".join(missing)
            )
        validation_root = analysis / "validation_candidates"
        commands: list[list[str]] = []
        for role, candidate in candidates.items():
            role_output = validation_root / role
            commands.append(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "evaluate_stage2_unet3d.py"),
                    str(candidate),
                    "--split",
                    "val",
                    "--select-threshold",
                    "--threshold-output",
                    str(role_output / "support_threshold.json"),
                    "--output-dir",
                    str(role_output),
                    "--device",
                    str(values.get("device", "auto")),
                    "--overwrite",
                ]
            )
        checkpoint = candidates["joint"]
        threshold_file = validation_root / "joint" / "support_threshold.json"
    else:
        checkpoint = output_dir / "best.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"missing Stage-2 best checkpoint: {checkpoint}")
        threshold_file = analysis / "validation" / "support_threshold.json"
        commands = [
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "evaluate_stage2_unet3d.py"),
                str(checkpoint),
                "--split",
                "val",
                "--select-threshold",
                "--threshold-output",
                str(threshold_file),
                "--output-dir",
                str(analysis / "validation"),
                "--device",
                str(values.get("device", "auto")),
                "--overwrite",
            ]
        ]

    if evaluate_test:
        commands.append(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "evaluate_stage2_unet3d.py"),
                str(checkpoint),
                "--split",
                "test",
                "--threshold-file",
                str(threshold_file),
                "--output-dir",
                str(analysis / "test"),
                "--device",
                str(values.get("device", "auto")),
                "--overwrite",
            ]
        )
    if visualize_test:
        commands.append(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "visualize_stage2_test_predictions.py"),
                str(checkpoint),
                "--threshold-file",
                str(threshold_file),
                "--output-dir",
                str(analysis / "test_predictions"),
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
            ]
        )
    environment = _postprocessing_environment()
    for command in commands:
        print(f"[stage2-postprocessing] running: {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)
    if compare_candidates:
        write_validation_candidate_comparison(validation_root, candidates)
    analysis.mkdir(parents=True, exist_ok=True)
    readme_lines = ["# Stage-2训练后自动分析", ""]
    if compare_candidates:
        readme_lines.extend(
            [
                "- [joint/support/dBZ三个checkpoint的完整val比较](validation_candidates/comparison.csv)",
                "- 各候选目录分别保存自己的val阈值、整轨指标和分区指标。",
                "- 当前实验不使用test选择checkpoint，需先人工分析val结果。",
            ]
        )
    else:
        readme_lines.append(
            "- [验证集支持域阈值与完整整轨指标](validation/metrics.json)"
        )
    if evaluate_test:
        readme_lines.append("- [固定val阈值的测试集整轨指标](test/metrics.json)")
    if visualize_test:
        readme_lines.append(
            "- [固定测试轨道支持域、dBZ、剖面、分布和相关性图](test_predictions/summary.md)"
        )
    (analysis / "README.md").write_text(
        "\n".join(readme_lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    training = config["training"]
    # Immutable epoch snapshots default to a disk-safe ten-epoch cadence.
    # The normalized value is written into resolved_config.json for auditing.
    checkpoint_every = validate_checkpoint_every(
        training.get("checkpoint_every", 10)
    )
    training["checkpoint_every"] = checkpoint_every
    output_dir = project_path(
        args.output_dir
        if args.output_dir is not None
        else config["experiment"]["output_dir"]
    )
    output_dir = validate_training_output_directory(
        output_dir,
        resume=args.resume,
        initialize_from=args.initialize_from,
    )
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size <= 0 or not 0 <= rank < world_size or local_rank < 0:
        raise ValueError("invalid torchrun rank environment")
    device = _resolve_device(args.device, local_rank, world_size)
    distributed_values = config.get("runtime", {}).get("distributed", {})
    backend = str(args.ddp_backend or distributed_values.get("backend", "auto")).lower()
    timeout = int(
        args.ddp_timeout_seconds
        if args.ddp_timeout_seconds is not None
        else distributed_values.get("timeout_seconds", 300)
    )
    active_backend = _initialize_distributed(
        device, world_size, backend=backend, timeout_seconds=timeout
    )
    is_main = rank == 0

    experiment = config["experiment"]
    data = config["data"]
    evaluation = config["evaluation"]
    if args.smoke_test:
        data["num_workers"] = 0
        data["persistent_workers"] = False
        data["pin_memory"] = device.type == "cuda"
    seed = int(experiment["seed"])
    seed_everything(seed + rank, deterministic=bool(config["runtime"]["deterministic"]))
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(
            json.dumps(
                {
                    "stage": 2,
                    "distributed": world_size > 1,
                    "world_size": world_size,
                    "backend": active_backend,
                    "device": str(device),
                    "device_name": (
                        torch.cuda.get_device_name(device)
                        if device.type == "cuda"
                        else "CPU"
                    ),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    input_channels = validate_stage2_input_channels(data.get("input_channels"))
    # E3's physical-dBZ weights are supervision-only Dataset outputs. The same
    # strictly parsed options must be used for train and validation so the
    # monitored weighted objective matches the optimizer objective.
    supervision_options = stage2_patch_dataset_kwargs(config["loss"])
    train_dataset = Stage2PatchDataset(
        project_path(data["train_index"]),
        project_path(data["normalization"]),
        cache_size=int(data["cache_size"]),
        input_channels=input_channels,
        **supervision_options,
    )
    val_dataset = Stage2PatchDataset(
        project_path(data["val_index"]),
        project_path(data["normalization"]),
        cache_size=int(data["cache_size"]),
        input_channels=input_channels,
        **supervision_options,
    )
    expected_channels = int(config["model"]["in_channels"])
    if len(train_dataset.feature_names) != expected_channels:
        raise ValueError("model.in_channels differs from Stage-2 Dataset channels")
    if train_dataset.feature_names != val_dataset.feature_names:
        raise ValueError("train and validation Stage-2 channels differ")
    # Persist the resolved order even when an older four-channel configuration
    # omitted data.input_channels. Evaluation then reconstructs the exact same
    # tensor contract from the checkpoint alone.
    config["data"]["input_channels"] = list(train_dataset.feature_names)
    train_loader, train_sampler, val_loader, val_sampler = build_loaders(
        train_dataset, val_dataset, config, seed=seed
    )
    if is_main:
        config["data"]["resolved_train_stratum_counts"] = train_sampler.stratum_counts
        config["data"]["resolved_train_epoch_quotas"] = train_sampler.epoch_quotas
        print(
            "[stage2-sampler] "
            + json.dumps(
                {
                    "counts": train_sampler.stratum_counts,
                    "quotas": train_sampler.epoch_quotas,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    model: torch.nn.Module = build_model(config).to(device)
    initialization_metadata: dict[str, Any] | None = None
    if args.initialize_from is not None:
        # Every DDP rank loads the same source weights before wrapping. The
        # optimizer is constructed only afterwards, guaranteeing an empty
        # AdamW state and the learning rate from the fine-tuning config.
        initialization_metadata = initialize_model_weights(
            args.initialize_from, model, map_location="cpu"
        )
        config["experiment"]["initialization"] = initialization_metadata
        if is_main:
            print(
                "[stage2-initialize] "
                + json.dumps(initialization_metadata, ensure_ascii=False),
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
    scheduler = (
        None
        if scheduler_name == "none"
        else torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=float(scheduler_values["eta_min"])
        )
    )
    criterion = build_stage2_loss(config["loss"])
    amp_enabled = bool(training["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
        init_scale=float(training.get("amp_initial_scale", 1024.0)),
    )
    dpr_mean = train_dataset.dpr_standardizer.mean.tolist()
    dpr_std = train_dataset.dpr_standardizer.std.tolist()
    support_threshold = float(evaluation.get("training_support_threshold", 0.5))
    # Patch-edge neighborhoods are not exact full-orbit FSS and are expensive
    # every epoch. Keep them disabled by default; postprocessing computes the
    # configured 1/2/4-radius FSS after complete-orbit reconstruction.
    training_fss_radii = tuple(
        int(value) for value in evaluation.get("training_fss_radii", ())
    )

    start_epoch = global_step = 0
    best_losses = BestValidationLosses()
    bad_epochs = 0
    early = training.get("early_stopping", {})
    if str(early.get("monitor", "val_loss")).lower() != "val_loss":
        raise ValueError("Stage-2 early stopping currently monitors val_loss only")
    patience = int(early.get("patience", 15))
    min_delta = float(early.get("min_delta", 0.0))
    task_checkpoint_min_delta = float(
        training.get("task_checkpoint_min_delta", 0.0)
    )
    if patience <= 0 or min_delta < 0.0 or not math.isfinite(min_delta):
        raise ValueError("invalid Stage-2 early-stopping settings")
    if task_checkpoint_min_delta < 0.0 or not math.isfinite(
        task_checkpoint_min_delta
    ):
        raise ValueError("training.task_checkpoint_min_delta must be finite and >=0")
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
        best_losses = BestValidationLosses.from_metrics(previous)
        bad_epochs = int(previous.get("early_stopping_bad_epochs", 0))
        source_config = checkpoint.get("config") or {}
        source_experiment = source_config.get("experiment", {})
        if "initialization" in source_experiment:
            initialization_metadata = dict(source_experiment["initialization"])
            config["experiment"]["initialization"] = initialization_metadata

    if is_main:
        (output_dir / "resolved_config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    max_batches = 2 if args.smoke_test else evaluation.get("max_batches")
    training_completed = False
    try:
        for epoch in range(start_epoch, epochs):
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)
            train_result = train_stage2_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                dpr_mean=dpr_mean,
                dpr_std=dpr_std,
                support_threshold=support_threshold,
                fss_radii=(),
                scaler=scaler,
                use_amp=amp_enabled,
                grad_clip_norm=float(training["grad_clip_norm"]),
                accumulation_steps=int(training["accumulation_steps"]),
                max_batches=max_batches,
            )
            val_result = evaluate_stage2_one_epoch(
                model,
                val_loader,
                criterion,
                device,
                dpr_mean=dpr_mean,
                dpr_std=dpr_std,
                support_threshold=support_threshold,
                fss_radii=training_fss_radii,
                use_amp=amp_enabled,
                max_batches=max_batches,
            )
            if scheduler is not None and train_result.optimizer_steps > 0:
                scheduler.step()
            global_step += train_result.optimizer_steps
            support_val_loss = float(val_result.loss_components["support"])
            reflectivity_val_loss = float(
                val_result.loss_components["reflectivity_standardized_dbz"]
            )
            improved = best_losses.update(
                joint=val_result.loss,
                support=support_val_loss,
                reflectivity=reflectivity_val_loss,
                joint_min_delta=min_delta,
                task_min_delta=task_checkpoint_min_delta,
            )
            if improved["joint"]:
                bad_epochs = 0
            else:
                bad_epochs += 1
            should_stop = bool(early.get("enabled", True)) and bad_epochs >= patience
            if is_main:
                record = _json_safe(
                    {
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "epoch": epoch,
                        "global_step": global_step,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "train": train_result.to_dict(),
                        "val": val_result.to_dict(),
                        **best_losses.to_dict(),
                        "checkpoint_improved": improved,
                        "early_stopping_bad_epochs": bad_epochs,
                        "early_stopping_triggered": should_stop,
                    }
                )
                print(json.dumps(record, ensure_ascii=False), flush=True)
                with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                checkpoint_metrics = {
                    **best_losses.to_dict(),
                    "early_stopping_bad_epochs": bad_epochs,
                    "initialization": initialization_metadata,
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
                if improved["joint"]:
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
                    save_checkpoint(
                        output_dir / "best_joint.pt",
                        model,
                        epoch=epoch,
                        global_step=global_step,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        config=config,
                        metrics=checkpoint_metrics,
                    )
                if improved["support"]:
                    save_checkpoint(
                        output_dir / "best_support.pt",
                        model,
                        epoch=epoch,
                        global_step=global_step,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        config=config,
                        metrics=checkpoint_metrics,
                    )
                if improved["reflectivity"]:
                    save_checkpoint(
                        output_dir / "best_dbz.pt",
                        model,
                        epoch=epoch,
                        global_step=global_step,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        config=config,
                        metrics=checkpoint_metrics,
                    )
                if should_save_periodic_checkpoint(epoch, checkpoint_every):
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
                        "[stage2-early-stopping] "
                        f"epoch={epoch}, best_val_loss={best_losses.joint:.6f}",
                        flush=True,
                    )
                break
        training_completed = True
    finally:
        if distributed.is_available() and distributed.is_initialized():
            distributed.destroy_process_group()

    if training_completed and is_main:
        del model, optimizer, scheduler, scaler
        if device.type == "cuda":
            torch.cuda.empty_cache()
        try:
            run_postprocessing(output_dir, config, skip=args.skip_postprocessing)
        except Exception as error:
            print(f"[stage2-postprocessing] ERROR: {error}", file=sys.stderr, flush=True)
            failure = output_dir / "analysis" / "POSTPROCESSING_FAILED.txt"
            failure.parent.mkdir(parents=True, exist_ok=True)
            failure.write_text(
                f"训练完成，但Stage 2自动分析失败：{type(error).__name__}: {error}\n",
                encoding="utf-8",
            )
            if bool(config.get("postprocessing", {}).get("fail_on_error", False)):
                raise


if __name__ == "__main__":
    main()
