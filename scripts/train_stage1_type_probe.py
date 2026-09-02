#!/usr/bin/env python3
"""Train frozen-backbone Pool or Ordered-3D precipitation-type probes."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as distributed
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from precipitation_inversion.data.patch_dataset import (  # noqa: E402
    Stage1PatchDataset,
    stage1_patch_dataset_kwargs,
)
from precipitation_inversion.data.precipitation_type import (  # noqa: E402
    TYPE_NAMES,
    inverse_sqrt_class_weights,
)
from precipitation_inversion.data.samplers import FileBlockBatchSampler  # noqa: E402
from precipitation_inversion.losses.masked_classification import (  # noqa: E402
    MaskedCrossEntropyLoss,
)
from precipitation_inversion.metrics.classification import (  # noqa: E402
    MulticlassConfusionMetrics,
)
from precipitation_inversion.models.type_heads import build_type_head  # noqa: E402
from precipitation_inversion.models.unet3d import Stage1UNet3D  # noqa: E402
from precipitation_inversion.training.engine import (  # noqa: E402
    move_batch_to_device,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--head", choices=("pool", "ordered_3d"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def resolve_device(value: str, local_rank: int, world_size: int) -> torch.device:
    if value == "auto":
        value = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    elif value == "cuda":
        value = f"cuda:{local_rank}"
    elif world_size > 1 and value.startswith("cuda:"):
        raise ValueError("under torchrun use --device auto and CUDA_VISIBLE_DEVICES")
    device = torch.device(value)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        torch.cuda.set_device(device)
    return device


def initialize_distributed(device: torch.device, world_size: int) -> None:
    if world_size <= 1:
        return
    distributed.init_process_group(
        backend="nccl" if device.type == "cuda" else "gloo",
        init_method="env://",
        timeout=timedelta(seconds=600),
        device_id=device if device.type == "cuda" else None,
    )


def build_backbone(config: dict[str, Any]) -> Stage1UNet3D:
    values = config["model"]
    return Stage1UNet3D(
        in_channels=int(values["in_channels"]),
        out_channels=int(values["out_channels"]),
        base_channels=int(values["base_channels"]),
        channel_multipliers=tuple(values["channel_multipliers"]),
        max_groups=int(values["max_groups"]),
        bottleneck_dropout=float(values["bottleneck_dropout"]),
    )


class FrozenBackboneTypeProbe(nn.Module):
    """Freeze all U-Net weights and train only a profile classification head."""

    def __init__(self, backbone: Stage1UNet3D, head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True) -> "FrozenBackboneTypeProbe":
        super().train(mode)
        self.backbone.eval()  # frozen GroupNorm/dropout state is always fixed
        self.head.train(mode)
        return self

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.backbone.forward_features(inputs)
        return self.head(features)


def build_loader(
    dataset: Stage1PatchDataset,
    data_config: dict[str, Any],
    *,
    training: bool,
    seed: int,
) -> tuple[torch.utils.data.DataLoader, FileBlockBatchSampler]:
    sampler = FileBlockBatchSampler(
        dataset,
        batch_size=int(data_config["batch_size"]),
        block_size=int(data_config["block_size"]),
        shuffle=training,
        drop_last=training,
        seed=seed,
        even_batches=training,
    )
    workers = int(data_config["num_workers"])
    return (
        torch.utils.data.DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=workers,
            pin_memory=bool(data_config["pin_memory"]),
            persistent_workers=bool(data_config["persistent_workers"]) and workers > 0,
        ),
        sampler,
    )


def class_counts(
    dataset: Stage1PatchDataset,
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    max_samples: int | None = None,
) -> torch.Tensor:
    counts = torch.zeros(len(TYPE_NAMES), dtype=torch.int64)
    stop = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    for index in range(rank, stop, world_size):
        item = dataset[index]
        counts += torch.bincount(
            item["type_target"][item["type_loss_mask"]], minlength=len(TYPE_NAMES)
        )
    counts = counts.to(device)
    if world_size > 1:
        distributed.all_reduce(counts, op=distributed.ReduceOp.SUM)
    return counts


def reduce_pair(total: float, count: int, device: torch.device) -> tuple[float, int]:
    values = torch.tensor([total, float(count)], dtype=torch.float64, device=device)
    if distributed.is_initialized():
        distributed.all_reduce(values, op=distributed.ReduceOp.SUM)
    return float(values[0].item()), int(round(values[1].item()))


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: MaskedCrossEntropyLoss,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
    max_batches: int | None,
    height_permutation: torch.Tensor | None = None,
) -> tuple[float, dict[str, Any]]:
    training = optimizer is not None
    model.train(training)
    inference_model = model.module if hasattr(model, "module") and not training else model
    metrics = MulticlassConfusionMetrics(TYPE_NAMES)
    loss_sum = 0.0
    profile_count = 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch_index, source in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            batch = move_batch_to_device(source, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                if height_permutation is None:
                    logits = inference_model(batch["inputs"])
                else:
                    raw_model = inference_model.module if hasattr(inference_model, "module") else inference_model
                    with torch.no_grad():
                        features = raw_model.backbone.forward_features(batch["inputs"])
                    if not hasattr(raw_model.head, "height_levels"):
                        logits = raw_model.head(features)
                    else:
                        logits = raw_model.head(features, height_permutation=height_permutation)
                loss = criterion(logits, batch["type_target"], batch["type_loss_mask"])
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            current = int(batch["type_loss_mask"].sum().item())
            loss_sum += float(loss.detach().item()) * current
            profile_count += current
            metrics.update(logits.detach(), batch["type_target"], batch["type_loss_mask"])
    loss_sum, profile_count = reduce_pair(loss_sum, profile_count, device)
    return (
        loss_sum / profile_count if profile_count else float("nan"),
        metrics.compute(synchronize=True),
    )


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("invalid probe optimization parameters")
    requested_output = args.output_dir.expanduser().resolve()
    if (requested_output / "metrics.jsonl").exists():
        raise FileExistsError(
            f"probe history already exists in {requested_output}; choose a new output directory"
        )
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    device = resolve_device(args.device, local_rank, world_size)
    initialize_distributed(device, world_size)
    is_main = rank == 0
    seed_everything(args.seed + rank)
    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("checkpoint has no embedded configuration")
    data = dict(config["data"])
    if args.smoke_test:
        # Keep the diagnostic runnable in restricted CI/sandbox environments
        # where multiprocessing resource-sharing sockets are unavailable.
        data["num_workers"] = 0
        data["persistent_workers"] = False
        data["pin_memory"] = device.type == "cuda"
    loss = config["loss"]
    options = stage1_patch_dataset_kwargs(data, loss)
    train_dataset = Stage1PatchDataset(
        project_path(data["train_index"]),
        project_path(data["normalization"]),
        positive_only=bool(data["train_positive_only"]),
        cache_size=int(data["cache_size"]),
        **options,
    )
    val_dataset = Stage1PatchDataset(
        project_path(data["val_index"]),
        project_path(data["normalization"]),
        positive_only=bool(data["val_positive_only"]),
        cache_size=int(data["cache_size"]),
        **options,
    )
    train_loader, train_sampler = build_loader(train_dataset, data, training=True, seed=args.seed)
    val_loader, val_sampler = build_loader(val_dataset, data, training=False, seed=args.seed)
    counts = class_counts(
        train_dataset,
        rank=rank,
        world_size=world_size,
        device=device,
        max_samples=64 if args.smoke_test else None,
    )
    if args.smoke_test:
        # A tiny prefix need not contain every rare class. The smoke run checks
        # wiring only; production always uses exact complete-train counts.
        counts = counts.clamp_min(1)
    weights = inverse_sqrt_class_weights(counts.cpu().tolist()).tolist()
    criterion = MaskedCrossEntropyLoss(weights).to(device)

    backbone = build_backbone(config)
    backbone_state = {
        key: value for key, value in checkpoint["model"].items()
        if not key.startswith("type_head.")
    }
    missing, unexpected = backbone.load_state_dict(backbone_state, strict=False)
    if unexpected or any(not key.startswith("type_head.") for key in missing):
        raise ValueError(f"checkpoint is incompatible: missing={missing}, unexpected={unexpected}")
    head_config = {
        "height_levels": int(train_dataset.z.size),
        "num_classes": len(TYPE_NAMES),
        "compressed_channels": 8,
        "hidden_2d_channels": 32,
    }
    model: nn.Module = FrozenBackboneTypeProbe(
        backbone, build_type_head(args.head, backbone.channels[0], head_config)
    ).to(device)
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            output_device=device.index if device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=False,
            static_graph=True,
        )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    output_dir = requested_output
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    max_batches = 2 if args.smoke_test else None
    best_f1 = -math.inf
    epochs = 1 if args.smoke_test else args.epochs
    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        train_loss, train_metrics = run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer, max_batches=max_batches
        )
        val_loss, val_metrics = run_epoch(
            model, val_loader, criterion, device, optimizer=None, max_batches=max_batches
        )
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train": train_metrics,
            "val": val_metrics,
        }
        current_f1 = float(val_metrics["macro_f1"])
        improved = math.isfinite(current_f1) and current_f1 > best_f1
        if improved:
            best_f1 = current_f1
        if is_main:
            print(json.dumps(record, ensure_ascii=False), flush=True)
            with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            payload = {
                "format_version": 1,
                "epoch": epoch,
                "head_kind": args.head,
                "head_config": head_config,
                "parent_checkpoint": str(checkpoint_path),
                "class_names": list(TYPE_NAMES),
                "class_counts": counts.cpu().tolist(),
                "class_weights": weights,
                "model": (model.module if hasattr(model, "module") else model).state_dict(),
                "metrics": record,
            }
            torch.save(payload, output_dir / "last.pt")
            if improved:
                torch.save(payload, output_dir / "best.pt")
        if world_size > 1:
            distributed.barrier(device_ids=[device.index] if device.type == "cuda" else None)

    # Test whether the learned head actually uses ordered height information.
    best = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    (model.module if hasattr(model, "module") else model).load_state_dict(best["model"])
    normal_loss, normal = run_epoch(model, val_loader, criterion, device, optimizer=None, max_batches=max_batches)
    reverse = torch.arange(train_dataset.z.size - 1, -1, -1, device=device)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    random_order = torch.randperm(train_dataset.z.size, generator=generator).to(device)
    reverse_loss, reverse_metrics = run_epoch(
        model, val_loader, criterion, device, optimizer=None, max_batches=max_batches, height_permutation=reverse
    )
    random_loss, random_metrics = run_epoch(
        model, val_loader, criterion, device, optimizer=None, max_batches=max_batches, height_permutation=random_order
    )
    if is_main:
        diagnostic = {
            "normal": {"loss": normal_loss, **normal},
            "reverse_height": {"loss": reverse_loss, **reverse_metrics},
            "random_height": {"loss": random_loss, **random_metrics},
            "interpretation": "ordered head should degrade under height permutation; pool control should be invariant",
        }
        (output_dir / "height_shuffle_metrics.json").write_text(
            json.dumps(diagnostic, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if distributed.is_initialized():
        distributed.destroy_process_group()


if __name__ == "__main__":
    main()
