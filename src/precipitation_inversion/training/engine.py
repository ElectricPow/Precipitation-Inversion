"""Reusable AMP/DDP-aware training and validation loops.

Model inputs have shape ``(B,3,64,64,60)`` and outputs/targets/masks have
shape ``(B,1,64,64,60)``. The engine never flattens spatial dimensions before
the model; flattening occurs only inside masked loss/metric reductions.
"""

from __future__ import annotations

import os
import random
import tempfile
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import torch.distributed as distributed
from torch import nn

from precipitation_inversion.metrics.regression import (
    PrecipitationRegressionMetrics,
)


@dataclass(frozen=True)
class EpochResult:
    """Serializable summary of one train or validation pass."""

    loss: float
    batch_count: int
    optimizer_steps: int
    valid_voxels: int
    metrics: dict[str, Any]
    duration_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy, CPU, and every visible CUDA device."""

    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def move_batch_to_device(
    batch: Mapping[str, Any], device: torch.device, *, non_blocking: bool = True
) -> dict[str, Any]:
    """Move tensor fields while preserving strings and other metadata."""

    return {
        key: (
            value.to(device, non_blocking=non_blocking)
            if isinstance(value, torch.Tensor)
            else value
        )
        for key, value in batch.items()
    }


def _distributed_active() -> bool:
    return (
        distributed.is_available()
        and distributed.is_initialized()
        and distributed.get_world_size() > 1
    )


def _reduce_totals(values: list[float], device: torch.device) -> list[float]:
    if not _distributed_active():
        return values
    backend = str(distributed.get_backend()).lower()
    reduce_device = device if "nccl" in backend else torch.device("cpu")
    tensor = torch.tensor(values, dtype=torch.float64, device=reduce_device)
    distributed.all_reduce(tensor, op=distributed.ReduceOp.SUM)
    return tensor.cpu().tolist()


def _batch_limit(loader: Iterable[Any], max_batches: int | None) -> int | None:
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive")
    try:
        available = len(loader)  # type: ignore[arg-type]
    except TypeError:
        return max_batches
    return available if max_batches is None else min(available, max_batches)


def _autocast(device: torch.device, enabled: bool):
    return torch.autocast(
        device_type=device.type,
        enabled=enabled and device.type == "cuda",
    )


def train_one_epoch(
    model: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device | str,
    *,
    scaler: Any | None = None,
    use_amp: bool = True,
    grad_clip_norm: float | None = 1.0,
    accumulation_steps: int = 1,
    thresholds_mm_h: tuple[float, ...] = (1, 5, 10, 30),
    max_batches: int | None = None,
) -> EpochResult:
    """Train one epoch and report voxel-weighted loss and masked metrics."""

    if accumulation_steps <= 0:
        raise ValueError("accumulation_steps must be positive")
    if grad_clip_norm is not None and grad_clip_norm <= 0:
        raise ValueError("grad_clip_norm must be positive or None")
    resolved_device = torch.device(device)
    total_batches = _batch_limit(loader, max_batches)
    if accumulation_steps > 1 and total_batches is None:
        raise TypeError("gradient accumulation requires a loader with __len__")

    model.train()
    metrics = PrecipitationRegressionMetrics(thresholds_mm_h)
    loss_weighted_sum = 0.0
    valid_voxels = 0
    batch_count = 0
    optimizer_steps = 0
    group_size = accumulation_steps
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()

    for batch_index, source_batch in enumerate(loader):
        if total_batches is not None and batch_index >= total_batches:
            break
        if batch_index % accumulation_steps == 0 and total_batches is not None:
            group_size = min(accumulation_steps, total_batches - batch_index)
        batch = move_batch_to_device(source_batch, resolved_device)
        # inputs=(B,3,D,H,Z), target/loss_mask=(B,1,D,H,Z).
        inputs = batch["inputs"]
        target = batch["target"]
        mask = batch["loss_mask"]
        current_valid = int(mask.sum().item())
        final_in_group = (
            (batch_index + 1) % accumulation_steps == 0
            or (total_batches is not None and batch_index + 1 == total_batches)
        )
        sync_context = (
            model.no_sync()  # type: ignore[attr-defined]
            if not final_in_group and hasattr(model, "no_sync")
            else nullcontext()
        )
        with sync_context:
            with _autocast(resolved_device, use_amp):
                prediction = model(inputs)
                loss = criterion(prediction, target, mask)
            if loss.ndim != 0 or not bool(torch.isfinite(loss)):
                raise FloatingPointError("training loss must be a finite scalar")
            scaled_loss = loss / group_size
            if scaler is not None and scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

        if final_in_group:
            step_succeeded = True
            if scaler is not None and scaler.is_enabled():
                scaler.unscale_(optimizer)
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            if scaler is not None and scaler.is_enabled():
                previous_scale = float(scaler.get_scale())
                scaler.step(optimizer)
                scaler.update()
                # GradScaler lowers its scale when non-finite gradients cause
                # optimizer.step() to be skipped. Do not count that as an update.
                step_succeeded = float(scaler.get_scale()) >= previous_scale
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += int(step_succeeded)

        loss_weighted_sum += float(loss.detach().item()) * current_valid
        valid_voxels += current_valid
        batch_count += 1
        metrics.update_log(prediction.detach(), target.detach(), mask.detach())

    loss_weighted_sum, valid_value, batches_value, steps_value = _reduce_totals(
        [loss_weighted_sum, float(valid_voxels), float(batch_count), float(optimizer_steps)],
        resolved_device,
    )
    valid_voxels = int(round(valid_value))
    average_loss = loss_weighted_sum / valid_voxels if valid_voxels else float("nan")
    # All DDP ranks execute the same number of optimizer steps. The reduction
    # above sums them only for transport, so report the per-rank step count.
    world_size = distributed.get_world_size() if _distributed_active() else 1
    return EpochResult(
        loss=average_loss,
        batch_count=int(round(batches_value)),
        optimizer_steps=int(round(steps_value / world_size)),
        valid_voxels=valid_voxels,
        metrics=metrics.compute(synchronize=True),
        duration_seconds=time.perf_counter() - started,
    )


def evaluate_one_epoch(
    model: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    criterion: nn.Module,
    device: torch.device | str,
    *,
    use_amp: bool = True,
    thresholds_mm_h: tuple[float, ...] = (1, 5, 10, 30),
    max_batches: int | None = None,
) -> EpochResult:
    """Evaluate patches without gradients and restore the prior model mode."""

    resolved_device = torch.device(device)
    total_batches = _batch_limit(loader, max_batches)
    was_training = model.training
    model.eval()
    metrics = PrecipitationRegressionMetrics(thresholds_mm_h)
    loss_weighted_sum = 0.0
    valid_voxels = 0
    batch_count = 0
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            for batch_index, source_batch in enumerate(loader):
                if total_batches is not None and batch_index >= total_batches:
                    break
                batch = move_batch_to_device(source_batch, resolved_device)
                # No shape change on transfer: inputs=(B,3,D,H,Z).
                inputs = batch["inputs"]
                target = batch["target"]
                mask = batch["loss_mask"]
                with _autocast(resolved_device, use_amp):
                    prediction = model(inputs)
                    loss = criterion(prediction, target, mask)
                if loss.ndim != 0 or not bool(torch.isfinite(loss)):
                    raise FloatingPointError("validation loss must be a finite scalar")
                current_valid = int(mask.sum().item())
                loss_weighted_sum += float(loss.item()) * current_valid
                valid_voxels += current_valid
                batch_count += 1
                metrics.update_log(prediction, target, mask)
    finally:
        model.train(was_training)

    loss_weighted_sum, valid_value, batches_value = _reduce_totals(
        [loss_weighted_sum, float(valid_voxels), float(batch_count)], resolved_device
    )
    valid_voxels = int(round(valid_value))
    return EpochResult(
        loss=(loss_weighted_sum / valid_voxels if valid_voxels else float("nan")),
        batch_count=int(round(batches_value)),
        optimizer_steps=0,
        valid_voxels=valid_voxels,
        metrics=metrics.compute(synchronize=True),
        duration_seconds=time.perf_counter() - started,
    )


def _unwrapped_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    epoch: int,
    global_step: int,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    config: Mapping[str, Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
) -> None:
    """Atomically save model/training state without a DDP ``module.`` prefix."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format_version": 1,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model": _unwrapped_model(model).state_dict(),
        "optimizer": None if optimizer is None else optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "config": None if config is None else dict(config),
        "metrics": None if metrics is None else dict(metrics),
    }
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary_name = handle.name
        torch.save(payload, temporary_name)
        os.replace(temporary_name, destination)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Restore available training state and return the checkpoint metadata."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    checkpoint = torch.load(source, map_location=map_location, weights_only=False)
    if checkpoint.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format version")
    _unwrapped_model(model).load_state_dict(checkpoint["model"], strict=strict)
    for name, value in (
        ("optimizer", optimizer),
        ("scheduler", scheduler),
        ("scaler", scaler),
    ):
        state = checkpoint.get(name)
        if value is not None and state is not None:
            value.load_state_dict(state)
    return checkpoint
