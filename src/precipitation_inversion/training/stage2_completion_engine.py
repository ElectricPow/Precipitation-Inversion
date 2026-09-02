"""AMP/DDP-aware value-only engine for the R1-O spatial upper bound.

Every tensor keeps the height-preserving convention ``(B,C,D,H,Z)``.  The
model output, standardized target and DPR supervision mask are respectively
``(B,1,D,H,Z)``.  No support classification target is optimized here: true
DPR support is an oracle supervision/evaluation domain, not a model output.
"""

from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.distributed as distributed
from torch import nn

from precipitation_inversion.losses.stage2_completion_losses import (
    Stage2CompletionLoss,
)
from precipitation_inversion.metrics.regression import RegressionAccumulator
from precipitation_inversion.models.stage2_completion_unet3d import (
    stage2_completion_prediction_from_output,
)
from precipitation_inversion.training.engine import (
    _autocast,
    _batch_limit,
    _reduce_totals,
    _unwrapped_model,
    move_batch_to_device,
)
from precipitation_inversion.training.stage2_engine import (
    standardized_to_physical_dbz,
)


@dataclass(frozen=True)
class Stage2CompletionEpochResult:
    """Serializable exact-denominator result for one R1-O epoch."""

    loss: float
    batch_count: int
    optimizer_steps: int
    reflectivity_voxels: int
    metrics: dict[str, Any]
    duration_seconds: float
    loss_components: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _batch_contract(
    batch: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    missing = [
        name for name in ("inputs", "target_dbz", "regression_mask")
        if name not in batch
    ]
    if missing:
        raise KeyError("R1-O batch missing " + ", ".join(missing))
    inputs = batch["inputs"]
    target = batch["target_dbz"]
    mask = batch["regression_mask"]
    if not all(isinstance(value, torch.Tensor) for value in (inputs, target, mask)):
        raise TypeError("R1-O model fields must be tensors")
    if inputs.ndim != 5 or inputs.shape[1] <= 0:
        raise ValueError("R1-O inputs must have shape (B,C,D,H,Z)")
    expected = (inputs.shape[0], 1, *inputs.shape[2:])
    if tuple(target.shape) != expected or tuple(mask.shape) != expected:
        raise ValueError(f"R1-O target and mask must have shape {expected}")
    if not torch.is_floating_point(inputs) or not torch.is_floating_point(target):
        raise TypeError("R1-O inputs and target must be floating-point")
    if mask.dtype != torch.bool:
        raise TypeError("R1-O regression_mask must be boolean")
    weights = batch.get("regression_weights")
    if weights is not None:
        if not isinstance(weights, torch.Tensor) or not torch.is_floating_point(weights):
            raise TypeError("R1-O regression_weights must be floating-point")
        if weights.shape != mask.shape:
            raise ValueError("R1-O regression_weights and mask shapes differ")
        selected = weights[mask]
        if selected.numel() and bool(
            (~torch.isfinite(selected) | (selected < 0.0)).any()
        ):
            raise ValueError("selected R1-O regression weights are invalid")
    return inputs, target, mask, weights


def _summarize(
    loss_sum: float,
    count: float,
    weight_sum: float,
) -> tuple[float, dict[str, Any]]:
    value = loss_sum / weight_sum if weight_sum > 0.0 else 0.0
    return value, {
        "reflectivity_standardized_dbz": value,
        "reflectivity_count": int(round(count)),
        "reflectivity_weight_sum": weight_sum,
        "total": value,
        "aggregation": (
            "weighted Smooth-L1 is reduced by the selected M_dbz weight sum; "
            "true DPR support is oracle supervision only"
        ),
    }


def _run_epoch(
    model: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    criterion: Stage2CompletionLoss,
    device: torch.device | str,
    *,
    dpr_mean: Sequence[float] | torch.Tensor,
    dpr_std: Sequence[float] | torch.Tensor,
    optimizer: torch.optim.Optimizer | None,
    scaler: Any | None,
    use_amp: bool,
    grad_clip_norm: float | None,
    accumulation_steps: int,
    max_batches: int | None,
) -> Stage2CompletionEpochResult:
    if accumulation_steps <= 0:
        raise ValueError("accumulation_steps must be positive")
    if grad_clip_norm is not None and grad_clip_norm <= 0.0:
        raise ValueError("grad_clip_norm must be positive or None")
    training = optimizer is not None
    resolved_device = torch.device(device)
    total_batches = _batch_limit(loader, max_batches)
    if training and accumulation_steps > 1 and total_batches is None:
        raise TypeError("gradient accumulation requires a loader with __len__")

    previous_mode = bool(model.training)
    model.train(training)
    inference_model = model if training else _unwrapped_model(model)
    metric = RegressionAccumulator()
    loss_sum = count = weight_sum = 0.0
    batch_count = optimizer_steps = 0
    if training:
        assert optimizer is not None
        optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    inference_context = nullcontext() if training else torch.inference_mode()
    try:
        with inference_context:
            for batch_index, source_batch in enumerate(loader):
                if total_batches is not None and batch_index >= total_batches:
                    break
                if training and batch_index % accumulation_steps == 0:
                    group_size = (
                        accumulation_steps
                        if total_batches is None
                        else min(accumulation_steps, total_batches - batch_index)
                    )
                else:
                    group_size = 1
                batch = move_batch_to_device(source_batch, resolved_device)
                inputs, target, mask, weights = _batch_contract(batch)
                final_in_group = (
                    not training
                    or (batch_index + 1) % accumulation_steps == 0
                    or (
                        total_batches is not None
                        and batch_index + 1 == total_batches
                    )
                )
                sync_context = (
                    model.no_sync()  # type: ignore[attr-defined]
                    if training and not final_in_group and hasattr(model, "no_sync")
                    else nullcontext()
                )
                with sync_context:
                    with _autocast(resolved_device, use_amp):
                        prediction = stage2_completion_prediction_from_output(
                            inference_model(inputs)
                        )
                        if prediction.shape != target.shape:
                            raise ValueError("R1-O output shape must match target")
                        if not bool(torch.isfinite(prediction).all()):
                            raise FloatingPointError("R1-O model output must be finite")
                        parts = criterion.compute_components(
                            prediction,
                            target,
                            mask,
                            regression_weights=weights,
                        )
                    if parts.total.ndim != 0 or not bool(torch.isfinite(parts.total)):
                        raise FloatingPointError("R1-O loss must be a finite scalar")
                    if training:
                        scaled = parts.total / group_size
                        if scaler is not None and scaler.is_enabled():
                            scaler.scale(scaled).backward()
                        else:
                            scaled.backward()

                if training and final_in_group:
                    assert optimizer is not None
                    step_succeeded = True
                    if scaler is not None and scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    if grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    if scaler is not None and scaler.is_enabled():
                        old_scale = float(scaler.get_scale())
                        scaler.step(optimizer)
                        scaler.update()
                        step_succeeded = float(scaler.get_scale()) >= old_scale
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_steps += int(step_succeeded)

                loss_sum += float(parts.reflectivity.detach()) * parts.reflectivity_weight_sum
                count += parts.reflectivity_count
                weight_sum += parts.reflectivity_weight_sum
                batch_count += 1
                prediction_dbz = standardized_to_physical_dbz(
                    prediction.detach(), dpr_mean, dpr_std
                )
                target_dbz = standardized_to_physical_dbz(
                    target.detach(), dpr_mean, dpr_std
                )
                metric.update(prediction_dbz, target_dbz, mask)
    finally:
        model.train(previous_mode)

    reduced = _reduce_totals(
        [loss_sum, count, weight_sum, float(batch_count), float(optimizer_steps)],
        resolved_device,
    )
    loss_sum, count, weight_sum, batches_value, steps_value = reduced
    average, components = _summarize(loss_sum, count, weight_sum)
    world_size = (
        distributed.get_world_size()
        if distributed.is_available() and distributed.is_initialized()
        else 1
    )
    metrics = metric.compute(synchronize=True)
    return Stage2CompletionEpochResult(
        loss=average,
        batch_count=int(round(batches_value)),
        optimizer_steps=int(round(steps_value / world_size)) if training else 0,
        reflectivity_voxels=int(round(count)),
        metrics={
            "reflectivity_on_oracle_support": {
                "count": metrics["count"],
                "mae_dbz": metrics["mae"],
                "rmse_dbz": metrics["rmse"],
                "bias_dbz": metrics["bias"],
                "r2": metrics["r2"],
                "pearson_r": metrics["pearson_r"],
                "ccc": metrics["ccc"],
            }
        },
        duration_seconds=time.perf_counter() - started,
        loss_components=components,
    )


def train_stage2_completion_one_epoch(
    model: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    optimizer: torch.optim.Optimizer,
    criterion: Stage2CompletionLoss,
    device: torch.device | str,
    *,
    dpr_mean: Sequence[float] | torch.Tensor,
    dpr_std: Sequence[float] | torch.Tensor,
    scaler: Any | None = None,
    use_amp: bool = True,
    grad_clip_norm: float | None = 1.0,
    accumulation_steps: int = 1,
    max_batches: int | None = None,
) -> Stage2CompletionEpochResult:
    """Optimize one R1-O epoch."""

    return _run_epoch(
        model, loader, criterion, device,
        dpr_mean=dpr_mean, dpr_std=dpr_std, optimizer=optimizer,
        scaler=scaler, use_amp=use_amp, grad_clip_norm=grad_clip_norm,
        accumulation_steps=accumulation_steps, max_batches=max_batches,
    )


def evaluate_stage2_completion_one_epoch(
    model: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    criterion: Stage2CompletionLoss,
    device: torch.device | str,
    *,
    dpr_mean: Sequence[float] | torch.Tensor,
    dpr_std: Sequence[float] | torch.Tensor,
    use_amp: bool = True,
    max_batches: int | None = None,
) -> Stage2CompletionEpochResult:
    """Evaluate disjoint R1-O patches and restore the model mode."""

    return _run_epoch(
        model, loader, criterion, device,
        dpr_mean=dpr_mean, dpr_std=dpr_std, optimizer=None,
        scaler=None, use_amp=use_amp, grad_clip_norm=None,
        accumulation_steps=1, max_batches=max_batches,
    )


__all__ = [
    "Stage2CompletionEpochResult",
    "train_stage2_completion_one_epoch",
    "evaluate_stage2_completion_one_epoch",
]
