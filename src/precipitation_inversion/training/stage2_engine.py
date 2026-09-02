"""AMP/DDP-aware loops for Stage-2 support and DPR-dBZ learning.

Tensor convention is ``(B,C,D,H,Z)``. Inputs are normally
``(B,C,64,64,60)`` with a configuration-defined GR-only channel count;
both model outputs and every task target/mask are ``(B,1,64,64,60)``.
Support classification and dBZ regression deliberately keep separate masks
and denominators throughout logging and distributed reduction.
"""

from __future__ import annotations

import math
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.distributed as distributed
from torch import nn

from precipitation_inversion.losses.stage2_losses import (
    Stage2CompositeLoss,
    Stage2LossComponents,
)
from precipitation_inversion.metrics.stage2_reflectivity import (
    Stage2ReflectivityMetrics,
)
from precipitation_inversion.models.stage2_unet3d import (
    stage2_predictions_from_output,
)
from precipitation_inversion.training.engine import (
    _autocast,
    _batch_limit,
    _reduce_totals,
    _unwrapped_model,
    move_batch_to_device,
)


@dataclass(frozen=True)
class Stage2EpochResult:
    """Serializable result of one Stage-2 train or validation pass."""

    loss: float
    batch_count: int
    optimizer_steps: int
    support_voxels: int
    support_positive_voxels: int
    reflectivity_voxels: int
    metrics: dict[str, Any]
    duration_seconds: float
    loss_components: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_threshold(value: float) -> float:
    threshold = float(value)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("support_threshold must lie in [0,1]")
    return threshold


def _normalization_vectors(
    mean: Sequence[float] | torch.Tensor,
    std: Sequence[float] | torch.Tensor,
    *,
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return broadcastable DPR statistics ``(1,1,1,1,Z)``."""

    mean_tensor = torch.as_tensor(mean, dtype=torch.float32, device=reference.device)
    std_tensor = torch.as_tensor(std, dtype=torch.float32, device=reference.device)
    expected = (reference.shape[-1],)
    if mean_tensor.shape != expected or std_tensor.shape != expected:
        raise ValueError(f"DPR normalization vectors must have shape {expected}")
    if not bool(
        torch.isfinite(mean_tensor).all()
        and torch.isfinite(std_tensor).all()
        and (std_tensor > 0.0).all()
    ):
        raise ValueError("DPR normalization statistics must be finite with std > 0")
    shape = (1, 1, 1, 1, reference.shape[-1])
    return mean_tensor.reshape(shape), std_tensor.reshape(shape)


def standardized_to_physical_dbz(
    values: torch.Tensor,
    mean: Sequence[float] | torch.Tensor,
    std: Sequence[float] | torch.Tensor,
) -> torch.Tensor:
    """Invert per-height standardization without changing ``(B,1,D,H,Z)``."""

    if values.ndim != 5 or values.shape[1] != 1:
        raise ValueError("standardized dBZ must have shape (B,1,D,H,Z)")
    mean_tensor, std_tensor = _normalization_vectors(mean, std, reference=values)
    return values.float() * std_tensor + mean_tensor


def _batch_contract(batch: Mapping[str, Any]) -> tuple[torch.Tensor, ...]:
    names = (
        "inputs",
        "target_support",
        "target_dbz",
        "support_loss_mask",
        "regression_mask",
    )
    missing = [name for name in names if name not in batch]
    if missing:
        raise KeyError("Stage-2 batch missing " + ", ".join(missing))
    inputs, target_support, target_dbz, support_mask, regression_mask = (
        batch[name] for name in names
    )
    if not all(isinstance(value, torch.Tensor) for value in (
        inputs, target_support, target_dbz, support_mask, regression_mask
    )):
        raise TypeError("Stage-2 model fields must all be tensors")
    if inputs.ndim != 5 or inputs.shape[1] <= 0:
        raise ValueError("Stage-2 inputs must have shape (B,C,D,H,Z) with C>0")
    expected = (inputs.shape[0], 1, *inputs.shape[2:])
    for name, value in (
        ("target_support", target_support),
        ("target_dbz", target_dbz),
        ("support_loss_mask", support_mask),
        ("regression_mask", regression_mask),
    ):
        if tuple(value.shape) != expected:
            raise ValueError(f"{name} must have shape {expected}")
    if not torch.is_floating_point(inputs):
        raise TypeError("inputs must be floating-point")
    if not torch.is_floating_point(target_support) or not torch.is_floating_point(target_dbz):
        raise TypeError("Stage-2 targets must be floating-point")
    if support_mask.dtype != torch.bool or regression_mask.dtype != torch.bool:
        raise TypeError("Stage-2 loss masks must be boolean")
    return inputs, target_support, target_dbz, support_mask, regression_mask


def _regression_weights(
    batch: Mapping[str, Any], regression_mask: torch.Tensor
) -> torch.Tensor | None:
    weights = batch.get("regression_weights")
    if weights is None:
        return None
    if not isinstance(weights, torch.Tensor) or not torch.is_floating_point(weights):
        raise TypeError("regression_weights must be a floating-point tensor")
    if weights.shape != regression_mask.shape:
        raise ValueError("regression_weights and regression_mask shapes differ")
    selected = weights[regression_mask]
    if selected.numel() and bool((~torch.isfinite(selected) | (selected < 0.0)).any()):
        raise ValueError("selected regression weights must be finite and non-negative")
    return weights


def _metric_state(metric: Stage2ReflectivityMetrics) -> list[float]:
    values: list[float] = [
        float(metric.support.true_positive),
        float(metric.support.false_positive),
        float(metric.support.false_negative),
        float(metric.support.true_negative),
    ]
    for accumulator in (metric.reflectivity_target, metric.reflectivity):
        values.extend(float(getattr(accumulator, name)) for name in accumulator.__dataclass_fields__)
    for accumulator in metric.fss.values():
        values.extend(
            [
                float(accumulator.count),
                accumulator.sum_squared_difference,
                accumulator.sum_squared_reference,
            ]
        )
    return values


def _load_metric_state(metric: Stage2ReflectivityMetrics, values: Sequence[float]) -> None:
    iterator = iter(values)
    metric.support.true_positive = int(round(next(iterator)))
    metric.support.false_positive = int(round(next(iterator)))
    metric.support.false_negative = int(round(next(iterator)))
    metric.support.true_negative = int(round(next(iterator)))
    for accumulator in (metric.reflectivity_target, metric.reflectivity):
        for name in accumulator.__dataclass_fields__:
            value = next(iterator)
            setattr(accumulator, name, int(round(value)) if name == "count" else float(value))
    for accumulator in metric.fss.values():
        accumulator.count = int(round(next(iterator)))
        accumulator.sum_squared_difference = float(next(iterator))
        accumulator.sum_squared_reference = float(next(iterator))
    try:
        next(iterator)
    except StopIteration:
        return
    raise ValueError("unexpected trailing Stage-2 metric state")


def _synchronize_metric(metric: Stage2ReflectivityMetrics, device: torch.device) -> None:
    """All-reduce additive Stage-2 metric state across initialized DDP ranks."""

    if not (
        distributed.is_available()
        and distributed.is_initialized()
        and distributed.get_world_size() > 1
    ):
        return
    backend = str(distributed.get_backend()).lower()
    reduce_device = device if "nccl" in backend else torch.device("cpu")
    tensor = torch.tensor(_metric_state(metric), dtype=torch.float64, device=reduce_device)
    distributed.all_reduce(tensor, op=distributed.ReduceOp.SUM)
    _load_metric_state(metric, tensor.cpu().tolist())


def _update_metrics(
    metric: Stage2ReflectivityMetrics,
    *,
    support_logits: torch.Tensor,
    reflectivity_prediction: torch.Tensor,
    target_support: torch.Tensor,
    target_dbz: torch.Tensor,
    support_mask: torch.Tensor,
    support_threshold: float,
    dpr_mean: Sequence[float] | torch.Tensor,
    dpr_std: Sequence[float] | torch.Tensor,
) -> None:
    # Standardized tensors keep shape (B,1,D,H,Z); inverse transformation
    # broadcasts train-only (Z,) statistics to (1,1,1,1,Z).
    prediction_physical = standardized_to_physical_dbz(
        reflectivity_prediction.detach(), dpr_mean, dpr_std
    )
    target_physical = standardized_to_physical_dbz(
        target_dbz.detach(), dpr_mean, dpr_std
    )
    predicted_support = torch.sigmoid(support_logits.detach().float()) >= support_threshold
    metric.update(
        prediction_physical.cpu().numpy(),
        predicted_support.cpu().numpy(),
        target_physical.cpu().numpy(),
        (target_support.detach() >= 0.5).cpu().numpy(),
        support_mask.detach().cpu().numpy(),
    )


def _summarize_losses(
    criterion: Stage2CompositeLoss,
    *,
    support_sum: float,
    support_count: float,
    reflectivity_sum: float,
    reflectivity_count: float,
    reflectivity_weight_sum: float,
) -> tuple[float, dict[str, Any]]:
    support = support_sum / support_count if support_count > 0.0 else 0.0
    reflectivity = (
        reflectivity_sum / reflectivity_weight_sum
        if reflectivity_weight_sum > 0.0
        else 0.0
    )
    weighted_support = criterion.support_weight * support
    weighted_reflectivity = criterion.reflectivity_weight * reflectivity
    total = weighted_support + weighted_reflectivity
    return total, {
        "support": support,
        "reflectivity_standardized_dbz": reflectivity,
        "support_weight": criterion.support_weight,
        "reflectivity_weight": criterion.reflectivity_weight,
        "weighted_support": weighted_support,
        "weighted_reflectivity": weighted_reflectivity,
        "support_count": int(round(support_count)),
        "reflectivity_count": int(round(reflectivity_count)),
        "reflectivity_weight_sum": reflectivity_weight_sum,
        "total": total,
        "aggregation": (
            "support is reduced by M_support count; dBZ is reduced by the "
            "selected regression-weight sum (equal to M_dbz count when unweighted)"
        ),
    }


def train_stage2_one_epoch(
    model: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    optimizer: torch.optim.Optimizer,
    criterion: Stage2CompositeLoss,
    device: torch.device | str,
    *,
    dpr_mean: Sequence[float] | torch.Tensor,
    dpr_std: Sequence[float] | torch.Tensor,
    support_threshold: float = 0.5,
    fss_radii: tuple[int, ...] = (),
    scaler: Any | None = None,
    use_amp: bool = True,
    grad_clip_norm: float | None = 1.0,
    accumulation_steps: int = 1,
    max_batches: int | None = None,
) -> Stage2EpochResult:
    """Train one epoch with separate support and reflectivity supervision."""

    threshold = _validate_threshold(support_threshold)
    if accumulation_steps <= 0:
        raise ValueError("accumulation_steps must be positive")
    if grad_clip_norm is not None and grad_clip_norm <= 0.0:
        raise ValueError("grad_clip_norm must be positive or None")
    resolved_device = torch.device(device)
    total_batches = _batch_limit(loader, max_batches)
    if accumulation_steps > 1 and total_batches is None:
        raise TypeError("gradient accumulation requires a loader with __len__")

    model.train()
    metric = Stage2ReflectivityMetrics(
        fss_radii=fss_radii, dense_prediction=True
    )
    support_sum = reflectivity_sum = 0.0
    support_count = support_positive_count = reflectivity_count = 0.0
    reflectivity_weight_sum = 0.0
    batch_count = optimizer_steps = 0
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()

    for batch_index, source_batch in enumerate(loader):
        if total_batches is not None and batch_index >= total_batches:
            break
        if batch_index % accumulation_steps == 0:
            group_size = (
                accumulation_steps
                if total_batches is None
                else min(accumulation_steps, total_batches - batch_index)
            )
        batch = move_batch_to_device(source_batch, resolved_device)
        inputs, target_support, target_dbz, support_mask, regression_mask = _batch_contract(batch)
        regression_weights = _regression_weights(batch, regression_mask)
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
                support_logits, reflectivity = stage2_predictions_from_output(model(inputs))
                if not bool(torch.isfinite(support_logits).all() and torch.isfinite(reflectivity).all()):
                    raise FloatingPointError("Stage-2 model outputs must be finite")
                parts = criterion.compute_components(
                    support_logits,
                    reflectivity,
                    target_support,
                    target_dbz,
                    support_mask,
                    regression_mask,
                    regression_weights=regression_weights,
                )
            if parts.total.ndim != 0 or not bool(torch.isfinite(parts.total)):
                raise FloatingPointError("Stage-2 training loss must be a finite scalar")
            scaled = parts.total / group_size
            if scaler is not None and scaler.is_enabled():
                scaler.scale(scaled).backward()
            else:
                scaled.backward()

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
                step_succeeded = float(scaler.get_scale()) >= previous_scale
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += int(step_succeeded)

        support_sum += float(parts.support.detach()) * parts.support_count
        reflectivity_sum += (
            float(parts.reflectivity.detach()) * parts.reflectivity_weight_sum
        )
        support_count += parts.support_count
        support_positive_count += parts.support_positive_count
        reflectivity_count += parts.reflectivity_count
        reflectivity_weight_sum += parts.reflectivity_weight_sum
        batch_count += 1
        _update_metrics(
            metric,
            support_logits=support_logits,
            reflectivity_prediction=reflectivity,
            target_support=target_support,
            target_dbz=target_dbz,
            support_mask=support_mask,
            support_threshold=threshold,
            dpr_mean=dpr_mean,
            dpr_std=dpr_std,
        )

    reduced = _reduce_totals(
        [
            support_sum,
            support_count,
            reflectivity_sum,
            reflectivity_count,
            reflectivity_weight_sum,
            support_positive_count,
            float(batch_count),
            float(optimizer_steps),
        ],
        resolved_device,
    )
    (
        support_sum,
        support_count,
        reflectivity_sum,
        reflectivity_count,
        reflectivity_weight_sum,
        support_positive_count,
        batches_value,
        steps_value,
    ) = reduced
    _synchronize_metric(metric, resolved_device)
    average_loss, components = _summarize_losses(
        criterion,
        support_sum=support_sum,
        support_count=support_count,
        reflectivity_sum=reflectivity_sum,
        reflectivity_count=reflectivity_count,
        reflectivity_weight_sum=reflectivity_weight_sum,
    )
    world_size = (
        distributed.get_world_size()
        if distributed.is_available() and distributed.is_initialized()
        else 1
    )
    return Stage2EpochResult(
        loss=average_loss,
        batch_count=int(round(batches_value)),
        optimizer_steps=int(round(steps_value / world_size)),
        support_voxels=int(round(support_count)),
        support_positive_voxels=int(round(support_positive_count)),
        reflectivity_voxels=int(round(reflectivity_count)),
        metrics={
            "support_threshold": threshold,
            **metric.compute(),
        },
        duration_seconds=time.perf_counter() - started,
        loss_components=components,
    )


def evaluate_stage2_one_epoch(
    model: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    criterion: Stage2CompositeLoss,
    device: torch.device | str,
    *,
    dpr_mean: Sequence[float] | torch.Tensor,
    dpr_std: Sequence[float] | torch.Tensor,
    support_threshold: float = 0.5,
    fss_radii: tuple[int, ...] = (1, 2, 4),
    use_amp: bool = True,
    max_batches: int | None = None,
) -> Stage2EpochResult:
    """Evaluate disjoint patches and restore the model's prior mode."""

    threshold = _validate_threshold(support_threshold)
    resolved_device = torch.device(device)
    total_batches = _batch_limit(loader, max_batches)
    was_training = bool(model.training)
    model.eval()
    inference_model = _unwrapped_model(model)
    metric = Stage2ReflectivityMetrics(
        fss_radii=fss_radii, dense_prediction=True
    )
    support_sum = reflectivity_sum = 0.0
    support_count = support_positive_count = reflectivity_count = 0.0
    reflectivity_weight_sum = 0.0
    batch_count = 0
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            for batch_index, source_batch in enumerate(loader):
                if total_batches is not None and batch_index >= total_batches:
                    break
                batch = move_batch_to_device(source_batch, resolved_device)
                inputs, target_support, target_dbz, support_mask, regression_mask = _batch_contract(batch)
                regression_weights = _regression_weights(batch, regression_mask)
                with _autocast(resolved_device, use_amp):
                    support_logits, reflectivity = stage2_predictions_from_output(
                        inference_model(inputs)
                    )
                    if not bool(torch.isfinite(support_logits).all() and torch.isfinite(reflectivity).all()):
                        raise FloatingPointError("Stage-2 model outputs must be finite")
                    parts = criterion.compute_components(
                        support_logits,
                        reflectivity,
                        target_support,
                        target_dbz,
                        support_mask,
                        regression_mask,
                        regression_weights=regression_weights,
                    )
                if parts.total.ndim != 0 or not bool(torch.isfinite(parts.total)):
                    raise FloatingPointError("Stage-2 validation loss must be a finite scalar")
                support_sum += float(parts.support) * parts.support_count
                reflectivity_sum += (
                    float(parts.reflectivity) * parts.reflectivity_weight_sum
                )
                support_count += parts.support_count
                support_positive_count += parts.support_positive_count
                reflectivity_count += parts.reflectivity_count
                reflectivity_weight_sum += parts.reflectivity_weight_sum
                batch_count += 1
                _update_metrics(
                    metric,
                    support_logits=support_logits,
                    reflectivity_prediction=reflectivity,
                    target_support=target_support,
                    target_dbz=target_dbz,
                    support_mask=support_mask,
                    support_threshold=threshold,
                    dpr_mean=dpr_mean,
                    dpr_std=dpr_std,
                )
    finally:
        model.train(was_training)

    reduced = _reduce_totals(
        [
            support_sum,
            support_count,
            reflectivity_sum,
            reflectivity_count,
            reflectivity_weight_sum,
            support_positive_count,
            float(batch_count),
        ],
        resolved_device,
    )
    (
        support_sum,
        support_count,
        reflectivity_sum,
        reflectivity_count,
        reflectivity_weight_sum,
        support_positive_count,
        batches_value,
    ) = reduced
    _synchronize_metric(metric, resolved_device)
    average_loss, components = _summarize_losses(
        criterion,
        support_sum=support_sum,
        support_count=support_count,
        reflectivity_sum=reflectivity_sum,
        reflectivity_count=reflectivity_count,
        reflectivity_weight_sum=reflectivity_weight_sum,
    )
    return Stage2EpochResult(
        loss=average_loss,
        batch_count=int(round(batches_value)),
        optimizer_steps=0,
        support_voxels=int(round(support_count)),
        support_positive_voxels=int(round(support_positive_count)),
        reflectivity_voxels=int(round(reflectivity_count)),
        metrics={
            "support_threshold": threshold,
            **metric.compute(),
        },
        duration_seconds=time.perf_counter() - started,
        loss_components=components,
    )
