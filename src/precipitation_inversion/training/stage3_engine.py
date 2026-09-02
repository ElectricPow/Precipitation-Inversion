"""Training engines for controlled Stage-3 one-sided adaptation."""

from __future__ import annotations

import math
import time
from collections.abc import Iterable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from precipitation_inversion.losses.stage3_losses import (
    Stage3C2CompositeLoss,
    Stage3D0CompositeLoss,
)
from precipitation_inversion.metrics.regression import PrecipitationRegressionMetrics
from precipitation_inversion.metrics.stage2_reflectivity import Stage2ReflectivityMetrics
from precipitation_inversion.models.stage3_cascade import (
    assert_c1_freeze_contract,
    assert_c2_freeze_contract,
    stage3_c2_predictions_from_output,
)
from precipitation_inversion.models.stage3_direct import (
    assert_d0_trainable_contract,
    d0_shared_audit_parameters,
    stage3_d0_predictions_from_output,
)

from .engine import (
    EpochResult,
    _autocast,
    _batch_limit,
    _loss_normalizer,
    _loss_weights,
    _metric_mask,
    _reduce_totals,
    _summarize_loss_components,
    _unwrapped_model,
    evaluate_one_epoch,
    move_batch_to_device,
    train_one_epoch,
)
from .stage2_engine import (
    _summarize_losses as _summarize_stage2_losses,
    _synchronize_metric as _synchronize_stage2_metric,
    _update_metrics as _update_stage2_metrics,
)


def train_stage3_c1_one_epoch(
    model: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device | str,
    **kwargs: Any,
) -> EpochResult:
    """Train only Stage 1 while preserving the C1 freeze contract."""

    assert_c1_freeze_contract(model)
    return train_one_epoch(model, loader, optimizer, criterion, device, **kwargs)


def evaluate_stage3_c1_one_epoch(
    model: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    criterion: nn.Module,
    device: torch.device | str,
    **kwargs: Any,
) -> EpochResult:
    """Evaluate the same oracle-support cascade and independent rain masks."""

    assert_c1_freeze_contract(model)
    return evaluate_one_epoch(model, loader, criterion, device, **kwargs)


@dataclass(frozen=True)
class Stage3C2EpochResult:
    """One C2 epoch with final-rain and Stage-2 task diagnostics."""

    loss: float
    batch_count: int
    optimizer_steps: int
    rain_voxels: int
    support_voxels: int
    support_positive_voxels: int
    reflectivity_voxels: int
    metrics: dict[str, Any]
    duration_seconds: float
    loss_components: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Stage3D0EpochResult(Stage3C2EpochResult):
    """One direct multi-head epoch with rain and physical diagnostics."""


def _c2_batch_fields(batch: Mapping[str, Any]) -> dict[str, torch.Tensor | None]:
    required = (
        "inputs",
        "target",
        "loss_mask",
        "reliable_loss_mask",
        "height_km",
        "stage2_target_support",
        "stage2_target_dbz",
        "stage2_support_loss_mask",
        "stage2_regression_mask",
    )
    missing = [name for name in required if name not in batch]
    if missing:
        raise KeyError("C2-O batch missing " + ", ".join(missing))
    tensors = {name: batch[name] for name in required}
    if not all(isinstance(value, torch.Tensor) for value in tensors.values()):
        raise TypeError("all required C2-O batch fields must be tensors")
    inputs = tensors["inputs"]
    assert isinstance(inputs, torch.Tensor)
    if inputs.ndim != 5 or inputs.shape[1] <= 2:
        raise ValueError("C2-O packed inputs must have shape (B,C2+2,D,H,Z)")
    expected = (inputs.shape[0], 1, *inputs.shape[2:])
    for name in (
        "target",
        "loss_mask",
        "reliable_loss_mask",
        "stage2_target_support",
        "stage2_target_dbz",
        "stage2_support_loss_mask",
        "stage2_regression_mask",
    ):
        value = tensors[name]
        assert isinstance(value, torch.Tensor)
        if tuple(value.shape) != expected:
            raise ValueError(f"{name} must have shape {expected}")
    for name in (
        "loss_mask",
        "reliable_loss_mask",
        "stage2_support_loss_mask",
        "stage2_regression_mask",
    ):
        value = tensors[name]
        assert isinstance(value, torch.Tensor)
        if value.dtype != torch.bool:
            raise TypeError(f"{name} must be boolean")
    for name in ("inputs", "target", "stage2_target_support", "stage2_target_dbz"):
        value = tensors[name]
        assert isinstance(value, torch.Tensor)
        if not torch.is_floating_point(value):
            raise TypeError(f"{name} must be floating-point")
    height = tensors["height_km"]
    assert isinstance(height, torch.Tensor)
    if not torch.is_floating_point(height) or height.shape[-1] != inputs.shape[-1]:
        raise ValueError("height_km must be floating-point and retain Z levels")

    rain_mask = tensors["loss_mask"]
    reliable = tensors["reliable_loss_mask"]
    assert isinstance(rain_mask, torch.Tensor) and isinstance(reliable, torch.Tensor)
    if bool((reliable & ~rain_mask).any()):
        raise ValueError("reliable_loss_mask must be a subset of loss_mask")

    rain_weights = _loss_weights(batch, rain_mask)
    regression_weights = batch.get("stage2_regression_weights")
    if regression_weights is not None:
        if not isinstance(regression_weights, torch.Tensor):
            raise TypeError("stage2_regression_weights must be a tensor")
        if regression_weights.shape != tensors["stage2_regression_mask"].shape:
            raise ValueError("Stage-2 regression weights/mask shapes differ")
        if not torch.is_floating_point(regression_weights):
            raise TypeError("stage2_regression_weights must be floating-point")
        selected = regression_weights[tensors["stage2_regression_mask"]]
        if selected.numel() and bool(
            (~torch.isfinite(selected) | (selected < 0.0)).any()
        ):
            raise ValueError("selected Stage-2 regression weights are invalid")
    tensors["rain_weights"] = rain_weights
    tensors["stage2_regression_weights"] = regression_weights
    return tensors


def _compute_c2_parts(
    model: nn.Module,
    criterion: Stage3C2CompositeLoss,
    fields: Mapping[str, torch.Tensor | None],
) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs = fields["inputs"]
    assert isinstance(inputs, torch.Tensor)
    rain, support_logits, reflectivity = stage3_c2_predictions_from_output(
        model(inputs)
    )
    if not bool(
        torch.isfinite(rain).all()
        and torch.isfinite(support_logits).all()
        and torch.isfinite(reflectivity).all()
    ):
        raise FloatingPointError("C2-O model outputs must be finite")
    parts = criterion.compute_components(
        rain_prediction=rain,
        support_logits=support_logits,
        reflectivity_prediction=reflectivity,
        rain_target=fields["target"],  # type: ignore[arg-type]
        rain_mask=fields["loss_mask"],  # type: ignore[arg-type]
        rain_weights=fields["rain_weights"],
        reliable_rain_mask=fields["reliable_loss_mask"],  # type: ignore[arg-type]
        height_km=fields["height_km"],  # type: ignore[arg-type]
        target_support=fields["stage2_target_support"],  # type: ignore[arg-type]
        target_dbz=fields["stage2_target_dbz"],  # type: ignore[arg-type]
        support_mask=fields["stage2_support_loss_mask"],  # type: ignore[arg-type]
        regression_mask=fields["stage2_regression_mask"],  # type: ignore[arg-type]
        regression_weights=fields["stage2_regression_weights"],
    )
    return parts, rain, support_logits, reflectivity


def _compute_d0_parts(
    model: nn.Module,
    criterion: Stage3D0CompositeLoss,
    fields: Mapping[str, torch.Tensor | None],
) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run D0 on GR-only inputs and retain all three independent heads."""

    inputs = fields["inputs"]
    assert isinstance(inputs, torch.Tensor)
    rain, support_logits, reflectivity = stage3_d0_predictions_from_output(
        model(inputs)
    )
    if not bool(
        torch.isfinite(rain).all()
        and torch.isfinite(support_logits).all()
        and torch.isfinite(reflectivity).all()
    ):
        raise FloatingPointError("D0 model outputs must be finite")
    parts = criterion.compute_components(
        rain_prediction=rain,
        support_logits=support_logits,
        reflectivity_prediction=reflectivity,
        rain_target=fields["target"],  # type: ignore[arg-type]
        rain_mask=fields["loss_mask"],  # type: ignore[arg-type]
        rain_weights=fields["rain_weights"],
        reliable_rain_mask=fields["reliable_loss_mask"],  # type: ignore[arg-type]
        height_km=fields["height_km"],  # type: ignore[arg-type]
        target_support=fields["stage2_target_support"],  # type: ignore[arg-type]
        target_dbz=fields["stage2_target_dbz"],  # type: ignore[arg-type]
        support_mask=fields["stage2_support_loss_mask"],  # type: ignore[arg-type]
        regression_mask=fields["stage2_regression_mask"],  # type: ignore[arg-type]
        regression_weights=fields["stage2_regression_weights"],
    )
    return parts, rain, support_logits, reflectivity


def _gradient_norm(
    loss: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
    *,
    retain_graph: bool,
) -> float:
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    squared = torch.zeros((), device=loss.device, dtype=torch.float64)
    for gradient in gradients:
        if gradient is not None:
            squared += gradient.detach().double().square().sum()
    return float(torch.sqrt(squared).item())


def audit_stage3_c2_gradient_scale(
    model: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    criterion: Stage3C2CompositeLoss,
    device: torch.device | str,
    *,
    target_gradient_ratio: float = 0.25,
    min_rain_weight: float = 1e-4,
    max_rain_weight: float = 1.0,
    valid_batches_per_rank: int = 4,
    max_candidate_batches_per_rank: int = 32,
) -> dict[str, Any]:
    """Select ``lambda_R`` from train-only gradient norms.

    For each non-empty rain batch, gradients of the unweighted Stage-2 anchor
    and sealed rain objective are measured with respect to the exact trainable
    Stage-2 parameters. The geometric mean of ``anchor_norm/rain_norm`` is
    multiplied by ``target_gradient_ratio`` and clamped to the configured safe
    interval. DDP ranks contribute disjoint train batches through a final
    all-reduce; validation and test data are never inspected.
    """

    for name, value in (
        ("target_gradient_ratio", target_gradient_ratio),
        ("min_rain_weight", min_rain_weight),
        ("max_rain_weight", max_rain_weight),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if min_rain_weight > max_rain_weight:
        raise ValueError("min_rain_weight cannot exceed max_rain_weight")
    if valid_batches_per_rank <= 0 or max_candidate_batches_per_rank <= 0:
        raise ValueError("gradient-audit batch limits must be positive")
    assert_c2_freeze_contract(model)
    candidate = _unwrapped_model(model)
    parameters = [value for value in candidate.parameters() if value.requires_grad]
    if not parameters:
        raise RuntimeError("C2-O gradient audit has no trainable parameters")

    was_training = bool(model.training)
    model.train(True)
    sums = {
        "log_ratio": 0.0,
        "anchor_norm": 0.0,
        "rain_norm": 0.0,
        "anchor_loss": 0.0,
        "rain_loss": 0.0,
        "count": 0.0,
        "candidate_count": 0.0,
    }
    resolved_device = torch.device(device)
    try:
        for batch_index, source_batch in enumerate(loader):
            if batch_index >= max_candidate_batches_per_rank:
                break
            sums["candidate_count"] += 1.0
            batch = move_batch_to_device(source_batch, resolved_device)
            fields = _c2_batch_fields(batch)
            rain_mask = fields["loss_mask"]
            assert isinstance(rain_mask, torch.Tensor)
            if not bool(rain_mask.any()):
                continue
            parts, _, _, _ = _compute_c2_parts(candidate, criterion, fields)
            anchor_norm = _gradient_norm(
                parts.stage2.total, parameters, retain_graph=True
            )
            rain_norm = _gradient_norm(
                parts.rain.total, parameters, retain_graph=False
            )
            if not (
                math.isfinite(anchor_norm)
                and math.isfinite(rain_norm)
                and anchor_norm > 0.0
                and rain_norm > 0.0
            ):
                continue
            sums["log_ratio"] += math.log(anchor_norm / rain_norm)
            sums["anchor_norm"] += anchor_norm
            sums["rain_norm"] += rain_norm
            sums["anchor_loss"] += float(parts.stage2.total.detach())
            sums["rain_loss"] += float(parts.rain.total.detach())
            sums["count"] += 1.0
            if sums["count"] >= valid_batches_per_rank:
                break
    finally:
        model.train(was_training)

    names = tuple(sums)
    reduced = _reduce_totals([sums[name] for name in names], resolved_device)
    global_values = dict(zip(names, reduced))
    count = global_values["count"]
    if count <= 0.0:
        raise RuntimeError("C2-O gradient audit found no finite positive-rain batch")
    anchor_to_rain = math.exp(global_values["log_ratio"] / count)
    unclamped = target_gradient_ratio * anchor_to_rain
    selected = min(max(unclamped, min_rain_weight), max_rain_weight)
    return {
        "selection_scope": "train_batches_only",
        "aggregation": "geometric_mean_of_per_batch_anchor_norm_over_rain_norm",
        "valid_batch_count_global": int(round(count)),
        "candidate_batch_count_global": int(
            round(global_values["candidate_count"])
        ),
        "target_weighted_rain_to_anchor_gradient_ratio": target_gradient_ratio,
        "mean_anchor_gradient_norm": global_values["anchor_norm"] / count,
        "mean_unweighted_rain_gradient_norm": global_values["rain_norm"] / count,
        "geometric_mean_anchor_to_rain_gradient_ratio": anchor_to_rain,
        "mean_anchor_loss": global_values["anchor_loss"] / count,
        "mean_unweighted_rain_loss": global_values["rain_loss"] / count,
        "unclamped_rain_weight": unclamped,
        "rain_weight_bounds": [min_rain_weight, max_rain_weight],
        "selected_rain_weight": selected,
        "selected_weight_was_clamped": not math.isclose(
            selected, unclamped, rel_tol=0.0, abs_tol=1e-15
        ),
    }


def audit_stage3_d0_gradient_scale(
    model: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    criterion: Stage3D0CompositeLoss,
    device: torch.device | str,
    *,
    target_gradient_ratio: float = 0.25,
    min_rain_weight: float = 1e-4,
    max_rain_weight: float = 1.0,
    valid_batches_per_rank: int = 4,
    max_candidate_batches_per_rank: int = 32,
    scaled_task: str = "rain",
) -> dict[str, Any]:
    """Select one D0 task coefficient from train-only decoder gradients.

    Head-specific parameters are intentionally excluded: support/dBZ heads do
    not receive rain gradients and the rain head does not receive physical
    anchor gradients. Comparing norms on those disjoint heads would not
    measure task competition. The complete trainable decoder is the common
    representation affected by every objective. ``scaled_task='rain'`` keeps
    the legacy anchor-primary objective and selects the rain coefficient.
    ``scaled_task='stage2'`` keeps rain at one and selects the physical
    support+dBZ coefficient so that RainPrimary cannot silently invert its
    intended main/auxiliary task hierarchy.
    """

    normalized_scaled_task = str(scaled_task).strip().lower()
    if normalized_scaled_task not in ("rain", "stage2"):
        raise ValueError("scaled_task must be 'rain' or 'stage2'")

    for name, value in (
        ("target_gradient_ratio", target_gradient_ratio),
        ("min_rain_weight", min_rain_weight),
        ("max_rain_weight", max_rain_weight),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if min_rain_weight > max_rain_weight:
        raise ValueError("min_rain_weight cannot exceed max_rain_weight")
    if valid_batches_per_rank <= 0 or max_candidate_batches_per_rank <= 0:
        raise ValueError("gradient-audit batch limits must be positive")
    assert_d0_trainable_contract(model)
    candidate = _unwrapped_model(model)
    parameters = d0_shared_audit_parameters(candidate)
    if not parameters:
        raise RuntimeError("D0 gradient audit has no shared decoder parameters")

    was_training = bool(model.training)
    model.train(True)
    sums = {
        "log_ratio": 0.0,
        "anchor_norm": 0.0,
        "rain_norm": 0.0,
        "anchor_loss": 0.0,
        "rain_loss": 0.0,
        "count": 0.0,
        "candidate_count": 0.0,
    }
    resolved_device = torch.device(device)
    try:
        for batch_index, source_batch in enumerate(loader):
            if batch_index >= max_candidate_batches_per_rank:
                break
            sums["candidate_count"] += 1.0
            batch = move_batch_to_device(source_batch, resolved_device)
            fields = _c2_batch_fields(batch)
            rain_mask = fields["loss_mask"]
            assert isinstance(rain_mask, torch.Tensor)
            if not bool(rain_mask.any()):
                continue
            parts, _, _, _ = _compute_d0_parts(candidate, criterion, fields)
            anchor_norm = _gradient_norm(
                parts.stage2.total, parameters, retain_graph=True
            )
            rain_norm = _gradient_norm(
                parts.rain.total, parameters, retain_graph=False
            )
            if not (
                math.isfinite(anchor_norm)
                and math.isfinite(rain_norm)
                and anchor_norm > 0.0
                and rain_norm > 0.0
            ):
                continue
            sums["log_ratio"] += math.log(anchor_norm / rain_norm)
            sums["anchor_norm"] += anchor_norm
            sums["rain_norm"] += rain_norm
            sums["anchor_loss"] += float(parts.stage2.total.detach())
            sums["rain_loss"] += float(parts.rain.total.detach())
            sums["count"] += 1.0
            if sums["count"] >= valid_batches_per_rank:
                break
    finally:
        model.train(was_training)

    names = tuple(sums)
    reduced = _reduce_totals([sums[name] for name in names], resolved_device)
    values = dict(zip(names, reduced))
    count = values["count"]
    if count <= 0.0:
        raise RuntimeError("D0 gradient audit found no finite positive-rain batch")
    anchor_to_rain = math.exp(values["log_ratio"] / count)
    if normalized_scaled_task == "rain":
        # lambda_R * |g_rain| / |g_anchor| = requested ratio.
        unclamped = target_gradient_ratio * anchor_to_rain
        selected_rain_weight_key = "selected_rain_weight"
        unclamped_weight_key = "unclamped_rain_weight"
        weight_bounds_key = "rain_weight_bounds"
        objective_mode = "anchor_primary"
    else:
        # lambda_phys * |g_anchor| / |g_rain| = requested ratio.
        unclamped = target_gradient_ratio / anchor_to_rain
        selected_rain_weight_key = "selected_stage2_weight"
        unclamped_weight_key = "unclamped_stage2_weight"
        weight_bounds_key = "stage2_weight_bounds"
        objective_mode = "rain_primary"
    selected = min(max(unclamped, min_rain_weight), max_rain_weight)
    result = {
        "selection_scope": "train_batches_only_shared_decoder",
        "aggregation": "geometric_mean_of_per_batch_anchor_norm_over_rain_norm",
        "objective_mode": objective_mode,
        "scaled_task": normalized_scaled_task,
        "valid_batch_count_global": int(round(count)),
        "candidate_batch_count_global": int(round(values["candidate_count"])),
        "mean_anchor_gradient_norm": values["anchor_norm"] / count,
        "mean_unweighted_rain_gradient_norm": values["rain_norm"] / count,
        "geometric_mean_anchor_to_rain_gradient_ratio": anchor_to_rain,
        "mean_anchor_loss": values["anchor_loss"] / count,
        "mean_unweighted_rain_loss": values["rain_loss"] / count,
        "selected_weight_was_clamped": not math.isclose(
            selected, unclamped, rel_tol=0.0, abs_tol=1e-15
        ),
        "gradient_parameter_scope": "trainable_decoder_only_excluding_task_heads",
    }
    result[unclamped_weight_key] = unclamped
    result[weight_bounds_key] = [min_rain_weight, max_rain_weight]
    result[selected_rain_weight_key] = selected
    if normalized_scaled_task == "rain":
        result["selected_stage2_weight"] = 1.0
        result["target_weighted_rain_to_anchor_gradient_ratio"] = target_gradient_ratio
    else:
        result["selected_rain_weight"] = 1.0
        result["target_weighted_stage2_to_rain_gradient_ratio"] = target_gradient_ratio
    return result


def _summarize_c2(
    criterion: Stage3C2CompositeLoss,
    values: Mapping[str, float],
) -> tuple[float, dict[str, Any]]:
    stage2_loss, stage2_components = _summarize_stage2_losses(
        criterion.stage2_criterion,
        support_sum=values["support_sum"],
        support_count=values["support_count"],
        reflectivity_sum=values["reflectivity_sum"],
        reflectivity_count=values["reflectivity_count"],
        reflectivity_weight_sum=values["reflectivity_weight_sum"],
    )
    rain_loss, rain_components = _summarize_loss_components(
        primary_sum=values["rain_primary_sum"],
        primary_normalizer=values["rain_primary_normalizer"],
        physical_gradient_sum=values["rain_gradient_sum"],
        physical_gradient_pair_count=values["rain_gradient_pair_count"],
        physical_gradient_weight=criterion.rain_criterion.physical_gradient_weight,
    )
    weighted_stage2 = criterion.stage2_weight * stage2_loss
    weighted_rain = criterion.rain_weight * rain_loss
    total = weighted_stage2 + weighted_rain
    return total, {
        "stage2_anchor": stage2_components,
        "rain_task": rain_components,
        "stage2_weight": criterion.stage2_weight,
        "weighted_stage2_anchor": weighted_stage2,
        "weighted_stage2_fraction": weighted_stage2 / total if total else float("nan"),
        "rain_weight": criterion.rain_weight,
        "weighted_rain_task": weighted_rain,
        "weighted_rain_fraction": weighted_rain / total if total else float("nan"),
        "total": total,
        "aggregation": (
            "Stage-2 support/dBZ and Stage-1 I/G retain independent masks and "
            "denominators; explicit coefficients multiply the two fully "
            "reduced task groups"
        ),
    }


def _run_stage3_three_head_epoch(
    model: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    criterion: Stage3C2CompositeLoss,
    device: torch.device | str,
    *,
    optimizer: torch.optim.Optimizer | None,
    dpr_mean: Sequence[float] | torch.Tensor,
    dpr_std: Sequence[float] | torch.Tensor,
    support_threshold: float,
    thresholds_mm_h: tuple[float, ...],
    scaler: Any | None,
    use_amp: bool,
    grad_clip_norm: float | None,
    accumulation_steps: int,
    max_batches: int | None,
    direct_d0: bool,
) -> Stage3C2EpochResult:
    training = optimizer is not None
    if accumulation_steps <= 0:
        raise ValueError("accumulation_steps must be positive")
    if grad_clip_norm is not None and grad_clip_norm <= 0.0:
        raise ValueError("grad_clip_norm must be positive or None")
    if not 0.0 <= support_threshold <= 1.0:
        raise ValueError("support_threshold must lie in [0,1]")
    if direct_d0:
        assert_d0_trainable_contract(model)
    else:
        assert_c2_freeze_contract(model)
    resolved_device = torch.device(device)
    total_batches = _batch_limit(loader, max_batches)
    if training and accumulation_steps > 1 and total_batches is None:
        raise TypeError("gradient accumulation requires a loader with __len__")

    was_training = bool(model.training)
    model.train(training)
    execution_model = model if training else _unwrapped_model(model)
    rain_metric = PrecipitationRegressionMetrics(thresholds_mm_h)
    stage2_metric = Stage2ReflectivityMetrics(fss_radii=(), dense_prediction=True)
    totals = {
        "support_sum": 0.0,
        "support_count": 0.0,
        "support_positive_count": 0.0,
        "reflectivity_sum": 0.0,
        "reflectivity_count": 0.0,
        "reflectivity_weight_sum": 0.0,
        "rain_primary_sum": 0.0,
        "rain_primary_normalizer": 0.0,
        "rain_gradient_sum": 0.0,
        "rain_gradient_pair_count": 0.0,
        "rain_voxels": 0.0,
        "batch_count": 0.0,
        "optimizer_steps": 0.0,
    }
    if training:
        optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    group_size = accumulation_steps

    context = nullcontext() if training else torch.inference_mode()
    try:
        with context:
            for batch_index, source_batch in enumerate(loader):
                if total_batches is not None and batch_index >= total_batches:
                    break
                if batch_index % accumulation_steps == 0 and total_batches is not None:
                    group_size = min(
                        accumulation_steps, total_batches - batch_index
                    )
                batch = move_batch_to_device(source_batch, resolved_device)
                fields = _c2_batch_fields(batch)
                final_in_group = (
                    (batch_index + 1) % accumulation_steps == 0
                    or (total_batches is not None and batch_index + 1 == total_batches)
                )
                sync_context = (
                    execution_model.no_sync()  # type: ignore[attr-defined]
                    if training
                    and not final_in_group
                    and hasattr(execution_model, "no_sync")
                    else nullcontext()
                )
                with sync_context:
                    with _autocast(resolved_device, use_amp):
                        compute_parts = _compute_d0_parts if direct_d0 else _compute_c2_parts
                        parts, rain, support_logits, reflectivity = compute_parts(
                            execution_model, criterion, fields  # type: ignore[arg-type]
                        )
                    if parts.total.ndim != 0 or not bool(torch.isfinite(parts.total)):
                        label = "D0" if direct_d0 else "C2-O"
                        raise FloatingPointError(f"{label} loss must be a finite scalar")
                    if training:
                        scaled = parts.total / group_size
                        if scaler is not None and scaler.is_enabled():
                            scaler.scale(scaled).backward()
                        else:
                            scaled.backward()

                if training and final_in_group:
                    step_succeeded = True
                    if scaler is not None and scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    if grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in execution_model.parameters() if p.requires_grad],
                            grad_clip_norm,
                        )
                    if scaler is not None and scaler.is_enabled():
                        previous_scale = float(scaler.get_scale())
                        scaler.step(optimizer)
                        scaler.update()
                        step_succeeded = float(scaler.get_scale()) >= previous_scale
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    totals["optimizer_steps"] += float(step_succeeded)

                rain_mask = fields["loss_mask"]
                rain_weights = fields["rain_weights"]
                reliable = _metric_mask(batch, rain_mask)  # type: ignore[arg-type]
                rain_normalizer = _loss_normalizer(
                    rain_mask, rain_weights  # type: ignore[arg-type]
                )
                totals["rain_primary_sum"] += (
                    float(parts.rain.primary.detach()) * rain_normalizer
                )
                totals["rain_primary_normalizer"] += rain_normalizer
                totals["rain_gradient_sum"] += (
                    float(parts.rain.physical_gradient.detach())
                    * parts.rain.physical_gradient_pair_count
                )
                totals["rain_gradient_pair_count"] += (
                    parts.rain.physical_gradient_pair_count
                )
                totals["rain_voxels"] += float(rain_mask.sum().item())  # type: ignore[union-attr]
                totals["support_sum"] += (
                    float(parts.stage2.support.detach()) * parts.stage2.support_count
                )
                totals["support_count"] += parts.stage2.support_count
                totals["support_positive_count"] += parts.stage2.support_positive_count
                totals["reflectivity_sum"] += (
                    float(parts.stage2.reflectivity.detach())
                    * parts.stage2.reflectivity_weight_sum
                )
                totals["reflectivity_count"] += parts.stage2.reflectivity_count
                totals["reflectivity_weight_sum"] += parts.stage2.reflectivity_weight_sum
                totals["batch_count"] += 1.0

                rain_metric.update_log(
                    rain.detach(),
                    fields["target"].detach(),  # type: ignore[union-attr]
                    reliable.detach(),
                )
                _update_stage2_metrics(
                    stage2_metric,
                    support_logits=support_logits,
                    reflectivity_prediction=reflectivity,
                    target_support=fields["stage2_target_support"],  # type: ignore[arg-type]
                    target_dbz=fields["stage2_target_dbz"],  # type: ignore[arg-type]
                    support_mask=fields["stage2_support_loss_mask"],  # type: ignore[arg-type]
                    support_threshold=support_threshold,
                    dpr_mean=dpr_mean,
                    dpr_std=dpr_std,
                )
    finally:
        if not training:
            model.train(was_training)

    names = tuple(totals)
    reduced = _reduce_totals([totals[name] for name in names], resolved_device)
    totals = dict(zip(names, reduced))
    _synchronize_stage2_metric(stage2_metric, resolved_device)
    average_loss, loss_components = _summarize_c2(criterion, totals)
    world_size = (
        torch.distributed.get_world_size()
        if torch.distributed.is_available() and torch.distributed.is_initialized()
        else 1
    )
    metrics = rain_metric.compute(synchronize=True)
    metrics["stage2"] = stage2_metric.compute()
    result_type = Stage3D0EpochResult if direct_d0 else Stage3C2EpochResult
    return result_type(
        loss=average_loss,
        batch_count=int(round(totals["batch_count"])),
        optimizer_steps=int(round(totals["optimizer_steps"] / world_size)),
        rain_voxels=int(round(totals["rain_voxels"])),
        support_voxels=int(round(totals["support_count"])),
        support_positive_voxels=int(round(totals["support_positive_count"])),
        reflectivity_voxels=int(round(totals["reflectivity_count"])),
        metrics=metrics,
        duration_seconds=time.perf_counter() - started,
        loss_components=loss_components,
    )


def train_stage3_c2_one_epoch(
    model: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    optimizer: torch.optim.Optimizer,
    criterion: Stage3C2CompositeLoss,
    device: torch.device | str,
    **kwargs: Any,
) -> Stage3C2EpochResult:
    """Train the restricted Stage-2 scope through anchored cascade loss."""

    return _run_stage3_three_head_epoch(
        model,
        loader,
        criterion,
        device,
        optimizer=optimizer,
        direct_d0=False,
        **kwargs,
    )


def evaluate_stage3_c2_one_epoch(
    model: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    criterion: Stage3C2CompositeLoss,
    device: torch.device | str,
    **kwargs: Any,
) -> Stage3C2EpochResult:
    """Evaluate C2 with the same independent rain/support/dBZ masks."""

    return _run_stage3_three_head_epoch(
        model,
        loader,
        criterion,
        device,
        optimizer=None,
        scaler=None,
        grad_clip_norm=None,
        accumulation_steps=1,
        direct_d0=False,
        **kwargs,
    )


def train_stage3_d0_one_epoch(
    model: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    optimizer: torch.optim.Optimizer,
    criterion: Stage3D0CompositeLoss,
    device: torch.device | str,
    **kwargs: Any,
) -> Stage3D0EpochResult:
    """Train a deployable GR-only D0 model with three independent heads."""

    return _run_stage3_three_head_epoch(
        model,
        loader,
        criterion,
        device,
        optimizer=optimizer,
        direct_d0=True,
        **kwargs,
    )  # type: ignore[return-value]


def evaluate_stage3_d0_one_epoch(
    model: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    criterion: Stage3D0CompositeLoss,
    device: torch.device | str,
    **kwargs: Any,
) -> Stage3D0EpochResult:
    """Evaluate D0 rain plus its support/dBZ physical diagnostics."""

    return _run_stage3_three_head_epoch(
        model,
        loader,
        criterion,
        device,
        optimizer=None,
        scaler=None,
        grad_clip_norm=None,
        accumulation_steps=1,
        direct_d0=True,
        **kwargs,
    )  # type: ignore[return-value]
