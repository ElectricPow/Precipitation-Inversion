"""Reusable AMP/DDP-aware training and validation loops.

Model inputs have shape ``(B,C,64,64,60)`` and outputs/targets/masks have
shape ``(B,1,64,64,60)``. ``C`` is three for the baseline and four for the
signed-CFB-distance diagnostic ablation. The engine never flattens spatial
dimensions before the model; flattening occurs only inside masked
loss/metric reductions.
"""

from __future__ import annotations

import math
import os
import random
import tempfile
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import torch.distributed as distributed
from torch import nn

from precipitation_inversion.losses.masked_losses import (
    Stage1CompositeLoss,
    Stage1LossComponents,
)
from precipitation_inversion.metrics.classification import (
    MulticlassConfusionMetrics,
)
from precipitation_inversion.metrics.regression import (
    FilewisePrecipitationMetrics,
    PhysicalRainGradientMetrics,
    PrecipitationRegressionMetrics,
    StratifiedPrecipitationMetrics,
)
from precipitation_inversion.models.multitask_unet3d import (
    rain_prediction_from_output,
)
from precipitation_inversion.data.precipitation_type import TYPE_NAMES


DEFAULT_CHECKPOINT_EVERY = 10
"""Default interval for immutable ``epoch_XXXX.pt`` snapshots."""


def validate_checkpoint_every(value: Any = DEFAULT_CHECKPOINT_EVERY) -> int:
    """Validate the disk-safe periodic-checkpoint interval.

    ``0`` explicitly disables immutable periodic snapshots.  A positive
    interval must be at least ten epochs so a missing or accidentally small
    configuration value cannot fill the training account with one checkpoint
    per epoch.  ``best*.pt`` and ``last.pt`` are managed separately and are
    therefore unaffected by this setting.
    """

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("checkpoint_every must be an integer")
    if value < 0 or 0 < value < DEFAULT_CHECKPOINT_EVERY:
        raise ValueError(
            "checkpoint_every must be 0 (disabled) or at least "
            f"{DEFAULT_CHECKPOINT_EVERY}"
        )
    return value


def should_save_periodic_checkpoint(
    epoch: int,
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
) -> bool:
    """Return whether a zero-based epoch needs an immutable snapshot.

    With the default interval, zero-based epochs 9, 19, 29, ... produce
    ``epoch_0009.pt``, ``epoch_0019.pt``, ``epoch_0029.pt``, ... .
    """

    if isinstance(epoch, bool) or not isinstance(epoch, int):
        raise TypeError("epoch must be an integer")
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    interval = validate_checkpoint_every(checkpoint_every)
    return interval > 0 and (epoch + 1) % interval == 0


def validate_training_output_directory(
    output_dir: str | Path,
    *,
    resume: str | Path | None = None,
    initialize_from: str | Path | None = None,
) -> Path:
    """Reject an accidental epoch-zero restart in an existing run directory.

    ``metrics.jsonl`` is the durable marker that training has already written
    epoch history.  Only ``--resume`` may append to such a directory.  A
    weights-only ``--initialize-from`` run deliberately starts at epoch zero,
    so it must use a new directory just like any other fresh experiment.

    This read-only check is designed to run before DDP, Dataset, and model
    initialization, making a mistaken command fail quickly on every rank.
    """

    destination = Path(output_dir).expanduser().resolve()
    history = destination / "metrics.jsonl"
    if not history.exists() or resume is not None:
        return destination
    if initialize_from is not None:
        raise FileExistsError(
            "weights-only initialization requires a new output directory: "
            f"{destination}; use --resume to continue the existing run"
        )
    raise FileExistsError(
        "refusing to start from epoch 0 in an output directory that already "
        f"contains metrics.jsonl: {destination}; pass --resume to continue or "
        "choose a new --output-dir"
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
    loss_components: dict[str, Any] = field(default_factory=dict)

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


def _loss_weights(batch: Mapping[str, Any], mask: torch.Tensor) -> torch.Tensor | None:
    """Return optional voxel weights after validating their tensor contract.

    Dataset ``loss_weights`` has the same ``(B,1,D,H,Z)`` shape as the loss
    mask.  A missing field deliberately means the historical unweighted
    objective, which keeps old datasets and tests backward compatible.
    """

    value = batch.get("loss_weights")
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        raise TypeError("batch loss_weights must be a tensor")
    if value.shape != mask.shape:
        raise ValueError(
            "loss_weights and loss_mask shapes differ: "
            f"{tuple(value.shape)} != {tuple(mask.shape)}"
        )
    if not torch.is_floating_point(value):
        raise TypeError("loss_weights must be floating-point")
    selected = value[mask]
    if selected.numel() and bool(
        (~torch.isfinite(selected) | (selected < 0)).any()
    ):
        raise ValueError("selected loss weights must be finite and non-negative")
    return value


def _loss_normalizer(mask: torch.Tensor, weights: torch.Tensor | None) -> float:
    """Denominator used by the criterion's masked weighted mean."""

    if weights is None:
        return float(mask.sum().item())
    return float(torch.where(mask, weights, torch.zeros_like(weights)).sum().item())


def _metric_mask(batch: Mapping[str, Any], loss_mask: torch.Tensor) -> torch.Tensor:
    """Use only high-confidence supervision for model selection metrics.

    Weak CFB labels may contribute a small training loss, but they must not
    change the primary validation support.  New datasets expose
    ``reliable_loss_mask`` for that purpose; historical batches fall back to
    their sole ``loss_mask``.
    """

    value = batch.get("reliable_loss_mask", loss_mask)
    if not isinstance(value, torch.Tensor) or value.dtype != torch.bool:
        raise TypeError("reliable_loss_mask must be a boolean tensor")
    if value.shape != loss_mask.shape:
        raise ValueError("reliable_loss_mask and loss_mask shapes differ")
    if bool((value & ~loss_mask).any()):
        raise ValueError("reliable_loss_mask must be a subset of loss_mask")
    return value


def _compute_loss_components(
    criterion: nn.Module,
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor | None,
    *,
    reliable_mask: torch.Tensor,
    height_km: Any | None,
) -> Stage1LossComponents:
    """Evaluate a legacy or composite objective through one engine contract."""

    if isinstance(criterion, Stage1CompositeLoss):
        return criterion.compute_components(
            prediction,
            target,
            mask,
            weights,
            reliable_mask=reliable_mask,
            height_km=height_km,
        )
    total = criterion(prediction, target, mask, weights)
    if not isinstance(total, torch.Tensor):
        raise TypeError("criterion must return a tensor")
    return Stage1LossComponents(
        total=total,
        primary=total,
        physical_gradient=total * 0.0,
        physical_gradient_weight=0.0,
        physical_gradient_pair_count=0,
    )


def _split_predictions(output: Any) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return rain ``(B,1,D,H,Z)`` and optional type logits ``(B,3,D,H)``."""

    rain = rain_prediction_from_output(output)
    type_logits = output.get("type_logits") if isinstance(output, Mapping) else None
    if type_logits is not None and not isinstance(type_logits, torch.Tensor):
        raise TypeError("model type_logits must be a tensor")
    return rain, type_logits


def _type_batch_fields(
    batch: Mapping[str, Any], logits: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor]:
    if logits is None:
        raise ValueError("type supervision was requested but model returned no type_logits")
    target = batch.get("type_target")
    mask = batch.get("type_loss_mask")
    if not isinstance(target, torch.Tensor) or not isinstance(mask, torch.Tensor):
        raise KeyError("multitask batch requires type_target and type_loss_mask")
    if target.dtype != torch.long or mask.dtype != torch.bool:
        raise TypeError("type_target must be int64 and type_loss_mask boolean")
    if target.shape != mask.shape or logits.shape[0] != target.shape[0] or logits.shape[2:] != target.shape[1:]:
        raise ValueError("type logits/target/mask shapes are inconsistent")
    return target, mask


def _finish_auxiliary_loss(
    rain_loss: float,
    components: dict[str, Any],
    *,
    classification_sum: float,
    classification_normalizer: float,
    classification_weight: float,
) -> tuple[float, dict[str, Any]]:
    if classification_weight <= 0.0:
        return rain_loss, components
    classification = (
        classification_sum / classification_normalizer
        if classification_normalizer > 0.0
        else 0.0
    )
    weighted = classification_weight * classification
    total = rain_loss + weighted
    updated = dict(components)
    updated.update(
        {
            "type_cross_entropy": classification,
            "type_valid_profiles": int(round(classification_normalizer)),
            "type_loss_weight": classification_weight,
            "weighted_type_cross_entropy": weighted,
            "weighted_type_fraction": weighted / total if total else float("nan"),
            "total": total,
        }
    )
    return total, updated


def _summarize_loss_components(
    *,
    primary_sum: float,
    primary_normalizer: float,
    physical_gradient_sum: float,
    physical_gradient_pair_count: float,
    physical_gradient_weight: float,
) -> tuple[float, dict[str, Any]]:
    """Combine globally reduced I/G statistics without mixing denominators."""

    primary = (
        primary_sum / primary_normalizer
        if primary_normalizer > 0.0
        else float("nan")
    )
    gradient = (
        physical_gradient_sum / physical_gradient_pair_count
        if physical_gradient_pair_count > 0.0
        else 0.0
    )
    weighted_gradient = physical_gradient_weight * gradient
    total = primary + weighted_gradient
    components = {
        "primary_log_smooth_l1": primary,
        "primary_weight_sum": primary_normalizer,
        "physical_drdz_enabled": physical_gradient_weight > 0.0,
        "physical_drdz_smooth_l1": gradient,
        "physical_drdz_pair_count": int(round(physical_gradient_pair_count)),
        "physical_drdz_weight": physical_gradient_weight,
        "weighted_physical_drdz": weighted_gradient,
        "weighted_physical_drdz_fraction": (
            weighted_gradient / total
            if math.isfinite(total) and total != 0.0
            else float("nan")
        ),
        "total": total,
        "aggregation": (
            "primary is reduced by selected voxel-weight sum; physical dR/dz "
            "is reduced by reliable adjacent-pair count"
        ),
    }
    return total, components


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
    type_criterion: nn.Module | None = None,
    type_loss_weight: float = 0.0,
) -> EpochResult:
    """Train one epoch and report voxel-weighted loss and masked metrics."""

    if accumulation_steps <= 0:
        raise ValueError("accumulation_steps must be positive")
    if grad_clip_norm is not None and grad_clip_norm <= 0:
        raise ValueError("grad_clip_norm must be positive or None")
    if type_loss_weight < 0.0 or not math.isfinite(type_loss_weight):
        raise ValueError("type_loss_weight must be finite and non-negative")
    if (type_criterion is None) != (type_loss_weight == 0.0):
        raise ValueError("type criterion and positive type_loss_weight must be enabled together")
    resolved_device = torch.device(device)
    total_batches = _batch_limit(loader, max_batches)
    if accumulation_steps > 1 and total_batches is None:
        raise TypeError("gradient accumulation requires a loader with __len__")

    model.train()
    metrics = PrecipitationRegressionMetrics(thresholds_mm_h)
    type_metrics = MulticlassConfusionMetrics(TYPE_NAMES) if type_criterion else None
    primary_loss_sum = 0.0
    primary_loss_denominator = 0.0
    physical_gradient_loss_sum = 0.0
    physical_gradient_pair_count = 0.0
    physical_gradient_weight = float(
        getattr(criterion, "physical_gradient_weight", 0.0)
    )
    classification_loss_sum = 0.0
    classification_loss_denominator = 0.0
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
        # inputs=(B,C,D,H,Z), target/loss_mask=(B,1,D,H,Z).
        inputs = batch["inputs"]
        target = batch["target"]
        mask = batch["loss_mask"]
        weights = _loss_weights(batch, mask)
        metric_mask = _metric_mask(batch, mask)
        current_valid = int(mask.sum().item())
        current_normalizer = _loss_normalizer(mask, weights)
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
                model_output = model(inputs)
                prediction, type_logits = _split_predictions(model_output)
                loss_parts = _compute_loss_components(
                    criterion,
                    prediction,
                    target,
                    mask,
                    weights,
                    reliable_mask=metric_mask,
                    height_km=batch.get("height_km"),
                )
                if type_criterion is not None:
                    type_target, type_mask = _type_batch_fields(batch, type_logits)
                    type_loss = type_criterion(type_logits, type_target, type_mask)
                else:
                    type_target = type_mask = None
                    type_loss = loss_parts.total * 0.0
                loss = loss_parts.total + type_loss_weight * type_loss
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

        primary_loss_sum += (
            float(loss_parts.primary.detach().item()) * current_normalizer
        )
        primary_loss_denominator += current_normalizer
        physical_gradient_loss_sum += (
            float(loss_parts.physical_gradient.detach().item())
            * loss_parts.physical_gradient_pair_count
        )
        physical_gradient_pair_count += loss_parts.physical_gradient_pair_count
        if type_criterion is not None:
            current_type_count = int(type_mask.sum().item())
            classification_loss_sum += float(type_loss.detach().item()) * current_type_count
            classification_loss_denominator += current_type_count
            assert type_metrics is not None and type_logits is not None
            type_metrics.update(type_logits.detach(), type_target.detach(), type_mask.detach())
        valid_voxels += current_valid
        batch_count += 1
        metrics.update_log(
            prediction.detach(), target.detach(), metric_mask.detach()
        )

    (
        primary_loss_sum,
        primary_loss_denominator,
        physical_gradient_loss_sum,
        physical_gradient_pair_count,
        valid_value,
        batches_value,
        steps_value,
        classification_loss_sum,
        classification_loss_denominator,
    ) = _reduce_totals(
        [
            primary_loss_sum,
            primary_loss_denominator,
            physical_gradient_loss_sum,
            physical_gradient_pair_count,
            float(valid_voxels),
            float(batch_count),
            float(optimizer_steps),
            classification_loss_sum,
            classification_loss_denominator,
        ],
        resolved_device,
    )
    valid_voxels = int(round(valid_value))
    average_loss, loss_components = _summarize_loss_components(
        primary_sum=primary_loss_sum,
        primary_normalizer=primary_loss_denominator,
        physical_gradient_sum=physical_gradient_loss_sum,
        physical_gradient_pair_count=physical_gradient_pair_count,
        physical_gradient_weight=physical_gradient_weight,
    )
    average_loss, loss_components = _finish_auxiliary_loss(
        average_loss,
        loss_components,
        classification_sum=classification_loss_sum,
        classification_normalizer=classification_loss_denominator,
        classification_weight=type_loss_weight,
    )
    # All DDP ranks execute the same number of optimizer steps. The reduction
    # above sums them only for transport, so report the per-rank step count.
    world_size = distributed.get_world_size() if _distributed_active() else 1
    computed_metrics = metrics.compute(synchronize=True)
    if type_metrics is not None:
        computed_metrics["precipitation_type"] = type_metrics.compute(synchronize=True)
    return EpochResult(
        loss=average_loss,
        batch_count=int(round(batches_value)),
        optimizer_steps=int(round(steps_value / world_size)),
        valid_voxels=valid_voxels,
        metrics=computed_metrics,
        duration_seconds=time.perf_counter() - started,
        loss_components=loss_components,
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
    stratified_metrics: StratifiedPrecipitationMetrics | None = None,
    filewise_metrics: FilewisePrecipitationMetrics | None = None,
    physical_gradient_metrics: PhysicalRainGradientMetrics | None = None,
    type_criterion: nn.Module | None = None,
    type_loss_weight: float = 0.0,
) -> EpochResult:
    """Evaluate patches without gradients and restore the prior model mode.

    When diagnostic accumulators are supplied, the Dataset must also return
    their requested height/CFB/type/file metadata. These tensors are
    evaluation-only and never enter the model. Physical ``dR/dz`` uses the
    reliable primary metric mask, never optional weak-CFB supervision.

    DDP validation may assign a different number of unique batches to each
    rank. Forward calls therefore use the already-synchronized wrapped module
    directly rather than the DDP wrapper; no forward collective remains, and
    a short rank can safely wait at the final metric reductions. This permits
    complete, non-duplicated validation coverage.
    """

    resolved_device = torch.device(device)
    if type_loss_weight < 0.0 or not math.isfinite(type_loss_weight):
        raise ValueError("type_loss_weight must be finite and non-negative")
    if (type_criterion is None) != (type_loss_weight == 0.0):
        raise ValueError("type criterion and positive type_loss_weight must be enabled together")
    total_batches = _batch_limit(loader, max_batches)
    was_training = model.training
    model.eval()
    inference_model = _unwrapped_model(model)
    metrics = PrecipitationRegressionMetrics(thresholds_mm_h)
    type_metrics = MulticlassConfusionMetrics(TYPE_NAMES) if type_criterion else None
    below_cfb_metrics = PrecipitationRegressionMetrics(thresholds_mm_h)
    below_cfb_diagnostic_batches = 0
    if stratified_metrics is not None:
        stratified_metrics.reset()
    if filewise_metrics is not None:
        filewise_metrics.reset()
    if physical_gradient_metrics is not None:
        physical_gradient_metrics.reset()
    primary_loss_sum = 0.0
    primary_loss_denominator = 0.0
    physical_gradient_loss_sum = 0.0
    physical_gradient_pair_count = 0.0
    physical_gradient_weight = float(
        getattr(criterion, "physical_gradient_weight", 0.0)
    )
    classification_loss_sum = 0.0
    classification_loss_denominator = 0.0
    valid_voxels = 0
    batch_count = 0
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            for batch_index, source_batch in enumerate(loader):
                if total_batches is not None and batch_index >= total_batches:
                    break
                batch = move_batch_to_device(source_batch, resolved_device)
                # No shape change on transfer: inputs=(B,C,D,H,Z).
                inputs = batch["inputs"]
                target = batch["target"]
                mask = batch["loss_mask"]
                weights = _loss_weights(batch, mask)
                metric_mask = _metric_mask(batch, mask)
                with _autocast(resolved_device, use_amp):
                    model_output = inference_model(inputs)
                    prediction, type_logits = _split_predictions(model_output)
                    loss_parts = _compute_loss_components(
                        criterion,
                        prediction,
                        target,
                        mask,
                        weights,
                        reliable_mask=metric_mask,
                        height_km=batch.get("height_km"),
                    )
                    if type_criterion is not None:
                        type_target, type_mask = _type_batch_fields(batch, type_logits)
                        type_loss = type_criterion(type_logits, type_target, type_mask)
                    else:
                        type_target = type_mask = None
                        type_loss = loss_parts.total * 0.0
                    loss = loss_parts.total + type_loss_weight * type_loss
                if loss.ndim != 0 or not bool(torch.isfinite(loss)):
                    raise FloatingPointError("validation loss must be a finite scalar")
                current_valid = int(mask.sum().item())
                current_normalizer = _loss_normalizer(mask, weights)
                primary_loss_sum += (
                    float(loss_parts.primary.item()) * current_normalizer
                )
                primary_loss_denominator += current_normalizer
                physical_gradient_loss_sum += (
                    float(loss_parts.physical_gradient.item())
                    * loss_parts.physical_gradient_pair_count
                )
                physical_gradient_pair_count += (
                    loss_parts.physical_gradient_pair_count
                )
                if type_criterion is not None:
                    current_type_count = int(type_mask.sum().item())
                    classification_loss_sum += float(type_loss.item()) * current_type_count
                    classification_loss_denominator += current_type_count
                    assert type_metrics is not None and type_logits is not None
                    type_metrics.update(type_logits, type_target, type_mask)
                valid_voxels += current_valid
                batch_count += 1
                metrics.update_log(prediction, target, metric_mask)
                diagnostic_names = (
                    "diagnostic_target",
                    "native_positive_mask",
                    "cfb_distance_km",
                )
                diagnostic_present = [name in batch for name in diagnostic_names]
                if any(diagnostic_present):
                    if not all(diagnostic_present):
                        missing = [
                            name
                            for name, present in zip(
                                diagnostic_names, diagnostic_present
                            )
                            if not present
                        ]
                        raise KeyError(
                            "below-CFB diagnostic batch is missing "
                            + ", ".join(missing)
                        )
                    diagnostic_target = batch["diagnostic_target"]
                    native_positive = batch["native_positive_mask"]
                    cfb_distance = batch["cfb_distance_km"]
                    if (
                        not isinstance(diagnostic_target, torch.Tensor)
                        or diagnostic_target.shape != prediction.shape
                    ):
                        raise ValueError(
                            "diagnostic_target must match prediction shape"
                        )
                    if (
                        not isinstance(native_positive, torch.Tensor)
                        or native_positive.dtype != torch.bool
                        or native_positive.shape != prediction.shape
                    ):
                        raise ValueError(
                            "native_positive_mask must be boolean and match "
                            "prediction shape"
                        )
                    if (
                        not isinstance(cfb_distance, torch.Tensor)
                        or cfb_distance.shape != prediction.shape
                    ):
                        raise ValueError(
                            "cfb_distance_km must match prediction shape"
                        )
                    # This support is intentionally independent from loss_mask:
                    # native positive labels below CFB are diagnostic-only and
                    # cannot affect the primary metric or checkpoint selection.
                    below_mask = (
                        native_positive
                        & torch.isfinite(cfb_distance)
                        & (cfb_distance < 0)
                    )
                    below_cfb_metrics.update_log(
                        prediction, diagnostic_target, below_mask
                    )
                    below_cfb_diagnostic_batches += 1
                if stratified_metrics is not None:
                    missing = [
                        name
                        for name in (
                            "height_km",
                            "cfb_distance_km",
                            "precipitation_type",
                        )
                        if name not in batch
                    ]
                    if missing:
                        raise KeyError(
                            "stratified evaluation batch is missing "
                            + ", ".join(missing)
                        )
                    stratified_metrics.update_log(
                        prediction,
                        target,
                        metric_mask,
                        height_km=batch["height_km"],
                        cfb_distance_km=batch["cfb_distance_km"],
                        precipitation_type=batch["precipitation_type"],
                    )
                if filewise_metrics is not None:
                    if "file_id" not in batch:
                        raise KeyError(
                            "filewise evaluation batch is missing file_id"
                        )
                    filewise_metrics.update_log(
                        prediction,
                        target,
                        metric_mask,
                        file_id=batch["file_id"],
                    )
                if physical_gradient_metrics is not None:
                    if "height_km" not in batch:
                        raise KeyError(
                            "physical dR/dz evaluation batch is missing height_km"
                        )
                    physical_gradient_metrics.update_log(
                        prediction,
                        target,
                        metric_mask,
                        height_km=batch["height_km"],
                        cfb_distance_km=batch.get("cfb_distance_km"),
                        precipitation_type=batch.get("precipitation_type"),
                        file_id=batch.get("file_id"),
                    )
    finally:
        model.train(was_training)

    (
        primary_loss_sum,
        primary_loss_denominator,
        physical_gradient_loss_sum,
        physical_gradient_pair_count,
        valid_value,
        batches_value,
        diagnostic_batches_value,
        classification_loss_sum,
        classification_loss_denominator,
    ) = _reduce_totals(
        [
            primary_loss_sum,
            primary_loss_denominator,
            physical_gradient_loss_sum,
            physical_gradient_pair_count,
            float(valid_voxels),
            float(batch_count),
            float(below_cfb_diagnostic_batches),
            classification_loss_sum,
            classification_loss_denominator,
        ],
        resolved_device,
    )
    valid_voxels = int(round(valid_value))
    computed_metrics = metrics.compute(synchronize=True)
    if type_metrics is not None:
        computed_metrics["precipitation_type"] = type_metrics.compute(synchronize=True)
    if stratified_metrics is not None:
        computed_metrics["stratified"] = stratified_metrics.compute(
            synchronize=True
        )
    # Every rank executes this collective even if its uneven shard happened to
    # contain no batch. Only expose the result when diagnostic fields existed
    # somewhere in the globally evaluated Dataset.
    below_cfb_result = below_cfb_metrics.compute(synchronize=True)
    if diagnostic_batches_value > 0:
        computed_metrics["diagnostics"] = {
            "below_cfb_native_positive": {
                "definition": (
                    "native positive DPR precipitation where signed CFB "
                    "distance is negative; excluded from primary metrics and "
                    "model selection"
                ),
                **below_cfb_result,
            }
        }
    if filewise_metrics is not None:
        computed_metrics["filewise"] = filewise_metrics.compute(
            synchronize=True
        )
    if physical_gradient_metrics is not None:
        computed_metrics["physical_drdz"] = physical_gradient_metrics.compute(
            synchronize=True
        )
    average_loss, loss_components = _summarize_loss_components(
        primary_sum=primary_loss_sum,
        primary_normalizer=primary_loss_denominator,
        physical_gradient_sum=physical_gradient_loss_sum,
        physical_gradient_pair_count=physical_gradient_pair_count,
        physical_gradient_weight=physical_gradient_weight,
    )
    average_loss, loss_components = _finish_auxiliary_loss(
        average_loss,
        loss_components,
        classification_sum=classification_loss_sum,
        classification_normalizer=classification_loss_denominator,
        classification_weight=type_loss_weight,
    )
    return EpochResult(
        loss=average_loss,
        batch_count=int(round(batches_value)),
        optimizer_steps=0,
        valid_voxels=valid_voxels,
        metrics=computed_metrics,
        duration_seconds=time.perf_counter() - started,
        loss_components=loss_components,
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
    extra: Mapping[str, Any] | None = None,
) -> None:
    """Atomically save model/training state without a DDP ``module.`` prefix.

    ``extra`` is reserved for auditable experiment metadata such as frozen
    upstream checkpoint hashes. It may not overwrite the stable checkpoint
    schema used by training and evaluation.
    """

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
    if extra is not None:
        overlap = sorted(set(extra).intersection(payload))
        if overlap:
            raise ValueError(
                "extra checkpoint metadata overlaps reserved keys: "
                + ", ".join(overlap)
            )
        payload.update(dict(extra))
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
