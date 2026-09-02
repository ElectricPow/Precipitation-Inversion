"""Masked regression losses for zero-filled dense precipitation tensors.

Dataset targets have shape ``(B,1,D,H,Z)`` and contain neutral zeros outside
their supervision domain. These functions use the accompanying boolean mask so
halo, missing observations, clutter and padding never contribute to gradients.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from precipitation_inversion.metrics.regression import (
    physical_vertical_rain_gradient,
)


Reduction = Literal["none", "sum", "mean"]
EmptyPolicy = Literal["zero", "raise"]


@dataclass(frozen=True)
class Stage1LossComponents:
    """Differentiable components of the configured stage-one objective.

    ``primary`` is the historical masked Smooth-L1 in ``log1p(R)`` space and
    may use I's per-voxel intensity weights. ``physical_gradient`` is an
    unweighted Smooth-L1 over reliable adjacent-height ``dR/dz`` pairs in
    physical ``mm h^-1 km^-1``. The optimizer differentiates ``total``.
    """

    total: torch.Tensor
    primary: torch.Tensor
    physical_gradient: torch.Tensor
    physical_gradient_weight: float
    physical_gradient_pair_count: int

    @property
    def weighted_physical_gradient(self) -> torch.Tensor:
        return self.physical_gradient * self.physical_gradient_weight


def _broadcast_mask(mask: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    if mask.dtype != torch.bool:
        raise TypeError("mask must have dtype torch.bool")
    try:
        return torch.broadcast_to(mask, shape)
    except RuntimeError as error:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} cannot broadcast to {tuple(shape)}"
        ) from error


def _broadcast_weights(
    weights: torch.Tensor | None,
    *,
    reference: torch.Tensor,
) -> torch.Tensor | None:
    if weights is None:
        return None
    if not torch.is_floating_point(weights):
        weights = weights.to(dtype=reference.dtype)
    try:
        result = torch.broadcast_to(weights.to(reference.device), reference.shape)
    except RuntimeError as error:
        raise ValueError(
            f"weights shape {tuple(weights.shape)} cannot broadcast to "
            f"{tuple(reference.shape)}"
        ) from error
    return result.to(dtype=reference.dtype)


def _masked_reduce(
    element_loss: torch.Tensor,
    mask: torch.Tensor,
    *,
    weights: torch.Tensor | None,
    reduction: Reduction,
    empty: EmptyPolicy,
) -> torch.Tensor:
    if reduction not in ("none", "sum", "mean"):
        raise ValueError("reduction must be 'none', 'sum', or 'mean'")
    if empty not in ("zero", "raise"):
        raise ValueError("empty must be 'zero' or 'raise'")

    selected = _broadcast_mask(mask, element_loss.shape)
    selected_weights = _broadcast_weights(weights, reference=element_loss)
    if selected_weights is not None:
        invalid_weights = selected & (
            ~torch.isfinite(selected_weights) | (selected_weights < 0)
        )
        if bool(invalid_weights.any()):
            raise ValueError("selected weights must be finite and non-negative")
        # Neutralize ignored weights *before* multiplication. Multiplying an
        # ignored NaN by a finite loss and masking afterwards can still create
        # NaN gradients (0 * NaN) during backward.
        safe_weights = torch.where(
            selected, selected_weights, torch.zeros_like(selected_weights)
        )
        weighted = element_loss * safe_weights
        denominator = safe_weights.sum()
    else:
        weighted = element_loss
        denominator = selected.sum().to(dtype=element_loss.dtype)

    # torch.where, unlike multiplication by zero, prevents NaN values outside
    # the mask from contaminating a valid loss sum.
    masked = torch.where(selected, weighted, torch.zeros_like(weighted))
    if reduction == "none":
        return masked
    numerator = masked.sum()
    if reduction == "sum":
        return numerator
    if bool(denominator <= 0):
        if empty == "raise":
            raise ValueError("masked loss received no positive-weight elements")
        # Remain connected to the model graph, yielding exactly zero gradient.
        return numerator * 0.0
    return numerator / denominator


def _validate_prediction_target(
    prediction: torch.Tensor, target: torch.Tensor
) -> None:
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target shapes differ: {tuple(prediction.shape)} != "
            f"{tuple(target.shape)}"
        )
    if prediction.device != target.device:
        raise ValueError("prediction and target must be on the same device")
    if not torch.is_floating_point(prediction) or not torch.is_floating_point(target):
        raise TypeError("prediction and target must be floating-point tensors")


def masked_smooth_l1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    beta: float = 0.2,
    weights: torch.Tensor | None = None,
    reduction: Reduction = "mean",
    empty: EmptyPolicy = "zero",
) -> torch.Tensor:
    """Smooth-L1 loss over selected ``(B,1,D,H,Z)`` log-rain voxels.

    Optional non-negative ``weights`` support future strong-rain reweighting.
    For a weighted mean the denominator is the selected weight sum, keeping the
    loss scale stable when the number or class of valid voxels changes.
    """

    _validate_prediction_target(prediction, target)
    if beta <= 0:
        raise ValueError("beta must be positive")
    selected = _broadcast_mask(mask, prediction.shape)
    # Missing/padded targets may contain arbitrary values. Neutralize both
    # operands before constructing element-wise loss so ignored NaN/Inf never
    # appears in the autograd graph.
    safe_prediction = torch.where(selected, prediction, torch.zeros_like(prediction))
    safe_target = torch.where(selected, target, torch.zeros_like(target))
    element_loss = F.smooth_l1_loss(
        safe_prediction, safe_target, beta=beta, reduction="none"
    )
    return _masked_reduce(
        element_loss,
        selected,
        weights=weights,
        reduction=reduction,
        empty=empty,
    )


def masked_mse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
    reduction: Reduction = "mean",
    empty: EmptyPolicy = "zero",
) -> torch.Tensor:
    """Squared error reduced only over selected voxels."""

    _validate_prediction_target(prediction, target)
    selected = _broadcast_mask(mask, prediction.shape)
    safe_prediction = torch.where(selected, prediction, torch.zeros_like(prediction))
    safe_target = torch.where(selected, target, torch.zeros_like(target))
    return _masked_reduce(
        torch.square(safe_prediction - safe_target),
        selected,
        weights=weights,
        reduction=reduction,
        empty=empty,
    )


def masked_mae_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
    reduction: Reduction = "mean",
    empty: EmptyPolicy = "zero",
) -> torch.Tensor:
    """Absolute error reduced only over selected voxels."""

    _validate_prediction_target(prediction, target)
    selected = _broadcast_mask(mask, prediction.shape)
    safe_prediction = torch.where(selected, prediction, torch.zeros_like(prediction))
    safe_target = torch.where(selected, target, torch.zeros_like(target))
    return _masked_reduce(
        torch.abs(safe_prediction - safe_target),
        selected,
        weights=weights,
        reduction=reduction,
        empty=empty,
    )


def _physical_gradient_loss_and_count(
    prediction_log: torch.Tensor,
    target_log: torch.Tensor,
    reliable_mask: torch.Tensor,
    *,
    height_km: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, int]:
    """Return physical ``dR/dz`` Smooth-L1 and its reliable pair count.

    Shapes change as follows::

        prediction_log/target_log/reliable_mask: (B,1,D,H,Z)
        prediction_drdz/target_drdz/pair_mask:    (B,1,D,H,Z-1)

    Only endpoints belonging to at least one reliable adjacent pair are sent
    through ``expm1``. This is mathematically identical on the selected
    support, while preventing arbitrary values in padding or missing cells
    from overflowing before the mask is applied. Physical conversion is kept
    in float32 even when the U-Net forward pass uses CUDA AMP.
    """

    _validate_prediction_target(prediction_log, target_log)
    if reliable_mask.dtype != torch.bool:
        raise TypeError("reliable_mask must have dtype torch.bool")
    if reliable_mask.shape != prediction_log.shape:
        raise ValueError(
            "reliable_mask and prediction shapes differ: "
            f"{tuple(reliable_mask.shape)} != {tuple(prediction_log.shape)}"
        )
    if reliable_mask.device != prediction_log.device:
        raise ValueError("reliable_mask and prediction must be on the same device")
    if not isinstance(height_km, torch.Tensor):
        raise TypeError("height_km must be a tensor")
    if beta <= 0 or not math.isfinite(beta):
        raise ValueError("physical-gradient beta must be finite and positive")

    pair_mask = reliable_mask[..., 1:] & reliable_mask[..., :-1]
    endpoint_mask = torch.zeros_like(reliable_mask)
    endpoint_mask[..., :-1] |= pair_mask
    endpoint_mask[..., 1:] |= pair_mask

    # Disable autocast explicitly: expm1 in float16 can overflow at ordinary
    # heavy-rain log values. The cast remains differentiable back to the model.
    with torch.autocast(device_type=prediction_log.device.type, enabled=False):
        prediction_float = prediction_log.float()
        target_float = target_log.float()
        zero = torch.zeros((), dtype=torch.float32, device=prediction_log.device)
        safe_prediction_log = torch.where(
            endpoint_mask, prediction_float.clamp_min(0.0), zero
        )
        safe_target_log = torch.where(
            endpoint_mask, target_float.clamp_min(0.0), zero
        )
        prediction_rain = torch.expm1(safe_prediction_log)
        target_rain = torch.expm1(safe_target_log)
        prediction_drdz, target_drdz, shared_pair_mask, _ = (
            physical_vertical_rain_gradient(
                prediction_rain,
                target_rain,
                reliable_mask,
                height_km=height_km,
            )
        )
        # The shared metric helper derives the same endpoint conjunction. Keep
        # this assertion as a guard against future metric/loss protocol drift.
        if not torch.equal(shared_pair_mask, pair_mask):
            raise RuntimeError("physical dR/dz metric and loss pair masks differ")
        value = masked_smooth_l1_loss(
            prediction_drdz,
            target_drdz,
            pair_mask,
            beta=beta,
            # Deliberately no intensity/height/CFB weights: G is a single
            # auxiliary factor on top of I, not a second strong-rain weighting.
            weights=None,
        )
    return value, int(pair_mask.sum().item())


def masked_physical_gradient_smooth_l1_loss(
    prediction_log: torch.Tensor,
    target_log: torch.Tensor,
    reliable_mask: torch.Tensor,
    *,
    height_km: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    """Smooth-L1 of physical vertical rain gradients on reliable pairs.

    ``prediction_log`` and ``target_log`` are ``log1p(mm/h)`` volumes. The
    physical conversion and upward adjacent-height difference exactly match
    :class:`~precipitation_inversion.metrics.regression.PhysicalRainGradientMetrics`.
    """

    value, _ = _physical_gradient_loss_and_count(
        prediction_log,
        target_log,
        reliable_mask,
        height_km=height_km,
        beta=beta,
    )
    return value


class Stage1CompositeLoss(nn.Module):
    """Historical I objective plus an optional physical ``dR/dz`` auxiliary.

    The primary term is always the existing masked Smooth-L1 over log-rain and
    continues to consume Dataset ``loss_weights``. When enabled, G uses only
    ``reliable_mask`` and real ``height_km``; it cannot consume weak-CFB labels
    or I's intensity weights.
    """

    def __init__(
        self,
        *,
        primary_beta: float = 0.2,
        physical_gradient_weight: float = 0.0,
        physical_gradient_beta: float = 1.0,
    ) -> None:
        super().__init__()
        if primary_beta <= 0 or not math.isfinite(primary_beta):
            raise ValueError("primary beta must be finite and positive")
        if (
            physical_gradient_weight < 0
            or not math.isfinite(physical_gradient_weight)
        ):
            raise ValueError(
                "physical-gradient weight must be finite and non-negative"
            )
        if physical_gradient_beta <= 0 or not math.isfinite(
            physical_gradient_beta
        ):
            raise ValueError("physical-gradient beta must be finite and positive")
        self.primary_beta = float(primary_beta)
        self.physical_gradient_weight = float(physical_gradient_weight)
        self.physical_gradient_beta = float(physical_gradient_beta)

    @property
    def physical_gradient_enabled(self) -> bool:
        return self.physical_gradient_weight > 0.0

    def compute_components(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        weights: torch.Tensor | None = None,
        *,
        reliable_mask: torch.Tensor | None = None,
        height_km: torch.Tensor | None = None,
    ) -> Stage1LossComponents:
        primary = masked_smooth_l1_loss(
            prediction,
            target,
            mask,
            beta=self.primary_beta,
            weights=weights,
        )
        if self.physical_gradient_enabled:
            if reliable_mask is None:
                raise KeyError(
                    "physical-gradient loss requires reliable_loss_mask"
                )
            if height_km is None:
                raise KeyError("physical-gradient loss requires height_km")
            physical_gradient, pair_count = _physical_gradient_loss_and_count(
                prediction,
                target,
                reliable_mask,
                height_km=height_km,
                beta=self.physical_gradient_beta,
            )
        else:
            # Stay connected to the prediction graph while adding exactly zero.
            physical_gradient = prediction.sum() * 0.0
            pair_count = 0
        total = primary + self.physical_gradient_weight * physical_gradient
        return Stage1LossComponents(
            total=total,
            primary=primary,
            physical_gradient=physical_gradient,
            physical_gradient_weight=self.physical_gradient_weight,
            physical_gradient_pair_count=pair_count,
        )

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        weights: torch.Tensor | None = None,
        *,
        reliable_mask: torch.Tensor | None = None,
        height_km: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.compute_components(
            prediction,
            target,
            mask,
            weights,
            reliable_mask=reliable_mask,
            height_km=height_km,
        ).total


def build_stage1_loss(loss_config: Mapping[str, Any]) -> Stage1CompositeLoss:
    """Build the same stage-one criterion for training and evaluation.

    Legacy configurations without ``loss.physical_gradient`` resolve to the
    historical objective with weight zero. An enabled block must declare a
    strictly positive weight, preventing a configuration from claiming to run
    G while silently optimizing the baseline.
    """

    if str(loss_config.get("name", "")).lower() != "masked_smooth_l1":
        raise ValueError("only masked_smooth_l1 is currently supported")
    physical = loss_config.get("physical_gradient", {})
    if not isinstance(physical, Mapping):
        raise TypeError("loss.physical_gradient must be a mapping")
    enabled = physical.get("enabled", False)
    if not isinstance(enabled, bool):
        raise TypeError("loss.physical_gradient.enabled must be boolean")
    weight = float(physical.get("weight", 0.0))
    beta = float(physical.get("beta", 1.0))
    if enabled and weight <= 0.0:
        raise ValueError(
            "enabled physical-gradient loss requires a positive weight"
        )
    if not enabled and weight != 0.0:
        raise ValueError(
            "disabled physical-gradient loss must have weight zero or omit it"
        )
    return Stage1CompositeLoss(
        primary_beta=float(loss_config["beta"]),
        physical_gradient_weight=weight if enabled else 0.0,
        physical_gradient_beta=beta,
    )


class MaskedSmoothL1Loss(nn.Module):
    """``nn.Module`` wrapper for the stage-one primary objective."""

    def __init__(
        self,
        *,
        beta: float = 0.2,
        reduction: Reduction = "mean",
        empty: EmptyPolicy = "zero",
    ) -> None:
        super().__init__()
        if beta <= 0:
            raise ValueError("beta must be positive")
        self.beta = float(beta)
        self.reduction = reduction
        self.empty = empty

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return masked_smooth_l1_loss(
            prediction,
            target,
            mask,
            beta=self.beta,
            weights=weights,
            reduction=self.reduction,
            empty=self.empty,
        )
