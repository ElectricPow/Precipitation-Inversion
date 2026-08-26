"""Masked regression losses for zero-filled dense precipitation tensors.

Dataset targets have shape ``(B,1,D,H,Z)`` and contain neutral zeros outside
their supervision domain. These functions use the accompanying boolean mask so
halo, missing observations, clutter and padding never contribute to gradients.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F


Reduction = Literal["none", "sum", "mean"]
EmptyPolicy = Literal["zero", "raise"]


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
        weighted = element_loss * selected_weights
        denominator = torch.where(
            selected, selected_weights, torch.zeros_like(selected_weights)
        ).sum()
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
    element_loss = F.smooth_l1_loss(
        prediction, target, beta=beta, reduction="none"
    )
    return _masked_reduce(
        element_loss,
        mask,
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
    return _masked_reduce(
        torch.square(prediction - target),
        mask,
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
    return _masked_reduce(
        torch.abs(prediction - target),
        mask,
        weights=weights,
        reduction=reduction,
        empty=empty,
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
