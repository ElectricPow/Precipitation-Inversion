"""Single masked dBZ objective for the R1-O spatial-completion control."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn

from .masked_losses import masked_smooth_l1_loss


def _selected_mask(mask: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
        raise TypeError("R1-O regression_mask must be a boolean tensor")
    try:
        return torch.broadcast_to(mask, shape)
    except RuntimeError as error:
        raise ValueError(
            f"regression_mask shape {tuple(mask.shape)} cannot broadcast to "
            f"{tuple(shape)}"
        ) from error


@dataclass(frozen=True)
class Stage2CompletionLossComponents:
    """Differentiable loss and exact global-aggregation denominators."""

    total: torch.Tensor
    reflectivity: torch.Tensor
    reflectivity_count: int
    reflectivity_weight_sum: float


class Stage2CompletionLoss(nn.Module):
    """Weighted Smooth-L1 on true DPR support only.

    ``prediction`` and ``target_dbz`` are standardized tensors shaped
    ``(B,1,D,H,Z)``.  Physical-dBZ intensity weights are optional Dataset-side
    metadata and are never included in the R1-O model input.
    """

    def __init__(self, *, beta: float = 0.2) -> None:
        super().__init__()
        if not math.isfinite(beta) or beta <= 0.0:
            raise ValueError("R1-O reflectivity beta must be finite and positive")
        self.beta = float(beta)

    def compute_components(
        self,
        prediction: torch.Tensor,
        target_dbz: torch.Tensor,
        regression_mask: torch.Tensor,
        *,
        regression_weights: torch.Tensor | None = None,
    ) -> Stage2CompletionLossComponents:
        if prediction.shape != target_dbz.shape:
            raise ValueError("R1-O prediction and target_dbz shapes differ")
        if prediction.ndim != 5 or prediction.shape[1] != 1:
            raise ValueError("R1-O prediction must have shape (B,1,D,H,Z)")
        if not torch.is_floating_point(prediction) or not torch.is_floating_point(
            target_dbz
        ):
            raise TypeError("R1-O prediction and target_dbz must be floating-point")
        selected = _selected_mask(regression_mask, prediction.shape)
        selected_target = target_dbz[selected]
        if selected_target.numel() and not bool(torch.isfinite(selected_target).all()):
            raise ValueError("selected R1-O dBZ targets must be finite")

        loss = masked_smooth_l1_loss(
            prediction,
            target_dbz,
            selected,
            beta=self.beta,
            weights=regression_weights,
        )
        count = int(selected.sum().item())
        if regression_weights is None:
            weight_sum = float(count)
        else:
            if not isinstance(regression_weights, torch.Tensor) or not torch.is_floating_point(
                regression_weights
            ):
                raise TypeError("regression_weights must be a floating-point tensor")
            try:
                weights = torch.broadcast_to(
                    regression_weights.to(device=prediction.device, dtype=torch.float32),
                    prediction.shape,
                )
            except RuntimeError as error:
                raise ValueError(
                    "regression_weights cannot broadcast to the R1-O prediction"
                ) from error
            chosen = weights[selected]
            if chosen.numel() and bool(
                (~torch.isfinite(chosen) | (chosen < 0.0)).any()
            ):
                raise ValueError(
                    "selected regression_weights must be finite and non-negative"
                )
            weight_sum = float(chosen.sum().detach().item())
        return Stage2CompletionLossComponents(
            total=loss,
            reflectivity=loss,
            reflectivity_count=count,
            reflectivity_weight_sum=weight_sum,
        )

    def forward(
        self,
        prediction: torch.Tensor,
        target_dbz: torch.Tensor,
        regression_mask: torch.Tensor,
        *,
        regression_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.compute_components(
            prediction,
            target_dbz,
            regression_mask,
            regression_weights=regression_weights,
        ).total


def build_stage2_completion_loss(config: Mapping[str, Any]) -> Stage2CompletionLoss:
    """Strictly parse the R1-O value-only loss mapping."""

    if not isinstance(config, Mapping):
        raise TypeError("R1-O loss configuration must be a mapping")
    if set(config) != {"reflectivity"}:
        raise ValueError("R1-O loss must contain only the reflectivity section")
    values = config["reflectivity"]
    if not isinstance(values, Mapping):
        raise TypeError("R1-O reflectivity loss configuration must be a mapping")
    allowed = {"beta", "intensity_bin_edges_dbz", "intensity_bin_weights"}
    unknown = sorted(set(values).difference(allowed))
    if unknown:
        raise ValueError(
            "unknown R1-O reflectivity loss options: " + ", ".join(unknown)
        )
    return Stage2CompletionLoss(beta=float(values.get("beta", 0.2)))


__all__ = [
    "Stage2CompletionLoss",
    "Stage2CompletionLossComponents",
    "build_stage2_completion_loss",
]
