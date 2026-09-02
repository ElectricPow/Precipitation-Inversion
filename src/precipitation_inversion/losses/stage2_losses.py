"""Dual masked objectives for Stage-2 support and DPR dBZ prediction.

The support head is supervised throughout the trustworthy occurrence domain,
including valid background zeros.  The reflectivity head is supervised only
where a physical DPR dBZ target exists.  The two masks are deliberately
independent, and predicted support never hard-gates reflectivity loss.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .masked_losses import masked_smooth_l1_loss


SupportLossName = Literal["bce", "focal"]


def _validate_bool_mask(mask: torch.Tensor, shape: torch.Size, *, name: str) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
        raise TypeError(f"{name} must be a boolean tensor")
    try:
        return torch.broadcast_to(mask, shape)
    except RuntimeError as error:
        raise ValueError(
            f"{name} shape {tuple(mask.shape)} cannot broadcast to {tuple(shape)}"
        ) from error


def _support_target(
    target: torch.Tensor,
    *,
    reference: torch.Tensor,
    selected: torch.Tensor,
) -> torch.Tensor:
    if target.shape != reference.shape:
        raise ValueError(
            f"support target/logit shapes differ: {tuple(target.shape)} != "
            f"{tuple(reference.shape)}"
        )
    if target.device != reference.device:
        raise ValueError("support target and logits must share a device")
    if target.dtype == torch.bool:
        result = target.to(dtype=reference.dtype)
    elif torch.is_floating_point(target):
        result = target.to(dtype=reference.dtype)
    else:
        raise TypeError("support target must be boolean or floating-point")
    selected_values = result[selected]
    if selected_values.numel() and bool(
        (~torch.isfinite(selected_values)
         | (selected_values < 0.0)
         | (selected_values > 1.0)).any()
    ):
        raise ValueError("selected support targets must be finite values in [0,1]")
    return result


def _masked_mean(
    element_loss: torch.Tensor,
    selected: torch.Tensor,
) -> torch.Tensor:
    # Mask before summation and retain a graph-connected zero for empty patches.
    safe = torch.where(selected, element_loss, torch.zeros_like(element_loss))
    count = selected.sum().to(dtype=element_loss.dtype)
    if bool(count == 0):
        return safe.sum() * 0.0
    return safe.sum() / count


def masked_support_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    pos_weight: float = 1.0,
) -> torch.Tensor:
    """Masked binary cross entropy for ``(B,1,D,H,Z)`` support logits."""

    if not torch.is_floating_point(logits):
        raise TypeError("support logits must be floating-point")
    if not math.isfinite(pos_weight) or pos_weight <= 0.0:
        raise ValueError("pos_weight must be finite and positive")
    selected = _validate_bool_mask(mask, logits.shape, name="support mask")
    values = _support_target(target, reference=logits, selected=selected)
    # Neutralize ignored logits/targets before BCE so arbitrary NaN/Inf padding
    # cannot enter the autograd graph even transiently.
    safe_logits = torch.where(selected, logits, torch.zeros_like(logits))
    safe_target = torch.where(selected, values, torch.zeros_like(values))
    element = F.binary_cross_entropy_with_logits(
        safe_logits,
        safe_target,
        pos_weight=torch.as_tensor(pos_weight, device=logits.device, dtype=logits.dtype),
        reduction="none",
    )
    return _masked_mean(element, selected)


def masked_support_focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    gamma: float = 2.0,
    alpha: float | None = 0.25,
) -> torch.Tensor:
    """Masked sigmoid focal loss with optional positive-class alpha."""

    if not torch.is_floating_point(logits):
        raise TypeError("support logits must be floating-point")
    if not math.isfinite(gamma) or gamma < 0.0:
        raise ValueError("focal gamma must be finite and non-negative")
    if alpha is not None and (not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0):
        raise ValueError("focal alpha must be null or lie in [0,1]")
    selected = _validate_bool_mask(mask, logits.shape, name="support mask")
    values = _support_target(target, reference=logits, selected=selected)
    safe_logits = torch.where(selected, logits, torch.zeros_like(logits))
    safe_target = torch.where(selected, values, torch.zeros_like(values))
    bce = F.binary_cross_entropy_with_logits(safe_logits, safe_target, reduction="none")
    probability = torch.sigmoid(safe_logits)
    probability_target = probability * safe_target + (1.0 - probability) * (
        1.0 - safe_target
    )
    element = bce * (1.0 - probability_target).pow(gamma)
    if alpha is not None:
        alpha_target = alpha * safe_target + (1.0 - alpha) * (1.0 - safe_target)
        element = element * alpha_target
    return _masked_mean(element, selected)


class MaskedSupportLoss(nn.Module):
    """Configurable BCE or focal support-domain loss."""

    def __init__(
        self,
        *,
        name: SupportLossName = "bce",
        pos_weight: float = 1.0,
        focal_gamma: float = 2.0,
        focal_alpha: float | None = 0.25,
    ) -> None:
        super().__init__()
        normalized = str(name).lower()
        if normalized not in {"bce", "focal"}:
            raise ValueError("support loss name must be 'bce' or 'focal'")
        if normalized == "focal" and pos_weight != 1.0:
            raise ValueError("focal loss uses alpha and cannot also use pos_weight")
        # Validate all numerical options at construction time.
        if not math.isfinite(pos_weight) or pos_weight <= 0.0:
            raise ValueError("pos_weight must be finite and positive")
        if not math.isfinite(focal_gamma) or focal_gamma < 0.0:
            raise ValueError("focal gamma must be finite and non-negative")
        if focal_alpha is not None and (
            not math.isfinite(focal_alpha) or not 0.0 <= focal_alpha <= 1.0
        ):
            raise ValueError("focal alpha must be null or lie in [0,1]")
        self.name: SupportLossName = normalized  # type: ignore[assignment]
        self.pos_weight = float(pos_weight)
        self.focal_gamma = float(focal_gamma)
        self.focal_alpha = None if focal_alpha is None else float(focal_alpha)

    def forward(
        self, logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        if self.name == "bce":
            return masked_support_bce_with_logits(
                logits, target, mask, pos_weight=self.pos_weight
            )
        return masked_support_focal_loss(
            logits,
            target,
            mask,
            gamma=self.focal_gamma,
            alpha=self.focal_alpha,
        )


@dataclass(frozen=True)
class Stage2LossComponents:
    """Differentiable components and exact supervision counts."""

    total: torch.Tensor
    support: torch.Tensor
    reflectivity: torch.Tensor
    support_weight: float
    reflectivity_weight: float
    support_count: int
    support_positive_count: int
    reflectivity_count: int
    reflectivity_weight_sum: float

    @property
    def weighted_support(self) -> torch.Tensor:
        return self.support * self.support_weight

    @property
    def weighted_reflectivity(self) -> torch.Tensor:
        return self.reflectivity * self.reflectivity_weight


class Stage2CompositeLoss(nn.Module):
    """Support BCE/focal plus masked standardized-dBZ Smooth-L1."""

    def __init__(
        self,
        *,
        support_loss: SupportLossName = "bce",
        support_weight: float = 1.0,
        reflectivity_weight: float = 1.0,
        support_pos_weight: float = 1.0,
        focal_gamma: float = 2.0,
        focal_alpha: float | None = 0.25,
        reflectivity_beta: float = 0.2,
    ) -> None:
        super().__init__()
        for name, value in (
            ("support_weight", support_weight),
            ("reflectivity_weight", reflectivity_weight),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if support_weight == 0.0 and reflectivity_weight == 0.0:
            raise ValueError("at least one Stage-2 task weight must be positive")
        if not math.isfinite(reflectivity_beta) or reflectivity_beta <= 0.0:
            raise ValueError("reflectivity_beta must be finite and positive")
        self.support_criterion = MaskedSupportLoss(
            name=support_loss,
            pos_weight=support_pos_weight,
            focal_gamma=focal_gamma,
            focal_alpha=focal_alpha,
        )
        self.support_weight = float(support_weight)
        self.reflectivity_weight = float(reflectivity_weight)
        self.reflectivity_beta = float(reflectivity_beta)

    def compute_components(
        self,
        support_logits: torch.Tensor,
        reflectivity_prediction: torch.Tensor,
        target_support: torch.Tensor,
        target_dbz: torch.Tensor,
        support_mask: torch.Tensor,
        regression_mask: torch.Tensor,
        *,
        regression_weights: torch.Tensor | None = None,
    ) -> Stage2LossComponents:
        if support_logits.shape != reflectivity_prediction.shape:
            raise ValueError("support and reflectivity prediction shapes differ")
        if target_dbz.shape != reflectivity_prediction.shape:
            raise ValueError("reflectivity prediction/target shapes differ")
        support_selected = _validate_bool_mask(
            support_mask, support_logits.shape, name="support mask"
        )
        regression_selected = _validate_bool_mask(
            regression_mask, reflectivity_prediction.shape, name="regression mask"
        )
        target_values = _support_target(
            target_support, reference=support_logits, selected=support_selected
        )
        if bool((regression_selected & ~support_selected).any()):
            raise ValueError("regression mask must be a subset of support mask")
        if bool((regression_selected & (target_values < 0.5)).any()):
            raise ValueError("regression mask must be a subset of positive support target")
        selected_dbz = target_dbz[regression_selected]
        if selected_dbz.numel() and not bool(torch.isfinite(selected_dbz).all()):
            raise ValueError("selected dBZ targets must be finite")

        support = self.support_criterion(
            support_logits, target_values, support_selected
        )
        reflectivity = masked_smooth_l1_loss(
            reflectivity_prediction,
            target_dbz,
            regression_selected,
            beta=self.reflectivity_beta,
            weights=regression_weights,
        )
        if regression_weights is None:
            reflectivity_weight_sum = float(regression_selected.sum().item())
        else:
            if not isinstance(regression_weights, torch.Tensor):
                raise TypeError("regression_weights must be a tensor")
            try:
                broadcast_weights = torch.broadcast_to(
                    regression_weights.to(
                        device=reflectivity_prediction.device,
                        # Do not sum the denominator in fp16 under AMP: a
                        # dense patch can contain more than 65,504 total
                        # weight and overflow half precision.
                        dtype=torch.float32,
                    ),
                    reflectivity_prediction.shape,
                )
            except RuntimeError as error:
                raise ValueError(
                    "regression_weights cannot broadcast to reflectivity shape"
                ) from error
            selected_weights = broadcast_weights[regression_selected]
            # masked_smooth_l1_loss performs the same validity check before
            # this point; the explicit sum is retained for exact epoch/DDP
            # aggregation of the weighted objective.
            reflectivity_weight_sum = float(selected_weights.sum().detach().item())
        total = (
            self.support_weight * support
            + self.reflectivity_weight * reflectivity
        )
        return Stage2LossComponents(
            total=total,
            support=support,
            reflectivity=reflectivity,
            support_weight=self.support_weight,
            reflectivity_weight=self.reflectivity_weight,
            support_count=int(support_selected.sum().item()),
            support_positive_count=int(
                ((target_values >= 0.5) & support_selected).sum().item()
            ),
            reflectivity_count=int(regression_selected.sum().item()),
            reflectivity_weight_sum=reflectivity_weight_sum,
        )

    def forward(
        self,
        support_logits: torch.Tensor,
        reflectivity_prediction: torch.Tensor,
        target_support: torch.Tensor,
        target_dbz: torch.Tensor,
        support_mask: torch.Tensor,
        regression_mask: torch.Tensor,
        *,
        regression_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.compute_components(
            support_logits,
            reflectivity_prediction,
            target_support,
            target_dbz,
            support_mask,
            regression_mask,
            regression_weights=regression_weights,
        ).total


def build_stage2_loss(config: Mapping[str, Any]) -> Stage2CompositeLoss:
    """Build a Stage-2 objective from a compact configuration mapping."""

    support = config.get("support", {})
    reflectivity = config.get("reflectivity", {})
    if not isinstance(support, Mapping) or not isinstance(reflectivity, Mapping):
        raise TypeError("support and reflectivity loss configuration must be mappings")
    support_allowed = {"name", "weight", "pos_weight", "gamma", "alpha"}
    reflectivity_allowed = {
        "weight",
        "beta",
        # Dataset-side, target-derived E3 options. They are accepted here so
        # the whole loss mapping has one strict schema, but never become model
        # inputs and do not alter the Smooth-L1 implementation itself.
        "intensity_bin_edges_dbz",
        "intensity_bin_weights",
    }
    unknown_support = sorted(set(support).difference(support_allowed))
    unknown_reflectivity = sorted(
        set(reflectivity).difference(reflectivity_allowed)
    )
    if unknown_support:
        raise ValueError(
            "unknown Stage-2 support loss options: " + ", ".join(unknown_support)
        )
    if unknown_reflectivity:
        raise ValueError(
            "unknown Stage-2 reflectivity loss options: "
            + ", ".join(unknown_reflectivity)
        )
    return Stage2CompositeLoss(
        support_loss=str(support.get("name", "bce")).lower(),  # type: ignore[arg-type]
        support_weight=float(support.get("weight", 1.0)),
        support_pos_weight=float(support.get("pos_weight", 1.0)),
        focal_gamma=float(support.get("gamma", 2.0)),
        focal_alpha=(
            None if support.get("alpha", 0.25) is None else float(support.get("alpha", 0.25))
        ),
        reflectivity_weight=float(reflectivity.get("weight", 1.0)),
        reflectivity_beta=float(reflectivity.get("beta", 0.2)),
    )
