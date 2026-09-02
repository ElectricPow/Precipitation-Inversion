"""Strict loss configuration for controlled Stage-3 experiments."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .masked_losses import (
    Stage1CompositeLoss,
    Stage1LossComponents,
    build_stage1_loss,
)
from .stage2_losses import (
    Stage2CompositeLoss,
    Stage2LossComponents,
    build_stage2_loss,
)


def validate_c1_oracle_loss(loss_config: Mapping[str, Any]) -> None:
    """Require the preregistered ``I + 0.02 G`` rain-only C1 objective."""

    if str(loss_config.get("name", "")).lower() != "masked_smooth_l1":
        raise ValueError("C1-O primary loss must be masked_smooth_l1")
    physical = loss_config.get("physical_gradient")
    if not isinstance(physical, Mapping) or physical.get("enabled") is not True:
        raise ValueError("C1-O requires enabled physical-gradient supervision")
    weight = float(physical.get("weight", float("nan")))
    if not math.isfinite(weight) or not math.isclose(
        weight, 0.02, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("C1-O physical-gradient weight must be exactly 0.02")


def build_stage3_c1_loss(loss_config: Mapping[str, Any]) -> Stage1CompositeLoss:
    """Build the existing Stage-1 I+G implementation after strict validation."""

    validate_c1_oracle_loss(loss_config)
    return build_stage1_loss(loss_config)


@dataclass(frozen=True)
class Stage3C2LossComponents:
    """Differentiable physical anchors and downstream rain objective."""

    total: torch.Tensor
    stage2: Stage2LossComponents
    rain: Stage1LossComponents
    stage2_weight: float
    rain_weight: float

    @property
    def weighted_stage2(self) -> torch.Tensor:
        return self.stage2.total * self.stage2_weight

    @property
    def weighted_rain(self) -> torch.Tensor:
        return self.rain.total * self.rain_weight


class Stage3C2CompositeLoss(nn.Module):
    """C2 objective: Stage-2 physical anchors plus task-aware rain loss.

    ``rain_weight`` is selected only from train-batch gradient norms. It is a
    scalar on the already sealed ``I + 0.02G`` Stage-1 objective; it never
    changes the W1.25 dBZ target weights or support BCE definition.
    """

    def __init__(
        self,
        stage2_criterion: Stage2CompositeLoss,
        rain_criterion: Stage1CompositeLoss,
        *,
        rain_weight: float,
        stage2_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if not math.isfinite(rain_weight) or rain_weight <= 0.0:
            raise ValueError("C2-O rain_weight must be finite and positive")
        if not math.isfinite(stage2_weight) or stage2_weight <= 0.0:
            raise ValueError("Stage-2 physical weight must be finite and positive")
        self.stage2_criterion = stage2_criterion
        self.rain_criterion = rain_criterion
        self.stage2_weight = float(stage2_weight)
        self.rain_weight = float(rain_weight)

    def compute_components(
        self,
        *,
        rain_prediction: torch.Tensor,
        support_logits: torch.Tensor,
        reflectivity_prediction: torch.Tensor,
        rain_target: torch.Tensor,
        rain_mask: torch.Tensor,
        rain_weights: torch.Tensor | None,
        reliable_rain_mask: torch.Tensor,
        height_km: torch.Tensor,
        target_support: torch.Tensor,
        target_dbz: torch.Tensor,
        support_mask: torch.Tensor,
        regression_mask: torch.Tensor,
        regression_weights: torch.Tensor | None,
    ) -> Stage3C2LossComponents:
        stage2 = self.stage2_criterion.compute_components(
            support_logits,
            reflectivity_prediction,
            target_support,
            target_dbz,
            support_mask,
            regression_mask,
            regression_weights=regression_weights,
        )
        rain = self.rain_criterion.compute_components(
            rain_prediction,
            rain_target,
            rain_mask,
            rain_weights,
            reliable_mask=reliable_rain_mask,
            height_km=height_km,
        )
        return Stage3C2LossComponents(
            total=self.stage2_weight * stage2.total + self.rain_weight * rain.total,
            stage2=stage2,
            rain=rain,
            stage2_weight=self.stage2_weight,
            rain_weight=self.rain_weight,
        )


def build_stage3_c2_loss(
    stage2_loss_config: Mapping[str, Any],
    stage1_loss_config: Mapping[str, Any],
    *,
    rain_weight: float,
) -> Stage3C2CompositeLoss:
    """Build C2 from the exact source Stage-2 and sealed Stage-1 objectives."""

    validate_c1_oracle_loss(stage1_loss_config)
    return Stage3C2CompositeLoss(
        build_stage2_loss(stage2_loss_config),
        build_stage1_loss(stage1_loss_config),
        rain_weight=rain_weight,
    )


class Stage3D0CompositeLoss(Stage3C2CompositeLoss):
    """D0 three-head objective with independent physical supervision masks.

    The mathematical components intentionally match C2 so differences are
    attributable to the direct high-dimensional rain path rather than a new
    loss definition. In a frozen-feature ``rain_head_only`` probe the Stage-2
    anchor is still reported but has no gradient; in decoder multi-task mode
    all three tasks constrain the shared decoder.
    """


def build_stage3_d0_loss(
    stage2_loss_config: Mapping[str, Any],
    rain_loss_config: Mapping[str, Any],
    *,
    rain_weight: float,
    stage2_weight: float = 1.0,
) -> Stage3D0CompositeLoss:
    """Build D0 with explicit coefficients for both independent task groups.

    Legacy D0-H/D0-D use ``stage2_weight=1`` and scale the rain term.  The
    RainPrimary experiment reverses that hierarchy: ``rain_weight=1`` while a
    train-only gradient audit selects the smaller ``stage2_weight``.  Keeping
    both coefficients explicit prevents an accidental return to the earlier
    anchor-primary objective when a checkpoint is resumed.
    """

    validate_c1_oracle_loss(rain_loss_config)
    return Stage3D0CompositeLoss(
        build_stage2_loss(stage2_loss_config),
        build_stage1_loss(rain_loss_config),
        rain_weight=rain_weight,
        stage2_weight=stage2_weight,
    )
