"""Masked objectives used by precipitation-inversion models."""

from .masked_losses import (
    MaskedSmoothL1Loss,
    Stage1CompositeLoss,
    Stage1LossComponents,
    build_stage1_loss,
    masked_mae_loss,
    masked_mse_loss,
    masked_physical_gradient_smooth_l1_loss,
    masked_smooth_l1_loss,
)
from .masked_classification import MaskedCrossEntropyLoss
from .stage2_losses import (
    MaskedSupportLoss,
    Stage2CompositeLoss,
    Stage2LossComponents,
    build_stage2_loss,
    masked_support_bce_with_logits,
    masked_support_focal_loss,
)
from .stage2_completion_losses import (
    Stage2CompletionLoss,
    Stage2CompletionLossComponents,
    build_stage2_completion_loss,
)
from .stage3_losses import (
    Stage3C2CompositeLoss,
    Stage3C2LossComponents,
    Stage3D0CompositeLoss,
    build_stage3_c1_loss,
    build_stage3_c2_loss,
    build_stage3_d0_loss,
    validate_c1_oracle_loss,
)

__all__ = [
    "MaskedSmoothL1Loss",
    "Stage1CompositeLoss",
    "Stage1LossComponents",
    "build_stage1_loss",
    "masked_mae_loss",
    "masked_mse_loss",
    "masked_physical_gradient_smooth_l1_loss",
    "masked_smooth_l1_loss",
    "MaskedCrossEntropyLoss",
    "MaskedSupportLoss",
    "Stage2CompositeLoss",
    "Stage2LossComponents",
    "build_stage2_loss",
    "masked_support_bce_with_logits",
    "masked_support_focal_loss",
    "Stage2CompletionLoss",
    "Stage2CompletionLossComponents",
    "build_stage2_completion_loss",
    "build_stage3_c1_loss",
    "validate_c1_oracle_loss",
    "Stage3C2CompositeLoss",
    "Stage3C2LossComponents",
    "build_stage3_c2_loss",
    "Stage3D0CompositeLoss",
    "build_stage3_d0_loss",
]
