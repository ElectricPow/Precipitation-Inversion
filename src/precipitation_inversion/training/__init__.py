"""Training-loop utilities for precipitation inversion."""

from .engine import (
    DEFAULT_CHECKPOINT_EVERY,
    EpochResult,
    evaluate_one_epoch,
    load_checkpoint,
    save_checkpoint,
    seed_everything,
    should_save_periodic_checkpoint,
    train_one_epoch,
    validate_checkpoint_every,
    validate_training_output_directory,
)
from .stage2_engine import (
    Stage2EpochResult,
    evaluate_stage2_one_epoch,
    standardized_to_physical_dbz,
    train_stage2_one_epoch,
)
from .stage2_completion_engine import (
    Stage2CompletionEpochResult,
    evaluate_stage2_completion_one_epoch,
    train_stage2_completion_one_epoch,
)
from .stage3_engine import (
    Stage3C2EpochResult,
    Stage3D0EpochResult,
    audit_stage3_c2_gradient_scale,
    audit_stage3_d0_gradient_scale,
    evaluate_stage3_c1_one_epoch,
    evaluate_stage3_c2_one_epoch,
    evaluate_stage3_d0_one_epoch,
    train_stage3_c1_one_epoch,
    train_stage3_c2_one_epoch,
    train_stage3_d0_one_epoch,
)

__all__ = [
    "DEFAULT_CHECKPOINT_EVERY",
    "EpochResult",
    "evaluate_one_epoch",
    "load_checkpoint",
    "save_checkpoint",
    "seed_everything",
    "should_save_periodic_checkpoint",
    "train_one_epoch",
    "validate_checkpoint_every",
    "validate_training_output_directory",
    "Stage2EpochResult",
    "evaluate_stage2_one_epoch",
    "standardized_to_physical_dbz",
    "train_stage2_one_epoch",
    "Stage2CompletionEpochResult",
    "evaluate_stage2_completion_one_epoch",
    "train_stage2_completion_one_epoch",
    "evaluate_stage3_c1_one_epoch",
    "train_stage3_c1_one_epoch",
    "Stage3C2EpochResult",
    "Stage3D0EpochResult",
    "audit_stage3_c2_gradient_scale",
    "audit_stage3_d0_gradient_scale",
    "evaluate_stage3_c2_one_epoch",
    "evaluate_stage3_d0_one_epoch",
    "train_stage3_c2_one_epoch",
    "train_stage3_d0_one_epoch",
]
