"""Neural-network architectures for precipitation inversion."""

from .blocks3d import (
    AnisotropicResidualBlock3D,
    HorizontalDownBlock3D,
    HorizontalUpBlock3D,
)
from .unet3d import Stage1UNet3D
from .multitask_unet3d import Stage1MultiTaskUNet3D, rain_prediction_from_output
from .type_heads import GlobalPoolTypeHead, Ordered3DMorphologyHead, build_type_head
from .stage2_unet3d import Stage2UNet3D, stage2_predictions_from_output
from .stage2_completion_unet3d import (
    Stage2CompletionUNet3D,
    stage2_completion_prediction_from_output,
)
from .stage2_partial_completion_unet3d import (
    PartialConv3D,
    SparseValuePartialStem3D,
    Stage2PartialCompletionUNet3D,
)
from .stage3_cascade import (
    STAGE3_C2_TRAINABLE_SCOPE,
    Stage3C1OracleCascade,
    Stage3C2OracleCascade,
    assert_c1_freeze_contract,
    assert_c2_freeze_contract,
    stage3_c2_predictions_from_output,
)
from .stage3_direct import (
    STAGE3_D0_DECODER_AND_HEADS,
    STAGE3_D0_RAIN_HEAD_ONLY,
    STAGE3_D0_TRAINABLE_SCOPES,
    Stage3DirectMultiHeadUNet3D,
    assert_d0_trainable_contract,
    d0_shared_audit_parameters,
    build_stage3_d0_model,
    load_stage2_source_into_d0,
    stage3_d0_predictions_from_output,
)

__all__ = [
    "AnisotropicResidualBlock3D",
    "HorizontalDownBlock3D",
    "HorizontalUpBlock3D",
    "Stage1UNet3D",
    "Stage1MultiTaskUNet3D",
    "GlobalPoolTypeHead",
    "Ordered3DMorphologyHead",
    "build_type_head",
    "rain_prediction_from_output",
    "Stage2UNet3D",
    "stage2_predictions_from_output",
    "Stage2CompletionUNet3D",
    "stage2_completion_prediction_from_output",
    "PartialConv3D",
    "SparseValuePartialStem3D",
    "Stage2PartialCompletionUNet3D",
    "Stage3C1OracleCascade",
    "Stage3C2OracleCascade",
    "STAGE3_C2_TRAINABLE_SCOPE",
    "assert_c1_freeze_contract",
    "assert_c2_freeze_contract",
    "stage3_c2_predictions_from_output",
    "STAGE3_D0_DECODER_AND_HEADS",
    "STAGE3_D0_RAIN_HEAD_ONLY",
    "STAGE3_D0_TRAINABLE_SCOPES",
    "Stage3DirectMultiHeadUNet3D",
    "assert_d0_trainable_contract",
    "d0_shared_audit_parameters",
    "build_stage3_d0_model",
    "load_stage2_source_into_d0",
    "stage3_d0_predictions_from_output",
]
