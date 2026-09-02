"""Inference helpers for reconstructing complete precipitation swaths."""

from .sliding_window import (
    CoreWindowReconstructor,
    predict_full_orbit,
    stitch_core_predictions,
)
from .stage2_sliding_window import (
    Stage3D0OrbitPrediction,
    Stage2OrbitPrediction,
    SupportThresholdSweep,
    ThresholdSelection,
    predict_stage2_full_orbit,
    predict_stage3_d0_full_orbit,
    reconstruct_stage2_fields,
    reconstruct_stage2_targets,
    select_support_threshold,
)
from .stage2_completion_sliding_window import (
    Stage2CompletionOrbitPrediction,
    predict_stage2_completion_full_orbit,
)
from .stage2_stage1_cascade import (
    Stage1CascadePrediction,
    Stage1CascadeWindow,
    iter_stage1_cascade_windows,
    predict_stage1_from_reflectivity_orbit,
)
from .stage2_oracles import (
    OracleComponent,
    RegionalOracleInput,
    build_regional_oracle_input,
)

__all__ = [
    "CoreWindowReconstructor",
    "predict_full_orbit",
    "stitch_core_predictions",
    "Stage2OrbitPrediction",
    "SupportThresholdSweep",
    "ThresholdSelection",
    "predict_stage2_full_orbit",
    "Stage3D0OrbitPrediction",
    "predict_stage3_d0_full_orbit",
    "reconstruct_stage2_fields",
    "reconstruct_stage2_targets",
    "select_support_threshold",
    "Stage2CompletionOrbitPrediction",
    "predict_stage2_completion_full_orbit",
    "Stage1CascadePrediction",
    "Stage1CascadeWindow",
    "iter_stage1_cascade_windows",
    "predict_stage1_from_reflectivity_orbit",
    "OracleComponent",
    "RegionalOracleInput",
    "build_regional_oracle_input",
]
