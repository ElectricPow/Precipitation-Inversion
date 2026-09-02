"""Evaluation metrics for precipitation inversion."""

from .regression import (
    FilewisePrecipitationMetrics,
    GroupedRegressionAccumulator,
    PhysicalRainGradientMetrics,
    PrecipitationRegressionMetrics,
    RegressionAccumulator,
    StratifiedPrecipitationMetrics,
    physical_vertical_rain_gradient,
)
from .classification import MulticlassConfusionMetrics
from .stage2_reflectivity import (
    FractionSkillAccumulator,
    ReflectivityRegressionAccumulator,
    Stage2ReflectivityMetrics,
    SupportConfusionAccumulator,
    finite_metrics_for_json,
    horizontal_window_sum,
)
from .stage2_decomposition import (
    CentroidDisplacementAccumulator,
    CfadAccumulator,
    EchoColumnAccumulator,
    MultiThresholdSpatialAccumulator,
    Stage2DecompositionDiagnostics,
    SupportProbabilityAccumulator,
)
from .stage2_local_shift import (
    LocalShiftOptions,
    aggregate_local_shift_audits,
    audit_orbit_local_shifts,
    finite_shift_offsets,
    non_overlapping_windows,
    select_best_shift,
)

__all__ = [
    "FilewisePrecipitationMetrics",
    "GroupedRegressionAccumulator",
    "PhysicalRainGradientMetrics",
    "PrecipitationRegressionMetrics",
    "RegressionAccumulator",
    "StratifiedPrecipitationMetrics",
    "physical_vertical_rain_gradient",
    "MulticlassConfusionMetrics",
    "FractionSkillAccumulator",
    "ReflectivityRegressionAccumulator",
    "Stage2ReflectivityMetrics",
    "SupportConfusionAccumulator",
    "finite_metrics_for_json",
    "horizontal_window_sum",
    "CentroidDisplacementAccumulator",
    "CfadAccumulator",
    "EchoColumnAccumulator",
    "MultiThresholdSpatialAccumulator",
    "Stage2DecompositionDiagnostics",
    "SupportProbabilityAccumulator",
    "LocalShiftOptions",
    "aggregate_local_shift_audits",
    "audit_orbit_local_shifts",
    "finite_shift_offsets",
    "non_overlapping_windows",
    "select_best_shift",
]
