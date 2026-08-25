"""Data reading, masking, indexing, and split utilities."""

from .masks import (
    clutter_mask_from_cfb,
    dpr_reflectivity_mask,
    gr_observation_mask,
    overlap_mask,
    precipitation_label_mask,
    positive_rain_mask,
    profile_has_observation,
    to_float_array,
    valid_cfb_mask,
    valid_numeric_mask,
    zero_rain_mask,
)
from .splits import (
    DEFAULT_BALANCE_WEIGHTS,
    DEFAULT_SPLIT_RATIOS,
    SplitResult,
    assert_valid_split,
    balanced_group_split,
    chronological_group_split,
)
from .nc_reader import (
    DEFAULT_VARIABLES,
    KNOWN_DIMENSIONS,
    NCSample,
    VariableMetadata,
    build_standard_masks,
    read_nc_sample,
)

__all__ = [
    "clutter_mask_from_cfb",
    "dpr_reflectivity_mask",
    "gr_observation_mask",
    "overlap_mask",
    "precipitation_label_mask",
    "positive_rain_mask",
    "profile_has_observation",
    "to_float_array",
    "valid_cfb_mask",
    "valid_numeric_mask",
    "zero_rain_mask",
    "DEFAULT_BALANCE_WEIGHTS",
    "DEFAULT_SPLIT_RATIOS",
    "SplitResult",
    "assert_valid_split",
    "balanced_group_split",
    "chronological_group_split",
    "DEFAULT_VARIABLES",
    "KNOWN_DIMENSIONS",
    "NCSample",
    "VariableMetadata",
    "build_standard_masks",
    "read_nc_sample",
]
