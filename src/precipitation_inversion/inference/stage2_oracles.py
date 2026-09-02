"""Counterfactual inputs for validation-only Stage-2 decomposition audits.

The helpers in this module deliberately consume DPR targets and are therefore
never deployable model preprocessing.  They construct one-factor-at-a-time
counterfactuals for a *frozen* Stage-1 model:

``value``
    Replace dBZ by the true DPR value in a selected true-echo region while
    keeping the caller-supplied support fixed.
``support``
    Replace support by the true DPR occurrence state in a selected region
    while keeping the dense Stage-2 dBZ prediction fixed.
``joint``
    Apply both replacements in the selected region.

All arrays keep complete-orbit shape ``(nscan, nray, z)``.  Returned arrays
are new allocations; caller-owned predictions and labels are never modified.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


OracleComponent = Literal["value", "support", "joint"]


@dataclass(frozen=True)
class RegionalOracleInput:
    """One physical-dBZ/support input pair for frozen Stage-1 inference."""

    reflectivity_dbz: np.ndarray
    reflectivity_support: np.ndarray
    component: OracleComponent


def build_regional_oracle_input(
    predicted_dbz: np.ndarray,
    predicted_support: np.ndarray,
    true_dpr_dbz: np.ndarray,
    true_dpr_support: np.ndarray,
    region_mask: np.ndarray,
    *,
    component: OracleComponent,
) -> RegionalOracleInput:
    """Build a strictly regional value/support/joint counterfactual.

    Parameters are complete physical-orbit arrays with identical
    ``(nscan,nray,z)`` shape. ``region_mask`` is target-derived audit metadata;
    it must already be restricted to the trustworthy Stage-2 label domain.

    For a ``value`` oracle, every selected voxel must be a true DPR echo,
    because DPR dBZ is undefined outside that support.  ``support`` and
    ``joint`` can also select no-echo regions such as Q10/Q00 in order to audit
    false-positive removal.  The dense Stage-2 dBZ field must be finite at
    every support voxel that the resulting input exposes to Stage 1.
    """

    prediction = np.asarray(predicted_dbz, dtype=np.float32)
    predicted = np.asarray(predicted_support)
    target = np.asarray(true_dpr_dbz, dtype=np.float32)
    truth = np.asarray(true_dpr_support)
    region = np.asarray(region_mask)
    if prediction.ndim != 3 or any(size <= 0 for size in prediction.shape):
        raise ValueError("predicted_dbz must have non-empty (nscan,nray,z) shape")
    if target.shape != prediction.shape:
        raise ValueError("true_dpr_dbz must match predicted_dbz")
    for name, values in (
        ("predicted_support", predicted),
        ("true_dpr_support", truth),
        ("region_mask", region),
    ):
        if values.shape != prediction.shape or values.dtype != np.bool_:
            raise TypeError(f"{name} must be boolean and match predicted_dbz")
    if component not in {"value", "support", "joint"}:
        raise ValueError("component must be 'value', 'support', or 'joint'")
    if component == "value" and np.any(region & ~truth):
        raise ValueError("a value oracle region must be a subset of true DPR support")

    # Copy first so the audit cannot alter Stage-2 outputs reused by later
    # factorial cells. Before/after shapes both remain (nscan,nray,z).
    output_dbz = prediction.copy()
    output_support = predicted.copy()
    if component in {"support", "joint"}:
        output_support[region] = truth[region]
    if component in {"value", "joint"}:
        selected_values = region & truth
        if np.any(selected_values & ~np.isfinite(target)):
            raise ValueError("regional oracle selects an undefined true DPR dBZ")
        output_dbz[selected_values] = target[selected_values]
    if np.any(output_support & ~np.isfinite(output_dbz)):
        raise ValueError("regional oracle exposes a non-finite dBZ value to Stage 1")

    return RegionalOracleInput(
        reflectivity_dbz=output_dbz,
        reflectivity_support=output_support.astype(bool, copy=False),
        component=component,
    )


__all__ = [
    "OracleComponent",
    "RegionalOracleInput",
    "build_regional_oracle_input",
]
