"""Utilities for DPR profile-level precipitation-type supervision.

The source ``typePrecip`` field is two-dimensional ``(nscan, nray)``.  DPR
codes 1/2/3 mean stratiform/convective/other, while -1111 means no
precipitation and -9999 is used locally for unavailable metadata.  The model
uses zero-based targets only at explicitly selected profiles; source codes are
never inserted into the input tensor.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


TYPE_NAMES = ("stratiform", "convective", "other")
SOURCE_TYPE_CODES = (1, 2, 3)
TYPE_IGNORE_INDEX = -100
TYPE_NO_PRECIPITATION = -1111
TYPE_UNKNOWN = -9999


def build_type_target_and_mask(
    source_codes: np.ndarray,
    *,
    core_profile_mask: np.ndarray,
    dpr_profile_support: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map source codes to ``0..2`` and build the strict training mask.

    Parameters are all profile fields with shape ``(nscan, nray)``.  A profile
    is supervised only when it belongs to the patch's non-overlapping output
    core, has a valid DPR class, and contains at least one native DPR echo.
    Thus padded/halo profiles, true missingness, no-precipitation, and unknown
    class metadata cannot contribute to cross entropy.
    """

    codes = np.asarray(source_codes)
    core = np.asarray(core_profile_mask, dtype=bool)
    support = np.asarray(dpr_profile_support, dtype=bool)
    if codes.ndim != 2 or core.shape != codes.shape or support.shape != codes.shape:
        raise ValueError("type codes and profile masks must share shape (nscan,nray)")

    target = np.full(codes.shape, TYPE_IGNORE_INDEX, dtype=np.int64)
    valid_class = np.zeros(codes.shape, dtype=bool)
    for target_index, source_code in enumerate(SOURCE_TYPE_CODES):
        selected = codes == source_code
        target[selected] = target_index
        valid_class |= selected
    mask = core & support & valid_class
    target[~mask] = TYPE_IGNORE_INDEX
    return target, mask


def inverse_sqrt_class_weights(
    counts: Sequence[int | float], *, normalize: bool = True
) -> np.ndarray:
    """Return finite inverse-square-root weights for the three DPR classes.

    The optional normalization makes the count-weighted mean equal to one, so
    the auxiliary-loss scale remains interpretable when class balance changes.
    Empty classes are rejected rather than receiving an unstable large weight.
    """

    values = np.asarray(tuple(counts), dtype=np.float64)
    if values.shape != (len(TYPE_NAMES),) or np.any(~np.isfinite(values)):
        raise ValueError("counts must contain three finite values")
    if np.any(values <= 0):
        raise ValueError("every precipitation type must have at least one sample")
    weights = 1.0 / np.sqrt(values)
    if normalize:
        weights /= float(np.sum(weights * values) / np.sum(values))
    return weights.astype(np.float32)

