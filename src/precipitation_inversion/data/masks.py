"""Canonical validity masks for the collocated GR--DPR NetCDF dataset.

The project must keep *missing observations* separate from numerical zero.
These helpers provide one shared interpretation for dataset auditing, training,
and evaluation so that each script does not silently invent a different rule.

The source files use both NetCDF masks/NaN and occasional large negative fill
values.  Real ground-radar weak echoes may be negative dBZ, so only values at or
below ``-9990`` are considered legacy fill values; ordinary negative dBZ values
remain valid observations.
"""

from __future__ import annotations

from typing import Any

import numpy as np


# Values near -9999 and -9999.9 occur as fill codes in related products.  The
# cutoff is deliberately far below physically meaningful negative reflectivity.
LEGACY_FILL_CUTOFF = -9990.0


def to_float_array(values: Any, *, mask_legacy_fill: bool = True) -> np.ndarray:
    """Return ``values`` as a float64 array with invalid entries represented by NaN.

    Parameters
    ----------
    values:
        A NumPy array, masked array, or a slice already read from a NetCDF
        variable.  Passing the variable itself is also supported because it can
        be sliced with ``[:]``.
    mask_legacy_fill:
        Replace values less than or equal to :data:`LEGACY_FILL_CUTOFF` by NaN.
        Keep this enabled for physical variables and category variables in this
        dataset: ``typePrecip=-1111`` remains valid, while ``-9999`` is removed.

    Notes
    -----
    This function intentionally does not convert weak/negative dBZ to zero.
    Noise clipping is a model preprocessing decision, not a missing-data rule.
    """

    if hasattr(values, "dimensions") and hasattr(values, "__getitem__"):
        values = values[:]
    # Convert before filling so a masked integer variable can represent NaN.
    # Calling ``filled(np.nan)`` on an integer masked array raises a TypeError.
    masked = np.ma.asarray(values, dtype=np.float64)
    array = np.asarray(np.ma.filled(masked, np.nan), dtype=np.float64)
    # Work on a writable copy when np.asarray returned a read-only view.
    if not array.flags.writeable:
        array = array.copy()
    array[~np.isfinite(array)] = np.nan
    if mask_legacy_fill:
        array[array <= LEGACY_FILL_CUTOFF] = np.nan
    return array


def valid_numeric_mask(values: Any) -> np.ndarray:
    """Identify finite, non-fill numerical values.

    Masked entries, NaN/Inf, and legacy fill codes around ``-9999`` are invalid.
    Values such as ``-30 dBZ`` and the valid no-precipitation code ``-1111`` are
    retained.
    """

    return np.isfinite(to_float_array(values))


def gr_observation_mask(dbz_gr: Any) -> np.ndarray:
    """Return locations with a direct or interpolated GR reflectivity value.

    The function only answers whether a numeric observation exists.  It does not
    apply the later 9 dBZ model-noise threshold.
    """

    return valid_numeric_mask(dbz_gr)


def dpr_reflectivity_mask(dbz_dpr: Any) -> np.ndarray:
    """Return locations where the derived DPR reflectivity is retained."""

    return valid_numeric_mask(dbz_dpr)


def precipitation_label_mask(
    pre_dpr: Any,
    *,
    cfb: Any | None = None,
    z: Any | None = None,
    exclude_clutter: bool = True,
) -> np.ndarray:
    """Return valid non-negative precipitation-rate label locations.

    When ``cfb`` and ``z`` are supplied, bins below the clutter-free bottom are
    removed by default.  A zero precipitation rate remains a valid label.
    Negative finite precipitation rates are rejected as physically invalid.
    """

    precipitation = to_float_array(pre_dpr)
    valid = np.isfinite(precipitation) & (precipitation >= 0.0)

    if exclude_clutter:
        if (cfb is None) != (z is None):
            raise ValueError("cfb and z must be provided together")
        if cfb is not None:
            clutter = clutter_mask_from_cfb(cfb, z)
            if clutter.shape != valid.shape:
                raise ValueError(
                    "pre_dpr and cfb-derived clutter mask must have the same shape: "
                    f"{valid.shape} != {clutter.shape}"
                )
            valid &= ~clutter
    return valid


def zero_rain_mask(
    pre_dpr: Any, *, valid_mask: np.ndarray | None = None
) -> np.ndarray:
    """Return valid locations explicitly labelled as zero precipitation."""

    precipitation = to_float_array(pre_dpr)
    valid = (
        precipitation_label_mask(precipitation, exclude_clutter=False)
        if valid_mask is None
        else np.asarray(valid_mask, dtype=bool)
    )
    if valid.shape != precipitation.shape:
        raise ValueError("valid_mask and pre_dpr must have the same shape")
    return valid & (precipitation == 0.0)


def positive_rain_mask(
    pre_dpr: Any,
    *,
    threshold: float = 0.0,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return valid locations whose precipitation exceeds ``threshold``."""

    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    precipitation = to_float_array(pre_dpr)
    valid = (
        precipitation_label_mask(precipitation, exclude_clutter=False)
        if valid_mask is None
        else np.asarray(valid_mask, dtype=bool)
    )
    if valid.shape != precipitation.shape:
        raise ValueError("valid_mask and pre_dpr must have the same shape")
    return valid & (precipitation > threshold)


def clutter_mask_from_cfb(cfb: Any, z: Any) -> np.ndarray:
    """Build the three-dimensional mask for bins below the clutter-free bottom.

    ``cfb`` contains an index into the ascending one-dimensional height array
    ``z`` for each ``(nscan, nray)`` profile.  Following the existing dataset
    loader, a bin is contaminated when ``z_bin < z[cfb]``.  Invalid or out-of-
    range CFB indices do not mark any height; callers can audit those profiles
    separately with :func:`valid_cfb_mask`.
    """

    height = to_float_array(z)
    if height.ndim != 1 or height.size == 0:
        raise ValueError("z must be a non-empty one-dimensional height array")
    if not np.all(np.diff(height[np.isfinite(height)]) > 0):
        raise ValueError("finite z values must be strictly increasing")

    cfb_values = to_float_array(cfb)
    valid_cfb = valid_cfb_mask(cfb_values, height.size)
    safe_indices = np.zeros(cfb_values.shape, dtype=np.int64)
    safe_indices[valid_cfb] = cfb_values[valid_cfb].astype(np.int64)

    boundary_height = np.full(cfb_values.shape, np.nan, dtype=np.float64)
    boundary_height[valid_cfb] = height[safe_indices[valid_cfb]]
    return valid_cfb[..., np.newaxis] & (
        height.reshape((1,) * cfb_values.ndim + (height.size,))
        < boundary_height[..., np.newaxis]
    )


def valid_cfb_mask(cfb: Any, z_size: int) -> np.ndarray:
    """Return profiles containing an integer CFB index inside ``[0, z_size)``."""

    if z_size <= 0:
        raise ValueError("z_size must be positive")
    values = to_float_array(cfb)
    return (
        np.isfinite(values)
        & (values >= 0)
        & (values < z_size)
        & (values == np.floor(values))
    )


def overlap_mask(first: Any, second: Any) -> np.ndarray:
    """Return locations valid in both arrays, requiring identical shapes."""

    first_mask = valid_numeric_mask(first)
    second_mask = valid_numeric_mask(second)
    if first_mask.shape != second_mask.shape:
        raise ValueError(
            f"overlap inputs must have identical shapes: {first_mask.shape} != "
            f"{second_mask.shape}"
        )
    return first_mask & second_mask


def profile_has_observation(
    voxel_mask: Any, *, vertical_axis: int = -1
) -> np.ndarray:
    """Collapse a voxel mask to profiles containing at least one valid height."""

    mask = np.asarray(voxel_mask, dtype=bool)
    if mask.ndim == 0:
        raise ValueError("voxel_mask must have at least one dimension")
    return np.any(mask, axis=vertical_axis)
