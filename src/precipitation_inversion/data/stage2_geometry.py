"""GR-only sparse-geometry features for Stage-2 learning.

All arrays use ``(nscan, nray, z)``. Geometry is computed independently at
each physical height: a GR observation at another height never shortens the
distance. This keeps the feature deployable and preserves the special meaning
of the vertical axis.
"""

from __future__ import annotations

import numpy as np


DEFAULT_MAX_GR_DISTANCE = 8
DEFAULT_GR_LOCAL_DENSITY_RADIUS = 2


def _validate_observation_mask(observed: np.ndarray) -> np.ndarray:
    mask = np.asarray(observed)
    if mask.ndim != 3:
        raise ValueError("observation mask must have shape (nscan,nray,z)")
    if mask.dtype != np.bool_:
        raise TypeError("observation mask must be boolean")
    return mask


def _validate_max_distance(max_distance: int) -> int:
    if isinstance(max_distance, (bool, np.bool_)) or not isinstance(
        max_distance, (int, np.integer)
    ):
        raise TypeError("max_distance must be an integer")
    value = int(max_distance)
    if value <= 0:
        raise ValueError("max_distance must be positive")
    if value > np.iinfo(np.uint16).max:
        raise OverflowError("max_distance exceeds uint16 capacity")
    return value


def _validate_density_radius(radius: int) -> int:
    if isinstance(radius, (bool, np.bool_)) or not isinstance(
        radius, (int, np.integer)
    ):
        raise TypeError("radius must be an integer")
    value = int(radius)
    if value < 0:
        raise ValueError("radius must be non-negative")
    # Counts are returned as uint32. Reject an impossible window definition
    # before either allocating a huge padding buffer or overflowing the count.
    window_area = (2 * value + 1) ** 2
    if window_area > np.iinfo(np.uint32).max:
        raise OverflowError("density window exceeds uint32 count capacity")
    return value


def dilate_horizontal_once(observed: np.ndarray) -> np.ndarray:
    """Dilate one Chebyshev cell in scan/ray without mixing height levels."""

    mask = _validate_observation_mask(observed)
    nscan, nray, z_size = mask.shape
    padded = np.pad(mask, ((1, 1), (1, 1), (0, 0)), constant_values=False)
    result = np.zeros((nscan, nray, z_size), dtype=bool)
    for scan_offset in range(3):
        for ray_offset in range(3):
            result |= padded[
                scan_offset : scan_offset + nscan,
                ray_offset : ray_offset + nray,
                :,
            ]
    return result


def clipped_horizontal_chebyshev_distance(
    observed: np.ndarray,
    *,
    max_distance: int = DEFAULT_MAX_GR_DISTANCE,
) -> np.ndarray:
    """Return distance to the nearest direct GR observation, clipped at max.

    The returned ``uint16`` tensor has the same ``(nscan,nray,z)`` shape as the
    input. Direct observations are 0. Locations farther than ``max_distance``
    and complete height planes without observations both equal
    ``max_distance``. Iterative horizontal dilation gives exact Chebyshev
    distance up to the clipping radius without crossing the height axis.
    """

    mask = _validate_observation_mask(observed)
    limit = _validate_max_distance(max_distance)
    distance = np.full(mask.shape, limit, dtype=np.uint16)
    distance[mask] = 0
    reached = mask.copy()
    unreached = ~mask
    for radius in range(1, limit + 1):
        reached = dilate_horizontal_once(reached)
        newly_reached = reached & unreached
        distance[newly_reached] = radius
        unreached &= ~newly_reached
        if not np.any(unreached):
            break
    return distance


def scaled_horizontal_distance_to_observation(
    observed: np.ndarray,
    *,
    max_distance: int = DEFAULT_MAX_GR_DISTANCE,
) -> np.ndarray:
    """Return clipped GR distance as finite ``float32`` values in ``[0,1]``.

    ``0`` means a direct physical GR dBZ exists at the voxel; ``1`` means the
    nearest same-height direct observation is at least ``max_distance`` cells
    away, or the height plane contains no direct observation. Values between
    them retain distance order as ``distance / max_distance``.
    """

    limit = _validate_max_distance(max_distance)
    distance = clipped_horizontal_chebyshev_distance(
        observed, max_distance=limit
    )
    return distance.astype(np.float32) / np.float32(limit)


def horizontal_observation_count(
    observed: np.ndarray,
    *,
    radius: int = DEFAULT_GR_LOCAL_DENSITY_RADIUS,
) -> np.ndarray:
    """Count direct GR observations in a same-height horizontal window.

    Parameters
    ----------
    observed:
        Boolean direct-GR value mask with shape ``(nscan,nray,z)``.
    radius:
        Horizontal radius in grid cells. The default radius 2 gives a 5x5
        scan/ray window. Height is never pooled or mixed.

    Returns
    -------
    numpy.ndarray
        ``uint32`` counts with the same ``(nscan,nray,z)`` shape. Space beyond
        the physical orbit/ray boundary is treated as unobserved, so it adds
        zero to the count.

    Notes
    -----
    A two-dimensional summed-area table is evaluated independently for every
    height level. This is equivalent to a horizontal all-ones convolution but
    avoids introducing a SciPy dependency into the training data pipeline.
    """

    mask = _validate_observation_mask(observed)
    window_radius = _validate_density_radius(radius)
    if window_radius == 0:
        return mask.astype(np.uint32)

    window_size = 2 * window_radius + 1
    padded = np.pad(
        mask.astype(np.uint8, copy=False),
        (
            (window_radius, window_radius),
            (window_radius, window_radius),
            (0, 0),
        ),
        mode="constant",
        constant_values=0,
    )
    # Leading zero row/column makes each window sum a four-corner lookup.
    # Tensor shape changes as follows:
    # (nscan+2r,nray+2r,z) -> (nscan+2r+1,nray+2r+1,z).
    integral = np.pad(
        padded,
        ((1, 0), (1, 0), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    integral = integral.cumsum(axis=0, dtype=np.uint32).cumsum(
        axis=1, dtype=np.uint32
    )
    counts = (
        integral[window_size:, window_size:, :]
        + integral[:-window_size, :-window_size, :]
        - integral[:-window_size, window_size:, :]
        - integral[window_size:, :-window_size, :]
    )
    if counts.shape != mask.shape:  # defensive check against slicing mistakes
        raise RuntimeError("local-density window count changed the source shape")
    return counts.astype(np.uint32, copy=False)


def scaled_horizontal_observation_density(
    observed: np.ndarray,
    *,
    radius: int = DEFAULT_GR_LOCAL_DENSITY_RADIUS,
) -> np.ndarray:
    """Return fixed-area local GR observation density in ``[0,1]``.

    The output has shape ``(nscan,nray,z)`` and is computed as

    ``same-height direct-GR count / (2*radius+1)^2``.

    For the Stage-2 ``S2-3V-rho`` experiment, ``radius=2`` therefore means
    direct observations in a 5x5 horizontal window divided by 25. A completely
    unobserved height plane is all zero. The denominator remains 25 at physical
    scan/ray boundaries, because positions outside the available grid are not
    observations and must not make a boundary voxel appear artificially dense.
    """

    window_radius = _validate_density_radius(radius)
    counts = horizontal_observation_count(observed, radius=window_radius)
    window_area = np.float32((2 * window_radius + 1) ** 2)
    return counts.astype(np.float32) / window_area
