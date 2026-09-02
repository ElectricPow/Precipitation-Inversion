"""Storage-aware masks for stage-two GR-to-DPR reflectivity experiments.

Stage one only needs to know whether a physical reflectivity value exists, so
the shared :mod:`precipitation_inversion.data.masks` helpers deliberately merge
NetCDF masks/NaN and legacy values near ``-9999.9`` into one missing state.
Stage two must audit those states *before* they are merged because the source
files encode them separately and the old project loader treated them
differently.

This module uses storage-semantic names.  In particular, ``sentinel`` means
only "a finite value at or below the legacy cutoff".  It must not be renamed to
``no_echo`` or ``elevation_gap`` until the data producer confirms that physical
meaning.  The helpers are pure NumPy operations and never modify their input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .masks import LEGACY_FILL_CUTOFF, precipitation_label_mask, to_float_array


@dataclass(frozen=True)
class ReflectivityStorageMasks:
    """Mutually exclusive masks retaining the three source storage states.

    All arrays have the same source shape, normally ``(nscan, nray, z)``:

    ``native_missing``
        A NetCDF masked entry or a non-finite raw value (NaN/Inf).
    ``sentinel``
        An unmasked finite entry at or below ``sentinel_cutoff``.
    ``value``
        An unmasked finite entry above ``sentinel_cutoff``.  Negative and weak
        dBZ values remain physical values; no 9 dBZ clipping is performed.
    """

    native_missing: np.ndarray
    sentinel: np.ndarray
    value: np.ndarray
    sentinel_cutoff: float = LEGACY_FILL_CUTOFF

    def __post_init__(self) -> None:
        arrays = (self.native_missing, self.sentinel, self.value)
        if any(array.dtype != np.bool_ for array in arrays):
            raise TypeError("reflectivity storage masks must have boolean dtype")
        if len({array.shape for array in arrays}) != 1:
            raise ValueError("reflectivity storage masks must have identical shapes")
        membership = sum(array.astype(np.uint8, copy=False) for array in arrays)
        if not np.all(membership == 1):
            raise ValueError(
                "native_missing, sentinel, and value must be mutually exclusive "
                "and exhaustive"
            )
        if not np.isfinite(self.sentinel_cutoff):
            raise ValueError("sentinel_cutoff must be finite")

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the common source shape."""

        return self.value.shape

    @property
    def size(self) -> int:
        """Return the number of source elements."""

        return self.value.size

    @property
    def native_available(self) -> np.ndarray:
        """Return entries not represented by the native mask/non-finite state.

        This includes both physical values and legacy sentinels.  The name is
        intentionally about storage availability, not verified radar coverage.
        """

        return self.sentinel | self.value

    @property
    def physical_missing(self) -> np.ndarray:
        """Return entries without a retained physical reflectivity value."""

        return self.native_missing | self.sentinel

    def counts(self) -> dict[str, int]:
        """Return stable integer counts for audit JSON/CSV output."""

        return {
            "total": self.size,
            "native_missing": int(self.native_missing.sum()),
            "sentinel": int(self.sentinel.sum()),
            "value": int(self.value.sum()),
            "native_available": int(self.native_available.sum()),
        }


def _masked_data_and_native_missing(values: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return float64 raw data and the unreduced native/non-finite mask."""

    if hasattr(values, "dimensions") and hasattr(values, "__getitem__"):
        values = values[:]
    masked = np.ma.asarray(values)
    data = np.asarray(np.ma.getdata(masked), dtype=np.float64)
    source_mask = np.asarray(np.ma.getmaskarray(masked), dtype=bool)
    # Some files declare ``_FillValue=NaN`` while others may arrive as plain
    # ndarrays.  Treating every non-finite raw value as native missing makes the
    # result stable across both NetCDF representations without merging finite
    # -9999.9 sentinels into the same state.
    native_missing = source_mask | ~np.isfinite(data)
    return data, native_missing


def classify_reflectivity_storage(
    values: Any,
    *,
    sentinel_cutoff: float = LEGACY_FILL_CUTOFF,
) -> ReflectivityStorageMasks:
    """Classify a reflectivity array into native, sentinel, and value states.

    The returned masks preserve weak and negative dBZ values as ``value``.  No
    numerical replacement is performed and the input array is not modified.
    """

    if not np.isfinite(sentinel_cutoff):
        raise ValueError("sentinel_cutoff must be finite")
    data, native_missing = _masked_data_and_native_missing(values)
    sentinel = ~native_missing & (data <= sentinel_cutoff)
    value = ~native_missing & (data > sentinel_cutoff)
    return ReflectivityStorageMasks(
        native_missing=np.asarray(native_missing, dtype=bool),
        sentinel=np.asarray(sentinel, dtype=bool),
        value=np.asarray(value, dtype=bool),
        sentinel_cutoff=float(sentinel_cutoff),
    )


def physical_reflectivity_values(
    values: Any,
    *,
    masks: ReflectivityStorageMasks | None = None,
    dtype: str | np.dtype = np.float32,
    fill_value: float = np.nan,
) -> np.ndarray:
    """Return physical reflectivity values and replace both missing states.

    Parameters
    ----------
    values:
        Source masked array, ndarray, or NetCDF variable.
    masks:
        Optional result of :func:`classify_reflectivity_storage`.  Supplying it
        avoids classifying the same source twice.
    dtype:
        Floating output dtype.  Integer outputs are rejected because the
        default missing representation is NaN.
    fill_value:
        Value written where ``masks.value`` is false.  Stage-two normalization
        will later use ``0`` *in standardized space*; raw physical arrays should
        normally keep the default NaN.
    """

    output_dtype = np.dtype(dtype)
    if output_dtype.kind != "f":
        raise TypeError("dtype must be floating-point")
    data, _ = _masked_data_and_native_missing(values)
    storage = masks or classify_reflectivity_storage(values)
    if data.shape != storage.shape:
        raise ValueError(
            f"values and masks must have identical shapes: {data.shape} != "
            f"{storage.shape}"
        )
    output = np.full(data.shape, fill_value, dtype=output_dtype)
    output[storage.value] = data[storage.value].astype(output_dtype, copy=False)
    return output


def build_stage2_spatial_masks(
    dbz_gr_sparse: Any,
    dbz_dpr: Any,
    *,
    dbz_gr_interp: Any | None = None,
    pre_dpr: Any | None = None,
    sentinel_cutoff: float = LEGACY_FILL_CUTOFF,
) -> dict[str, np.ndarray]:
    """Build stage-two source-state, support, and recoverability masks.

    All reflectivity arrays must share ``(nscan, nray, z)``.  When ``pre_dpr``
    is supplied, ``occupancy_domain`` marks finite non-negative precipitation
    labels, including valid zeros.  It is a target-domain mask and must never be
    passed to the deployable model as an input channel.
    """

    gr = classify_reflectivity_storage(
        dbz_gr_sparse, sentinel_cutoff=sentinel_cutoff
    )
    dpr = classify_reflectivity_storage(dbz_dpr, sentinel_cutoff=sentinel_cutoff)
    if gr.shape != dpr.shape:
        raise ValueError(
            f"GR sparse and DPR reflectivity shapes differ: {gr.shape} != {dpr.shape}"
        )

    result: dict[str, np.ndarray] = {
        "gr_native_missing": gr.native_missing,
        "gr_native_available": gr.native_available,
        "gr_sentinel": gr.sentinel,
        "gr_value": gr.value,
        "dpr_native_missing": dpr.native_missing,
        "dpr_native_available": dpr.native_available,
        "dpr_sentinel": dpr.sentinel,
        "dpr_value": dpr.value,
        # Four exhaustive relationships under the physical-value definition.
        "q11_overlap": gr.value & dpr.value,
        "q01_dpr_only": ~gr.value & dpr.value,
        "q10_gr_only": gr.value & ~dpr.value,
        "q00_neither": ~gr.value & ~dpr.value,
    }

    if dbz_gr_interp is not None:
        interp = classify_reflectivity_storage(
            dbz_gr_interp, sentinel_cutoff=sentinel_cutoff
        )
        if interp.shape != gr.shape:
            raise ValueError(
                "GR interpolated and sparse reflectivity shapes differ: "
                f"{interp.shape} != {gr.shape}"
            )
        gap_proxy = ~gr.value & interp.value
        outside_proxy = ~gr.value & ~interp.value
        result.update(
            {
                "gr_interp_native_missing": interp.native_missing,
                "gr_interp_native_available": interp.native_available,
                "gr_interp_sentinel": interp.sentinel,
                "gr_interp_value": interp.value,
                # These are interpolation-reachability proxies, not verified
                # elevation-gap and true-outside-coverage labels.
                "gap_proxy": gap_proxy,
                "outside_proxy": outside_proxy,
                "dpr_only_gap_proxy": dpr.value & gap_proxy,
                "dpr_only_outside_proxy": dpr.value & outside_proxy,
            }
        )

    if pre_dpr is not None:
        precipitation = to_float_array(pre_dpr)
        if precipitation.shape != gr.shape:
            raise ValueError(
                f"pre_dpr and reflectivity shapes differ: {precipitation.shape} "
                f"!= {gr.shape}"
            )
        occupancy_domain = precipitation_label_mask(
            precipitation, exclude_clutter=False
        )
        result["occupancy_domain"] = occupancy_domain
        result["pre_positive"] = occupancy_domain & (precipitation > 0.0)

    return result


def mask_counts(masks: Mapping[str, np.ndarray]) -> dict[str, int]:
    """Return integer true counts after validating a common mask shape."""

    if not masks:
        raise ValueError("masks must not be empty")
    shapes = {np.asarray(mask).shape for mask in masks.values()}
    if len(shapes) != 1:
        raise ValueError(f"all masks must have one shape, got {sorted(shapes)}")
    counts: dict[str, int] = {}
    for name, mask in masks.items():
        array = np.asarray(mask)
        if array.dtype != np.bool_:
            raise TypeError(f"mask {name!r} must have boolean dtype")
        counts[name] = int(array.sum())
    return counts
