"""Selective and mask-aware reader for collocated GR--DPR NetCDF samples.

The dataset contains files with a variable ``nscan`` length, while ``nray=49``
and ``z=60`` are currently stable.  This module never assumes a fixed ``nscan``
and can read a contiguous scan interval without loading the complete swath.

Missing reflectivity, valid zero precipitation, positive precipitation, and the
CFB clutter region are returned as separate masks.  The source NetCDF file is
opened read-only and closed before :class:`NCSample` is returned.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from netCDF4 import Dataset

from .masks import (
    clutter_mask_from_cfb,
    dpr_reflectivity_mask,
    gr_observation_mask,
    precipitation_label_mask,
    positive_rain_mask,
    profile_has_observation,
    to_float_array,
    valid_cfb_mask,
    zero_rain_mask,
)


DEFAULT_VARIABLES: tuple[str, ...] = (
    "z",
    "lat",
    "lon",
    "dbz_gr_sparse",
    "dbz_gr_interp",
    "dbz_dpr",
    "pre_dpr",
    "cfb",
    "typePrecip",
    "flagPrecip",
    "p",
    "t",
    "q",
)

# Exact dimension contracts catch transposed or structurally incompatible files
# before they silently enter training. Unknown variables can still be read; they
# simply do not receive an exact-dimension check.
KNOWN_DIMENSIONS: dict[str, tuple[str, ...]] = {
    "z": ("z",),
    "p": ("nscan", "nray", "z"),
    "t": ("nscan", "nray", "z"),
    "q": ("nscan", "nray", "z"),
    "lat": ("nscan", "nray"),
    "lon": ("nscan", "nray"),
    "dbz_gr_sparse": ("nscan", "nray", "z"),
    "dbz_gr_sparse_min": ("nscan", "nray", "z"),
    "dbz_gr_sparse_max": ("nscan", "nray", "z"),
    "dbz_gr_interp": ("nscan", "nray", "z"),
    "sw_gr_sparse": ("nscan", "nray", "z"),
    "sw_gr_sparse_min": ("nscan", "nray", "z"),
    "sw_gr_sparse_max": ("nscan", "nray", "z"),
    "dbz_dpr": ("nscan", "nray", "z"),
    "pre_dpr": ("nscan", "nray", "z"),
    "cfb": ("nscan", "nray"),
    "scan_id": ("nscan",),
    "time": ("nscan",),
    "nsrr_dpr": ("nscan", "nray"),
    "srr_dpr": ("nscan", "nray"),
    "binRealSurface": ("nscan", "nray"),
    "typePrecip": ("nscan", "nray"),
    "flagPrecip": ("nscan", "nray"),
}


@dataclass(frozen=True)
class VariableMetadata:
    """Source and returned metadata for one variable."""

    name: str
    dimensions: tuple[str, ...]
    source_shape: tuple[int, ...]
    returned_shape: tuple[int, ...]
    source_dtype: str
    returned_dtype: str
    units: str | None
    long_name: str | None


@dataclass
class NCSample:
    """In-memory result of one selective NetCDF read."""

    path: Path
    source_dimensions: dict[str, int]
    dimensions: dict[str, int]
    scan_start: int
    scan_stop: int
    variables: dict[str, np.ndarray]
    metadata: dict[str, VariableMetadata]
    masks: dict[str, np.ndarray]

    @property
    def nscan(self) -> int:
        return self.dimensions["nscan"]

    @property
    def nray(self) -> int:
        return self.dimensions["nray"]

    @property
    def z_size(self) -> int:
        return self.dimensions["z"]

    @property
    def shape_3d(self) -> tuple[int, int, int]:
        return self.nscan, self.nray, self.z_size

    def require_variable(self, name: str) -> np.ndarray:
        """Return one variable or raise a descriptive error."""

        if name not in self.variables:
            raise KeyError(
                f"Variable {name!r} was not requested; available: "
                f"{sorted(self.variables)}"
            )
        return self.variables[name]

    def require_mask(self, name: str) -> np.ndarray:
        """Return one derived mask or raise a descriptive error."""

        if name not in self.masks:
            raise KeyError(
                f"Mask {name!r} cannot be built from the requested variables; "
                f"available: {sorted(self.masks)}"
            )
        return self.masks[name]


def normalize_variable_names(variables: Iterable[str]) -> tuple[str, ...]:
    """Validate names and remove duplicates without changing their order."""

    result: list[str] = []
    seen: set[str] = set()
    for value in variables:
        name = str(value).strip()
        if not name:
            raise ValueError("variable names must be non-empty")
        if name not in seen:
            seen.add(name)
            result.append(name)
    if not result:
        raise ValueError("at least one variable must be requested")
    return tuple(result)


def normalize_scan_slice(scan_slice: slice | None, nscan: int) -> tuple[slice, int, int]:
    """Resolve a contiguous scan slice and return ``(slice, start, stop)``."""

    if nscan <= 0:
        raise ValueError("the NetCDF nscan dimension must be positive")
    if scan_slice is None:
        return slice(0, nscan, 1), 0, nscan
    if not isinstance(scan_slice, slice):
        raise TypeError("scan_slice must be a slice or None")
    start, stop, step = scan_slice.indices(nscan)
    if step != 1:
        raise ValueError("scan_slice must be contiguous; step must equal 1")
    if stop <= start:
        raise ValueError(
            f"scan_slice selects no scans after normalization: start={start}, stop={stop}"
        )
    return slice(start, stop, 1), start, stop


def variable_index(
    dimensions: tuple[str, ...], scan_slice: slice
) -> tuple[slice, ...]:
    """Apply the scan selection only along a variable's ``nscan`` dimension."""

    index = [slice(None)] * len(dimensions)
    if "nscan" in dimensions:
        index[dimensions.index("nscan")] = scan_slice
    return tuple(index)


def _validate_known_dimensions(name: str, dimensions: tuple[str, ...]) -> None:
    expected = KNOWN_DIMENSIONS.get(name)
    if expected is not None and dimensions != expected:
        raise ValueError(
            f"Variable {name!r} has dimensions {dimensions}, expected {expected}"
        )


def _read_variable(
    variable: Any,
    *,
    name: str,
    scan_slice: slice,
    dtype: np.dtype,
    strict_dimensions: bool,
) -> tuple[np.ndarray, VariableMetadata]:
    dimensions = tuple(variable.dimensions)
    if strict_dimensions:
        _validate_known_dimensions(name, dimensions)
    values = variable[variable_index(dimensions, scan_slice)]
    array = to_float_array(values).astype(dtype, copy=False)
    metadata = VariableMetadata(
        name=name,
        dimensions=dimensions,
        source_shape=tuple(variable.shape),
        returned_shape=tuple(array.shape),
        source_dtype=str(variable.dtype),
        returned_dtype=str(array.dtype),
        units=str(variable.units) if hasattr(variable, "units") else None,
        long_name=str(variable.long_name) if hasattr(variable, "long_name") else None,
    )
    return array, metadata


def build_standard_masks(variables: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Build every standard mask supported by the variables currently loaded.

    Masks are named explicitly by semantics.  In particular,
    ``pre_valid_native`` preserves valid zero rain, while ``pre_valid_qc`` also
    removes bins below CFB.  No missing reflectivity is converted to zero.
    """

    masks: dict[str, np.ndarray] = {}

    if "dbz_gr_sparse" in variables:
        mask = gr_observation_mask(variables["dbz_gr_sparse"])
        masks["gr_sparse_observed"] = mask
        masks["gr_sparse_profile_observed"] = profile_has_observation(mask)
    if "dbz_gr_interp" in variables:
        mask = gr_observation_mask(variables["dbz_gr_interp"])
        masks["gr_interp_observed"] = mask
        masks["gr_interp_profile_observed"] = profile_has_observation(mask)
    if "dbz_dpr" in variables:
        mask = dpr_reflectivity_mask(variables["dbz_dpr"])
        masks["dpr_reflectivity_valid"] = mask
        masks["dpr_reflectivity_profile_valid"] = profile_has_observation(mask)

    if "cfb" in variables and "z" in variables:
        masks["cfb_profile_valid"] = valid_cfb_mask(
            variables["cfb"], variables["z"].size
        )
        masks["cfb_clutter"] = clutter_mask_from_cfb(
            variables["cfb"], variables["z"]
        )

    if "pre_dpr" in variables:
        precipitation = variables["pre_dpr"]
        native_valid = precipitation_label_mask(
            precipitation, exclude_clutter=False
        )
        masks["pre_valid_native"] = native_valid
        masks["pre_zero_native"] = zero_rain_mask(
            precipitation, valid_mask=native_valid
        )
        masks["pre_positive_native"] = positive_rain_mask(
            precipitation, valid_mask=native_valid
        )

        if "cfb" in variables and "z" in variables:
            qc_valid = precipitation_label_mask(
                precipitation, cfb=variables["cfb"], z=variables["z"]
            )
            masks["pre_valid_qc"] = qc_valid
            masks["pre_zero_qc"] = zero_rain_mask(
                precipitation, valid_mask=qc_valid
            )
            masks["pre_positive_qc"] = positive_rain_mask(
                precipitation, valid_mask=qc_valid
            )

    if "gr_sparse_observed" in masks and "dpr_reflectivity_valid" in masks:
        masks["gr_dpr_overlap"] = (
            masks["gr_sparse_observed"] & masks["dpr_reflectivity_valid"]
        )
    return masks


def validate_sample(sample: NCSample) -> None:
    """Validate returned shapes and the ascending height coordinate."""

    for name, array in sample.variables.items():
        dimensions = sample.metadata[name].dimensions
        expected_shape = tuple(sample.dimensions[dimension] for dimension in dimensions)
        if array.shape != expected_shape:
            raise ValueError(
                f"Returned variable {name!r} has shape {array.shape}, "
                f"expected {expected_shape} from dimensions {dimensions}"
            )
    if "z" in sample.variables:
        z = sample.variables["z"]
        if not np.all(np.isfinite(z)):
            raise ValueError("z contains missing or non-finite heights")
        if not np.all(np.diff(z) > 0):
            raise ValueError("z must be strictly increasing")


def read_nc_sample(
    path: str | Path,
    *,
    variables: Iterable[str] = DEFAULT_VARIABLES,
    scan_slice: slice | None = None,
    dtype: str | np.dtype = np.float32,
    build_masks: bool = True,
    strict_dimensions: bool = True,
) -> NCSample:
    """Read selected variables from one NetCDF sample.

    Parameters
    ----------
    path:
        Source ``.nc`` file. It is never modified.
    variables:
        Variable names to load. Only these variables are read into memory.
        Derived masks are created only when their source variables are present.
    scan_slice:
        Optional contiguous slice along ``nscan`` (for example ``slice(100,
        132)``). Coordinates and every other scan-dependent variable receive the
        identical selection; ``z`` remains unchanged.
    dtype:
        Returned floating dtype. ``float32`` is the memory-conscious default.
    build_masks:
        Build the standard masks described in :func:`build_standard_masks`.
    strict_dimensions:
        Require known variables to use their documented dimension order.
    """

    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"NetCDF sample not found: {source_path}")
    requested = normalize_variable_names(variables)
    output_dtype = np.dtype(dtype)
    if output_dtype.kind != "f":
        raise TypeError("dtype must be a floating-point dtype so NaN can represent missing data")

    loaded: dict[str, np.ndarray] = {}
    metadata: dict[str, VariableMetadata] = {}
    with Dataset(source_path, "r") as dataset:
        source_dimensions = {
            name: len(dimension) for name, dimension in dataset.dimensions.items()
        }
        for required_dimension in ("nscan", "nray", "z"):
            if required_dimension not in source_dimensions:
                raise KeyError(
                    f"NetCDF sample is missing dimension {required_dimension!r}: "
                    f"{source_path}"
                )
        normalized_slice, scan_start, scan_stop = normalize_scan_slice(
            scan_slice, source_dimensions["nscan"]
        )
        missing_variables = sorted(set(requested).difference(dataset.variables))
        if missing_variables:
            raise KeyError(
                "NetCDF sample is missing requested variables: "
                + ", ".join(missing_variables)
            )
        for name in requested:
            loaded[name], metadata[name] = _read_variable(
                dataset.variables[name],
                name=name,
                scan_slice=normalized_slice,
                dtype=output_dtype,
                strict_dimensions=strict_dimensions,
            )

    dimensions = dict(source_dimensions)
    dimensions["nscan"] = scan_stop - scan_start
    sample = NCSample(
        path=source_path,
        source_dimensions=source_dimensions,
        dimensions=dimensions,
        scan_start=scan_start,
        scan_stop=scan_stop,
        variables=loaded,
        metadata=metadata,
        masks=build_standard_masks(loaded) if build_masks else {},
    )
    validate_sample(sample)
    return sample
