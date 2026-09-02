"""Core-with-halo patches for stage-two sparse-GR to dense-DPR learning.

Source arrays use ``(nscan,nray,z)``.  Returned PyTorch tensors are channel
first ``(C,D,H,Z)``.  Only scan/ray are padded; all 60 physical height levels
remain unchanged.  Raw reflectivity is classified before conversion so native
missing, finite sentinel, and physical dBZ states cannot be conflated.
"""

from __future__ import annotations

import json
import math
from collections import OrderedDict
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from netCDF4 import Dataset as NetCDFDataset

from .dataset import sha256_file
from .masks import clutter_mask_from_cfb, to_float_array
from .patch_dataset import ceil_to_multiple, core_starts, save_patch_index
from .stage2_masks import (
    build_stage2_spatial_masks,
    classify_reflectivity_storage,
    physical_reflectivity_values,
)
from .stage2_geometry import (
    DEFAULT_GR_LOCAL_DENSITY_RADIUS,
    DEFAULT_MAX_GR_DISTANCE,
    scaled_horizontal_observation_density,
    scaled_horizontal_distance_to_observation,
)
from .transforms import PerLevelStandardizer


try:
    import torch
    from torch.utils.data import Dataset as TorchDataset
except (ImportError, OSError):  # pragma: no cover - index creation needs no torch
    torch = None

    class TorchDataset:  # type: ignore[no-redef]
        """Fallback permitting NumPy-only index construction."""


STAGE2_PATCH_INDEX_FORMAT_VERSION = 1
# Historical patch-index metadata records the original four-channel baseline.
# Keep this tuple stable so existing indices/checkpoints remain reproducible.
STAGE2_INPUT_CHANNELS = (
    "dbz_gr_sparse_standardized",
    "gr_value_mask",
    "gr_native_available",
    "height_scaled",
)
# Runtime channel construction supports a canonical superset. Experiments may
# select an ordered subset without rebuilding the patch index because all
# channels are derived lazily from the same source NetCDF file.
STAGE2_AVAILABLE_INPUT_CHANNELS = (
    "dbz_gr_sparse_standardized",
    "gr_value_mask",
    "gr_native_available",
    "gr_nearest_distance_scaled",
    "gr_local_density_scaled",
    "dbz_gr_interp_standardized",
    "gr_interp_value_mask",
    # R1-O non-deployable spatial-completion upper bound.  These channels
    # reveal true DPR values only where the original GR geometry has a direct
    # physical sample and DPR also has a physical target.  They must never be
    # combined with deployable GR-value channels in one experiment.
    "dbz_dpr_sparse_anchor_standardized",
    "dpr_sparse_anchor_mask",
    "dpr_sparse_anchor_distance_scaled",
    "height_scaled",
)
STAGE2_THREE_CHANNEL_NATIVE_INPUT_CHANNELS = (
    "dbz_gr_sparse_standardized",
    "gr_native_available",
    "height_scaled",
)
STAGE2_FOUR_CHANNEL_DISTANCE_INPUT_CHANNELS = (
    "dbz_gr_sparse_standardized",
    "gr_value_mask",
    "gr_nearest_distance_scaled",
    "height_scaled",
)
STAGE2_FOUR_CHANNEL_DENSITY_INPUT_CHANNELS = (
    "dbz_gr_sparse_standardized",
    "gr_value_mask",
    "gr_local_density_scaled",
    "height_scaled",
)
STAGE2_FIVE_CHANNEL_INTERP_INPUT_CHANNELS = (
    "dbz_gr_sparse_standardized",
    "gr_value_mask",
    "dbz_gr_interp_standardized",
    "gr_interp_value_mask",
    "height_scaled",
)
STAGE2_R1_ORACLE_SPARSE_VALUE_INPUT_CHANNELS = (
    "dbz_dpr_sparse_anchor_standardized",
    "dpr_sparse_anchor_mask",
    "dpr_sparse_anchor_distance_scaled",
    "height_scaled",
)
STAGE2_INPUT_VARIABLE = "dbz_gr_sparse"
STAGE2_INTERP_VARIABLE = "dbz_gr_interp"
STAGE2_TARGET_VARIABLE = "dbz_dpr"
STAGE2_PATCH_INDEX_DTYPE = np.dtype(
    [
        ("file_id", "<u2"),
        ("core_start", "<u2"),
        ("core_length", "u1"),
        ("gr_count", "<u4"),
        ("dpr_count", "<u4"),
        ("q11_count", "<u4"),
        ("q01_count", "<u4"),
        ("q10_count", "<u4"),
        ("gap_target_count", "<u4"),
        ("outside_target_count", "<u4"),
        ("strong_dpr_count", "<u4"),
    ]
)


def _finite_float_sequence(value: Any, *, name: str) -> tuple[float, ...]:
    """Strictly parse one numerical configuration sequence.

    Strings and booleans are rejected instead of being silently converted to
    floats. This matters for loss weights because a configuration typo would
    otherwise change the training objective without changing tensor shapes.
    """

    if isinstance(value, (str, bytes)) or not isinstance(
        value, (Sequence, np.ndarray)
    ):
        raise TypeError(f"{name} must be a numerical sequence")
    result: list[float] = []
    for item in value:
        if isinstance(item, (bool, np.bool_)) or not isinstance(item, Real):
            raise TypeError(f"{name} must contain only real numbers")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{name} must contain only finite values")
        result.append(number)
    return tuple(result)


def validate_stage2_reflectivity_weighting(
    bin_edges_dbz: Sequence[float] = (),
    bin_weights: Sequence[float] = (),
) -> tuple[np.ndarray, np.ndarray]:
    """Validate physical-dBZ bins used only by the regression objective.

    Empty edges and weights disable weighting. Otherwise ``N`` increasing
    physical-dBZ edges require exactly ``N+1`` strictly positive weights.
    Negative dBZ edges remain legal because reflectivity itself can be
    negative; E3 happens to use the edges 25 and 35 dBZ.
    """

    edges_values = _finite_float_sequence(
        bin_edges_dbz, name="reflectivity intensity_bin_edges_dbz"
    )
    weight_values = _finite_float_sequence(
        bin_weights, name="reflectivity intensity_bin_weights"
    )
    edges = np.asarray(edges_values, dtype=np.float32)
    weights = np.asarray(weight_values, dtype=np.float32)
    if edges.size == 0 and weights.size == 0:
        return edges, weights
    if np.any(np.diff(edges) <= 0.0):
        raise ValueError(
            "reflectivity intensity_bin_edges_dbz must be strictly increasing"
        )
    if weights.size != edges.size + 1:
        raise ValueError(
            "reflectivity intensity_bin_weights must contain one more value "
            "than intensity_bin_edges_dbz"
        )
    if np.any(weights <= 0.0):
        raise ValueError("reflectivity intensity_bin_weights must be positive")
    return edges, weights


def physical_dbz_regression_weights(
    physical_dbz: np.ndarray,
    regression_mask: np.ndarray,
    *,
    bin_edges_dbz: Sequence[float],
    bin_weights: Sequence[float],
) -> np.ndarray:
    """Build supervision-only weights without changing source geometry.

    ``physical_dbz`` and ``regression_mask`` are both ``(D,H,Z)``. Binning
    uses raw physical DPR dBZ before per-height standardization. The returned
    float32 array has the same shape and is exactly zero outside ``M_dbz`` so
    missing values, halo, and later padding cannot affect loss. ``side=right``
    gives the E3 boundary contract ``25 -> middle`` and ``35 -> strongest``.
    """

    values = np.asarray(physical_dbz)
    selected = np.asarray(regression_mask)
    if values.shape != selected.shape:
        raise ValueError("physical dBZ and regression mask shapes differ")
    if selected.dtype != np.bool_:
        raise TypeError("regression_mask must be boolean")
    edges, weights = validate_stage2_reflectivity_weighting(
        bin_edges_dbz, bin_weights
    )
    result = np.zeros(values.shape, dtype=np.float32)
    if weights.size == 0 or not bool(selected.any()):
        return result
    selected_values = values[selected]
    if not np.all(np.isfinite(selected_values)):
        raise ValueError("selected physical DPR dBZ values must be finite")
    bins = np.searchsorted(edges, selected_values, side="right")
    result[selected] = weights[bins]
    return result


def stage2_patch_dataset_kwargs(loss_config: Mapping[str, Any]) -> dict[str, Any]:
    """Map Stage-2 loss configuration to Dataset supervision options.

    Reflectivity intensity weighting is label-derived and therefore generated
    by the Dataset, but it remains part of the loss configuration and never an
    input channel. Parsing is centralized so train and validation cannot use
    different defaults.
    """

    if not isinstance(loss_config, Mapping):
        raise TypeError("Stage-2 loss configuration must be a mapping")
    reflectivity = loss_config.get("reflectivity", {})
    if not isinstance(reflectivity, Mapping):
        raise TypeError("Stage-2 reflectivity loss configuration must be a mapping")
    edges = _finite_float_sequence(
        reflectivity.get("intensity_bin_edges_dbz", ()),
        name="reflectivity intensity_bin_edges_dbz",
    )
    weights = _finite_float_sequence(
        reflectivity.get("intensity_bin_weights", ()),
        name="reflectivity intensity_bin_weights",
    )
    # Validate before opening the dataset and again in its constructor so
    # direct API users receive the same strict contract.
    validate_stage2_reflectivity_weighting(edges, weights)
    return {
        "reflectivity_intensity_bin_edges_dbz": edges,
        "reflectivity_intensity_bin_weights": weights,
    }


def validate_stage2_input_channels(
    input_channels: Sequence[str] | None,
) -> tuple[str, ...]:
    """Return a safe ordered subset of the channels available in the index.

    ``None`` preserves the four-channel baseline.  Requiring canonical order
    prevents a configuration typo from silently changing checkpoint channel
    semantics. Reflectivity and physical height remain mandatory. The
    interpolated GR value and its value mask must always be selected together
    so a neutral standardized placeholder cannot be mistaken for an echo.
    """

    if input_channels is None:
        return STAGE2_INPUT_CHANNELS
    if isinstance(input_channels, (str, bytes)):
        raise TypeError("Stage-2 input_channels must be a sequence of names")
    selected = tuple(str(name) for name in input_channels)
    if len(selected) != len(set(selected)):
        raise ValueError("Stage-2 input_channels must not contain duplicates")
    unknown = sorted(set(selected).difference(STAGE2_AVAILABLE_INPUT_CHANNELS))
    if unknown:
        raise ValueError(
            "unknown Stage-2 input channels: " + ", ".join(unknown)
        )
    oracle_set = set(STAGE2_R1_ORACLE_SPARSE_VALUE_INPUT_CHANNELS)
    # ``height_scaled`` is shared by every Stage-2 experiment; it cannot be
    # used as the switch that identifies an Oracle-DPRSparse configuration.
    oracle_specific = oracle_set.difference({"height_scaled"})
    oracle_selected = oracle_specific.intersection(selected)
    if oracle_selected:
        missing = sorted(oracle_set.difference(selected))
        if missing:
            raise ValueError(
                "R1 Oracle-DPRSparse input_channels missing required channels: "
                + ", ".join(missing)
            )
        deployable_value_channels = {
            "dbz_gr_sparse_standardized",
            "gr_value_mask",
            "gr_native_available",
            "gr_nearest_distance_scaled",
            "gr_local_density_scaled",
            "dbz_gr_interp_standardized",
            "gr_interp_value_mask",
        }
        leaked = sorted(deployable_value_channels.intersection(selected))
        if leaked:
            raise ValueError(
                "R1 Oracle-DPRSparse channels cannot be mixed with GR-value "
                "experiment channels: " + ", ".join(leaked)
            )
    else:
        required = {"dbz_gr_sparse_standardized", "height_scaled"}
        missing = sorted(required.difference(selected))
        if missing:
            raise ValueError(
                "Stage-2 input_channels missing required channels: "
                + ", ".join(missing)
            )
    interp_pair = {
        "dbz_gr_interp_standardized",
        "gr_interp_value_mask",
    }
    if len(interp_pair.intersection(selected)) == 1:
        raise ValueError(
            "Stage-2 interpolation value and mask channels must be selected together"
        )
    canonical_subset = tuple(
        name for name in STAGE2_AVAILABLE_INPUT_CHANNELS if name in selected
    )
    if selected != canonical_subset:
        raise ValueError(
            "Stage-2 input_channels must follow canonical channel order: "
            + ", ".join(STAGE2_AVAILABLE_INPUT_CHANNELS)
        )
    return selected


def _validate_sizes(core_size: int, halo_size: int = 0) -> None:
    if core_size <= 0:
        raise ValueError("core_size must be positive")
    if core_size > np.iinfo(STAGE2_PATCH_INDEX_DTYPE["core_length"]).max:
        raise OverflowError("core_size exceeds uint8 core_length capacity")
    if halo_size < 0:
        raise ValueError("halo_size must be non-negative")


@dataclass
class _DecodedStage2File:
    """Storage-aware full-file arrays in source geometry."""

    z: np.ndarray
    gr_dbz: np.ndarray
    gr_interp_dbz: np.ndarray
    dpr_dbz: np.ndarray
    masks: dict[str, np.ndarray]
    cfb_clutter: np.ndarray
    gr_nearest_distance_scaled: np.ndarray | None = None
    gr_local_density_scaled: np.ndarray | None = None
    dpr_sparse_anchor_distance_scaled: np.ndarray | None = None

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.gr_dbz.shape


def _decode_stage2_file(
    path: Path,
    *,
    include_gr_distance: bool = False,
    include_gr_density: bool = False,
    include_dpr_sparse_anchor_distance: bool = False,
) -> _DecodedStage2File:
    """Read one file without collapsing raw reflectivity storage states."""

    with NetCDFDataset(path, "r") as dataset:
        required = {
            "z",
            "dbz_gr_sparse",
            "dbz_gr_interp",
            "dbz_dpr",
            "pre_dpr",
            "cfb",
        }
        missing = sorted(required.difference(dataset.variables))
        if missing:
            raise KeyError(f"{path.name} missing variables: {', '.join(missing)}")
        z = to_float_array(dataset["z"][:]).astype(np.float32, copy=False)
        gr_raw = dataset["dbz_gr_sparse"][:]
        interp_raw = dataset["dbz_gr_interp"][:]
        dpr_raw = dataset["dbz_dpr"][:]
        pre_raw = dataset["pre_dpr"][:]
        cfb_raw = dataset["cfb"][:]

    expected_shape = (gr_raw.shape[0], gr_raw.shape[1], z.size)
    for name, values in (
        ("dbz_gr_interp", interp_raw),
        ("dbz_dpr", dpr_raw),
        ("pre_dpr", pre_raw),
    ):
        if values.shape != expected_shape:
            raise ValueError(f"{path.name}:{name} shape {values.shape} != {expected_shape}")
    if z.ndim != 1 or not np.all(np.isfinite(z)) or not np.all(np.diff(z) > 0):
        raise ValueError(f"{path.name} has an invalid height coordinate")

    gr_states = classify_reflectivity_storage(gr_raw)
    interp_states = classify_reflectivity_storage(interp_raw)
    dpr_states = classify_reflectivity_storage(dpr_raw)
    masks = build_stage2_spatial_masks(
        gr_raw,
        dpr_raw,
        dbz_gr_interp=interp_raw,
        pre_dpr=pre_raw,
    )
    return _DecodedStage2File(
        z=z,
        gr_dbz=physical_reflectivity_values(gr_raw, masks=gr_states),
        gr_interp_dbz=physical_reflectivity_values(
            interp_raw, masks=interp_states
        ),
        dpr_dbz=physical_reflectivity_values(dpr_raw, masks=dpr_states),
        masks=masks,
        cfb_clutter=clutter_mask_from_cfb(cfb_raw, z),
        gr_nearest_distance_scaled=(
            scaled_horizontal_distance_to_observation(
                gr_states.value,
                max_distance=DEFAULT_MAX_GR_DISTANCE,
            )
            if include_gr_distance
            else None
        ),
        gr_local_density_scaled=(
            scaled_horizontal_observation_density(
                gr_states.value,
                radius=DEFAULT_GR_LOCAL_DENSITY_RADIUS,
            )
            if include_gr_density
            else None
        ),
        dpr_sparse_anchor_distance_scaled=(
            scaled_horizontal_distance_to_observation(
                masks["gr_value"] & masks["dpr_value"],
                max_distance=DEFAULT_MAX_GR_DISTANCE,
            )
            if include_dpr_sparse_anchor_distance
            else None
        ),
    )


def build_stage2_patch_index_records(
    paths: Sequence[Path],
    *,
    core_size: int = 32,
    strong_dbz_threshold: float = 35.0,
) -> tuple[np.ndarray, list[dict[str, Any]], np.ndarray, int, int]:
    """Build one compact record per non-overlapping scan core.

    Counts are restricted to the unique core, not its future halo.  They retain
    direct overlap, fill targets, interpolation proxies, and strong-DPR targets
    needed by the later sampler without dropping background-only patches.
    """

    _validate_sizes(core_size)
    if not paths:
        raise ValueError("at least one NetCDF path is required")
    if not np.isfinite(strong_dbz_threshold):
        raise ValueError("strong_dbz_threshold must be finite")
    if len(paths) - 1 > np.iinfo(STAGE2_PATCH_INDEX_DTYPE["file_id"]).max:
        raise OverflowError("too many files for uint16 file_id")

    record_parts: list[np.ndarray] = []
    files: list[dict[str, Any]] = []
    reference_z: np.ndarray | None = None
    reference_nray: int | None = None
    cumulative = 0
    count_names = STAGE2_PATCH_INDEX_DTYPE.names[3:]
    assert count_names is not None

    for file_id, path_value in enumerate(paths):
        path = Path(path_value).expanduser().resolve()
        decoded = _decode_stage2_file(path)
        nscan, nray, z_size = decoded.shape
        if reference_z is None:
            reference_z = decoded.z.astype(np.float64)
            reference_nray = nray
        elif (
            nray != reference_nray
            or decoded.z.shape != reference_z.shape
            or not np.allclose(decoded.z, reference_z, rtol=0.0, atol=1e-6)
        ):
            raise ValueError(f"grid differs from earlier files: {path}")

        domain = decoded.masks["occupancy_domain"]
        gr = decoded.masks["gr_value"]
        dpr = decoded.masks["dpr_value"]
        q11 = domain & gr & dpr
        q01 = domain & ~gr & dpr
        q10 = domain & gr & ~dpr
        gap = domain & decoded.masks["dpr_only_gap_proxy"]
        outside = domain & decoded.masks["dpr_only_outside_proxy"]
        strong = domain & dpr & (decoded.dpr_dbz >= strong_dbz_threshold)
        starts = core_starts(nscan, core_size)
        records = np.empty(len(starts), dtype=STAGE2_PATCH_INDEX_DTYPE)
        for position, start in enumerate(starts):
            stop = min(start + core_size, nscan)
            section = slice(start, stop)
            records[position] = (
                file_id,
                start,
                stop - start,
                int(gr[section].sum()),
                int(dpr[section].sum()),
                int(q11[section].sum()),
                int(q01[section].sum()),
                int(q10[section].sum()),
                int(gap[section].sum()),
                int(outside[section].sum()),
                int(strong[section].sum()),
            )
        index_start = cumulative
        cumulative += len(records)
        entry: dict[str, Any] = {
            "file_id": file_id,
            "sample_id": path.stem,
            "file_name": path.name,
            "file_path": str(path),
            "nscan": nscan,
            "nray": nray,
            "z_size": z_size,
            "patch_count": len(records),
            "target_patch_count": int(np.count_nonzero(records["dpr_count"])),
            "strong_patch_count": int(
                np.count_nonzero(records["strong_dpr_count"])
            ),
            "index_start": index_start,
            "index_stop": cumulative,
        }
        for name in count_names:
            entry[name] = int(records[name].sum(dtype=np.uint64))
        files.append(entry)
        record_parts.append(records)
        print(
            f"OK [{file_id + 1}/{len(paths)}] {path.name} "
            f"patches={len(records)} dpr={entry['dpr_count']:,}",
            flush=True,
        )

    assert reference_z is not None and reference_nray is not None
    return np.concatenate(record_parts), files, reference_z, reference_nray, len(paths)


class Stage2PatchDataset(TorchDataset):
    """Return configurable GR-only input channels and dual Stage-2 targets.

    ``inputs`` has shape ``(C,Dp,Hp,Z)``, where ``C`` is the selected channel
    count. Every target/mask has shape ``(1,Dp,Hp,Z)``. DPR/pre_dpr/CFB are
    returned only as supervision or diagnostics for ordinary experiments.
    Nearest-observation distance and optional interpolation channels are
    derived entirely from GR and are deployable inputs for controlled
    experiments. The explicitly named ``dpr_sparse_anchor`` R1-O channels are
    the sole exception: they form a non-deployable spatial-recovery upper
    bound. Local density is computed in a 5x5 scan/ray window independently at
    each physical height.
    """

    def __init__(
        self,
        index_metadata: str | Path,
        normalization_stats: str | Path,
        *,
        cache_size: int = 1,
        verify_hashes: bool = True,
        input_channels: Sequence[str] | None = None,
        reflectivity_intensity_bin_edges_dbz: Sequence[float] = (),
        reflectivity_intensity_bin_weights: Sequence[float] = (),
    ) -> None:
        if torch is None:
            raise ModuleNotFoundError("PyTorch is required for Stage2PatchDataset")
        if cache_size <= 0:
            raise ValueError("cache_size must be positive")
        self.metadata_path = Path(index_metadata).expanduser().resolve()
        self.normalization_path = Path(normalization_stats).expanduser().resolve()
        self.feature_names = list(validate_stage2_input_channels(input_channels))
        (
            self.reflectivity_intensity_bin_edges_dbz,
            self.reflectivity_intensity_bin_weights,
        ) = validate_stage2_reflectivity_weighting(
            reflectivity_intensity_bin_edges_dbz,
            reflectivity_intensity_bin_weights,
        )
        self.include_gr_distance = (
            "gr_nearest_distance_scaled" in self.feature_names
        )
        self.include_gr_density = "gr_local_density_scaled" in self.feature_names
        self.include_dpr_sparse_anchor_distance = (
            "dpr_sparse_anchor_distance_scaled" in self.feature_names
        )
        self.cache_size = cache_size
        self._cache: OrderedDict[int, _DecodedStage2File] = OrderedDict()
        self.index_metadata = self._load_json(self.metadata_path)
        self.normalization = self._load_json(self.normalization_path)
        self._validate_metadata()

        self.index_path = self.metadata_path.parent / self.index_metadata["index_file"]
        if not self.index_path.is_file():
            raise FileNotFoundError(f"patch index not found: {self.index_path}")
        if verify_hashes and sha256_file(self.index_path) != self.index_metadata["index_sha256"]:
            raise ValueError(f"patch index SHA-256 mismatch: {self.index_path}")
        self.records = np.load(self.index_path, mmap_mode="r", allow_pickle=False)
        if self.records.dtype != STAGE2_PATCH_INDEX_DTYPE or self.records.ndim != 1:
            raise ValueError("stage-two patch index has unexpected dtype or shape")
        if len(self.records) != int(self.index_metadata["patch_count"]):
            raise ValueError("patch index length differs from metadata")

        self.files = list(self.index_metadata["files"])
        self.core_size = int(self.index_metadata["core_size"])
        self.halo_size = int(self.index_metadata["halo_size"])
        self.input_size = self.core_size + 2 * self.halo_size
        self.horizontal_multiple = int(self.index_metadata["horizontal_multiple"])
        self.nray = int(self.index_metadata["nray"])
        self.z = np.asarray(self.index_metadata["heights_km"], dtype=np.float32)
        self.padded_shape = (
            ceil_to_multiple(self.input_size, self.horizontal_multiple),
            ceil_to_multiple(self.nray, self.horizontal_multiple),
            self.z.size,
        )
        self.gr_standardizer = self._standardizer(STAGE2_INPUT_VARIABLE)
        # Existing 3/4-channel experiments must not require interpolation
        # statistics. Load them only when the configured input actually uses
        # the GR-derived interpolation field.
        self.gr_interp_standardizer = (
            self._standardizer(STAGE2_INTERP_VARIABLE)
            if "dbz_gr_interp_standardized" in self.feature_names
            else None
        )
        self.dpr_standardizer = self._standardizer(STAGE2_TARGET_VARIABLE)

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        if not path.is_file():
            raise FileNotFoundError(f"JSON metadata not found: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _standardizer(self, variable: str) -> PerLevelStandardizer:
        statistics = self.normalization["variables"].get(variable)
        if statistics is None:
            raise KeyError(f"normalization statistics missing {variable!r}")
        heights = np.asarray(statistics.get("heights_km"), dtype=np.float64)
        if heights.shape != self.z.shape or not np.allclose(
            heights, self.z, rtol=0.0, atol=1e-6
        ):
            raise ValueError(f"{variable} normalization heights differ from index")
        standardizer = PerLevelStandardizer.from_dict(statistics)
        if not (
            np.all(np.isfinite(standardizer.mean))
            and np.all(np.isfinite(standardizer.std))
            and np.all(standardizer.std > standardizer.epsilon)
        ):
            raise ValueError(
                f"{variable} must have fitted non-constant statistics at every height"
            )
        return standardizer

    def _validate_metadata(self) -> None:
        metadata = self.index_metadata
        if metadata.get("format_version") != STAGE2_PATCH_INDEX_FORMAT_VERSION:
            raise ValueError("unsupported stage-two patch-index format version")
        if metadata.get("stage") != 2:
            raise ValueError("patch index must declare stage=2")
        if metadata.get("input_variable") != STAGE2_INPUT_VARIABLE:
            raise ValueError("patch index must use dbz_gr_sparse input")
        if metadata.get("target_variable") != STAGE2_TARGET_VARIABLE:
            raise ValueError("patch index must use dbz_dpr target")
        if tuple(metadata.get("input_channels", ())) != STAGE2_INPUT_CHANNELS:
            raise ValueError("patch index has unexpected stage-two input channels")
        _validate_sizes(int(metadata["core_size"]), int(metadata["halo_size"]))
        if int(metadata.get("horizontal_multiple", 0)) <= 0:
            raise ValueError("horizontal_multiple must be positive")
        if int(metadata.get("height_padding", -1)) != 0:
            raise ValueError("stage-two patches must not pad height")
        expected_shape = [
            ceil_to_multiple(
                int(metadata["core_size"]) + 2 * int(metadata["halo_size"]),
                int(metadata["horizontal_multiple"]),
            ),
            ceil_to_multiple(
                int(metadata["nray"]), int(metadata["horizontal_multiple"])
            ),
            int(metadata["z_size"]),
        ]
        if metadata.get("padded_patch_shape") != expected_shape:
            raise ValueError("padded_patch_shape does not match horizontal-only padding")
        if self.normalization.get("stage") != 2:
            raise ValueError("normalization must declare stage=2")
        if self.normalization.get("scope") != "training_split_only":
            raise ValueError("normalization must be fitted on the training split only")
        if self.normalization.get("selection_mask") != "reflectivity_storage_value":
            raise ValueError("stage-two normalization has unexpected selection mask")
        if self.normalization.get("processed_file_count") != self.normalization.get(
            "validated_file_count"
        ):
            raise ValueError("partial --max-files statistics cannot be used for training")
        index_hash = metadata.get("split_manifest_sha256")
        stats_hash = self.normalization.get("split_manifest_sha256")
        if index_hash is not None and stats_hash is not None and index_hash != stats_hash:
            raise ValueError("patch index and normalization use different split manifests")

    def __len__(self) -> int:
        return len(self.records)

    def file_index_range(self, file_id: int) -> range:
        entry = self.files[file_id]
        return range(int(entry["index_start"]), int(entry["index_stop"]))

    @property
    def cached_file_ids(self) -> tuple[int, ...]:
        return tuple(self._cache)

    def clear_cache(self) -> None:
        self._cache.clear()

    def _load_file(self, file_id: int) -> _DecodedStage2File:
        cached = self._cache.pop(file_id, None)
        if cached is not None:
            self._cache[file_id] = cached
            return cached
        entry = self.files[file_id]
        decoded = _decode_stage2_file(
            Path(entry["file_path"]),
            include_gr_distance=self.include_gr_distance,
            include_gr_density=self.include_gr_density,
            include_dpr_sparse_anchor_distance=(
                self.include_dpr_sparse_anchor_distance
            ),
        )
        if (
            decoded.shape != (int(entry["nscan"]), self.nray, self.z.size)
            or not np.allclose(decoded.z, self.z, rtol=0.0, atol=1e-6)
        ):
            raise ValueError(f"source grid changed for {entry['file_path']}")
        self._cache[file_id] = decoded
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return decoded

    def _extract_3d(
        self,
        source: np.ndarray,
        *,
        core_start: int,
        fill_value: float | bool | int,
    ) -> np.ndarray:
        if source.ndim != 3 or source.shape[1:] != (self.nray, self.z.size):
            raise ValueError(f"unexpected source shape: {source.shape}")
        window_start = core_start - self.halo_size
        window_stop = window_start + self.input_size
        source_start = max(window_start, 0)
        source_stop = min(window_stop, source.shape[0])
        destination_start = source_start - window_start
        destination_stop = destination_start + source_stop - source_start
        output = np.full(
            (self.input_size, self.nray, self.z.size),
            fill_value,
            dtype=source.dtype,
        )
        output[destination_start:destination_stop] = source[source_start:source_stop]
        return output

    def _geometry(self, *, nscan: int, core_start: int) -> np.ndarray:
        window_start = core_start - self.halo_size
        source_start = max(window_start, 0)
        source_stop = min(window_start + self.input_size, nscan)
        destination_start = source_start - window_start
        result = np.zeros((self.input_size, self.nray, self.z.size), dtype=bool)
        result[
            destination_start : destination_start + source_stop - source_start
        ] = True
        return result

    def _pad(self, array: np.ndarray, *, fill_value: float | bool | int) -> np.ndarray:
        if array.shape != (self.input_size, self.nray, self.z.size):
            raise ValueError(f"unexpected unpadded patch shape: {array.shape}")
        return np.pad(
            array,
            tuple(
                (0, target - current)
                for current, target in zip(array.shape, self.padded_shape)
            ),
            mode="constant",
            constant_values=fill_value,
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        record = self.records[index]
        file_id = int(record["file_id"])
        core_start = int(record["core_start"])
        core_length = int(record["core_length"])
        decoded = self._load_file(file_id)

        # Every source field below is (file_nscan,nray,z).  Extraction changes
        # it to fixed (input_size,nray,z), e.g. (64,49,60).
        gr_dbz = self._extract_3d(
            decoded.gr_dbz, core_start=core_start, fill_value=np.nan
        )
        gr_interp_dbz = self._extract_3d(
            decoded.gr_interp_dbz, core_start=core_start, fill_value=np.nan
        )
        # Distance is computed over the complete source orbit before Patch
        # extraction, so Patch boundaries do not create artificial distance.
        # Orbit-external halo is maximally far (1.0), not a direct observation.
        if decoded.gr_nearest_distance_scaled is None:
            gr_distance_scaled = np.zeros_like(gr_dbz, dtype=np.float32)
        else:
            gr_distance_scaled = self._extract_3d(
                decoded.gr_nearest_distance_scaled,
                core_start=core_start,
                fill_value=1.0,
            )
        # Density is also computed over the complete orbit before Patch
        # extraction. It is the fraction of direct GR values in a same-height
        # 5x5 horizontal window. Orbit-external halo is unobserved density 0.
        if decoded.gr_local_density_scaled is None:
            gr_density_scaled = np.zeros_like(gr_dbz, dtype=np.float32)
        else:
            gr_density_scaled = self._extract_3d(
                decoded.gr_local_density_scaled,
                core_start=core_start,
                fill_value=0.0,
            )
        # R1-O distance is computed from the complete-orbit Oracle anchor
        # geometry before Patch extraction.  An anchor exists iff GR has a
        # direct physical value and DPR has a physical dBZ at the same voxel.
        # Orbit-external context is maximally far, as for the GR distance
        # channel. Shape remains (input_size,nray,z).
        if decoded.dpr_sparse_anchor_distance_scaled is None:
            dpr_anchor_distance_scaled = np.zeros_like(gr_dbz, dtype=np.float32)
        else:
            dpr_anchor_distance_scaled = self._extract_3d(
                decoded.dpr_sparse_anchor_distance_scaled,
                core_start=core_start,
                fill_value=1.0,
            )
        dpr_dbz = self._extract_3d(
            decoded.dpr_dbz, core_start=core_start, fill_value=np.nan
        )
        windows = {
            name: self._extract_3d(values, core_start=core_start, fill_value=False)
            for name, values in decoded.masks.items()
        }
        clutter = self._extract_3d(
            decoded.cfb_clutter, core_start=core_start, fill_value=False
        )
        geometry = self._geometry(nscan=decoded.shape[0], core_start=core_start)
        core = np.zeros_like(geometry)
        core[self.halo_size : self.halo_size + core_length] = True
        core &= geometry

        gr_normalized, effective_gr = self.gr_standardizer.transform(
            gr_dbz,
            valid_mask=windows["gr_value"] & geometry,
            fill_value=0.0,
            dtype=np.float32,
        )
        target_dbz, effective_dpr = self.dpr_standardizer.transform(
            dpr_dbz,
            valid_mask=windows["dpr_value"] & geometry,
            fill_value=0.0,
            dtype=np.float32,
        )
        if not np.array_equal(effective_gr, windows["gr_value"] & geometry):
            raise RuntimeError("fitted GR statistics unexpectedly removed physical values")
        if not np.array_equal(effective_dpr, windows["dpr_value"] & geometry):
            raise RuntimeError("fitted DPR statistics unexpectedly removed target values")

        # Non-deployable R1-O sparse-DPR anchor input.  ``target_dbz`` is
        # already standardized with train-only DPR per-height statistics.
        # Only colocated physical GR/DPR voxels are exposed; every other value
        # is neutral 0 and is disambiguated by the explicit anchor mask.
        # Before stacking: all arrays are (input_size,nray,z).
        dpr_sparse_anchor_mask = (
            windows["gr_value"] & effective_dpr & geometry
        )
        dpr_sparse_anchor = np.where(
            dpr_sparse_anchor_mask, target_dbz, 0.0
        ).astype(np.float32, copy=False)

        # Optional interpolation tensors before padding are
        # (input_size,nray,z). Missing/sentinel/orbit-halo positions receive
        # neutral standardized 0.0 while the paired mask remains False.
        interp_normalized = np.zeros_like(gr_normalized, dtype=np.float32)
        effective_interp = windows["gr_interp_value"] & geometry
        if self.gr_interp_standardizer is not None:
            interp_normalized, transformed_interp = (
                self.gr_interp_standardizer.transform(
                    gr_interp_dbz,
                    valid_mask=effective_interp,
                    fill_value=0.0,
                    dtype=np.float32,
                )
            )
            if not np.array_equal(transformed_interp, effective_interp):
                raise RuntimeError(
                    "fitted interpolated-GR statistics unexpectedly removed "
                    "physical values"
                )

        occupancy = core & windows["occupancy_domain"]
        regression = core & effective_dpr
        overlap = occupancy & windows["q11_overlap"]
        dpr_only = occupancy & windows["q01_dpr_only"]
        gr_only = occupancy & windows["q10_gr_only"]
        neither = occupancy & windows["q00_neither"]
        gap_target = occupancy & windows["dpr_only_gap_proxy"]
        outside_target = occupancy & windows["dpr_only_outside_proxy"]
        below_cfb = core & effective_dpr & clutter

        # Supervision-only E3 weights are derived from raw physical DPR dBZ,
        # not from the standardized target tensor. Before padding each array is
        # (input_size,nray,z); only core DPR values in M_dbz receive a positive
        # weight. Nothing here is stacked into ``inputs``.
        regression_weights: np.ndarray | None = None
        if self.reflectivity_intensity_bin_weights.size:
            regression_weights = physical_dbz_regression_weights(
                dpr_dbz,
                regression,
                bin_edges_dbz=self.reflectivity_intensity_bin_edges_dbz,
                bin_weights=self.reflectivity_intensity_bin_weights,
            )

        z_min, z_max = float(self.z[0]), float(self.z[-1])
        height_levels = 2.0 * (self.z - z_min) / (z_max - z_min) - 1.0
        height = np.broadcast_to(
            height_levels[np.newaxis, np.newaxis, :], geometry.shape
        ).copy()
        # Orbit-boundary and high-ray padding receive neutral zero in ordinary
        # continuous channels. Distance is the exception: outside geometry is
        # maximally far (1.0), which is paired with gr_value_mask=0.
        height[~geometry] = 0.0

        gr_value = effective_gr
        gr_interp_value = effective_interp
        gr_native_available = windows["gr_native_available"] & geometry
        # Each entry is (input_size,nray,z), for example (64,49,60).
        # Stack only the configured GR-deployable subset to obtain
        # (C,input_size,nray,z). Existing channel order remains stable; each
        # controlled experiment inserts only its configured GR-derived fields
        # before the shared height channel.
        available_input_channels = {
            "dbz_gr_sparse_standardized": gr_normalized,
            "gr_value_mask": gr_value.astype(np.float32),
            "gr_native_available": gr_native_available.astype(np.float32),
            "gr_nearest_distance_scaled": gr_distance_scaled,
            "gr_local_density_scaled": gr_density_scaled,
            "dbz_gr_interp_standardized": interp_normalized,
            "gr_interp_value_mask": gr_interp_value.astype(np.float32),
            "dbz_dpr_sparse_anchor_standardized": dpr_sparse_anchor,
            "dpr_sparse_anchor_mask": dpr_sparse_anchor_mask.astype(np.float32),
            "dpr_sparse_anchor_distance_scaled": dpr_anchor_distance_scaled,
            "height_scaled": height.astype(np.float32),
        }
        padding_values = {
            "gr_nearest_distance_scaled": 1.0,
            "dpr_sparse_anchor_distance_scaled": 1.0,
        }
        inputs = np.stack(
            [
                self._pad(
                    available_input_channels[name],
                    fill_value=padding_values.get(name, 0.0),
                )
                for name in self.feature_names
            ],
            axis=0,
        ).astype(np.float32, copy=False)

        def bool_tensor(values: np.ndarray) -> Any:
            return torch.from_numpy(self._pad(values, fill_value=False)[np.newaxis])

        def float_tensor(values: np.ndarray) -> Any:
            return torch.from_numpy(
                self._pad(values.astype(np.float32), fill_value=0.0)[np.newaxis]
            )

        # target_support is a float label for BCE; target_valid is the exact
        # boolean DPR state.  Neither one is included in the model input.
        target_valid = effective_dpr & geometry
        item = {
            "inputs": torch.from_numpy(np.ascontiguousarray(inputs)),
            "target_dbz": float_tensor(target_dbz),
            "target_support": float_tensor(target_valid),
            "target_valid": bool_tensor(target_valid),
            "support_loss_mask": bool_tensor(occupancy),
            "occupancy_domain_mask": bool_tensor(occupancy),
            "regression_mask": bool_tensor(regression),
            "overlap_mask": bool_tensor(overlap),
            "dpr_only_mask": bool_tensor(dpr_only),
            "gr_only_mask": bool_tensor(gr_only),
            "neither_mask": bool_tensor(neither),
            "gap_proxy_mask": bool_tensor(gap_target),
            "outside_proxy_mask": bool_tensor(outside_target),
            "below_cfb_target_mask": bool_tensor(below_cfb),
            "core_mask": bool_tensor(core),
            "geometry_mask": bool_tensor(geometry),
            "gr_value_mask": bool_tensor(gr_value),
            "dpr_sparse_anchor_mask": bool_tensor(dpr_sparse_anchor_mask),
            "dpr_sparse_anchor_distance_scaled": float_tensor(
                dpr_anchor_distance_scaled
            ),
            "gr_interp_value_mask": bool_tensor(gr_interp_value),
            "gr_native_available": bool_tensor(gr_native_available),
            "gr_native_missing": bool_tensor(
                windows["gr_native_missing"] & geometry
            ),
            "gr_sentinel": bool_tensor(windows["gr_sentinel"] & geometry),
            "height_km": torch.from_numpy(
                self.z[np.newaxis, np.newaxis, np.newaxis, :].copy()
            ),
            "patch_index": torch.tensor(index, dtype=torch.int64),
            "file_id": torch.tensor(file_id, dtype=torch.int64),
            "core_start": torch.tensor(core_start, dtype=torch.int64),
            "core_length": torch.tensor(core_length, dtype=torch.int64),
            "original_shape": torch.tensor(decoded.shape, dtype=torch.int64),
            "unpadded_shape": torch.tensor(
                [self.input_size, self.nray, self.z.size], dtype=torch.int64
            ),
            "padded_shape": torch.tensor(self.padded_shape, dtype=torch.int64),
        }
        if regression_weights is not None:
            # (input_size,nray,z) -> padded (Dp,Hp,Z) -> (1,Dp,Hp,Z).
            # Halo, native missing, and horizontal padding remain exactly 0.
            item["regression_weights"] = float_tensor(regression_weights)
        return item

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["records"] = None
        state["_cache"] = OrderedDict()
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.records = np.load(self.index_path, mmap_mode="r", allow_pickle=False)


def save_stage2_patch_index(
    index_path: Path,
    metadata_path: Path,
    records: np.ndarray,
    metadata: Mapping[str, Any],
) -> None:
    """Save a typed stage-two index and hash-linked JSON metadata."""

    if records.dtype != STAGE2_PATCH_INDEX_DTYPE or records.ndim != 1:
        raise ValueError("records must be a 1-D STAGE2_PATCH_INDEX_DTYPE array")
    save_patch_index(index_path, metadata_path, records, metadata)
