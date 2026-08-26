"""Core-with-halo patches for stage-one 3D U-Net precipitation inversion.

Each source orbit is partitioned into non-overlapping cores along ``nscan``.
Every core is read with left/right context (a halo), producing a fixed-length
input window. Loss is evaluated only in the core, so every physical voxel is
owned by exactly one patch while still receiving context across core borders.

Array convention in this module is ``(nscan, nray, z)``. Returned PyTorch
tensors use channel-first ``(C, nscan, nray, z)`` as required by ``Conv3d``.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from netCDF4 import Dataset as NetCDFDataset

from .dataset import atomic_save_json, atomic_save_npy, sha256_file
from .nc_reader import NCSample, read_nc_sample
from .transforms import PerLevelStandardizer


try:
    import torch
    from torch.utils.data import Dataset as TorchDataset
except (ImportError, OSError):  # pragma: no cover - index building needs no torch
    torch = None

    class TorchDataset:  # type: ignore[no-redef]
        """Fallback allowing NumPy-only patch-index construction."""


PATCH_INDEX_FORMAT_VERSION = 2
PATCH_INPUT_VARIABLE = "dbz_dpr"
PATCH_LABEL_VARIABLE = "pre_dpr"
PATCH_INPUT_CHANNELS = (
    "dbz_dpr_standardized",
    "dbz_dpr_valid",
    "height_scaled",
)
CFB_DISTANCE_INPUT_CHANNEL = "signed_cfb_distance_scaled"
CFB_INPUT_MODES = frozenset(
    {"baseline", "mask_below_cfb", "signed_distance"}
)

# Values returned in ``cfb_quality_region``.  This diagnostic tensor describes
# CFB quality independently of DPR/rain availability: missing observations and
# a physically valid profile can therefore coexist at the same voxel.
CFB_QUALITY_UNKNOWN = 0
CFB_QUALITY_RELIABLE = 1
CFB_QUALITY_WEAK = 2
CFB_QUALITY_CLUTTER = 3
PRECIPITATION_TYPE_UNKNOWN = -9999
PATCH_INDEX_DTYPE = np.dtype(
    [
        ("file_id", "<u2"),
        ("core_start", "<u2"),
        ("core_length", "u1"),
        ("input_count", "<u4"),
        ("positive_count", "<u4"),
    ]
)


def ceil_to_multiple(value: int, multiple: int) -> int:
    """Return the smallest multiple greater than or equal to ``value``."""

    if value <= 0 or multiple <= 0:
        raise ValueError("value and multiple must be positive")
    return ((value + multiple - 1) // multiple) * multiple


def core_starts(nscan: int, core_size: int) -> tuple[int, ...]:
    """Return non-overlapping core starts that cover ``[0, nscan)`` once."""

    if nscan <= 0 or core_size <= 0:
        raise ValueError("nscan and core_size must be positive")
    return tuple(range(0, nscan, core_size))


def _validate_patch_sizes(core_size: int, halo_size: int) -> None:
    if core_size <= 0:
        raise ValueError("core_size must be positive")
    if core_size > np.iinfo(PATCH_INDEX_DTYPE["core_length"]).max:
        raise OverflowError("core_size exceeds uint8 core_length capacity")
    if halo_size < 0:
        raise ValueError("halo_size must be non-negative")


def _validate_loss_weight_configuration(
    *,
    weak_cfb_layer_weights: Sequence[float],
    height_loss_weighting: str,
    height_loss_weight_min: float,
    height_loss_weight_max: float,
    intensity_loss_bin_edges: Sequence[float],
    intensity_loss_bin_weights: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and canonicalize loss-weight options as float32 arrays."""

    weak = np.asarray(tuple(weak_cfb_layer_weights), dtype=np.float32)
    if weak.ndim != 1 or np.any(~np.isfinite(weak)):
        raise ValueError("weak_cfb_layer_weights must be a finite 1-D sequence")
    if np.any((weak < 0.0) | (weak > 1.0)):
        raise ValueError("weak CFB weights must lie inside [0,1]")

    if height_loss_weighting not in {"none", "inverse_sqrt_frequency"}:
        raise ValueError(
            "height_loss_weighting must be 'none' or "
            "'inverse_sqrt_frequency'"
        )
    if (
        not np.isfinite(height_loss_weight_min)
        or not np.isfinite(height_loss_weight_max)
        or height_loss_weight_min <= 0.0
        or height_loss_weight_max < height_loss_weight_min
        or not (height_loss_weight_min <= 1.0 <= height_loss_weight_max)
    ):
        raise ValueError(
            "height loss weight limits must be finite, positive, ordered, "
            "and contain 1"
        )

    edges = np.asarray(tuple(intensity_loss_bin_edges), dtype=np.float32)
    bin_weights = np.asarray(tuple(intensity_loss_bin_weights), dtype=np.float32)
    if edges.ndim != 1 or np.any(~np.isfinite(edges)):
        raise ValueError("intensity_loss_bin_edges must be a finite 1-D sequence")
    if np.any(edges < 0.0) or np.any(np.diff(edges) <= 0.0):
        raise ValueError(
            "intensity loss bin edges must be non-negative and strictly increasing"
        )
    if bin_weights.ndim != 1 or np.any(~np.isfinite(bin_weights)):
        raise ValueError("intensity_loss_bin_weights must be a finite 1-D sequence")
    if edges.size == 0 and bin_weights.size == 0:
        return weak, edges, bin_weights
    if bin_weights.size != edges.size + 1:
        raise ValueError(
            "intensity_loss_bin_weights must contain one more value than edges"
        )
    if np.any(bin_weights <= 0.0):
        raise ValueError("intensity loss bin weights must be positive")
    return weak, edges, bin_weights


def _bounded_inverse_sqrt_frequency_weights(
    counts: np.ndarray,
    *,
    minimum: float,
    maximum: float,
) -> np.ndarray:
    """Return clipped inverse-sqrt height weights with count-weighted mean 1.

    ``counts`` comes from train-only normalization metadata.  Empty height
    levels receive ``maximum`` but cannot affect the normalization because
    their count is zero.  A scalar is found by bisection so clipping does not
    silently move the weighted average away from one.
    """

    values = np.asarray(counts, dtype=np.float64)
    if values.ndim != 1 or np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("normalization height counts must be finite and non-negative")
    positive = values > 0.0
    if not np.any(positive):
        raise ValueError("normalization contains no fitted height levels")
    reference = float(np.mean(values[positive]))
    raw = np.full(values.shape, np.inf, dtype=np.float64)
    raw[positive] = np.sqrt(reference / values[positive])

    def weighted_mean(scale: float) -> float:
        candidate = np.clip(raw * scale, minimum, maximum)
        return float(np.sum(candidate[positive] * values[positive]) / values[positive].sum())

    lower, upper = 0.0, 1.0
    while weighted_mean(upper) < 1.0:
        upper *= 2.0
    for _ in range(80):
        midpoint = 0.5 * (lower + upper)
        if weighted_mean(midpoint) < 1.0:
            lower = midpoint
        else:
            upper = midpoint
    result = np.clip(raw * (0.5 * (lower + upper)), minimum, maximum)
    return result.astype(np.float32)


def _height_loss_reference_counts(
    normalization: Mapping[str, Any],
    variable_statistics: Mapping[str, Any],
    heights_km: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Load reliable-label height counts independently of input statistics.

    New input-normalization files store an explicit
    ``height_loss_weight_reference`` selected by ``pre_positive_qc``. Legacy
    files fitted directly with that same mask may safely reuse their variable
    counts because input and reliable-label supports were identical there.
    Native/variable-valid input counts are never silently reused as target-loss
    frequencies.
    """

    reference = normalization.get("height_loss_weight_reference")
    if reference is not None:
        if not isinstance(reference, Mapping):
            raise TypeError("height_loss_weight_reference must be a mapping or null")
        if reference.get("selection_mask") != "pre_positive_qc":
            raise ValueError(
                "height-loss reference must use pre_positive_qc reliable labels"
            )
        reference_heights = np.asarray(reference.get("heights_km"), dtype=float)
        if reference_heights.shape != heights_km.shape or not np.allclose(
            reference_heights, heights_km, rtol=0.0, atol=1e-6
        ):
            raise ValueError(
                "height-loss reference and patch height coordinates differ"
            )
        counts = np.asarray(reference.get("count"), dtype=np.float64)
        source = "height_loss_weight_reference:pre_positive_qc"
        expected_total = reference.get("total_count")
        if expected_total is not None and int(expected_total) != int(counts.sum()):
            raise ValueError("height-loss reference total_count is inconsistent")
    elif normalization.get("selection_mask") == "pre_positive_qc":
        if "count" not in variable_statistics:
            raise KeyError(
                "legacy pre_positive_qc statistics are missing per-height count"
            )
        counts = np.asarray(variable_statistics["count"], dtype=np.float64)
        source = "legacy_input_statistics:pre_positive_qc"
    else:
        raise KeyError(
            "inverse_sqrt_frequency with variable-valid input normalization "
            "requires an independent height_loss_weight_reference selected by "
            "pre_positive_qc"
        )

    if (
        counts.shape != heights_km.shape
        or np.any(~np.isfinite(counts))
        or np.any(counts < 0.0)
        or np.any(counts != np.floor(counts))
    ):
        raise ValueError(
            "height-loss reference counts must be finite non-negative integers "
            "with one value per height"
        )
    return counts, source


def stage1_patch_dataset_kwargs(
    data_config: Mapping[str, Any],
    loss_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Map experiment configuration to :class:`Stage1PatchDataset` options.

    Training, evaluation, and visualization scripts should use this one helper
    instead of duplicating defaults.  CFB input/weak-label decisions live in
    the ``data`` section; objective reweighting lives in ``loss``.
    """

    return {
        "cfb_input_mode": str(data_config.get("cfb_input_mode", "baseline")),
        "cfb_distance_scale_km": float(
            data_config.get("cfb_distance_scale_km", 2.0)
        ),
        "weak_cfb_layer_weights": tuple(
            float(value)
            for value in data_config.get("weak_cfb_layer_weights", ())
        ),
        "height_loss_weighting": str(
            loss_config.get("height_loss_weighting", "none")
        ),
        "height_loss_weight_min": float(
            loss_config.get("height_loss_weight_min", 0.5)
        ),
        "height_loss_weight_max": float(
            loss_config.get("height_loss_weight_max", 3.0)
        ),
        "intensity_loss_bin_edges": tuple(
            float(value)
            for value in loss_config.get("intensity_loss_bin_edges", ())
        ),
        "intensity_loss_bin_weights": tuple(
            float(value)
            for value in loss_config.get("intensity_loss_bin_weights", ())
        ),
    }


def build_stage1_patch_index_records(
    paths: Sequence[str | Path],
    *,
    core_size: int = 32,
) -> tuple[np.ndarray, list[dict[str, Any]], np.ndarray, int, int]:
    """Build one record per non-overlapping scan core.

    Counts are measured only inside each core, never inside its future halo.
    ``input_count`` is DPR-valid and above CFB; ``positive_count`` is the
    reference-label ``pre_positive_qc`` count. Their equality is audited but is
    not assumed by the Dataset implementation.

    Returns ``(records, file_entries, heights, nray, processed_file_count)``.
    """

    _validate_patch_sizes(core_size, halo_size=0)
    if not paths:
        raise ValueError("at least one NetCDF path is required")
    if len(paths) - 1 > np.iinfo(PATCH_INDEX_DTYPE["file_id"]).max:
        raise OverflowError("too many files for uint16 file_id")

    record_parts: list[np.ndarray] = []
    file_entries: list[dict[str, Any]] = []
    reference_z: np.ndarray | None = None
    reference_nray: int | None = None
    cumulative = 0

    for file_id, path_value in enumerate(paths):
        path = Path(path_value).expanduser().resolve()
        sample = read_nc_sample(
            path,
            variables=("z", "dbz_dpr", "pre_dpr", "cfb"),
            dtype=np.float32,
            build_masks=True,
        )
        z = sample.variables["z"].astype(np.float64)
        if reference_z is None:
            reference_z = z.copy()
            reference_nray = sample.nray
        elif (
            sample.nray != reference_nray
            or z.shape != reference_z.shape
            or not np.allclose(z, reference_z, rtol=0.0, atol=1e-6)
        ):
            raise ValueError(f"Grid differs from earlier files: {path}")

        # Both masks have source shape (nscan, nray, z).
        positive = sample.masks["pre_positive_qc"]
        dpr_qc = sample.masks["dpr_reflectivity_valid"] & ~sample.masks["cfb_clutter"]
        starts = core_starts(sample.nscan, core_size)
        file_records = np.empty(len(starts), dtype=PATCH_INDEX_DTYPE)
        for record_position, start in enumerate(starts):
            stop = min(start + core_size, sample.nscan)
            file_records[record_position] = (
                file_id,
                start,
                stop - start,
                int(dpr_qc[start:stop].sum()),
                int(positive[start:stop].sum()),
            )

        record_parts.append(file_records)
        index_start = cumulative
        cumulative += len(file_records)
        file_entries.append(
            {
                "file_id": file_id,
                "sample_id": path.stem,
                "file_name": path.name,
                "file_path": str(path),
                "nscan": sample.nscan,
                "nray": sample.nray,
                "z_size": sample.z_size,
                "patch_count": len(file_records),
                "positive_patch_count": int(
                    np.count_nonzero(file_records["positive_count"])
                ),
                "input_count": int(file_records["input_count"].sum()),
                "positive_count": int(file_records["positive_count"].sum()),
                "index_start": index_start,
                "index_stop": cumulative,
            }
        )
        print(
            f"OK [{file_id + 1}/{len(paths)}] {path.name} "
            f"patches={len(file_records)} positive_patches="
            f"{file_entries[-1]['positive_patch_count']}",
            flush=True,
        )

    assert reference_z is not None and reference_nray is not None
    records = np.concatenate(record_parts)
    return records, file_entries, reference_z, reference_nray, len(paths)


@dataclass
class _CachedPatchFile:
    """Decoded full-file arrays, all in source shape ``(nscan,nray,z)``."""

    sample: NCSample


class Stage1PatchDataset(TorchDataset):
    """Return fixed 3D core-with-halo patches for stage-one U-Net training.

    Output tensor shapes are fixed for every source orbit:

    - ``inputs``: ``(C, padded_nscan, padded_nray, z)`` where ``C=3`` for
      baseline/CFB-masked input and ``C=4`` for signed-CFB-distance input;
    - ``target`` and masks: ``(1, padded_nscan, padded_nray, z)``.

    Input channel 0 is standardized DPR reflectivity with invalid cells filled
    by zero *after* standardization. Channel 1 is its explicit validity mask.
    Channel 2 is height scaled to ``[-1,1]``. In ``signed_distance`` mode,
    channel 3 is ``(height - CFB height) / cfb_distance_scale_km`` clipped to
    ``[-1,1]``; invalid-CFB and padding positions receive neutral zero.

    Only scan/ray are padded; height retains all 60 physical levels. Reliable
    loss occupies positive DPR labels at/above CFB in the central core. Optional
    weak CFB weights may add the first bins below CFB without conflating those
    lower-confidence labels with truly missing observations.
    """

    def __init__(
        self,
        index_metadata: str | Path,
        normalization_stats: str | Path,
        *,
        positive_only: bool = False,
        cache_size: int = 1,
        verify_hashes: bool = True,
        cfb_input_mode: str = "baseline",
        cfb_distance_scale_km: float = 2.0,
        weak_cfb_layer_weights: Sequence[float] = (),
        height_loss_weighting: str = "none",
        height_loss_weight_min: float = 0.5,
        height_loss_weight_max: float = 3.0,
        intensity_loss_bin_edges: Sequence[float] = (),
        intensity_loss_bin_weights: Sequence[float] = (),
    ) -> None:
        if torch is None:
            raise ModuleNotFoundError("PyTorch is required for Stage1PatchDataset")
        if cache_size <= 0:
            raise ValueError("cache_size must be positive")
        cfb_input_mode = str(cfb_input_mode).strip().lower()
        if cfb_input_mode not in CFB_INPUT_MODES:
            raise ValueError(
                f"cfb_input_mode must be one of {sorted(CFB_INPUT_MODES)}"
            )
        if not np.isfinite(cfb_distance_scale_km) or cfb_distance_scale_km <= 0.0:
            raise ValueError("cfb_distance_scale_km must be finite and positive")
        weak_weights, intensity_edges, intensity_weights = (
            _validate_loss_weight_configuration(
                weak_cfb_layer_weights=weak_cfb_layer_weights,
                height_loss_weighting=height_loss_weighting,
                height_loss_weight_min=height_loss_weight_min,
                height_loss_weight_max=height_loss_weight_max,
                intensity_loss_bin_edges=intensity_loss_bin_edges,
                intensity_loss_bin_weights=intensity_loss_bin_weights,
            )
        )
        self.metadata_path = Path(index_metadata).expanduser().resolve()
        self.normalization_path = Path(normalization_stats).expanduser().resolve()
        self.positive_only = positive_only
        self.cache_size = cache_size
        self.cfb_input_mode = cfb_input_mode
        self.cfb_distance_scale_km = float(cfb_distance_scale_km)
        self.weak_cfb_layer_weights = weak_weights
        self.height_loss_weighting = height_loss_weighting
        self.height_loss_weight_min = float(height_loss_weight_min)
        self.height_loss_weight_max = float(height_loss_weight_max)
        self.intensity_loss_bin_edges = intensity_edges
        self.intensity_loss_bin_weights = intensity_weights
        self._cache: OrderedDict[int, _CachedPatchFile] = OrderedDict()

        self.index_metadata = self._load_json(self.metadata_path)
        self.normalization = self._load_json(self.normalization_path)
        self._validate_metadata()
        self.input_normalization_selection = str(
            self.normalization["selection_mask"]
        )

        self.index_path = self.metadata_path.parent / self.index_metadata["index_file"]
        if not self.index_path.is_file():
            raise FileNotFoundError(f"Patch index not found: {self.index_path}")
        if verify_hashes and sha256_file(self.index_path) != self.index_metadata["index_sha256"]:
            raise ValueError(f"Patch index SHA-256 mismatch: {self.index_path}")
        self.records = np.load(self.index_path, mmap_mode="r", allow_pickle=False)
        if self.records.dtype != PATCH_INDEX_DTYPE or self.records.ndim != 1:
            raise ValueError("patch index has an unexpected dtype or shape")
        if len(self.records) != int(self.index_metadata["patch_count"]):
            raise ValueError("patch index length differs from metadata")

        self.source_files = list(self.index_metadata["files"])
        self.record_indices: np.ndarray | None = None
        if positive_only:
            self.record_indices = np.flatnonzero(
                self.records["positive_count"] > 0
            ).astype(np.int64)
        self.files = self._selected_file_entries()

        self.core_size = int(self.index_metadata["core_size"])
        self.halo_size = int(self.index_metadata["halo_size"])
        self.input_size = self.core_size + 2 * self.halo_size
        self.horizontal_multiple = int(
            self.index_metadata["horizontal_multiple"]
        )
        self.nray = int(self.index_metadata["nray"])
        self.z = np.asarray(self.index_metadata["heights_km"], dtype=np.float32)
        self.padded_shape = (
            ceil_to_multiple(self.input_size, self.horizontal_multiple),
            ceil_to_multiple(self.nray, self.horizontal_multiple),
            self.z.size,
        )
        self.feature_names = list(PATCH_INPUT_CHANNELS)
        if self.cfb_input_mode == "signed_distance":
            self.feature_names.append(CFB_DISTANCE_INPUT_CHANNEL)

        statistics = self.normalization["variables"].get(PATCH_INPUT_VARIABLE)
        if statistics is None:
            raise KeyError("normalization statistics are missing 'dbz_dpr'")
        if not np.allclose(
            np.asarray(statistics["heights_km"], dtype=float),
            self.z,
            rtol=0.0,
            atol=1e-6,
        ):
            raise ValueError("normalization and patch index height coordinates differ")
        self.standardizer = PerLevelStandardizer.from_dict(statistics)
        if self.height_loss_weighting == "inverse_sqrt_frequency":
            counts, self.height_loss_weight_source = _height_loss_reference_counts(
                self.normalization, statistics, self.z
            )
            self.height_loss_weights = _bounded_inverse_sqrt_frequency_weights(
                counts,
                minimum=self.height_loss_weight_min,
                maximum=self.height_loss_weight_max,
            )
        else:
            self.height_loss_weights = np.ones(self.z.shape, dtype=np.float32)
            self.height_loss_weight_source = "uniform"

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        if not path.is_file():
            raise FileNotFoundError(f"JSON metadata not found: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _validate_metadata(self) -> None:
        if self.index_metadata.get("format_version") != PATCH_INDEX_FORMAT_VERSION:
            raise ValueError("unsupported patch-index format version")
        if self.index_metadata.get("input_variable") != PATCH_INPUT_VARIABLE:
            raise ValueError("patch index must use dbz_dpr")
        if self.index_metadata.get("label_variable") != PATCH_LABEL_VARIABLE:
            raise ValueError("patch index must use pre_dpr")
        if self.index_metadata.get("label_transform") != "log1p":
            raise ValueError("patch index must use log1p labels")
        _validate_patch_sizes(
            int(self.index_metadata["core_size"]),
            int(self.index_metadata["halo_size"]),
        )
        if int(self.index_metadata["horizontal_multiple"]) <= 0:
            raise ValueError("horizontal_multiple must be positive")
        if tuple(self.index_metadata.get("input_channels", ())) != PATCH_INPUT_CHANNELS:
            raise ValueError("patch index has unexpected input channels")
        if int(self.index_metadata.get("height_padding", -1)) != 0:
            raise ValueError("patch index must not pad the height axis")
        input_size = int(self.index_metadata["core_size"]) + 2 * int(
            self.index_metadata["halo_size"]
        )
        horizontal_multiple = int(self.index_metadata["horizontal_multiple"])
        expected_shape = [
            ceil_to_multiple(input_size, horizontal_multiple),
            ceil_to_multiple(int(self.index_metadata["nray"]), horizontal_multiple),
            int(self.index_metadata["z_size"]),
        ]
        if self.index_metadata.get("padded_patch_shape") != expected_shape:
            raise ValueError(
                "padded_patch_shape does not match horizontal-only padding"
            )
        if self.normalization.get("scope") != "training_split_only":
            raise ValueError("normalization must be fitted on the training split only")
        processed_files = self.normalization.get("processed_file_count")
        validated_files = self.normalization.get("validated_file_count")
        if (
            processed_files is not None
            and validated_files is not None
            and processed_files != validated_files
        ):
            raise ValueError(
                "normalization processed_file_count differs from "
                "validated_file_count; debug --max-files statistics cannot be "
                "used for formal training"
            )
        normalization_selection = self.normalization.get("selection_mask")
        if normalization_selection not in {"variable_valid", "pre_positive_qc"}:
            raise ValueError(
                "stage-one normalization must use variable_valid input values "
                "or legacy pre_positive_qc values"
            )
        if (
            normalization_selection == "variable_valid"
            and self.normalization.get("label_qc_applied_to_input_statistics")
            is True
        ):
            raise ValueError(
                "variable_valid input normalization cannot apply label QC"
            )
        index_hash = self.index_metadata.get("split_manifest_sha256")
        stats_hash = self.normalization.get("split_manifest_sha256")
        if index_hash is not None and stats_hash is not None and index_hash != stats_hash:
            raise ValueError("patch index and normalization use different split manifests")

    def _selected_file_entries(self) -> list[dict[str, Any]]:
        """Build contiguous logical ranges, including after positive filtering."""

        result: list[dict[str, Any]] = []
        cumulative = 0
        for expected_id, source_entry in enumerate(self.source_files):
            if int(source_entry["file_id"]) != expected_id:
                raise ValueError("file_id values must be contiguous")
            raw_start = int(source_entry["index_start"])
            raw_stop = int(source_entry["index_stop"])
            if self.record_indices is None:
                selected_count = raw_stop - raw_start
            else:
                selected_count = int(
                    np.count_nonzero(
                        (self.record_indices >= raw_start)
                        & (self.record_indices < raw_stop)
                    )
                )
            entry = dict(source_entry)
            entry["index_start"] = cumulative
            cumulative += selected_count
            entry["index_stop"] = cumulative
            entry["sample_count"] = selected_count
            result.append(entry)
        if cumulative != len(self):
            raise ValueError("selected file ranges do not cover the logical Dataset")
        return result

    def __len__(self) -> int:
        return len(self.records) if self.record_indices is None else len(self.record_indices)

    def file_index_range(self, file_id: int) -> range:
        entry = self.files[file_id]
        return range(int(entry["index_start"]), int(entry["index_stop"]))

    @property
    def cached_file_ids(self) -> tuple[int, ...]:
        return tuple(self._cache)

    def clear_cache(self) -> None:
        self._cache.clear()

    def _load_file(self, file_id: int) -> _CachedPatchFile:
        cached = self._cache.pop(file_id, None)
        if cached is not None:
            self._cache[file_id] = cached
            return cached
        entry = self.source_files[file_id]
        # ``typePrecip`` is diagnostic-only.  Production files contain it, but
        # retaining an unknown fallback keeps older/minimal NetCDF fixtures
        # readable and prevents a reporting field from blocking training.
        with NetCDFDataset(entry["file_path"], "r") as source_dataset:
            has_precipitation_type = "typePrecip" in source_dataset.variables
        variables = ["z", "dbz_dpr", "pre_dpr", "cfb"]
        if has_precipitation_type:
            variables.append("typePrecip")
        sample = read_nc_sample(
            entry["file_path"],
            variables=variables,
            dtype=np.float32,
            build_masks=True,
        )
        if (
            sample.nscan != int(entry["nscan"])
            or sample.nray != self.nray
            or sample.z_size != self.z.size
            or not np.allclose(sample.variables["z"], self.z, rtol=0.0, atol=1e-6)
        ):
            raise ValueError(f"Source grid changed for {entry['file_path']}")
        cached = _CachedPatchFile(sample=sample)
        self._cache[file_id] = cached
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return cached

    def _record(self, logical_index: int) -> tuple[int, np.void]:
        if logical_index < 0:
            logical_index += len(self)
        if logical_index < 0 or logical_index >= len(self):
            raise IndexError(logical_index)
        record_index = (
            logical_index
            if self.record_indices is None
            else int(self.record_indices[logical_index])
        )
        return record_index, self.records[record_index]

    def _extract_source_window(
        self, source: np.ndarray, *, core_start: int, fill_value: float | bool
    ) -> np.ndarray:
        """Extract source ``(nscan,nray,z)`` into fixed ``(input_size,nray,z)``."""

        window_start = core_start - self.halo_size
        window_stop = window_start + self.input_size
        source_start = max(window_start, 0)
        source_stop = min(window_stop, source.shape[0])
        destination_start = source_start - window_start
        destination_stop = destination_start + (source_stop - source_start)

        # Before: source=(file_nscan,nray,z). After: fixed=(input_size,nray,z).
        output = np.full(
            (self.input_size, self.nray, self.z.size),
            fill_value,
            dtype=source.dtype,
        )
        output[destination_start:destination_stop] = source[source_start:source_stop]
        return output

    def _extract_profile_window(
        self,
        source: np.ndarray,
        *,
        core_start: int,
        fill_value: float | int,
    ) -> np.ndarray:
        """Extract ``(nscan,nray)`` profiles into ``(input_size,nray)``."""

        if source.ndim != 2 or source.shape[1] != self.nray:
            raise ValueError(f"unexpected profile array shape: {source.shape}")
        window_start = core_start - self.halo_size
        window_stop = window_start + self.input_size
        source_start = max(window_start, 0)
        source_stop = min(window_stop, source.shape[0])
        destination_start = source_start - window_start
        destination_stop = destination_start + (source_stop - source_start)
        output = np.full(
            (self.input_size, self.nray), fill_value, dtype=source.dtype
        )
        output[destination_start:destination_stop] = source[source_start:source_stop]
        return output

    def _pad_horizontal(
        self, array: np.ndarray, *, fill_value: float | bool
    ) -> np.ndarray:
        """Pad scan/ray at the high end while retaining all physical z levels."""

        if array.shape != (self.input_size, self.nray, self.z.size):
            raise ValueError(f"unexpected unpadded patch shape: {array.shape}")
        pads = tuple(
            (0, target - current)
            for current, target in zip(array.shape, self.padded_shape)
        )
        # Before=(input_size,nray,z); after=(padded_nscan,padded_nray,z).
        # The vertical axis is intentionally never padded because the
        # anisotropic U-Net does not downsample height.
        return np.pad(array, pads, mode="constant", constant_values=fill_value)

    def _pad_profiles(
        self, array: np.ndarray, *, fill_value: float | int
    ) -> np.ndarray:
        """Pad profile data ``(input_size,nray)`` to ``(Dp,Hp)``."""

        if array.shape != (self.input_size, self.nray):
            raise ValueError(f"unexpected unpadded profile shape: {array.shape}")
        return np.pad(
            array,
            (
                (0, self.padded_shape[0] - self.input_size),
                (0, self.padded_shape[1] - self.nray),
            ),
            mode="constant",
            constant_values=fill_value,
        )

    def _geometry_window(self, *, nscan: int, core_start: int) -> np.ndarray:
        """Mark physical source cells in local shape ``(input_size,nray,z)``."""

        window_start = core_start - self.halo_size
        window_stop = window_start + self.input_size
        source_start = max(window_start, 0)
        source_stop = min(window_stop, nscan)
        destination_start = source_start - window_start
        destination_stop = destination_start + (source_stop - source_start)
        geometry = np.zeros(
            (self.input_size, self.nray, self.z.size), dtype=bool
        )
        geometry[destination_start:destination_stop] = True
        return geometry

    def __getitem__(self, index: int) -> dict[str, Any]:
        record_index, record = self._record(index)
        file_id = int(record["file_id"])
        core_start = int(record["core_start"])
        core_length = int(record["core_length"])
        sample = self._load_file(file_id).sample

        # Source arrays/masks below all have shape (file_nscan,nray,z).
        dbz = sample.variables["dbz_dpr"]
        dpr_valid = sample.masks["dpr_reflectivity_valid"]
        clutter = sample.masks["cfb_clutter"]
        positive_qc = sample.masks["pre_positive_qc"]
        positive_native = sample.masks["pre_positive_native"]
        rain = sample.variables["pre_dpr"]
        cfb = sample.variables["cfb"]
        cfb_profile_valid = sample.masks["cfb_profile_valid"]

        # Fixed context windows have shape (input_size,nray,z), e.g. (64,49,60).
        dbz_window = self._extract_source_window(
            dbz, core_start=core_start, fill_value=np.nan
        )
        valid_window = self._extract_source_window(
            dpr_valid, core_start=core_start, fill_value=False
        )
        clutter_window = self._extract_source_window(
            clutter, core_start=core_start, fill_value=False
        )
        positive_qc_window = self._extract_source_window(
            positive_qc, core_start=core_start, fill_value=False
        )
        positive_native_window = self._extract_source_window(
            positive_native, core_start=core_start, fill_value=False
        )
        rain_window = self._extract_source_window(
            rain, core_start=core_start, fill_value=np.nan
        )
        cfb_window = self._extract_profile_window(
            cfb, core_start=core_start, fill_value=np.nan
        )
        cfb_profile_valid_window = self._extract_profile_window(
            cfb_profile_valid, core_start=core_start, fill_value=False
        )
        # Geometry has shape (input_size,nray,z). It distinguishes real grid
        # points (including missing DPR cells) from scan positions outside the
        # beginning/end of this variable-length source orbit.
        geometry_window = self._geometry_window(
            nscan=sample.nscan, core_start=core_start
        )

        # Convert each profile's CFB index into two signed diagnostics.  Before:
        # cfb_window=(input_size,nray). After: both arrays=(input_size,nray,z).
        # Negative values are below CFB, zero is the CFB bin, and positive
        # values are above CFB. Invalid CFB profiles deliberately remain NaN.
        safe_cfb_index = np.zeros(cfb_window.shape, dtype=np.int64)
        safe_cfb_index[cfb_profile_valid_window] = cfb_window[
            cfb_profile_valid_window
        ].astype(np.int64)
        cfb_height = np.full(cfb_window.shape, np.nan, dtype=np.float32)
        cfb_height[cfb_profile_valid_window] = self.z[
            safe_cfb_index[cfb_profile_valid_window]
        ]
        cfb_distance_km = (
            self.z[np.newaxis, np.newaxis, :] - cfb_height[..., np.newaxis]
        ).astype(np.float32)
        relative_cfb_level = (
            np.arange(self.z.size, dtype=np.float32)[np.newaxis, np.newaxis, :]
            - safe_cfb_index[..., np.newaxis]
        )
        cfb_voxel_valid = (
            cfb_profile_valid_window[..., np.newaxis] & geometry_window
        )
        cfb_distance_km[~cfb_voxel_valid] = np.nan
        relative_cfb_level[~cfb_voxel_valid] = np.nan

        # This categorical diagnostic separates profile quality from whether a
        # DPR/rain value exists.  It is never included in the model input.
        cfb_quality_region = np.full(
            geometry_window.shape, CFB_QUALITY_UNKNOWN, dtype=np.int8
        )
        cfb_quality_region[cfb_voxel_valid & (relative_cfb_level >= 0)] = (
            CFB_QUALITY_RELIABLE
        )
        cfb_quality_region[cfb_voxel_valid & (relative_cfb_level < 0)] = (
            CFB_QUALITY_CLUTTER
        )
        if self.weak_cfb_layer_weights.size:
            near_cfb = (
                cfb_voxel_valid
                & (relative_cfb_level < 0)
                & (relative_cfb_level >= -self.weak_cfb_layer_weights.size)
            )
            cfb_quality_region[near_cfb] = CFB_QUALITY_WEAK

        # Standardization is per z level. Invalid/unfitted values become 0 only
        # after ``effective_input`` records their missingness.
        transform_valid = valid_window
        if self.cfb_input_mode == "mask_below_cfb":
            # E1: below-CFB echoes are removed from both reflectivity and its
            # validity channel. True missingness and CFB masking remain
            # distinguishable through the returned CFB diagnostic tensors.
            transform_valid = valid_window & ~clutter_window
        normalized_dbz, effective_input = self.standardizer.transform(
            dbz_window,
            valid_mask=transform_valid,
            fill_value=0.0,
            dtype=np.float32,
        )

        # Geometry mask: (input_size,nray,z), true only for the non-padded core.
        core_mask = np.zeros_like(effective_input, dtype=bool)
        core_stop_in_window = self.halo_size + core_length
        core_mask[self.halo_size:core_stop_in_window, :, :] = True

        # Reliable supervision preserves the historical baseline: valid model
        # input + positive QC label at/above CFB, restricted to the unique core.
        reliable_output_mask = core_mask & effective_input & ~clutter_window
        reliable_loss_mask = (
            reliable_output_mask
            & positive_qc_window
            & np.isfinite(rain_window)
        )

        # Optional weak supervision considers only the explicitly configured
        # first N bins below CFB. It requires a native positive rain label and a
        # native DPR echo but remains separate from the reliable mask. Thus a
        # weight of zero is equivalent to no supervision, not a zero-rain label.
        weak_loss_mask = np.zeros_like(reliable_loss_mask, dtype=bool)
        # Inference support must never depend on the precipitation target.  It
        # is therefore built separately from ``weak_loss_mask`` using only the
        # physical core, native DPR availability, valid CFB geometry, and the
        # configured positive-weight layers below CFB.
        weak_output_mask = np.zeros_like(reliable_loss_mask, dtype=bool)
        quality_loss_weights = np.ones_like(normalized_dbz, dtype=np.float32)
        for layer_offset, layer_weight in enumerate(
            self.weak_cfb_layer_weights, start=1
        ):
            if layer_weight <= 0.0:
                continue
            layer_output_support = (
                core_mask
                & valid_window
                & cfb_voxel_valid
                & (relative_cfb_level == -layer_offset)
            )
            weak_output_mask |= layer_output_support
            layer_mask = (
                layer_output_support
                & positive_native_window
                & np.isfinite(rain_window)
            )
            weak_loss_mask |= layer_mask
            quality_loss_weights[layer_mask] = float(layer_weight)

        loss_mask = reliable_loss_mask | weak_loss_mask
        output_mask = reliable_output_mask | weak_output_mask
        target = np.zeros_like(normalized_dbz, dtype=np.float32)
        target[loss_mask] = np.log1p(rain_window[loss_mask]).astype(np.float32)

        # ``diagnostic_target`` exposes every native positive core label in
        # model space, including CFB-cluttered labels. Its explicit mask keeps
        # neutral fill zeros from being mistaken for observed zero rain.
        native_positive_mask = (
            core_mask & positive_native_window & np.isfinite(rain_window)
        )
        diagnostic_target = np.zeros_like(normalized_dbz, dtype=np.float32)
        diagnostic_target[native_positive_mask] = np.log1p(
            rain_window[native_positive_mask]
        ).astype(np.float32)

        # Combine independent height, intensity and CFB-quality factors. The
        # criterion divides by the selected weight sum, so zeros outside the
        # loss mask are safe and do not alter the loss scale.
        loss_weights = np.zeros_like(normalized_dbz, dtype=np.float32)
        combined_weights = np.broadcast_to(
            self.height_loss_weights[np.newaxis, np.newaxis, :],
            normalized_dbz.shape,
        ).copy()
        if self.intensity_loss_bin_weights.size:
            intensity_bin = np.searchsorted(
                self.intensity_loss_bin_edges, rain_window, side="right"
            )
            combined_weights *= self.intensity_loss_bin_weights[intensity_bin]
        combined_weights *= quality_loss_weights
        loss_weights[loss_mask] = combined_weights[loss_mask]

        # Height coordinate: (z,) -> broadcast (input_size,nray,z). Real grid
        # cells retain their physical height even when DPR is missing; orbit
        # boundary fill positions are neutral zero and have an invalid mask.
        z_min = float(self.z[0])
        z_max = float(self.z[-1])
        height_levels = 2.0 * (self.z - z_min) / (z_max - z_min) - 1.0
        height_scaled = np.broadcast_to(
            height_levels[np.newaxis, np.newaxis, :], normalized_dbz.shape
        ).copy()
        height_scaled[~geometry_window] = 0.0

        signed_cfb_distance_scaled = np.clip(
            cfb_distance_km / self.cfb_distance_scale_km, -1.0, 1.0
        )
        signed_cfb_distance_scaled[~np.isfinite(signed_cfb_distance_scaled)] = 0.0

        # typePrecip is a DPR diagnostic (1=stratiform, 2=convective, 3=other,
        # -1111=no precipitation), never a model feature. Missing/unsupported
        # files use -9999 so unknown is not confused with no precipitation.
        if "typePrecip" in sample.variables:
            precipitation_type_source = np.full(
                (sample.nscan, sample.nray),
                PRECIPITATION_TYPE_UNKNOWN,
                dtype=np.int16,
            )
            source_type = sample.variables["typePrecip"]
            finite_type = np.isfinite(source_type)
            precipitation_type_source[finite_type] = np.rint(
                source_type[finite_type]
            ).astype(np.int16)
        else:
            precipitation_type_source = np.full(
                (sample.nscan, sample.nray),
                PRECIPITATION_TYPE_UNKNOWN,
                dtype=np.int16,
            )
        precipitation_type_window = self._extract_profile_window(
            precipitation_type_source,
            core_start=core_start,
            fill_value=PRECIPITATION_TYPE_UNKNOWN,
        )

        # Pad only scan/ray: (input_size,nray,z) -> (Dp,Hp,z), e.g.
        # (64,49,60) -> (64,64,60). No artificial height levels are added.
        dbz_padded = self._pad_horizontal(normalized_dbz, fill_value=0.0)
        input_mask_padded = self._pad_horizontal(
            effective_input, fill_value=False
        )
        height_padded = self._pad_horizontal(height_scaled, fill_value=0.0)
        target_padded = self._pad_horizontal(target, fill_value=0.0)
        loss_mask_padded = self._pad_horizontal(loss_mask, fill_value=False)
        loss_weights_padded = self._pad_horizontal(loss_weights, fill_value=0.0)
        output_mask_padded = self._pad_horizontal(output_mask, fill_value=False)
        core_mask_padded = self._pad_horizontal(core_mask, fill_value=False)
        reliable_loss_mask_padded = self._pad_horizontal(
            reliable_loss_mask, fill_value=False
        )
        weak_loss_mask_padded = self._pad_horizontal(
            weak_loss_mask, fill_value=False
        )
        native_positive_mask_padded = self._pad_horizontal(
            native_positive_mask, fill_value=False
        )
        diagnostic_target_padded = self._pad_horizontal(
            diagnostic_target, fill_value=0.0
        )
        cfb_distance_km_padded = self._pad_horizontal(
            cfb_distance_km, fill_value=np.nan
        )
        relative_cfb_level_padded = self._pad_horizontal(
            relative_cfb_level, fill_value=np.nan
        )
        cfb_quality_region_padded = self._pad_horizontal(
            cfb_quality_region, fill_value=CFB_QUALITY_UNKNOWN
        )
        signed_cfb_distance_padded = self._pad_horizontal(
            signed_cfb_distance_scaled, fill_value=0.0
        )
        cfb_profile_valid_padded = self._pad_profiles(
            cfb_profile_valid_window, fill_value=False
        )
        precipitation_type_padded = self._pad_profiles(
            precipitation_type_window,
            fill_value=PRECIPITATION_TYPE_UNKNOWN,
        )

        # Stack three (Dp,Hp,z) arrays -> inputs=(3,Dp,Hp,z). The height
        # channel is geometric information, not an additional meteorological
        # observation such as p/t/q.
        input_channels = [
            dbz_padded,
            input_mask_padded.astype(np.float32),
            height_padded,
        ]
        if self.cfb_input_mode == "signed_distance":
            input_channels.append(signed_cfb_distance_padded)
        inputs = np.stack(input_channels, axis=0).astype(np.float32, copy=False)

        return {
            "inputs": torch.from_numpy(np.ascontiguousarray(inputs)),
            "target": torch.from_numpy(target_padded[np.newaxis, ...]),
            "loss_mask": torch.from_numpy(loss_mask_padded[np.newaxis, ...]),
            "loss_weights": torch.from_numpy(
                loss_weights_padded[np.newaxis, ...]
            ),
            "output_mask": torch.from_numpy(output_mask_padded[np.newaxis, ...]),
            "core_mask": torch.from_numpy(core_mask_padded[np.newaxis, ...]),
            "reliable_loss_mask": torch.from_numpy(
                reliable_loss_mask_padded[np.newaxis, ...]
            ),
            "weak_loss_mask": torch.from_numpy(
                weak_loss_mask_padded[np.newaxis, ...]
            ),
            "diagnostic_target": torch.from_numpy(
                diagnostic_target_padded[np.newaxis, ...]
            ),
            "native_positive_mask": torch.from_numpy(
                native_positive_mask_padded[np.newaxis, ...]
            ),
            # height diagnostics use compact broadcastable shapes
            # (1,1,1,z); CFB distance varies by profile and is fully dense.
            "height_km": torch.from_numpy(
                self.z[np.newaxis, np.newaxis, np.newaxis, :].copy()
            ),
            "height_index": torch.from_numpy(
                np.arange(self.z.size, dtype=np.int64)[
                    np.newaxis, np.newaxis, np.newaxis, :
                ]
            ),
            "cfb_distance_km": torch.from_numpy(
                cfb_distance_km_padded[np.newaxis, ...]
            ),
            "relative_cfb_level": torch.from_numpy(
                relative_cfb_level_padded[np.newaxis, ...]
            ),
            "cfb_profile_valid": torch.from_numpy(
                cfb_profile_valid_padded[
                    np.newaxis, ..., np.newaxis
                ]
            ),
            "cfb_quality_region": torch.from_numpy(
                cfb_quality_region_padded[np.newaxis, ...]
            ),
            "precipitation_type": torch.from_numpy(
                precipitation_type_padded[np.newaxis, ..., np.newaxis]
            ),
            "patch_index": torch.tensor(record_index, dtype=torch.int64),
            "file_id": torch.tensor(file_id, dtype=torch.int64),
            "core_start": torch.tensor(core_start, dtype=torch.int64),
            "core_length": torch.tensor(core_length, dtype=torch.int64),
            "original_shape": torch.tensor(
                [sample.nscan, sample.nray, sample.z_size], dtype=torch.int64
            ),
            "unpadded_shape": torch.tensor(
                [self.input_size, self.nray, self.z.size], dtype=torch.int64
            ),
            "padded_shape": torch.tensor(self.padded_shape, dtype=torch.int64),
        }

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["records"] = None
        state["_cache"] = OrderedDict()
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.records = np.load(self.index_path, mmap_mode="r", allow_pickle=False)


def save_patch_index(
    index_path: Path,
    metadata_path: Path,
    records: np.ndarray,
    metadata: Mapping[str, Any],
) -> None:
    """Atomically save an index followed by metadata containing its hash."""

    atomic_save_npy(index_path, records)
    value = dict(metadata)
    value["index_file"] = index_path.name
    value["index_sha256"] = sha256_file(index_path)
    atomic_save_json(metadata_path, value)
