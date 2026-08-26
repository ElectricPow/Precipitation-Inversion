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

    - ``inputs``: ``(3, padded_nscan, padded_nray, z)``;
    - ``target`` and masks: ``(1, padded_nscan, padded_nray, z)``.

    Input channel 0 is standardized DPR reflectivity with invalid cells filled
    by zero *after* standardization. Channel 1 is its explicit validity mask.
    Channel 2 is height scaled to ``[-1,1]``. Only the two horizontal axes are
    padded; height retains the 60 physical levels throughout the model.
    Loss and reconstruction masks are non-zero only in the central core.
    """

    def __init__(
        self,
        index_metadata: str | Path,
        normalization_stats: str | Path,
        *,
        positive_only: bool = False,
        cache_size: int = 1,
        verify_hashes: bool = True,
    ) -> None:
        if torch is None:
            raise ModuleNotFoundError("PyTorch is required for Stage1PatchDataset")
        if cache_size <= 0:
            raise ValueError("cache_size must be positive")
        self.metadata_path = Path(index_metadata).expanduser().resolve()
        self.normalization_path = Path(normalization_stats).expanduser().resolve()
        self.positive_only = positive_only
        self.cache_size = cache_size
        self._cache: OrderedDict[int, _CachedPatchFile] = OrderedDict()

        self.index_metadata = self._load_json(self.metadata_path)
        self.normalization = self._load_json(self.normalization_path)
        self._validate_metadata()

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
        if self.normalization.get("selection_mask") != "pre_positive_qc":
            raise ValueError("normalization must use pre_positive_qc")
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
        sample = read_nc_sample(
            entry["file_path"],
            variables=("z", "dbz_dpr", "pre_dpr", "cfb"),
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
        positive = sample.masks["pre_positive_qc"]
        rain = sample.variables["pre_dpr"]

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
        positive_window = self._extract_source_window(
            positive, core_start=core_start, fill_value=False
        )
        rain_window = self._extract_source_window(
            rain, core_start=core_start, fill_value=np.nan
        )
        # Geometry has shape (input_size,nray,z). It distinguishes real grid
        # points (including missing DPR cells) from scan positions outside the
        # beginning/end of this variable-length source orbit.
        geometry_window = self._geometry_window(
            nscan=sample.nscan, core_start=core_start
        )

        # Standardization is per z level. Invalid/unfitted values become 0 only
        # after ``effective_input`` records their missingness.
        normalized_dbz, effective_input = self.standardizer.transform(
            dbz_window,
            valid_mask=valid_window,
            fill_value=0.0,
            dtype=np.float32,
        )

        # Geometry mask: (input_size,nray,z), true only for the non-padded core.
        core_mask = np.zeros_like(effective_input, dtype=bool)
        core_stop_in_window = self.halo_size + core_length
        core_mask[self.halo_size:core_stop_in_window, :, :] = True

        # The model predicts only observed DPR support above CFB. Loss further
        # requires a valid positive reference label.
        output_mask = core_mask & effective_input & ~clutter_window
        loss_mask = output_mask & positive_window & np.isfinite(rain_window)
        target = np.zeros_like(normalized_dbz, dtype=np.float32)
        target[loss_mask] = np.log1p(rain_window[loss_mask]).astype(np.float32)

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

        # Pad only scan/ray: (input_size,nray,z) -> (Dp,Hp,z), e.g.
        # (64,49,60) -> (64,64,60). No artificial height levels are added.
        dbz_padded = self._pad_horizontal(normalized_dbz, fill_value=0.0)
        input_mask_padded = self._pad_horizontal(
            effective_input, fill_value=False
        )
        height_padded = self._pad_horizontal(height_scaled, fill_value=0.0)
        target_padded = self._pad_horizontal(target, fill_value=0.0)
        loss_mask_padded = self._pad_horizontal(loss_mask, fill_value=False)
        output_mask_padded = self._pad_horizontal(output_mask, fill_value=False)
        core_mask_padded = self._pad_horizontal(core_mask, fill_value=False)

        # Stack three (Dp,Hp,z) arrays -> inputs=(3,Dp,Hp,z). The height
        # channel is geometric information, not an additional meteorological
        # observation such as p/t/q.
        inputs = np.stack(
            [
                dbz_padded,
                input_mask_padded.astype(np.float32),
                height_padded,
            ],
            axis=0,
        ).astype(np.float32, copy=False)

        return {
            "inputs": torch.from_numpy(np.ascontiguousarray(inputs)),
            "target": torch.from_numpy(target_padded[np.newaxis, ...]),
            "loss_mask": torch.from_numpy(loss_mask_padded[np.newaxis, ...]),
            "output_mask": torch.from_numpy(output_mask_padded[np.newaxis, ...]),
            "core_mask": torch.from_numpy(core_mask_padded[np.newaxis, ...]),
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
