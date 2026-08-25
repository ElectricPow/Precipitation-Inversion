"""Stage-one positive-rain sample index and PyTorch Dataset.

Stage one learns precipitation *intensity conditioned on detected rain*.  Index
records therefore contain only ``pre_positive_qc`` cells with finite DPR
reflectivity and auxiliary inputs.  They must not be used to report rain/no-rain
detection skill because the DPR reflectivity mask already reveals occurrence.

The compact NumPy index is sorted by file and memory-mapped.  Each DataLoader
worker owns a small LRU cache of decoded NetCDF files; no file handle is kept
open and cache contents are cleared when the Dataset is pickled for workers.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .nc_reader import NCSample, read_nc_sample
from .transforms import PerLevelStandardizer


try:  # Index construction remains usable before PyTorch is installed.
    import torch
    from torch.utils.data import Dataset as TorchDataset
except (ImportError, OSError):  # pragma: no cover - exercised without usable torch
    torch = None

    class TorchDataset:  # type: ignore[no-redef]
        """Placeholder allowing NumPy-only index construction."""


STAGE1_INPUT_VARIABLES: tuple[str, ...] = ("dbz_dpr", "p", "t", "q")
STAGE1_READ_VARIABLES: tuple[str, ...] = (
    "z",
    "dbz_dpr",
    "pre_dpr",
    "cfb",
    *STAGE1_INPUT_VARIABLES[1:],
)
STAGE1_INDEX_DTYPE = np.dtype(
    [
        ("file_id", "<u2"),
        ("scan", "<u2"),
        ("ray", "u1"),
        ("level", "u1"),
    ]
)
STAGE1_INDEX_FORMAT_VERSION = 1


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_path_list(path: str | Path) -> list[Path]:
    """Load, resolve, and validate a newline-separated NetCDF path list."""

    list_path = Path(path)
    if not list_path.is_file():
        raise FileNotFoundError(f"File list not found: {list_path}")
    paths = [
        Path(line.strip()).expanduser().resolve()
        for line in list_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not paths:
        raise ValueError(f"File list is empty: {list_path}")
    if len(paths) != len(set(paths)):
        raise ValueError(f"File list contains duplicate paths: {list_path}")
    missing = [source for source in paths if not source.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} NetCDF path(s) do not exist; first: {missing[0]}"
        )
    return paths


def _source_nscan(path: Path) -> int:
    # Reading z alone is inexpensive and also validates the three core dimensions.
    sample = read_nc_sample(path, variables=("z",), build_masks=False)
    return sample.source_dimensions["nscan"]


def _index_records(
    file_id: int, scan_start: int, selected: np.ndarray
) -> np.ndarray:
    positions = np.argwhere(selected)
    if not positions.size:
        return np.empty(0, dtype=STAGE1_INDEX_DTYPE)
    absolute_scan = positions[:, 0].astype(np.int64) + scan_start
    limits = {
        "file_id": file_id,
        "scan": int(absolute_scan.max()),
        "ray": int(positions[:, 1].max()),
        "level": int(positions[:, 2].max()),
    }
    for name, maximum in limits.items():
        if maximum > np.iinfo(STAGE1_INDEX_DTYPE[name]).max:
            raise OverflowError(
                f"{name}={maximum} exceeds compact index dtype "
                f"{STAGE1_INDEX_DTYPE[name]}"
            )
    records = np.empty(positions.shape[0], dtype=STAGE1_INDEX_DTYPE)
    records["file_id"] = file_id
    records["scan"] = absolute_scan
    records["ray"] = positions[:, 1]
    records["level"] = positions[:, 2]
    return records


def build_stage1_index_records(
    paths: Sequence[Path], *, scan_chunk_size: int = 64
) -> tuple[np.ndarray, list[dict[str, Any]], np.ndarray, int]:
    """Build compact records for all usable positive-rain cells.

    Returns ``(records, file_entries, z, chunk_count)``. ``positive_qc_count``
    records the task mask before auxiliary-variable checks, while
    ``excluded_missing_input_count`` reveals whether a candidate was removed
    because any model input was missing.
    """

    if scan_chunk_size <= 0:
        raise ValueError("scan_chunk_size must be positive")
    if not paths:
        raise ValueError("at least one NetCDF path is required")
    if len(paths) - 1 > np.iinfo(STAGE1_INDEX_DTYPE["file_id"]).max:
        raise OverflowError("too many files for the compact file_id field")

    all_records: list[np.ndarray] = []
    file_entries: list[dict[str, Any]] = []
    reference_z: np.ndarray | None = None
    cumulative = 0
    chunk_count = 0

    for file_id, path_value in enumerate(paths):
        path = Path(path_value).expanduser().resolve()
        nscan = _source_nscan(path)
        positive_qc_count = 0
        excluded_count = 0
        file_record_parts: list[np.ndarray] = []
        for start in range(0, nscan, scan_chunk_size):
            stop = min(start + scan_chunk_size, nscan)
            sample = read_nc_sample(
                path,
                variables=STAGE1_READ_VARIABLES,
                scan_slice=slice(start, stop),
                dtype=np.float32,
                build_masks=True,
            )
            z = sample.variables["z"].astype(np.float64)
            if reference_z is None:
                reference_z = z.copy()
            elif z.shape != reference_z.shape or not np.allclose(
                z, reference_z, rtol=0.0, atol=1e-6
            ):
                raise ValueError(f"Height coordinate differs from earlier files: {path}")

            positive_qc = sample.masks["pre_positive_qc"]
            positive_qc_count += int(positive_qc.sum())
            selected = positive_qc & sample.masks["dpr_reflectivity_valid"]
            for name in STAGE1_INPUT_VARIABLES:
                selected &= np.isfinite(sample.variables[name])
            excluded_count += int(positive_qc.sum() - selected.sum())
            records = _index_records(file_id, start, selected)
            if records.size:
                file_record_parts.append(records)
            chunk_count += 1

        file_records = (
            np.concatenate(file_record_parts)
            if file_record_parts
            else np.empty(0, dtype=STAGE1_INDEX_DTYPE)
        )
        all_records.append(file_records)
        start_offset = cumulative
        cumulative += int(file_records.size)
        file_entries.append(
            {
                "file_id": file_id,
                "sample_id": path.stem,
                "file_name": path.name,
                "file_path": str(path),
                "nscan": nscan,
                "positive_qc_count": positive_qc_count,
                "sample_count": int(file_records.size),
                "excluded_missing_input_count": excluded_count,
                "index_start": start_offset,
                "index_stop": cumulative,
            }
        )
        print(
            f"OK [{file_id + 1}/{len(paths)}] {path.name} "
            f"positive_qc={positive_qc_count:,} selected={file_records.size:,}",
            flush=True,
        )

    assert reference_z is not None
    records = (
        np.concatenate(all_records)
        if all_records
        else np.empty(0, dtype=STAGE1_INDEX_DTYPE)
    )
    return records, file_entries, reference_z, chunk_count


def atomic_save_npy(path: Path, records: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as handle:
        np.save(handle, records, allow_pickle=False)
    partial.replace(path)


def atomic_save_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
    partial.replace(path)


@dataclass
class _CachedFile:
    variables: dict[str, np.ndarray]


class Stage1IntensityDataset(TorchDataset):
    """Random-access PyTorch Dataset for conditional positive-rain intensity.

    The index is ordered by file. For efficient training, a later sampler should
    shuffle file blocks and then shuffle samples within each block; completely
    global random order would repeatedly evict the small per-worker file cache.
    """

    def __init__(
        self,
        index_metadata: str | Path,
        normalization_stats: str | Path,
        *,
        cache_size: int = 1,
        include_height: bool = True,
        verify_hashes: bool = True,
    ) -> None:
        if torch is None:
            raise ModuleNotFoundError(
                "PyTorch is required for Stage1IntensityDataset; install torch "
                "in the project virtual environment"
            )
        if cache_size <= 0:
            raise ValueError("cache_size must be positive")
        self.metadata_path = Path(index_metadata).expanduser().resolve()
        self.normalization_path = Path(normalization_stats).expanduser().resolve()
        self.cache_size = cache_size
        self.include_height = include_height
        self._cache: OrderedDict[int, _CachedFile] = OrderedDict()

        self.index_metadata = self._load_json(self.metadata_path)
        self.normalization = self._load_json(self.normalization_path)
        if self.index_metadata.get("format_version") != STAGE1_INDEX_FORMAT_VERSION:
            raise ValueError("unsupported stage-one index format version")
        if self.index_metadata.get("selection_mask") != "pre_positive_qc":
            raise ValueError("stage-one index must use the pre_positive_qc mask")
        if tuple(self.index_metadata.get("input_variables", ())) != STAGE1_INPUT_VARIABLES:
            raise ValueError("stage-one index input_variables differ from Dataset inputs")
        if self.index_metadata.get("label_variable") != "pre_dpr":
            raise ValueError("stage-one index must use pre_dpr as its label")
        if self.index_metadata.get("label_transform") != "log1p":
            raise ValueError("stage-one index must use the log1p label transform")
        if self.normalization.get("scope") != "training_split_only":
            raise ValueError("normalization statistics are not marked training_split_only")
        if self.normalization.get("selection_mask") != "pre_positive_qc":
            raise ValueError("stage-one Dataset requires pre_positive_qc statistics")
        index_manifest_hash = self.index_metadata.get("split_manifest_sha256")
        stats_manifest_hash = self.normalization.get("split_manifest_sha256")
        if (
            index_manifest_hash is not None
            and stats_manifest_hash is not None
            and index_manifest_hash != stats_manifest_hash
        ):
            raise ValueError("index and normalization use different split manifests")

        self.index_path = self.metadata_path.parent / self.index_metadata["index_file"]
        if not self.index_path.is_file():
            raise FileNotFoundError(f"Stage-one NumPy index not found: {self.index_path}")
        if verify_hashes and sha256_file(self.index_path) != self.index_metadata["index_sha256"]:
            raise ValueError(f"Index SHA-256 mismatch: {self.index_path}")
        self.records = np.load(self.index_path, mmap_mode="r", allow_pickle=False)
        if self.records.dtype != STAGE1_INDEX_DTYPE:
            raise ValueError(
                f"Index dtype {self.records.dtype} != expected {STAGE1_INDEX_DTYPE}"
            )
        if self.records.ndim != 1:
            raise ValueError("stage-one index must be one-dimensional")
        if len(self.records) != int(self.index_metadata["sample_count"]):
            raise ValueError("index length differs from metadata sample_count")

        self.files = list(self.index_metadata["files"])
        if [entry["file_id"] for entry in self.files] != list(range(len(self.files))):
            raise ValueError("file metadata must use contiguous file_id values")
        self.z = np.asarray(self.index_metadata["heights_km"], dtype=np.float32)
        if self.z.ndim != 1 or self.z.size < 2 or not np.all(np.diff(self.z) > 0):
            raise ValueError("heights_km must be a strictly increasing vector")
        self.standardizers: dict[str, PerLevelStandardizer] = {}
        for name in STAGE1_INPUT_VARIABLES:
            if name not in self.normalization["variables"]:
                raise KeyError(f"normalization statistics are missing {name!r}")
            statistics = self.normalization["variables"][name]
            if not np.allclose(
                np.asarray(statistics["heights_km"], dtype=float),
                self.z,
                rtol=0.0,
                atol=1e-6,
            ):
                raise ValueError(f"normalization height coordinate differs for {name}")
            self.standardizers[name] = PerLevelStandardizer.from_dict(statistics)
        self.feature_names = list(STAGE1_INPUT_VARIABLES)
        if include_height:
            self.feature_names.append("height_scaled")

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        if not path.is_file():
            raise FileNotFoundError(f"JSON metadata not found: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def __len__(self) -> int:
        return len(self.records)

    def clear_cache(self) -> None:
        self._cache.clear()

    @property
    def cached_file_ids(self) -> tuple[int, ...]:
        return tuple(self._cache)

    def file_index_range(self, file_id: int) -> range:
        entry = self.files[file_id]
        return range(int(entry["index_start"]), int(entry["index_stop"]))

    def _load_file(self, file_id: int) -> _CachedFile:
        cached = self._cache.pop(file_id, None)
        if cached is not None:
            self._cache[file_id] = cached
            return cached
        entry = self.files[file_id]
        sample: NCSample = read_nc_sample(
            entry["file_path"],
            variables=("z", *STAGE1_INPUT_VARIABLES, "pre_dpr"),
            dtype=np.float32,
            build_masks=False,
        )
        if not np.allclose(sample.variables["z"], self.z, rtol=0.0, atol=1e-6):
            raise ValueError(f"Height coordinate changed for {entry['file_path']}")
        cached = _CachedFile(variables=sample.variables)
        self._cache[file_id] = cached
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return cached

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        record = self.records[index]
        file_id = int(record["file_id"])
        scan = int(record["scan"])
        ray = int(record["ray"])
        level = int(record["level"])
        cached = self._load_file(file_id)

        feature_values: list[float] = []
        for name in STAGE1_INPUT_VARIABLES:
            raw_value = float(cached.variables[name][scan, ray, level])
            standardizer = self.standardizers[name]
            mean = float(standardizer.mean[level])
            std = float(standardizer.std[level])
            if not np.isfinite(raw_value) or not np.isfinite(mean) or std <= standardizer.epsilon:
                raise ValueError(
                    f"Indexed sample has unusable {name} at "
                    f"file={file_id}, scan={scan}, ray={ray}, level={level}"
                )
            feature_values.append((raw_value - mean) / std)
        if self.include_height:
            height_scaled = 2.0 * (float(self.z[level]) - float(self.z[0])) / (
                float(self.z[-1]) - float(self.z[0])
            ) - 1.0
            feature_values.append(height_scaled)

        rain_rate = float(cached.variables["pre_dpr"][scan, ray, level])
        if not np.isfinite(rain_rate) or rain_rate <= 0:
            raise ValueError("stage-one index points to a non-positive rain label")
        return {
            "features": torch.tensor(feature_values, dtype=torch.float32),
            "target": torch.tensor(np.log1p(rain_rate), dtype=torch.float32),
            "rain_rate": torch.tensor(rain_rate, dtype=torch.float32),
            "sample_index": torch.tensor(index, dtype=torch.int64),
            "file_id": torch.tensor(file_id, dtype=torch.int64),
            "scan": torch.tensor(scan, dtype=torch.int64),
            "ray": torch.tensor(ray, dtype=torch.int64),
            "level": torch.tensor(level, dtype=torch.int64),
        }

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        # Spawned workers reopen the memory map instead of serializing its contents.
        state["records"] = None
        # Do not copy large decoded NetCDF arrays from the parent into workers.
        state["_cache"] = OrderedDict()
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.records = np.load(self.index_path, mmap_mode="r", allow_pickle=False)
