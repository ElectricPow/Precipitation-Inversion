"""Cache-aware batch samplers for file-backed precipitation datasets.

The stage-one sample index is contiguous by source NetCDF file. Globally
shuffling millions of individual indices would make each DataLoader worker
continually evict and reload files. :class:`FileBlockBatchSampler` instead
shuffles files, fixed-size blocks inside each file, and indices inside each
block. This retains stochastic training while keeping every batch within one
source file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence

import numpy as np


try:
    import torch.distributed as distributed
    from torch.utils.data import Sampler
except (ImportError, OSError):  # pragma: no cover - NumPy-only environments
    distributed = None

    class Sampler:  # type: ignore[no-redef]
        """Minimal fallback so sampler ordering can be inspected without torch."""

        def __class_getitem__(cls, item: Any) -> type["Sampler"]:
            return cls


@dataclass(frozen=True)
class FileIndexRange:
    """Half-open global sample-index interval belonging to one source file."""

    file_id: int
    start: int
    stop: int

    @property
    def sample_count(self) -> int:
        return self.stop - self.start


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_integer(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def file_ranges_from_dataset(dataset: Any) -> tuple[FileIndexRange, ...]:
    """Validate and extract contiguous file ranges from a stage-one Dataset.

    The strict coverage check prevents silently omitting or duplicating samples
    when index metadata has gaps, overlaps, or inconsistent ``sample_count``.
    Empty source files are valid and remain represented by an empty interval.
    """

    if not hasattr(dataset, "files"):
        raise TypeError("dataset must expose file metadata through a 'files' attribute")
    try:
        dataset_size = len(dataset)
    except TypeError as error:
        raise TypeError("dataset must implement __len__") from error
    if dataset_size < 0:
        raise ValueError("dataset length cannot be negative")

    files: Sequence[Mapping[str, Any]] = dataset.files
    ranges: list[FileIndexRange] = []
    expected_start = 0
    for position, entry in enumerate(files):
        if not isinstance(entry, Mapping):
            raise TypeError(f"files[{position}] must be a mapping")
        try:
            file_id = _nonnegative_integer(entry["file_id"], name="file_id")
            start = _nonnegative_integer(entry["index_start"], name="index_start")
            stop = _nonnegative_integer(entry["index_stop"], name="index_stop")
        except KeyError as error:
            raise KeyError(
                f"files[{position}] is missing required field {error.args[0]!r}"
            ) from error
        if file_id != position:
            raise ValueError("file_id values must be contiguous and match file order")
        if start != expected_start:
            raise ValueError(
                f"file ranges must be contiguous: expected start {expected_start}, "
                f"got {start} for file_id={file_id}"
            )
        if stop < start:
            raise ValueError(f"index_stop precedes index_start for file_id={file_id}")
        if "sample_count" in entry and int(entry["sample_count"]) != stop - start:
            raise ValueError(f"sample_count differs from index range for file_id={file_id}")
        ranges.append(FileIndexRange(file_id=file_id, start=start, stop=stop))
        expected_start = stop
    if expected_start != dataset_size:
        raise ValueError(
            f"file ranges cover {expected_start} samples but dataset has {dataset_size}"
        )
    return tuple(ranges)


def _distributed_defaults(
    num_replicas: int | None, rank: int | None
) -> tuple[int, int]:
    """Resolve explicit or initialized torch.distributed world information."""

    active = (
        distributed is not None
        and distributed.is_available()
        and distributed.is_initialized()
    )
    if num_replicas is None:
        num_replicas = distributed.get_world_size() if active else 1
    if rank is None:
        rank = distributed.get_rank() if active else 0
    replicas = _positive_integer(num_replicas, name="num_replicas")
    resolved_rank = _nonnegative_integer(rank, name="rank")
    if resolved_rank >= replicas:
        raise ValueError(f"rank={resolved_rank} must be smaller than num_replicas={replicas}")
    return replicas, resolved_rank


class FileBlockBatchSampler(Sampler[list[int]]):
    """Yield cache-local batches with deterministic epoch-dependent shuffling.

    Parameters
    ----------
    dataset:
        Dataset exposing ``files`` entries with ``file_id``, ``index_start`` and
        ``index_stop``. :class:`Stage1IntensityDataset` satisfies this contract.
    batch_size:
        Number of voxel samples in a full batch.
    block_size:
        Number of contiguous indices shuffled together. It must be a multiple
        of ``batch_size``. The default is 16 batches, balancing local mixing and
        temporary-memory use.
    shuffle:
        Shuffle file order, block order, and indices within every block.
    drop_last:
        Drop each file's final incomplete batch. Batches never cross files.
    seed:
        Base random seed. Call :meth:`set_epoch` before every training epoch.
    num_replicas, rank:
        Batch-level DDP sharding. When omitted, an initialized
        ``torch.distributed`` process group is detected automatically.
    even_batches:
        In distributed mode, discard at most ``num_replicas - 1`` global
        batches so all ranks perform the same number of optimizer steps. The
        omitted batches change with the shuffled order in later epochs.
    """

    def __init__(
        self,
        dataset: Any,
        *,
        batch_size: int,
        block_size: int | None = None,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0,
        num_replicas: int | None = None,
        rank: int | None = None,
        even_batches: bool = True,
    ) -> None:
        self.file_ranges = file_ranges_from_dataset(dataset)
        self.batch_size = _positive_integer(batch_size, name="batch_size")
        self.block_size = _positive_integer(
            self.batch_size * 16 if block_size is None else block_size,
            name="block_size",
        )
        if self.block_size % self.batch_size != 0:
            raise ValueError("block_size must be a multiple of batch_size")
        if not isinstance(shuffle, bool):
            raise TypeError("shuffle must be bool")
        if not isinstance(drop_last, bool):
            raise TypeError("drop_last must be bool")
        if not isinstance(even_batches, bool):
            raise TypeError("even_batches must be bool")
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.even_batches = even_batches
        self.seed = _nonnegative_integer(seed, name="seed")
        self.num_replicas, self.rank = _distributed_defaults(num_replicas, rank)
        self.epoch = 0
        self._global_batch_count = sum(
            file_range.sample_count // self.batch_size
            if self.drop_last
            else (file_range.sample_count + self.batch_size - 1) // self.batch_size
            for file_range in self.file_ranges
        )

    def set_epoch(self, epoch: int) -> None:
        """Select a reproducible, different ordering for a training epoch."""

        self.epoch = _nonnegative_integer(epoch, name="epoch")

    def _rng(self) -> np.random.Generator:
        # SeedSequence avoids simple seed+epoch overflow and keeps both inputs explicit.
        return np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch]))

    def _iter_global_batches(self) -> Iterator[list[int]]:
        rng = self._rng()
        file_order = np.arange(len(self.file_ranges), dtype=np.int64)
        if self.shuffle:
            rng.shuffle(file_order)

        for range_position in file_order:
            file_range = self.file_ranges[int(range_position)]
            block_starts = list(
                range(file_range.start, file_range.stop, self.block_size)
            )
            if self.shuffle:
                rng.shuffle(block_starts)
            for block_start in block_starts:
                block_stop = min(block_start + self.block_size, file_range.stop)
                indices = np.arange(block_start, block_stop, dtype=np.int64)
                if self.shuffle:
                    rng.shuffle(indices)
                for batch_start in range(0, indices.size, self.batch_size):
                    batch = indices[batch_start : batch_start + self.batch_size]
                    if batch.size < self.batch_size and self.drop_last:
                        continue
                    yield batch.tolist()

    def __iter__(self) -> Iterator[list[int]]:
        usable_count = self._global_batch_count
        if self.even_batches:
            usable_count -= usable_count % self.num_replicas
        for global_index, batch in enumerate(self._iter_global_batches()):
            if global_index >= usable_count:
                break
            if global_index % self.num_replicas == self.rank:
                yield batch

    def __len__(self) -> int:
        if self.even_batches:
            return self._global_batch_count // self.num_replicas
        remaining = self._global_batch_count - self.rank
        return 0 if remaining <= 0 else (remaining + self.num_replicas - 1) // self.num_replicas

