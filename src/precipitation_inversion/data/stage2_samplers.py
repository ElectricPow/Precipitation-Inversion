"""Stratified, cache-aware batch sampling for Stage-2 patch training.

Stage-two patch records are highly imbalanced: many patches contain only
background, while strong DPR targets and GR-missing fill targets are rarer but
scientifically important.  This sampler assigns each patch to one mutually
exclusive stratum, draws deterministic epoch quotas, then optionally groups the
drawn indices by source file before batching to reduce NetCDF cache churn.

The global batch sequence is constructed identically on every DDP rank and
sharded by batch position.  Validation/test must not use this sampler; they
should iterate every index once in deterministic order.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np

from .samplers import _distributed_defaults, _nonnegative_integer, _positive_integer


try:
    from torch.utils.data import Sampler
except (ImportError, OSError):  # pragma: no cover - NumPy-only inspection

    class Sampler:  # type: ignore[no-redef]
        def __class_getitem__(cls, item: Any) -> type["Sampler"]:
            return cls


STAGE2_STRATA = (
    "background",
    "ordinary_target",
    "fill_dominant_target",
    "strong_target",
)
DEFAULT_STAGE2_STRATUM_WEIGHTS: dict[str, float] = {
    "background": 0.20,
    "ordinary_target": 0.10,
    "fill_dominant_target": 0.30,
    "strong_target": 0.40,
}
DEFAULT_FILL_FRACTION_THRESHOLD = 0.75
REQUIRED_RECORD_FIELDS = (
    "file_id",
    "dpr_count",
    "q11_count",
    "q01_count",
    "strong_dpr_count",
)


def classify_stage2_patch_strata(
    records: np.ndarray,
    *,
    fill_fraction_threshold: float = DEFAULT_FILL_FRACTION_THRESHOLD,
) -> np.ndarray:
    """Return one integer stratum id for every structured patch record.

    Priority is strong target, then non-strong fill-dominant target, then
    remaining non-strong DPR target, then background. ``fill_dominant_target``
    means ``q01_count/dpr_count >= fill_fraction_threshold``; it does not claim
    that every target in the patch is missing from direct GR.
    """

    values = np.asarray(records)
    if (
        not np.isfinite(fill_fraction_threshold)
        or not 0.0 < fill_fraction_threshold <= 1.0
    ):
        raise ValueError("fill_fraction_threshold must lie in (0,1]")
    if values.ndim != 1 or values.dtype.names is None:
        raise TypeError("records must be a one-dimensional structured array")
    missing = [name for name in REQUIRED_RECORD_FIELDS if name not in values.dtype.names]
    if missing:
        raise ValueError(f"records missing fields: {', '.join(missing)}")
    for name in REQUIRED_RECORD_FIELDS:
        if values.dtype[name].kind not in "ui":
            raise TypeError(f"record field {name!r} must be integer")

    dpr = values["dpr_count"].astype(np.uint64, copy=False)
    q11 = values["q11_count"].astype(np.uint64, copy=False)
    q01 = values["q01_count"].astype(np.uint64, copy=False)
    strong = values["strong_dpr_count"].astype(np.uint64, copy=False)
    if np.any(q11 + q01 != dpr):
        raise ValueError("each record must satisfy q11_count + q01_count = dpr_count")
    if np.any(strong > dpr):
        raise ValueError("strong_dpr_count cannot exceed dpr_count")

    # 0=background, 1=ordinary target, 2=fill-dominant target, 3=strong.
    strata = np.zeros(values.size, dtype=np.int8)
    nonstrong_target = (dpr > 0) & (strong == 0)
    strata[nonstrong_target] = 1
    fill_dominant = np.zeros(values.size, dtype=bool)
    fill_dominant[nonstrong_target] = (
        q01[nonstrong_target] / dpr[nonstrong_target]
        >= fill_fraction_threshold
    )
    strata[fill_dominant] = 2
    strata[strong > 0] = 3
    return strata


def _normalize_weights(weights: Mapping[str, float] | None) -> np.ndarray:
    source = DEFAULT_STAGE2_STRATUM_WEIGHTS if weights is None else weights
    if set(source) != set(STAGE2_STRATA):
        raise ValueError(f"stratum_weights must contain exactly {STAGE2_STRATA}")
    result = np.asarray([source[name] for name in STAGE2_STRATA], dtype=np.float64)
    if np.any(~np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError("stratum weights must be finite and non-negative")
    if result.sum() <= 0.0:
        raise ValueError("at least one stratum weight must be positive")
    return result / result.sum()


def allocate_stage2_epoch_quotas(
    epoch_size: int,
    normalized_weights: np.ndarray,
) -> np.ndarray:
    """Allocate exact integer quotas by the deterministic largest remainder rule."""

    size = _positive_integer(epoch_size, name="epoch_size")
    weights = np.asarray(normalized_weights, dtype=np.float64)
    if weights.shape != (len(STAGE2_STRATA),):
        raise ValueError(f"normalized_weights must have shape {(len(STAGE2_STRATA),)}")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("normalized_weights must be finite and non-negative")
    if not np.isclose(weights.sum(), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("normalized_weights must sum to one")
    exact = weights * size
    quotas = np.floor(exact).astype(np.int64)
    remaining = size - int(quotas.sum())
    if remaining:
        order = np.argsort(-(exact - quotas), kind="stable")
        quotas[order[:remaining]] += 1
    return quotas


def _draw_cyclically(
    candidates: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw shuffled full cycles so all candidates appear before repeats."""

    if count == 0:
        return np.empty(0, dtype=np.int64)
    if candidates.size == 0:
        raise ValueError("a positive quota was assigned to an empty stratum")
    pieces: list[np.ndarray] = []
    full_cycles, remainder = divmod(count, candidates.size)
    for _ in range(full_cycles):
        pieces.append(rng.permutation(candidates))
    if remainder:
        pieces.append(rng.permutation(candidates)[:remainder])
    return np.concatenate(pieces).astype(np.int64, copy=False)


class Stage2StratifiedBatchSampler(Sampler[list[int]]):
    """Draw balanced Stage-2 patch batches with deterministic DDP sharding.

    Oversampling is explicit: if an epoch quota exceeds a stratum's unique
    count, shuffled full cycles repeat its indices.  ``epoch_size`` defaults to
    the original Dataset size, so training-step count remains stable across
    ablations even when stratum weights change.
    """

    def __init__(
        self,
        dataset: Any,
        *,
        batch_size: int,
        stratum_weights: Mapping[str, float] | None = None,
        fill_fraction_threshold: float = DEFAULT_FILL_FRACTION_THRESHOLD,
        epoch_size: int | None = None,
        seed: int = 0,
        shuffle: bool = True,
        group_by_file: bool = True,
        drop_last: bool = False,
        num_replicas: int | None = None,
        rank: int | None = None,
        even_batches: bool = True,
    ) -> None:
        if not hasattr(dataset, "records"):
            raise TypeError("dataset must expose structured patch records")
        self.records = np.asarray(dataset.records)
        if len(dataset) != self.records.size:
            raise ValueError("dataset length differs from its patch records")
        if self.records.size == 0:
            raise ValueError("stage-two sampler requires at least one patch")
        self.fill_fraction_threshold = float(fill_fraction_threshold)
        self.strata = classify_stage2_patch_strata(
            self.records,
            fill_fraction_threshold=self.fill_fraction_threshold,
        )
        self.batch_size = _positive_integer(batch_size, name="batch_size")
        self.epoch_size = _positive_integer(
            self.records.size if epoch_size is None else epoch_size,
            name="epoch_size",
        )
        self.weights = _normalize_weights(stratum_weights)
        self.quotas = allocate_stage2_epoch_quotas(self.epoch_size, self.weights)
        counts = np.bincount(self.strata, minlength=len(STAGE2_STRATA))
        for index, (quota, count) in enumerate(zip(self.quotas, counts)):
            if quota > 0 and count == 0:
                raise ValueError(
                    f"stratum {STAGE2_STRATA[index]!r} has positive quota but no patches"
                )
        for name, value in (
            ("shuffle", shuffle),
            ("group_by_file", group_by_file),
            ("drop_last", drop_last),
            ("even_batches", even_batches),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be bool")
        self.shuffle = shuffle
        self.group_by_file = group_by_file
        self.drop_last = drop_last
        self.even_batches = even_batches
        self.seed = _nonnegative_integer(seed, name="seed")
        self.num_replicas, self.rank = _distributed_defaults(num_replicas, rank)
        self.epoch = 0
        self._global_batch_count = (
            self.epoch_size // self.batch_size
            if self.drop_last
            else (self.epoch_size + self.batch_size - 1) // self.batch_size
        )

    @property
    def stratum_counts(self) -> dict[str, int]:
        counts = np.bincount(self.strata, minlength=len(STAGE2_STRATA))
        return {name: int(counts[index]) for index, name in enumerate(STAGE2_STRATA)}

    @property
    def epoch_quotas(self) -> dict[str, int]:
        return {name: int(self.quotas[index]) for index, name in enumerate(STAGE2_STRATA)}

    def set_epoch(self, epoch: int) -> None:
        self.epoch = _nonnegative_integer(epoch, name="epoch")

    def _rng(self) -> np.random.Generator:
        return np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch]))

    def _draw_epoch_indices(self) -> np.ndarray:
        rng = self._rng()
        pieces = [
            _draw_cyclically(
                np.flatnonzero(self.strata == stratum_id).astype(np.int64),
                int(self.quotas[stratum_id]),
                rng,
            )
            for stratum_id in range(len(STAGE2_STRATA))
        ]
        selected = np.concatenate(pieces)
        if self.group_by_file:
            file_ids = self.records["file_id"][selected]
            file_order = np.unique(file_ids)
            if self.shuffle:
                rng.shuffle(file_order)
            grouped: list[np.ndarray] = []
            for file_id in file_order:
                current = selected[file_ids == file_id].copy()
                if self.shuffle:
                    rng.shuffle(current)
                grouped.append(current)
            selected = np.concatenate(grouped)
        elif self.shuffle:
            rng.shuffle(selected)
        return selected

    def _iter_global_batches(self) -> Iterator[list[int]]:
        indices = self._draw_epoch_indices()
        for start in range(0, indices.size, self.batch_size):
            batch = indices[start : start + self.batch_size]
            if batch.size < self.batch_size and self.drop_last:
                continue
            yield batch.tolist()

    def __iter__(self) -> Iterator[list[int]]:
        usable = self._global_batch_count
        if self.even_batches:
            usable -= usable % self.num_replicas
        for global_index, batch in enumerate(self._iter_global_batches()):
            if global_index >= usable:
                break
            if global_index % self.num_replicas == self.rank:
                yield batch

    def __len__(self) -> int:
        if self.even_batches:
            return self._global_batch_count // self.num_replicas
        remaining = self._global_batch_count - self.rank
        return 0 if remaining <= 0 else (remaining + self.num_replicas - 1) // self.num_replicas
