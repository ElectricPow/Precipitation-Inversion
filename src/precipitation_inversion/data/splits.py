"""Leakage-safe grouped splits for the GR--DPR dataset.

Neighbouring satellite swaths from the same date may describe the same weather
system.  Therefore records are assigned by a group key (``date`` by default),
never as independent files or voxels.  The balanced strategy searches many
deterministic random group partitions and retains the one whose file count,
coverage, rain-tail, and precipitation-type shares best match requested ratios.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


DEFAULT_SPLIT_RATIOS: dict[str, float] = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}

# File count is synthesized during grouping; the remaining names come directly
# from dataset_manifest.csv.  Rare heavy-rain and convective statistics receive
# larger weights so that they are not accidentally concentrated in one split.
DEFAULT_BALANCE_WEIGHTS: dict[str, float] = {
    "file_count": 1.0,
    "total_voxel_count": 1.0,
    "gr_sparse_valid_count": 1.0,
    "pre_positive_count": 2.0,
    "pre_gt_10_count": 2.0,
    "pre_gt_20_count": 3.0,
    "pre_gt_50_count": 4.0,
    "stratiform_profile_count": 1.0,
    "convective_profile_count": 2.0,
}


@dataclass(frozen=True)
class SplitResult:
    """One complete group-level split assignment."""

    split_names: tuple[str, ...]
    ratios: tuple[float, ...]
    group_field: str
    group_assignments: dict[str, str]
    record_assignments: dict[str, str]
    score: float
    strategy: str
    seed: int | None
    trials: int


def normalized_ratios(ratios: Mapping[str, float]) -> tuple[tuple[str, ...], np.ndarray]:
    """Validate and normalize positive split ratios while preserving order."""

    if len(ratios) < 2:
        raise ValueError("at least two splits are required")
    names = tuple(ratios)
    if len(set(names)) != len(names) or any(not name for name in names):
        raise ValueError("split names must be unique and non-empty")
    values = np.asarray([ratios[name] for name in names], dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("all split ratios must be finite and positive")
    return names, values / values.sum()


def allocate_group_counts(group_count: int, ratios: Sequence[float]) -> np.ndarray:
    """Allocate an integer number of groups using the largest-remainder method.

    Every positive-ratio split receives at least one group when enough groups are
    available.  The returned counts always sum exactly to ``group_count``.
    """

    values = np.asarray(ratios, dtype=np.float64)
    if group_count <= 0:
        raise ValueError("group_count must be positive")
    if values.ndim != 1 or values.size < 2:
        raise ValueError("ratios must be a one-dimensional sequence with 2+ values")
    if np.any(values <= 0) or not np.all(np.isfinite(values)):
        raise ValueError("all ratios must be finite and positive")
    if group_count < values.size:
        raise ValueError("there must be at least one group per split")

    values = values / values.sum()
    exact = values * group_count
    counts = np.floor(exact).astype(np.int64)
    remaining = group_count - int(counts.sum())
    # Stable sorting makes ties deterministic and respects split order.
    remainder_order = np.argsort(-(exact - counts), kind="stable")
    counts[remainder_order[:remaining]] += 1

    # A very small ratio can floor to zero. Borrow from the largest allocation.
    for empty_index in np.flatnonzero(counts == 0):
        donors = np.flatnonzero(counts > 1)
        if not donors.size:
            raise ValueError("cannot allocate at least one group to every split")
        donor = donors[np.argmax(counts[donors] - exact[donors])]
        counts[donor] -= 1
        counts[empty_index] += 1
    return counts


def _number(value: Any, field: str) -> float:
    if value is None or value == "":
        return 0.0
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"field {field!r} contains a non-numeric value: {value!r}") from exc
    if not np.isfinite(number) or number < 0:
        raise ValueError(f"field {field!r} must contain finite non-negative values")
    return number


def validate_records(
    records: Sequence[Mapping[str, Any]], *, group_field: str, id_field: str
) -> None:
    if not records:
        raise ValueError("manifest contains no records")
    identifiers: list[str] = []
    for index, record in enumerate(records):
        identifier = str(record.get(id_field, "")).strip()
        group = str(record.get(group_field, "")).strip()
        if not identifier:
            raise ValueError(f"record {index} has an empty {id_field!r}")
        if not group:
            raise ValueError(f"record {identifier!r} has an empty {group_field!r}")
        identifiers.append(identifier)
    duplicate_ids = sorted(
        identifier for identifier in set(identifiers) if identifiers.count(identifier) > 1
    )
    if duplicate_ids:
        raise ValueError(f"duplicate record identifiers: {duplicate_ids[:5]}")


def group_feature_matrix(
    records: Sequence[Mapping[str, Any]],
    *,
    group_field: str,
    balance_weights: Mapping[str, float],
) -> tuple[tuple[str, ...], tuple[str, ...], np.ndarray, np.ndarray]:
    """Aggregate manifest features into one row per leakage group."""

    feature_names = tuple(balance_weights)
    weights = np.asarray([balance_weights[name] for name in feature_names], dtype=float)
    if not feature_names or np.any(weights <= 0) or not np.all(np.isfinite(weights)):
        raise ValueError("balance feature weights must be finite and positive")

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record[group_field]), []).append(record)
    group_names = tuple(sorted(grouped))
    matrix = np.zeros((len(group_names), len(feature_names)), dtype=np.float64)
    for group_index, group_name in enumerate(group_names):
        group_records = grouped[group_name]
        for feature_index, field in enumerate(feature_names):
            if field == "file_count":
                matrix[group_index, feature_index] = len(group_records)
            else:
                matrix[group_index, feature_index] = sum(
                    _number(record.get(field), field) for record in group_records
                )
    return group_names, feature_names, matrix, weights


def assignment_score(
    assignment: np.ndarray,
    feature_matrix: np.ndarray,
    target_ratios: np.ndarray,
    feature_weights: np.ndarray,
) -> float:
    """Measure weighted squared deviation of feature shares from target ratios."""

    split_count = target_ratios.size
    if assignment.shape != (feature_matrix.shape[0],):
        raise ValueError("assignment length must equal the number of groups")
    split_totals = np.zeros((split_count, feature_matrix.shape[1]), dtype=np.float64)
    for split_index in range(split_count):
        split_totals[split_index] = feature_matrix[assignment == split_index].sum(axis=0)
    feature_totals = feature_matrix.sum(axis=0)
    usable = feature_totals > 0
    if not usable.any():
        return 0.0
    shares = split_totals[:, usable] / feature_totals[usable]
    differences = shares - target_ratios[:, np.newaxis]
    weights = feature_weights[usable]
    return float(np.sum(differences**2 * weights[np.newaxis, :]) / weights.sum())


def _assignment_from_order(order: np.ndarray, group_counts: np.ndarray) -> np.ndarray:
    assignment = np.empty(order.size, dtype=np.int64)
    start = 0
    for split_index, count in enumerate(group_counts):
        stop = start + int(count)
        assignment[order[start:stop]] = split_index
        start = stop
    return assignment


def _result_from_assignment(
    records: Sequence[Mapping[str, Any]],
    *,
    assignment: np.ndarray,
    group_names: Sequence[str],
    split_names: Sequence[str],
    ratios: np.ndarray,
    group_field: str,
    id_field: str,
    score: float,
    strategy: str,
    seed: int | None,
    trials: int,
) -> SplitResult:
    group_assignments = {
        group_name: split_names[int(assignment[index])]
        for index, group_name in enumerate(group_names)
    }
    record_assignments = {
        str(record[id_field]): group_assignments[str(record[group_field])]
        for record in records
    }
    return SplitResult(
        split_names=tuple(split_names),
        ratios=tuple(float(value) for value in ratios),
        group_field=group_field,
        group_assignments=group_assignments,
        record_assignments=record_assignments,
        score=score,
        strategy=strategy,
        seed=seed,
        trials=trials,
    )


def balanced_group_split(
    records: Sequence[Mapping[str, Any]],
    *,
    ratios: Mapping[str, float] = DEFAULT_SPLIT_RATIOS,
    group_field: str = "date",
    id_field: str = "sample_id",
    balance_weights: Mapping[str, float] = DEFAULT_BALANCE_WEIGHTS,
    seed: int = 20260824,
    trials: int = 10_000,
) -> SplitResult:
    """Choose the best of many deterministic random group partitions.

    Group counts are fixed by the requested ratios.  Random search is appropriate
    here because the real dataset has only 108 date groups; it is transparent,
    reproducible, and avoids pretending that an exact stratification exists for
    several competing long-tail statistics.
    """

    if trials <= 0:
        raise ValueError("trials must be positive")
    validate_records(records, group_field=group_field, id_field=id_field)
    split_names, target_ratios = normalized_ratios(ratios)
    group_names, _, matrix, weights = group_feature_matrix(
        records, group_field=group_field, balance_weights=balance_weights
    )
    group_counts = allocate_group_counts(len(group_names), target_ratios)
    rng = np.random.default_rng(seed)
    best_assignment: np.ndarray | None = None
    best_score = np.inf
    for _ in range(trials):
        order = rng.permutation(len(group_names))
        assignment = _assignment_from_order(order, group_counts)
        score = assignment_score(assignment, matrix, target_ratios, weights)
        if score < best_score:
            best_score = score
            best_assignment = assignment.copy()
    assert best_assignment is not None
    return _result_from_assignment(
        records,
        assignment=best_assignment,
        group_names=group_names,
        split_names=split_names,
        ratios=target_ratios,
        group_field=group_field,
        id_field=id_field,
        score=best_score,
        strategy="balanced",
        seed=seed,
        trials=trials,
    )


def chronological_group_split(
    records: Sequence[Mapping[str, Any]],
    *,
    ratios: Mapping[str, float] = DEFAULT_SPLIT_RATIOS,
    group_field: str = "date",
    id_field: str = "sample_id",
    balance_weights: Mapping[str, float] = DEFAULT_BALANCE_WEIGHTS,
) -> SplitResult:
    """Assign sorted groups consecutively, useful as a temporal-shift benchmark."""

    validate_records(records, group_field=group_field, id_field=id_field)
    split_names, target_ratios = normalized_ratios(ratios)
    group_names, _, matrix, weights = group_feature_matrix(
        records, group_field=group_field, balance_weights=balance_weights
    )
    group_counts = allocate_group_counts(len(group_names), target_ratios)
    assignment = _assignment_from_order(np.arange(len(group_names)), group_counts)
    score = assignment_score(assignment, matrix, target_ratios, weights)
    return _result_from_assignment(
        records,
        assignment=assignment,
        group_names=group_names,
        split_names=split_names,
        ratios=target_ratios,
        group_field=group_field,
        id_field=id_field,
        score=score,
        strategy="chronological",
        seed=None,
        trials=1,
    )


def assert_valid_split(
    records: Sequence[Mapping[str, Any]],
    result: SplitResult,
    *,
    id_field: str = "sample_id",
) -> None:
    """Raise when records are missing, duplicated, or groups leak across splits."""

    expected_ids = {str(record[id_field]) for record in records}
    if set(result.record_assignments) != expected_ids:
        missing = expected_ids.difference(result.record_assignments)
        extra = set(result.record_assignments).difference(expected_ids)
        raise AssertionError(f"record assignment mismatch; missing={missing}, extra={extra}")
    if set(result.record_assignments.values()) != set(result.split_names):
        raise AssertionError("every requested split must receive at least one record")
    for record in records:
        record_split = result.record_assignments[str(record[id_field])]
        group_split = result.group_assignments[str(record[result.group_field])]
        if record_split != group_split:
            raise AssertionError(
                f"group leakage for {record[result.group_field]!r}: "
                f"{record_split!r} != {group_split!r}"
            )

