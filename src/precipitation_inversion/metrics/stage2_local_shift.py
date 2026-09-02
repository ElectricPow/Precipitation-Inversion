"""Local finite-shift diagnostics for Stage-2 GR/DPR support alignment.

This module is deliberately diagnostic-only.  It uses the DPR target support to
select a best integer shift *after* observing the answer, so none of its oracle
shifts may be exposed to a trainable model or inference pipeline.

The audit differs from the historical split-wide, per-height shift summary in
three important ways:

* every orbit is processed independently;
* target space is divided into non-overlapping horizontal windows; and
* every candidate shift is scored on the same fixed target-domain cells.

Arrays use the native NetCDF order ``(nscan, nray, z)``.  A positive
``scan_shift`` means that GR at ``(i, j, z)`` is compared with DPR at
``(i + scan_shift, j + ray_shift, z)``.  Shifts never wrap around an edge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


COUNT_FIELDS = (
    "domain_count",
    "gr_count",
    "dpr_count",
    "true_positive",
    "false_positive",
    "false_negative",
    "true_negative",
    "union_count",
)


@dataclass(frozen=True)
class LocalShiftOptions:
    """Configuration for one-orbit local finite-shift auditing."""

    window_scan: int = 64
    window_ray: int = 49
    max_shift: int = 2
    min_dpr_support: int = 1
    min_gr_support: int = 1
    include_partial_windows: bool = True

    def __post_init__(self) -> None:
        if self.window_scan <= 0 or self.window_ray <= 0:
            raise ValueError("window dimensions must be positive")
        if self.max_shift < 0:
            raise ValueError("max_shift must be non-negative")
        if self.min_dpr_support <= 0 or self.min_gr_support <= 0:
            raise ValueError("minimum support counts must be positive")


def finite_shift_offsets(max_shift: int) -> tuple[tuple[int, int], ...]:
    """Return all finite horizontal integer shifts in deterministic order."""

    if max_shift < 0:
        raise ValueError("max_shift must be non-negative")
    return tuple(
        (scan_shift, ray_shift)
        for scan_shift in range(-max_shift, max_shift + 1)
        for ray_shift in range(-max_shift, max_shift + 1)
    )


def non_overlapping_windows(
    nscan: int,
    nray: int,
    *,
    window_scan: int,
    window_ray: int,
    include_partial: bool,
) -> tuple[dict[str, int], ...]:
    """Build deterministic, non-overlapping target-space horizontal windows."""

    if nscan <= 0 or nray <= 0:
        raise ValueError("horizontal dimensions must be positive")
    if window_scan <= 0 or window_ray <= 0:
        raise ValueError("window dimensions must be positive")

    rows: list[dict[str, int]] = []
    window_index = 0
    scan_block = 0
    for scan_start in range(0, nscan, window_scan):
        scan_end = min(scan_start + window_scan, nscan)
        if not include_partial and scan_end - scan_start < window_scan:
            continue
        ray_block = 0
        for ray_start in range(0, nray, window_ray):
            ray_end = min(ray_start + window_ray, nray)
            if not include_partial and ray_end - ray_start < window_ray:
                continue
            rows.append(
                {
                    "window_index": window_index,
                    "scan_block": scan_block,
                    "ray_block": ray_block,
                    "scan_start": scan_start,
                    "scan_end": scan_end,
                    "ray_start": ray_start,
                    "ray_end": ray_end,
                }
            )
            window_index += 1
            ray_block += 1
        scan_block += 1
    if not rows:
        raise ValueError("window configuration produced no windows")
    return tuple(rows)


def _as_boolean_volume(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 3:
        raise ValueError(f"{name} must have shape (nscan,nray,z)")
    if array.dtype != np.bool_:
        raise TypeError(f"{name} must have boolean dtype")
    return array


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else math.nan


def _counts_from_arrays(
    gr_support: np.ndarray,
    dpr_support: np.ndarray,
    domain: np.ndarray,
) -> dict[str, int | float]:
    """Return binary support counts on one fixed target-domain array."""

    predicted = np.asarray(gr_support, dtype=bool) & domain
    target = np.asarray(dpr_support, dtype=bool) & domain
    true_positive = int(np.count_nonzero(predicted & target))
    false_positive = int(np.count_nonzero(predicted & ~target & domain))
    false_negative = int(np.count_nonzero(~predicted & target & domain))
    true_negative = int(np.count_nonzero(~predicted & ~target & domain))
    union = true_positive + false_positive + false_negative
    return {
        "domain_count": int(np.count_nonzero(domain)),
        "gr_count": true_positive + false_positive,
        "dpr_count": true_positive + false_negative,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "union_count": union,
        "support_csi": _safe_ratio(true_positive, union),
    }


def _zero_counts() -> dict[str, int]:
    return {name: 0 for name in COUNT_FIELDS}


def _add_counts(
    destination: dict[str, int], source: Mapping[str, int | float]
) -> None:
    for name in COUNT_FIELDS:
        destination[name] += int(source[name])


def _counts_with_csi(counts: Mapping[str, int]) -> dict[str, int | float]:
    output: dict[str, int | float] = {name: int(counts[name]) for name in COUNT_FIELDS}
    output["support_csi"] = _safe_ratio(
        output["true_positive"], output["union_count"]
    )
    return output


def _candidate_key(candidate: Mapping[str, int | float]) -> tuple[float, ...]:
    """Prefer maximum CSI and then the smallest deterministic displacement."""

    csi = float(candidate["support_csi"])
    if not math.isfinite(csi):
        csi = -1.0
    scan_shift = int(candidate["scan_shift"])
    ray_shift = int(candidate["ray_shift"])
    return (
        csi,
        float(candidate["true_positive"]),
        -float(abs(scan_shift) + abs(ray_shift)),
        -float(abs(scan_shift)),
        -float(abs(ray_shift)),
        -float(scan_shift),
        -float(ray_shift),
    )


def select_best_shift(
    candidates: Sequence[Mapping[str, int | float]],
) -> Mapping[str, int | float]:
    """Select the maximum-CSI shift with deterministic zero-favouring ties."""

    if not candidates:
        raise ValueError("cannot select a shift from an empty candidate list")
    return max(candidates, key=_candidate_key)


def _distribution(values: Sequence[float]) -> dict[str, int | float]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if finite.size == 0:
        return {
            "count": 0,
            "mean": math.nan,
            "median": math.nan,
            "p75": math.nan,
            "p90": math.nan,
            "p95": math.nan,
            "max": math.nan,
            "positive_fraction": math.nan,
        }
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p75": float(np.percentile(finite, 75)),
        "p90": float(np.percentile(finite, 90)),
        "p95": float(np.percentile(finite, 95)),
        "max": float(np.max(finite)),
        "positive_fraction": float(np.mean(finite > 1e-12)),
    }


def _opposition_summary(
    shifts: Sequence[tuple[int, int]],
) -> dict[str, int | bool]:
    nonzero = [(dx, dy) for dx, dy in shifts if dx != 0 or dy != 0]
    unique = set(nonzero)
    positive_scan = sum(dx > 0 for dx, _ in nonzero)
    negative_scan = sum(dx < 0 for dx, _ in nonzero)
    positive_ray = sum(dy > 0 for _, dy in nonzero)
    negative_ray = sum(dy < 0 for _, dy in nonzero)
    opposite_vector = any((-dx, -dy) in unique for dx, dy in unique)
    return {
        "nonzero_best_shift_count": len(nonzero),
        "positive_scan_shift_count": positive_scan,
        "negative_scan_shift_count": negative_scan,
        "positive_ray_shift_count": positive_ray,
        "negative_ray_shift_count": negative_ray,
        "opposite_scan_signs_present": positive_scan > 0 and negative_scan > 0,
        "opposite_ray_signs_present": positive_ray > 0 and negative_ray > 0,
        "opposite_vector_pair_present": opposite_vector,
    }


def _prefixed_counts(
    prefix: str, counts: Mapping[str, int | float]
) -> dict[str, int | float]:
    return {
        f"{prefix}_{name}": value
        for name, value in counts.items()
        if name in COUNT_FIELDS or name == "support_csi"
    }


def _aggregate_candidates(
    candidate_lists: Sequence[Sequence[Mapping[str, int | float]]],
    offsets: Sequence[tuple[int, int]],
) -> list[dict[str, int | float]]:
    totals = [_zero_counts() for _ in offsets]
    for candidates in candidate_lists:
        if len(candidates) != len(offsets):
            raise ValueError("candidate list has an inconsistent shift count")
        for index, candidate in enumerate(candidates):
            if (
                int(candidate["scan_shift"]),
                int(candidate["ray_shift"]),
            ) != offsets[index]:
                raise ValueError("candidate shift ordering is inconsistent")
            _add_counts(totals[index], candidate)
    rows: list[dict[str, int | float]] = []
    for (scan_shift, ray_shift), counts in zip(offsets, totals):
        rows.append(
            {
                "scan_shift": scan_shift,
                "ray_shift": ray_shift,
                **_counts_with_csi(counts),
            }
        )
    return rows


def _fixed_target_candidate_counts(
    gr_support: np.ndarray,
    dpr_support: np.ndarray,
    domain: np.ndarray,
    window: Mapping[str, int],
    offsets: Sequence[tuple[int, int]],
    max_shift: int,
) -> list[list[dict[str, int | float]]]:
    """Return candidate counts for every height in one target-space window.

    The target cells are fixed for every candidate.  Only the source GR slice
    changes with ``(scan_shift, ray_shift)``.  Internal window boundaries do
    not discard cells: a source may be read from a neighbouring window.  Only
    the global orbit border of ``max_shift`` cells is excluded, which prevents
    wraparound and makes every candidate use exactly the same target domain.
    """

    nscan, nray, z_size = domain.shape
    scan_start = max(int(window["scan_start"]), max_shift)
    scan_end = min(int(window["scan_end"]), nscan - max_shift)
    ray_start = max(int(window["ray_start"]), max_shift)
    ray_end = min(int(window["ray_end"]), nray - max_shift)
    if scan_start >= scan_end or ray_start >= ray_end:
        return [[] for _ in range(z_size)]

    target_index = (slice(scan_start, scan_end), slice(ray_start, ray_end), slice(None))
    target = dpr_support[target_index]
    target_domain = domain[target_index]
    by_height: list[list[dict[str, int | float]]] = [
        [] for _ in range(z_size)
    ]
    for scan_shift, ray_shift in offsets:
        source_index = (
            slice(scan_start - scan_shift, scan_end - scan_shift),
            slice(ray_start - ray_shift, ray_end - ray_shift),
            slice(None),
        )
        source = gr_support[source_index]
        for height_index in range(z_size):
            counts = _counts_from_arrays(
                source[..., height_index],
                target[..., height_index],
                target_domain[..., height_index],
            )
            by_height[height_index].append(
                {
                    "scan_shift": scan_shift,
                    "ray_shift": ray_shift,
                    **counts,
                }
            )
    return by_height


def audit_orbit_local_shifts(
    gr_support: Any,
    dpr_support: Any,
    occupancy_domain: Any,
    heights_km: Sequence[float],
    *,
    sample_id: str,
    file_name: str,
    options: LocalShiftOptions = LocalShiftOptions(),
) -> dict[str, Any]:
    """Audit one validation orbit using local, per-height finite-shift oracles."""

    gr = _as_boolean_volume(gr_support, name="gr_support")
    dpr = _as_boolean_volume(dpr_support, name="dpr_support")
    domain = _as_boolean_volume(occupancy_domain, name="occupancy_domain")
    if gr.shape != dpr.shape or gr.shape != domain.shape:
        raise ValueError("GR, DPR, and occupancy-domain shapes must match")
    heights = np.asarray(heights_km, dtype=np.float64)
    if (
        heights.ndim != 1
        or heights.size != gr.shape[-1]
        or not np.all(np.isfinite(heights))
    ):
        raise ValueError("heights_km must be finite and match the z dimension")
    if np.any(gr & ~domain) or np.any(dpr & ~domain):
        raise ValueError("direct GR/DPR support must be a subset of occupancy_domain")
    if gr.shape[0] <= 2 * options.max_shift or gr.shape[1] <= 2 * options.max_shift:
        raise ValueError("orbit is too small for the configured max_shift")

    offsets = finite_shift_offsets(options.max_shift)
    zero_index = offsets.index((0, 0))
    windows = non_overlapping_windows(
        gr.shape[0],
        gr.shape[1],
        window_scan=options.window_scan,
        window_ray=options.window_ray,
        include_partial=options.include_partial_windows,
    )

    window_rows: list[dict[str, Any]] = []
    # Retain candidate lists only as compact counts; full volume tensors are
    # released as soon as this function returns.
    valid_items_by_height: list[list[dict[str, Any]]] = [
        [] for _ in range(gr.shape[-1])
    ]
    planned_count = len(windows) * gr.shape[-1]
    skipped_no_target = 0
    skipped_no_source = 0

    for window in windows:
        by_height = _fixed_target_candidate_counts(
            gr, dpr, domain, window, offsets, options.max_shift
        )
        for height_index, candidates in enumerate(by_height):
            if not candidates:
                skipped_no_target += 1
                continue
            dpr_count = int(candidates[zero_index]["dpr_count"])
            if dpr_count < options.min_dpr_support:
                skipped_no_target += 1
                continue
            max_gr_count = max(int(candidate["gr_count"]) for candidate in candidates)
            if max_gr_count < options.min_gr_support:
                skipped_no_source += 1
                continue

            exact = candidates[zero_index]
            best = select_best_shift(candidates)
            gain = float(best["support_csi"]) - float(exact["support_csi"])
            row = {
                "sample_id": sample_id,
                "file_name": file_name,
                **window,
                "height_index": height_index,
                "height_km": float(heights[height_index]),
                "max_gr_count_over_shifts": max_gr_count,
                **_prefixed_counts("exact", exact),
                "best_scan_shift": int(best["scan_shift"]),
                "best_ray_shift": int(best["ray_shift"]),
                "best_is_zero_shift": (
                    int(best["scan_shift"]) == 0 and int(best["ray_shift"]) == 0
                ),
                **_prefixed_counts("best", best),
                "support_csi_gain": gain,
            }
            window_rows.append(row)
            valid_items_by_height[height_index].append(
                {"candidates": candidates, "exact": exact, "best": best, "row": row}
            )

    per_height_rows: list[dict[str, Any]] = []
    orbit_height_shift_rows: list[dict[str, Any]] = []
    orbit_candidate_lists: list[Sequence[Mapping[str, int | float]]] = []
    orbit_exact = _zero_counts()
    orbit_local = _zero_counts()
    orbit_height_oracle = _zero_counts()
    all_local_shifts: list[tuple[int, int]] = []
    height_with_opposition = 0

    for height_index, items in enumerate(valid_items_by_height):
        if not items:
            per_height_rows.append(
                {
                    "sample_id": sample_id,
                    "file_name": file_name,
                    "height_index": height_index,
                    "height_km": float(heights[height_index]),
                    "valid_window_count": 0,
                    "best_scan_shift": None,
                    "best_ray_shift": None,
                    "exact_support_csi": math.nan,
                    "single_shift_oracle_support_csi": math.nan,
                    "local_oracle_support_csi": math.nan,
                    "single_shift_csi_gain": math.nan,
                    "local_oracle_csi_gain": math.nan,
                    "local_minus_single_shift_csi": math.nan,
                    **_opposition_summary(()),
                }
            )
            continue

        candidate_lists = [item["candidates"] for item in items]
        aggregated_candidates = _aggregate_candidates(candidate_lists, offsets)
        height_best = select_best_shift(aggregated_candidates)
        exact_counts = _zero_counts()
        local_counts = _zero_counts()
        local_shifts: list[tuple[int, int]] = []
        gains: list[float] = []
        for item in items:
            _add_counts(exact_counts, item["exact"])
            _add_counts(local_counts, item["best"])
            local_shifts.append(
                (
                    int(item["best"]["scan_shift"]),
                    int(item["best"]["ray_shift"]),
                )
            )
            gains.append(float(item["row"]["support_csi_gain"]))
            orbit_candidate_lists.append(item["candidates"])
        exact_metrics = _counts_with_csi(exact_counts)
        local_metrics = _counts_with_csi(local_counts)
        single_csi = float(height_best["support_csi"])
        exact_csi = float(exact_metrics["support_csi"])
        local_csi = float(local_metrics["support_csi"])
        opposition = _opposition_summary(local_shifts)
        if (
            opposition["opposite_scan_signs_present"]
            or opposition["opposite_ray_signs_present"]
            or opposition["opposite_vector_pair_present"]
        ):
            height_with_opposition += 1
        distribution = _distribution(gains)
        per_height_rows.append(
            {
                "sample_id": sample_id,
                "file_name": file_name,
                "height_index": height_index,
                "height_km": float(heights[height_index]),
                "valid_window_count": len(items),
                "best_scan_shift": int(height_best["scan_shift"]),
                "best_ray_shift": int(height_best["ray_shift"]),
                **_prefixed_counts("exact", exact_metrics),
                **_prefixed_counts("single_shift_oracle", height_best),
                **_prefixed_counts("local_oracle", local_metrics),
                "single_shift_csi_gain": single_csi - exact_csi,
                "local_oracle_csi_gain": local_csi - exact_csi,
                "local_minus_single_shift_csi": local_csi - single_csi,
                "mean_window_csi_gain": distribution["mean"],
                "median_window_csi_gain": distribution["median"],
                "p90_window_csi_gain": distribution["p90"],
                "positive_gain_window_fraction": distribution["positive_fraction"],
                **opposition,
            }
        )
        for candidate in aggregated_candidates:
            orbit_height_shift_rows.append(
                {
                    "sample_id": sample_id,
                    "file_name": file_name,
                    "height_index": height_index,
                    "height_km": float(heights[height_index]),
                    "valid_window_count": len(items),
                    **candidate,
                    "selected_as_best": (
                        int(candidate["scan_shift"])
                        == int(height_best["scan_shift"])
                        and int(candidate["ray_shift"])
                        == int(height_best["ray_shift"])
                    ),
                }
            )
        _add_counts(orbit_exact, exact_metrics)
        _add_counts(orbit_local, local_metrics)
        _add_counts(orbit_height_oracle, height_best)
        all_local_shifts.extend(local_shifts)

    if orbit_candidate_lists:
        orbit_candidates = _aggregate_candidates(orbit_candidate_lists, offsets)
        orbit_best = select_best_shift(orbit_candidates)
    else:
        orbit_candidates = [
            {"scan_shift": dx, "ray_shift": dy, **_counts_with_csi(_zero_counts())}
            for dx, dy in offsets
        ]
        orbit_best = orbit_candidates[zero_index]

    orbit_exact_metrics = _counts_with_csi(orbit_exact)
    orbit_local_metrics = _counts_with_csi(orbit_local)
    orbit_height_metrics = _counts_with_csi(orbit_height_oracle)
    exact_csi = float(orbit_exact_metrics["support_csi"])
    single_csi = float(orbit_best["support_csi"])
    per_height_csi = float(orbit_height_metrics["support_csi"])
    local_csi = float(orbit_local_metrics["support_csi"])
    gains = [float(row["support_csi_gain"]) for row in window_rows]
    opposition = _opposition_summary(all_local_shifts)
    gain_distribution = _distribution(gains)
    orbit_row = {
        "sample_id": sample_id,
        "file_name": file_name,
        "nscan": gr.shape[0],
        "nray": gr.shape[1],
        "z_size": gr.shape[2],
        "window_count": len(windows),
        "planned_window_height_count": planned_count,
        "valid_window_height_count": len(window_rows),
        "skipped_no_dpr_support_count": skipped_no_target,
        "skipped_no_gr_support_count": skipped_no_source,
        "valid_height_count": sum(bool(items) for items in valid_items_by_height),
        "height_with_opposing_local_shifts_count": height_with_opposition,
        "occupancy_domain_count": int(np.count_nonzero(domain)),
        "audited_domain_count": int(orbit_exact_metrics["domain_count"]),
        "best_scan_shift": int(orbit_best["scan_shift"]),
        "best_ray_shift": int(orbit_best["ray_shift"]),
        **_prefixed_counts("exact", orbit_exact_metrics),
        **_prefixed_counts("single_shift_oracle", orbit_best),
        **_prefixed_counts("per_height_oracle", orbit_height_metrics),
        **_prefixed_counts("local_oracle", orbit_local_metrics),
        "single_shift_csi_gain": single_csi - exact_csi,
        "per_height_oracle_csi_gain": per_height_csi - exact_csi,
        "local_oracle_csi_gain": local_csi - exact_csi,
        "local_minus_single_shift_csi": local_csi - single_csi,
        "local_minus_per_height_csi": local_csi - per_height_csi,
        "mean_window_csi_gain": gain_distribution["mean"],
        "median_window_csi_gain": gain_distribution["median"],
        "p90_window_csi_gain": gain_distribution["p90"],
        "positive_gain_window_fraction": gain_distribution["positive_fraction"],
        **opposition,
    }

    orbit_shift_rows = [
        {
            "sample_id": sample_id,
            "file_name": file_name,
            **candidate,
            "selected_as_best": (
                int(candidate["scan_shift"]) == int(orbit_best["scan_shift"])
                and int(candidate["ray_shift"]) == int(orbit_best["ray_shift"])
            ),
        }
        for candidate in orbit_candidates
    ]
    histogram: dict[tuple[int, int], int] = {offset: 0 for offset in offsets}
    for shift in all_local_shifts:
        histogram[shift] += 1

    return {
        "sample_id": sample_id,
        "file_name": file_name,
        "offsets": offsets,
        "window_height_rows": window_rows,
        "per_height_rows": per_height_rows,
        "orbit_height_shift_rows": orbit_height_shift_rows,
        "orbit_shift_rows": orbit_shift_rows,
        "orbit_row": orbit_row,
        "best_shift_histogram": histogram,
        # Kept for exact additive aggregation across validation orbits.
        "_orbit_candidates": orbit_candidates,
    }


def aggregate_local_shift_audits(
    audits: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate independently selected orbit audits without mixing their labels."""

    if not audits:
        raise ValueError("at least one orbit audit is required")
    offsets = tuple(tuple(value) for value in audits[0]["offsets"])
    for audit in audits[1:]:
        if tuple(tuple(value) for value in audit["offsets"]) != offsets:
            raise ValueError("orbit audits use inconsistent shift grids")

    window_rows = [row for audit in audits for row in audit["window_height_rows"]]
    per_height_rows = [row for audit in audits for row in audit["per_height_rows"]]
    orbit_height_shift_rows = [
        row for audit in audits for row in audit["orbit_height_shift_rows"]
    ]
    orbit_shift_rows = [row for audit in audits for row in audit["orbit_shift_rows"]]
    orbit_rows = [dict(audit["orbit_row"]) for audit in audits]

    all_validation_candidates = _aggregate_candidates(
        [audit["_orbit_candidates"] for audit in audits], offsets
    )
    validation_best = select_best_shift(all_validation_candidates)
    exact = _zero_counts()
    per_orbit = _zero_counts()
    per_height = _zero_counts()
    local = _zero_counts()
    for row in orbit_rows:
        for prefix, destination in (
            ("exact", exact),
            ("single_shift_oracle", per_orbit),
            ("per_height_oracle", per_height),
            ("local_oracle", local),
        ):
            _add_counts(
                destination,
                {name: row[f"{prefix}_{name}"] for name in COUNT_FIELDS},
            )
    exact_metrics = _counts_with_csi(exact)
    per_orbit_metrics = _counts_with_csi(per_orbit)
    per_height_metrics = _counts_with_csi(per_height)
    local_metrics = _counts_with_csi(local)

    gains = [float(row["support_csi_gain"]) for row in window_rows]
    orbit_local_minus_single = [
        float(row["local_minus_single_shift_csi"])
        for row in orbit_rows
        if math.isfinite(float(row["local_minus_single_shift_csi"]))
    ]
    cancellation_orbits = [
        row
        for row in orbit_rows
        if (
            row["opposite_scan_signs_present"]
            or row["opposite_ray_signs_present"]
            or row["opposite_vector_pair_present"]
        )
        and float(row["local_minus_single_shift_csi"]) > 1e-12
    ]

    histogram = {offset: 0 for offset in offsets}
    for audit in audits:
        for offset, count in audit["best_shift_histogram"].items():
            histogram[tuple(offset)] += int(count)
    histogram_rows = [
        {
            "scan_shift": scan_shift,
            "ray_shift": ray_shift,
            "selected_window_height_count": histogram[(scan_shift, ray_shift)],
            "fraction_of_valid_window_heights": _safe_ratio(
                histogram[(scan_shift, ray_shift)], len(window_rows)
            ),
        }
        for scan_shift, ray_shift in offsets
    ]

    exact_csi = float(exact_metrics["support_csi"])
    return {
        "window_height_rows": window_rows,
        "per_height_rows": per_height_rows,
        "orbit_height_shift_rows": orbit_height_shift_rows,
        "orbit_shift_rows": orbit_shift_rows,
        "orbit_rows": orbit_rows,
        "histogram_rows": histogram_rows,
        "all_validation_shift_rows": all_validation_candidates,
        "summary": {
            "orbit_count": len(audits),
            "valid_window_height_count": len(window_rows),
            "planned_window_height_count": sum(
                int(row["planned_window_height_count"]) for row in orbit_rows
            ),
            "opposing_local_shift_orbit_count": sum(
                bool(
                    row["opposite_scan_signs_present"]
                    or row["opposite_ray_signs_present"]
                    or row["opposite_vector_pair_present"]
                )
                for row in orbit_rows
            ),
            "cancellation_evidence_orbit_count": len(cancellation_orbits),
            "cancellation_evidence_sample_ids": [
                row["sample_id"] for row in cancellation_orbits
            ],
            "window_csi_gain_distribution": _distribution(gains),
            "orbit_local_minus_single_shift_distribution": _distribution(
                orbit_local_minus_single
            ),
            "all_validation_best_scan_shift": int(validation_best["scan_shift"]),
            "all_validation_best_ray_shift": int(validation_best["ray_shift"]),
            **_prefixed_counts("exact", exact_metrics),
            **_prefixed_counts("one_shift_all_validation_oracle", validation_best),
            **_prefixed_counts("per_orbit_single_shift_oracle", per_orbit_metrics),
            **_prefixed_counts("per_orbit_height_oracle", per_height_metrics),
            **_prefixed_counts("local_window_height_oracle", local_metrics),
            "one_shift_all_validation_csi_gain": (
                float(validation_best["support_csi"]) - exact_csi
            ),
            "per_orbit_single_shift_csi_gain": (
                float(per_orbit_metrics["support_csi"]) - exact_csi
            ),
            "per_orbit_height_csi_gain": (
                float(per_height_metrics["support_csi"]) - exact_csi
            ),
            "local_window_height_csi_gain": (
                float(local_metrics["support_csi"]) - exact_csi
            ),
        },
    }


__all__ = [
    "COUNT_FIELDS",
    "LocalShiftOptions",
    "aggregate_local_shift_audits",
    "audit_orbit_local_shifts",
    "finite_shift_offsets",
    "non_overlapping_windows",
    "select_best_shift",
]
