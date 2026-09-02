"""Streaming diagnostics for the decomposed Stage-2 inverse problem.

The original Stage-2 score combines occurrence recovery and conditional dBZ
regression into a small number of global metrics.  R0 needs additional,
strictly diagnostic views before a new model is trained:

* calibration of the support probability (Brier/AP/ECE);
* nested dBZ-event skill at several physical thresholds;
* vertical echo-top/base reconstruction;
* horizontal reflectivity-centroid displacement; and
* a height-by-dBZ contoured-frequency-by-altitude diagram (CFAD).

All updates operate on one complete orbit with shape ``(nscan, nray, z)``.
Only additive statistics are retained, so no complete-orbit tensor is kept in
memory after :meth:`update` returns.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .stage2_reflectivity import (
    FractionSkillAccumulator,
    ReflectivityRegressionAccumulator,
    SupportConfusionAccumulator,
)


def _as_bool(value: Any, *, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.bool_:
        raise TypeError(f"{name} must have boolean dtype")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} shape {array.shape} differs from {shape}")
    return array


def _as_float(value: Any, *, name: str, ndim: int = 3) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    return array


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else math.nan


class SupportProbabilityAccumulator:
    """Histogram-based probability calibration with exact Brier score.

    ``histogram_bins`` controls the deterministic approximation used for
    average precision.  Brier score and calibration-bin means remain exact
    with respect to the streamed values (up to float64 summation).
    """

    def __init__(self, *, histogram_bins: int = 1000, calibration_bins: int = 10) -> None:
        if histogram_bins <= 1 or calibration_bins <= 1:
            raise ValueError("probability bin counts must be greater than one")
        self.histogram_bins = int(histogram_bins)
        self.calibration_bins = int(calibration_bins)
        self.positive_histogram = np.zeros(self.histogram_bins, dtype=np.int64)
        self.negative_histogram = np.zeros(self.histogram_bins, dtype=np.int64)
        self.calibration_count = np.zeros(self.calibration_bins, dtype=np.int64)
        self.calibration_probability_sum = np.zeros(
            self.calibration_bins, dtype=np.float64
        )
        self.calibration_target_sum = np.zeros(
            self.calibration_bins, dtype=np.float64
        )
        self.count = 0
        self.positive_count = 0
        self.squared_error_sum = 0.0

    def update(self, probability: Any, target: Any, domain_mask: Any) -> None:
        values = _as_float(probability, name="support_probability")
        truth = _as_bool(target, name="target_support", shape=values.shape)
        domain = _as_bool(domain_mask, name="support_probability domain", shape=values.shape)
        selected = values[domain]
        selected_truth = truth[domain]
        if selected.size == 0:
            return
        if not np.all(np.isfinite(selected)) or np.any((selected < 0.0) | (selected > 1.0)):
            raise ValueError("selected support probabilities must be finite in [0,1]")

        histogram_index = np.minimum(
            (selected * self.histogram_bins).astype(np.int64),
            self.histogram_bins - 1,
        )
        self.positive_histogram += np.bincount(
            histogram_index[selected_truth], minlength=self.histogram_bins
        )
        self.negative_histogram += np.bincount(
            histogram_index[~selected_truth], minlength=self.histogram_bins
        )

        calibration_index = np.minimum(
            (selected * self.calibration_bins).astype(np.int64),
            self.calibration_bins - 1,
        )
        self.calibration_count += np.bincount(
            calibration_index, minlength=self.calibration_bins
        )
        self.calibration_probability_sum += np.bincount(
            calibration_index, weights=selected, minlength=self.calibration_bins
        )
        self.calibration_target_sum += np.bincount(
            calibration_index,
            weights=selected_truth.astype(np.float64),
            minlength=self.calibration_bins,
        )
        error = selected - selected_truth.astype(np.float64)
        self.count += int(selected.size)
        self.positive_count += int(np.count_nonzero(selected_truth))
        self.squared_error_sum += float(np.square(error).sum(dtype=np.float64))

    def merge(self, other: "SupportProbabilityAccumulator") -> None:
        if (
            self.histogram_bins != other.histogram_bins
            or self.calibration_bins != other.calibration_bins
        ):
            raise ValueError("cannot merge probability accumulators with different bins")
        self.positive_histogram += other.positive_histogram
        self.negative_histogram += other.negative_histogram
        self.calibration_count += other.calibration_count
        self.calibration_probability_sum += other.calibration_probability_sum
        self.calibration_target_sum += other.calibration_target_sum
        self.count += other.count
        self.positive_count += other.positive_count
        self.squared_error_sum += other.squared_error_sum

    def compute(self) -> dict[str, Any]:
        # Walk thresholds from probability 1 down to 0.  The step-integral of
        # precision over recall is the binned average-precision estimate.
        tp = np.cumsum(self.positive_histogram[::-1], dtype=np.float64)
        fp = np.cumsum(self.negative_histogram[::-1], dtype=np.float64)
        precision = np.divide(tp, tp + fp, out=np.ones_like(tp), where=(tp + fp) > 0)
        recall = np.divide(
            tp,
            float(self.positive_count),
            out=np.zeros_like(tp),
            where=self.positive_count > 0,
        )
        recall_previous = np.concatenate(([0.0], recall[:-1]))
        average_precision = float(np.sum((recall - recall_previous) * precision))

        calibration: list[dict[str, int | float]] = []
        ece_numerator = 0.0
        for index, count_value in enumerate(self.calibration_count):
            count = int(count_value)
            mean_probability = _safe_ratio(
                float(self.calibration_probability_sum[index]), count
            )
            observed_frequency = _safe_ratio(
                float(self.calibration_target_sum[index]), count
            )
            if count:
                ece_numerator += count * abs(mean_probability - observed_frequency)
            calibration.append(
                {
                    "bin_index": index,
                    "lower": index / self.calibration_bins,
                    "upper": (index + 1) / self.calibration_bins,
                    "count": count,
                    "mean_probability": mean_probability,
                    "observed_frequency": observed_frequency,
                }
            )
        return {
            "count": self.count,
            "positive_count": self.positive_count,
            "prevalence": _safe_ratio(self.positive_count, self.count),
            "brier_score": _safe_ratio(self.squared_error_sum, self.count),
            "average_precision_histogram": (
                average_precision if self.positive_count else math.nan
            ),
            "histogram_bins": self.histogram_bins,
            "expected_calibration_error": _safe_ratio(ece_numerator, self.count),
            "calibration_bins": calibration,
        }


class MultiThresholdSpatialAccumulator:
    """Occurrence/FSS skill for nested physical-dBZ events."""

    def __init__(
        self,
        thresholds_dbz: Sequence[float] = (15.0, 25.0, 35.0),
        *,
        fss_radii: Sequence[int] = (0, 1, 2, 4),
    ) -> None:
        thresholds = tuple(float(value) for value in thresholds_dbz)
        radii = tuple(int(value) for value in fss_radii)
        if not thresholds or any(not math.isfinite(value) for value in thresholds):
            raise ValueError("thresholds_dbz must contain finite values")
        if tuple(sorted(set(thresholds))) != thresholds:
            raise ValueError("thresholds_dbz must be sorted and unique")
        if any(value < 0 for value in radii) or len(set(radii)) != len(radii):
            raise ValueError("fss_radii must be unique non-negative integers")
        self.thresholds_dbz = thresholds
        self.fss_radii = radii
        self.confusion = {value: SupportConfusionAccumulator() for value in thresholds}
        self.fss = {
            value: {radius: FractionSkillAccumulator(radius) for radius in radii}
            for value in thresholds
        }

    def update(
        self,
        prediction_dbz: Any,
        predicted_support: Any,
        target_dbz: Any,
        target_support: Any,
        domain_mask: Any,
    ) -> None:
        prediction = _as_float(prediction_dbz, name="prediction_dbz")
        target = _as_float(target_dbz, name="target_dbz")
        if target.shape != prediction.shape:
            raise ValueError("prediction_dbz and target_dbz shapes differ")
        predicted = _as_bool(
            predicted_support, name="predicted_support", shape=prediction.shape
        )
        truth = _as_bool(target_support, name="target_support", shape=prediction.shape)
        domain = _as_bool(domain_mask, name="threshold domain", shape=prediction.shape)
        if np.any(domain & predicted & ~np.isfinite(prediction)):
            raise ValueError("selected predicted dBZ values must be finite")
        if np.any(domain & truth & ~np.isfinite(target)):
            raise ValueError("selected target dBZ values must be finite")
        for threshold in self.thresholds_dbz:
            predicted_event = predicted & np.isfinite(prediction) & (prediction >= threshold)
            target_event = truth & np.isfinite(target) & (target >= threshold)
            self.confusion[threshold].update(predicted_event, target_event, domain)
            for accumulator in self.fss[threshold].values():
                accumulator.update(predicted_event, target_event, domain)

    def compute(self) -> dict[str, Any]:
        return {
            f"{threshold:g}": {
                "threshold_dbz": threshold,
                "support": self.confusion[threshold].compute(),
                "fss": {
                    str(radius): accumulator.compute()
                    for radius, accumulator in self.fss[threshold].items()
                },
            }
            for threshold in self.thresholds_dbz
        }


class EchoColumnAccumulator:
    """Column occurrence and paired echo-top/base height errors."""

    def __init__(self, heights_km: Sequence[float]) -> None:
        heights = np.asarray(heights_km, dtype=np.float64)
        if heights.ndim != 1 or heights.size == 0 or not np.all(np.isfinite(heights)):
            raise ValueError("heights_km must be a finite non-empty 1-D array")
        if np.any(np.diff(heights) <= 0.0):
            raise ValueError("heights_km must be strictly increasing")
        self.heights_km = heights
        self.column_support = SupportConfusionAccumulator()
        self.top = ReflectivityRegressionAccumulator()
        self.base = ReflectivityRegressionAccumulator()

    def update(self, predicted_support: Any, target_support: Any, domain_mask: Any) -> None:
        predicted = _as_bool(predicted_support, name="predicted_support")
        if predicted.ndim != 3 or predicted.shape[-1] != self.heights_km.size:
            raise ValueError("support must have shape (nscan,nray,len(heights_km))")
        truth = _as_bool(target_support, name="target_support", shape=predicted.shape)
        domain = _as_bool(domain_mask, name="column domain", shape=predicted.shape)
        predicted_event = predicted & domain
        target_event = truth & domain
        column_domain = np.any(domain, axis=-1)
        predicted_column = np.any(predicted_event, axis=-1)
        target_column = np.any(target_event, axis=-1)
        self.column_support.update(predicted_column, target_column, column_domain)
        paired = column_domain & predicted_column & target_column
        if not np.any(paired):
            return

        height = self.heights_km.reshape(1, 1, -1)
        predicted_top = np.max(np.where(predicted_event, height, -np.inf), axis=-1)
        target_top = np.max(np.where(target_event, height, -np.inf), axis=-1)
        predicted_base = np.min(np.where(predicted_event, height, np.inf), axis=-1)
        target_base = np.min(np.where(target_event, height, np.inf), axis=-1)
        self.top.update(predicted_top, target_top, paired)
        self.base.update(predicted_base, target_base, paired)

    def compute(self) -> dict[str, Any]:
        def renamed(values: Mapping[str, int | float]) -> dict[str, int | float]:
            return {
                name.replace("_dbz", "_km"): value for name, value in values.items()
            }

        return {
            "column_support": self.column_support.compute(),
            "paired_echo_top": renamed(self.top.compute()),
            "paired_echo_base": renamed(self.base.compute()),
        }


@dataclass
class CentroidDisplacementAccumulator:
    """Per-height reflectivity-weighted centroid displacement in grid cells."""

    count: int = 0
    sum_distance: float = 0.0
    sum_squared_distance: float = 0.0
    sum_scan_offset: float = 0.0
    sum_ray_offset: float = 0.0

    @staticmethod
    def _linear_weights(dbz: np.ndarray, support: np.ndarray) -> np.ndarray:
        # Clipping only protects exponentiation.  Values outside support are
        # exactly zero and cannot affect the centroid.
        safe = np.clip(np.where(support, dbz, 0.0), -20.0, 60.0)
        return np.where(support, np.power(10.0, safe / 10.0), 0.0)

    def update(
        self,
        prediction_dbz: Any,
        predicted_support: Any,
        target_dbz: Any,
        target_support: Any,
        domain_mask: Any,
    ) -> None:
        prediction = _as_float(prediction_dbz, name="prediction_dbz")
        target = _as_float(target_dbz, name="target_dbz")
        if prediction.shape != target.shape:
            raise ValueError("centroid prediction and target shapes differ")
        predicted = _as_bool(
            predicted_support, name="predicted_support", shape=prediction.shape
        )
        truth = _as_bool(target_support, name="target_support", shape=prediction.shape)
        domain = _as_bool(domain_mask, name="centroid domain", shape=prediction.shape)
        # ``np.asarray`` may return a view of the caller-owned support arrays.
        # Never mask those arrays in place: the cascade evaluator reuses them
        # for the factorial and regional-oracle routes after this diagnostic.
        predicted = predicted & domain
        truth = truth & domain
        if np.any(predicted & ~np.isfinite(prediction)) or np.any(truth & ~np.isfinite(target)):
            raise ValueError("selected centroid dBZ values must be finite")

        scan_grid = np.arange(prediction.shape[0], dtype=np.float64)[:, None]
        ray_grid = np.arange(prediction.shape[1], dtype=np.float64)[None, :]
        for level in range(prediction.shape[-1]):
            predicted_weights = self._linear_weights(
                prediction[..., level], predicted[..., level]
            )
            target_weights = self._linear_weights(target[..., level], truth[..., level])
            predicted_sum = float(predicted_weights.sum(dtype=np.float64))
            target_sum = float(target_weights.sum(dtype=np.float64))
            if predicted_sum <= 0.0 or target_sum <= 0.0:
                continue
            predicted_scan = float((predicted_weights * scan_grid).sum() / predicted_sum)
            target_scan = float((target_weights * scan_grid).sum() / target_sum)
            predicted_ray = float((predicted_weights * ray_grid).sum() / predicted_sum)
            target_ray = float((target_weights * ray_grid).sum() / target_sum)
            scan_offset = predicted_scan - target_scan
            ray_offset = predicted_ray - target_ray
            distance = math.hypot(scan_offset, ray_offset)
            self.count += 1
            self.sum_distance += distance
            self.sum_squared_distance += distance * distance
            self.sum_scan_offset += scan_offset
            self.sum_ray_offset += ray_offset

    def compute(self) -> dict[str, int | float]:
        return {
            "paired_orbit_height_count": self.count,
            "mean_distance_grid_cells": _safe_ratio(self.sum_distance, self.count),
            "rmse_distance_grid_cells": (
                math.sqrt(self.sum_squared_distance / self.count)
                if self.count
                else math.nan
            ),
            "mean_scan_offset_grid_cells": _safe_ratio(
                self.sum_scan_offset, self.count
            ),
            "mean_ray_offset_grid_cells": _safe_ratio(self.sum_ray_offset, self.count),
        }


class CfadAccumulator:
    """Height-by-dBZ frequency table for target and predicted echo voxels."""

    def __init__(self, heights_km: Sequence[float], dbz_edges: Sequence[float]) -> None:
        heights = np.asarray(heights_km, dtype=np.float64)
        edges = np.asarray(dbz_edges, dtype=np.float64)
        if heights.ndim != 1 or heights.size == 0:
            raise ValueError("heights_km must be a non-empty 1-D array")
        if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0.0):
            raise ValueError("dbz_edges must be a strictly increasing 1-D array")
        if not np.all(np.isfinite(heights)) or not np.all(np.isfinite(edges)):
            raise ValueError("CFAD heights and dBZ edges must be finite")
        self.heights_km = heights
        self.dbz_edges = edges
        shape = (heights.size, edges.size - 1)
        self.prediction_histogram = np.zeros(shape, dtype=np.int64)
        self.target_histogram = np.zeros(shape, dtype=np.int64)

    def update(
        self,
        prediction_dbz: Any,
        predicted_support: Any,
        target_dbz: Any,
        target_support: Any,
        domain_mask: Any,
    ) -> None:
        prediction = _as_float(prediction_dbz, name="prediction_dbz")
        target = _as_float(target_dbz, name="target_dbz")
        if prediction.shape != target.shape or prediction.shape[-1] != self.heights_km.size:
            raise ValueError("CFAD tensors must share shape (..., len(heights_km))")
        predicted = _as_bool(
            predicted_support, name="predicted_support", shape=prediction.shape
        )
        truth = _as_bool(target_support, name="target_support", shape=prediction.shape)
        domain = _as_bool(domain_mask, name="CFAD domain", shape=prediction.shape)
        for level in range(self.heights_km.size):
            predicted_selected = predicted[..., level] & domain[..., level]
            target_selected = truth[..., level] & domain[..., level]
            predicted_values = prediction[..., level][predicted_selected]
            target_values = target[..., level][target_selected]
            if not np.all(np.isfinite(predicted_values)) or not np.all(np.isfinite(target_values)):
                raise ValueError("selected CFAD dBZ values must be finite")
            self.prediction_histogram[level] += np.histogram(
                predicted_values, bins=self.dbz_edges
            )[0]
            self.target_histogram[level] += np.histogram(
                target_values, bins=self.dbz_edges
            )[0]

    def rows(self) -> list[dict[str, int | float]]:
        rows: list[dict[str, int | float]] = []
        prediction_total = self.prediction_histogram.sum(axis=1)
        target_total = self.target_histogram.sum(axis=1)
        for level, height in enumerate(self.heights_km):
            for bin_index, (lower, upper) in enumerate(
                zip(self.dbz_edges[:-1], self.dbz_edges[1:])
            ):
                predicted_count = int(self.prediction_histogram[level, bin_index])
                target_count = int(self.target_histogram[level, bin_index])
                rows.append(
                    {
                        "height_index": level,
                        "height_km": float(height),
                        "dbz_bin_index": bin_index,
                        "dbz_lower": float(lower),
                        "dbz_upper": float(upper),
                        "prediction_count": predicted_count,
                        "target_count": target_count,
                        "prediction_fraction_at_height": _safe_ratio(
                            predicted_count, int(prediction_total[level])
                        ),
                        "target_fraction_at_height": _safe_ratio(
                            target_count, int(target_total[level])
                        ),
                    }
                )
        return rows


@dataclass
class Stage2DecompositionDiagnostics:
    """Complete R0 physical diagnostics for one Stage-2 prediction stream."""

    probability: SupportProbabilityAccumulator
    threshold_spatial: MultiThresholdSpatialAccumulator
    columns: EchoColumnAccumulator
    centroid: CentroidDisplacementAccumulator
    cfad: CfadAccumulator
    region_probability: dict[str, SupportProbabilityAccumulator] = field(
        default_factory=dict
    )

    @classmethod
    def create(
        cls,
        heights_km: Sequence[float],
        *,
        thresholds_dbz: Sequence[float] = (15.0, 25.0, 35.0),
        fss_radii: Sequence[int] = (0, 1, 2, 4),
        cfad_edges_dbz: Sequence[float] = tuple(np.arange(-10.0, 60.1, 5.0)),
        probability_histogram_bins: int = 1000,
        probability_calibration_bins: int = 10,
    ) -> "Stage2DecompositionDiagnostics":
        return cls(
            probability=SupportProbabilityAccumulator(
                histogram_bins=probability_histogram_bins,
                calibration_bins=probability_calibration_bins,
            ),
            threshold_spatial=MultiThresholdSpatialAccumulator(
                thresholds_dbz, fss_radii=fss_radii
            ),
            columns=EchoColumnAccumulator(heights_km),
            centroid=CentroidDisplacementAccumulator(),
            cfad=CfadAccumulator(heights_km, cfad_edges_dbz),
        )

    def update(
        self,
        support_probability: Any,
        prediction_dbz: Any,
        predicted_support: Any,
        target_dbz: Any,
        target_support: Any,
        label_domain_mask: Any,
        *,
        region_masks: Mapping[str, Any] | None = None,
    ) -> None:
        probability = _as_float(support_probability, name="support_probability")
        prediction = _as_float(prediction_dbz, name="prediction_dbz")
        if probability.shape != prediction.shape:
            raise ValueError("Stage-2 probability and dBZ shapes differ")
        predicted = _as_bool(
            predicted_support, name="predicted_support", shape=prediction.shape
        )
        target = _as_float(target_dbz, name="target_dbz")
        if target.shape != prediction.shape:
            raise ValueError("Stage-2 target and predicted dBZ shapes differ")
        truth = _as_bool(target_support, name="target_support", shape=prediction.shape)
        domain = _as_bool(
            label_domain_mask, name="label_domain_mask", shape=prediction.shape
        )
        self.probability.update(probability, truth, domain)
        self.threshold_spatial.update(
            prediction, predicted, target, truth, domain
        )
        self.columns.update(predicted, truth, domain)
        self.centroid.update(prediction, predicted, target, truth, domain)
        self.cfad.update(prediction, predicted, target, truth, domain)
        if region_masks is not None:
            for name, raw_mask in region_masks.items():
                mask = _as_bool(raw_mask, name=f"region {name}", shape=prediction.shape)
                accumulator = self.region_probability.setdefault(
                    str(name),
                    SupportProbabilityAccumulator(
                        histogram_bins=self.probability.histogram_bins,
                        calibration_bins=self.probability.calibration_bins,
                    ),
                )
                accumulator.update(probability, truth, domain & mask)

    def compute(self) -> dict[str, Any]:
        return {
            "support_probability": self.probability.compute(),
            "support_probability_by_region": {
                name: accumulator.compute()
                for name, accumulator in sorted(self.region_probability.items())
            },
            "nested_dbz_events": self.threshold_spatial.compute(),
            "echo_columns": self.columns.compute(),
            "reflectivity_centroid": self.centroid.compute(),
        }
