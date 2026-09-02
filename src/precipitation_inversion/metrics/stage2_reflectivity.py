"""Mask-aware metrics for stage-two GR-to-DPR reflectivity conversion.

Stage two has two coupled objectives that must be reported separately:

* support prediction: which voxels contain a retained DPR reflectivity value;
* reflectivity regression: how accurate dBZ is where a prediction and a DPR
  target are both available.

The accumulators below store only additive float64 statistics.  They therefore
work for complete orbits, file-by-file evaluation, and later distributed
aggregation without retaining dense predictions.  Arrays conventionally have
shape ``(B,1,D,H,Z)`` or ``(D,H,Z)``, but any identical shapes are accepted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


def _bool_array(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.bool_:
        raise TypeError(f"{name} must have boolean dtype")
    return array


def _same_shape(reference: np.ndarray, value: np.ndarray, *, name: str) -> None:
    if value.shape != reference.shape:
        raise ValueError(
            f"{name} shape {value.shape} differs from reference {reference.shape}"
        )


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else math.nan


@dataclass
class SupportConfusionAccumulator:
    """Streaming binary support confusion matrix inside an explicit domain."""

    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    true_negative: int = 0

    def update(
        self,
        predicted_support: Any,
        target_support: Any,
        domain_mask: Any,
    ) -> None:
        """Accumulate boolean support arrays of the same shape.

        ``domain_mask`` is normally the trustworthy DPR occurrence-label
        domain.  Values outside it must not become implicit true negatives.
        """

        predicted = _bool_array(predicted_support, name="predicted_support")
        target = _bool_array(target_support, name="target_support")
        domain = _bool_array(domain_mask, name="domain_mask")
        _same_shape(predicted, target, name="target_support")
        _same_shape(predicted, domain, name="domain_mask")
        self.true_positive += int(np.count_nonzero(domain & predicted & target))
        self.false_positive += int(np.count_nonzero(domain & predicted & ~target))
        self.false_negative += int(np.count_nonzero(domain & ~predicted & target))
        self.true_negative += int(np.count_nonzero(domain & ~predicted & ~target))

    def merge(self, other: "SupportConfusionAccumulator") -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def compute(self) -> dict[str, int | float]:
        tp = self.true_positive
        fp = self.false_positive
        fn = self.false_negative
        tn = self.true_negative
        count = tp + fp + fn + tn
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        return {
            "count": count,
            "target_positive_count": tp + fn,
            "predicted_positive_count": tp + fp,
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "true_negative": tn,
            "prevalence": _safe_divide(tp + fn, count),
            "accuracy": _safe_divide(tp + tn, count),
            "precision": precision,
            "recall": recall,
            "pod": recall,
            "false_alarm_ratio": _safe_divide(fp, tp + fp),
            "specificity": _safe_divide(tn, tn + fp),
            "f1": _safe_divide(2 * tp, 2 * tp + fp + fn),
            "csi": _safe_divide(tp, tp + fp + fn),
        }


@dataclass
class ReflectivityRegressionAccumulator:
    """Streaming dBZ regression statistics under a boolean selection mask."""

    count: int = 0
    sum_abs_error: float = 0.0
    sum_squared_error: float = 0.0
    sum_error: float = 0.0
    sum_prediction: float = 0.0
    sum_target: float = 0.0
    sum_prediction_squared: float = 0.0
    sum_target_squared: float = 0.0
    sum_product: float = 0.0

    def update(self, prediction_dbz: Any, target_dbz: Any, mask: Any) -> None:
        prediction = np.asarray(prediction_dbz, dtype=np.float64)
        target = np.asarray(target_dbz, dtype=np.float64)
        selected = _bool_array(mask, name="regression mask")
        _same_shape(prediction, target, name="target_dbz")
        _same_shape(prediction, selected, name="regression mask")
        if not np.any(selected):
            return
        predicted_values = prediction[selected]
        target_values = target[selected]
        if not (
            np.all(np.isfinite(predicted_values))
            and np.all(np.isfinite(target_values))
        ):
            raise ValueError("selected reflectivity values must be finite")
        error = predicted_values - target_values
        self.count += int(predicted_values.size)
        self.sum_abs_error += float(np.abs(error).sum(dtype=np.float64))
        self.sum_squared_error += float(np.square(error).sum(dtype=np.float64))
        self.sum_error += float(error.sum(dtype=np.float64))
        self.sum_prediction += float(predicted_values.sum(dtype=np.float64))
        self.sum_target += float(target_values.sum(dtype=np.float64))
        self.sum_prediction_squared += float(
            np.square(predicted_values).sum(dtype=np.float64)
        )
        self.sum_target_squared += float(
            np.square(target_values).sum(dtype=np.float64)
        )
        self.sum_product += float(
            (predicted_values * target_values).sum(dtype=np.float64)
        )

    def merge(self, other: "ReflectivityRegressionAccumulator") -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def compute(self) -> dict[str, int | float]:
        if self.count == 0:
            return {
                "count": 0,
                "mae_dbz": math.nan,
                "rmse_dbz": math.nan,
                "bias_dbz": math.nan,
                "r2": math.nan,
                "pearson_r": math.nan,
                "ccc": math.nan,
            }
        count = self.count
        target_centered = self.sum_target_squared - self.sum_target**2 / count
        prediction_centered = (
            self.sum_prediction_squared - self.sum_prediction**2 / count
        )
        covariance = self.sum_product - self.sum_prediction * self.sum_target / count
        pearson_denominator = math.sqrt(
            max(prediction_centered, 0.0) * max(target_centered, 0.0)
        )
        ccc_denominator = (
            prediction_centered
            + target_centered
            + (self.sum_prediction - self.sum_target) ** 2 / count
        )
        return {
            "count": count,
            "mae_dbz": self.sum_abs_error / count,
            "rmse_dbz": math.sqrt(self.sum_squared_error / count),
            "bias_dbz": self.sum_error / count,
            "r2": (
                1.0 - self.sum_squared_error / target_centered
                if target_centered > 0.0
                else math.nan
            ),
            "pearson_r": (
                covariance / pearson_denominator
                if pearson_denominator > 0.0
                else math.nan
            ),
            "ccc": (
                2.0 * covariance / ccc_denominator
                if ccc_denominator > 0.0
                else math.nan
            ),
        }


def horizontal_window_sum(values: Any, radius: int) -> np.ndarray:
    """Sum a horizontal square window independently at every height.

    Input and output shapes are ``(..., D, H, Z)``.  Only the final three axes
    participate: the operation never mixes physical height levels.  Values
    outside the array are treated as zero and edge windows are truncated.
    """

    if radius < 0:
        raise ValueError("radius must be non-negative")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 3:
        raise ValueError("horizontal_window_sum requires at least three dimensions")
    if radius == 0:
        return array.copy()
    leading = array.shape[:-3]
    d_size, h_size, z_size = array.shape[-3:]
    flat = array.reshape((-1, d_size, h_size, z_size))
    # Zero-pad by the requested radius, then add a leading zero row/column for
    # a fully vectorized summed-area table.  Four shifted slices calculate all
    # horizontal windows at once; the final Z axis remains untouched.
    window = 2 * radius + 1
    padded = np.pad(
        flat,
        ((0, 0), (radius, radius), (radius, radius), (0, 0)),
        mode="constant",
    )
    integral = np.pad(
        padded,
        ((0, 0), (1, 0), (1, 0), (0, 0)),
        mode="constant",
    ).cumsum(axis=1).cumsum(axis=2)
    output = (
        integral[:, window:, window:, :]
        - integral[:, :-window, window:, :]
        - integral[:, window:, :-window, :]
        + integral[:, :-window, :-window, :]
    )
    return output.reshape((*leading, d_size, h_size, z_size))


@dataclass
class FractionSkillAccumulator:
    """Streaming horizontal fraction skill score (FSS) at one radius."""

    radius: int
    count: int = 0
    sum_squared_difference: float = 0.0
    sum_squared_reference: float = 0.0

    def __post_init__(self) -> None:
        if self.radius < 0:
            raise ValueError("radius must be non-negative")

    def update(
        self,
        predicted_support: Any,
        target_support: Any,
        domain_mask: Any,
    ) -> None:
        predicted = _bool_array(predicted_support, name="predicted_support")
        target = _bool_array(target_support, name="target_support")
        domain = _bool_array(domain_mask, name="domain_mask")
        _same_shape(predicted, target, name="target_support")
        _same_shape(predicted, domain, name="domain_mask")

        # Domain-aware fractions prevent unavailable target cells from being
        # silently interpreted as observed no-echo.  All arrays keep their
        # original (...,D,H,Z) shape before the final boolean selection.
        window_domain = horizontal_window_sum(domain, self.radius)
        valid_centers = domain & (window_domain > 0.0)
        if not np.any(valid_centers):
            return
        predicted_fraction = horizontal_window_sum(
            predicted & domain, self.radius
        ) / np.maximum(window_domain, 1.0)
        target_fraction = horizontal_window_sum(
            target & domain, self.radius
        ) / np.maximum(window_domain, 1.0)
        predicted_values = predicted_fraction[valid_centers]
        target_values = target_fraction[valid_centers]
        self.count += int(predicted_values.size)
        self.sum_squared_difference += float(
            np.square(predicted_values - target_values).sum(dtype=np.float64)
        )
        self.sum_squared_reference += float(
            (np.square(predicted_values) + np.square(target_values)).sum(
                dtype=np.float64
            )
        )

    def merge(self, other: "FractionSkillAccumulator") -> None:
        if self.radius != other.radius:
            raise ValueError("cannot merge FSS accumulators with different radii")
        self.count += other.count
        self.sum_squared_difference += other.sum_squared_difference
        self.sum_squared_reference += other.sum_squared_reference

    def compute(self) -> dict[str, int | float]:
        return {
            "radius": self.radius,
            "window_size": 2 * self.radius + 1,
            "count": self.count,
            "mse": _safe_divide(self.sum_squared_difference, self.count),
            "reference_mse": _safe_divide(self.sum_squared_reference, self.count),
            "fss": (
                1.0
                - self.sum_squared_difference / self.sum_squared_reference
                if self.sum_squared_reference > 0.0
                else math.nan
            ),
        }


class Stage2ReflectivityMetrics:
    """Joint support, target/common-support dBZ, and multi-scale FSS metrics.

    ``reflectivity_on_target_support`` evaluates the continuous dBZ head on
    every trustworthy DPR target and is independent of the chosen occurrence
    threshold. ``reflectivity_on_common_support`` describes the final gated
    product and must be interpreted together with support recall.
    """

    def __init__(
        self,
        *,
        fss_radii: tuple[int, ...] = (1, 2, 4),
        dense_prediction: bool = False,
    ) -> None:
        radii = tuple(int(value) for value in fss_radii)
        if any(value < 0 for value in radii) or len(set(radii)) != len(radii):
            raise ValueError("fss_radii must be unique non-negative integers")
        self.support = SupportConfusionAccumulator()
        self.dense_prediction = bool(dense_prediction)
        self.reflectivity_target = ReflectivityRegressionAccumulator()
        self.reflectivity = ReflectivityRegressionAccumulator()
        self.fss = {radius: FractionSkillAccumulator(radius) for radius in radii}

    def update(
        self,
        prediction_dbz: Any,
        predicted_support: Any,
        target_dbz: Any,
        target_support: Any,
        domain_mask: Any,
    ) -> None:
        predicted = _bool_array(predicted_support, name="predicted_support")
        target = _bool_array(target_support, name="target_support")
        domain = _bool_array(domain_mask, name="domain_mask")
        self.support.update(predicted, target, domain)
        if self.dense_prediction:
            target_domain = domain & target
            self.reflectivity_target.update(
                prediction_dbz, target_dbz, target_domain
            )
        # Sparse baselines do not produce dBZ outside their support.  dBZ
        # quality is consequently computed on common support and must always be
        # interpreted together with support recall/target coverage.
        common_support = domain & predicted & target
        self.reflectivity.update(prediction_dbz, target_dbz, common_support)
        for accumulator in self.fss.values():
            accumulator.update(predicted, target, domain)

    def merge(self, other: "Stage2ReflectivityMetrics") -> None:
        if tuple(self.fss) != tuple(other.fss):
            raise ValueError("cannot merge Stage2 metrics with different FSS radii")
        if self.dense_prediction != other.dense_prediction:
            raise ValueError("cannot merge sparse and dense Stage2 metrics")
        self.support.merge(other.support)
        self.reflectivity_target.merge(other.reflectivity_target)
        self.reflectivity.merge(other.reflectivity)
        for radius, accumulator in self.fss.items():
            accumulator.merge(other.fss[radius])

    def compute(self) -> dict[str, Any]:
        return {
            "support": self.support.compute(),
            "reflectivity_on_target_support": self.reflectivity_target.compute(),
            "reflectivity_on_common_support": self.reflectivity.compute(),
            "fss": {
                str(radius): accumulator.compute()
                for radius, accumulator in self.fss.items()
            },
        }


def finite_metrics_for_json(value: Any) -> Any:
    """Recursively replace NaN/Inf metric values with JSON-safe ``None``."""

    if isinstance(value, Mapping):
        return {str(key): finite_metrics_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_metrics_for_json(item) for item in value]
    # bool is an int subclass in Python; preserve its semantic type before the
    # generic integer branch (R0 completeness/contract flags depend on this).
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    return value
