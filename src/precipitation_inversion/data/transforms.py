"""Train-only online statistics and reversible data transforms.

Normalization statistics are accumulated independently for every height level
with a parallel/mergeable form of Welford's algorithm.  Missing values never
contribute to statistics.  The same accumulator can process scan chunks and
files one at a time, avoiding an in-memory copy of the complete dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


def _json_values(values: np.ndarray) -> list[int | float | None]:
    """Convert a one-dimensional numeric array to strict JSON-safe values."""

    result: list[int | float | None] = []
    for value in np.asarray(values).tolist():
        if isinstance(value, int):
            result.append(value)
        elif np.isfinite(value):
            result.append(float(value))
        else:
            result.append(None)
    return result


@dataclass
class PerLevelRunningStats:
    """Mergeable population moments for arrays whose last axis is height.

    ``count``, ``mean`` and ``m2`` are maintained per level. ``m2`` stores the
    sum of squared deviations from the current mean, so separate chunks can be
    merged without losing the variation between their means.
    """

    count: np.ndarray
    mean: np.ndarray
    m2: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray

    @classmethod
    def empty(cls, level_count: int) -> "PerLevelRunningStats":
        if level_count <= 0:
            raise ValueError("level_count must be positive")
        return cls(
            count=np.zeros(level_count, dtype=np.int64),
            mean=np.zeros(level_count, dtype=np.float64),
            m2=np.zeros(level_count, dtype=np.float64),
            minimum=np.full(level_count, np.inf, dtype=np.float64),
            maximum=np.full(level_count, -np.inf, dtype=np.float64),
        )

    @property
    def level_count(self) -> int:
        return int(self.count.size)

    def _validate_state(self) -> None:
        expected = (self.level_count,)
        for name in ("count", "mean", "m2", "minimum", "maximum"):
            if getattr(self, name).shape != expected:
                raise ValueError(f"{name} must have shape {expected}")
        if np.any(self.count < 0):
            raise ValueError("count cannot be negative")

    def update(
        self, values: Any, *, valid_mask: np.ndarray | None = None
    ) -> "PerLevelRunningStats":
        """Update from one array, reducing every axis except the last.

        NaN/Inf values are always rejected. An optional mask can impose an
        additional task-specific selection such as ``pre_positive_qc``.
        """

        self._validate_state()
        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 0 or array.shape[-1] != self.level_count:
            raise ValueError(
                f"values must end with {self.level_count} height levels, got "
                f"shape {array.shape}"
            )
        valid = np.isfinite(array)
        if valid_mask is not None:
            selected = np.asarray(valid_mask, dtype=bool)
            if selected.shape != array.shape:
                raise ValueError(
                    f"valid_mask shape {selected.shape} != values shape {array.shape}"
                )
            valid &= selected

        flat = array.reshape(-1, self.level_count)
        flat_valid = valid.reshape(-1, self.level_count)
        batch_count = flat_valid.sum(axis=0, dtype=np.int64)
        has_values = batch_count > 0
        if not has_values.any():
            return self

        safe_values = np.where(flat_valid, flat, 0.0)
        batch_sum = safe_values.sum(axis=0, dtype=np.float64)
        batch_mean = np.zeros(self.level_count, dtype=np.float64)
        batch_mean[has_values] = batch_sum[has_values] / batch_count[has_values]
        deviations = np.where(
            flat_valid, flat - batch_mean[np.newaxis, :], 0.0
        )
        batch_m2 = np.square(deviations).sum(axis=0, dtype=np.float64)
        batch_min = np.min(np.where(flat_valid, flat, np.inf), axis=0)
        batch_max = np.max(np.where(flat_valid, flat, -np.inf), axis=0)

        incoming = PerLevelRunningStats(
            count=batch_count,
            mean=batch_mean,
            m2=batch_m2,
            minimum=batch_min,
            maximum=batch_max,
        )
        return self.merge(incoming)

    def merge(self, other: "PerLevelRunningStats") -> "PerLevelRunningStats":
        """Merge another independent accumulator into this one in place."""

        self._validate_state()
        other._validate_state()
        if other.level_count != self.level_count:
            raise ValueError(
                f"level counts differ: {self.level_count} != {other.level_count}"
            )
        active = other.count > 0
        if not active.any():
            return self

        old_count = self.count.copy()
        combined_count = old_count + other.count
        delta = other.mean - self.mean
        # The correction term accounts for the distance between chunk means.
        correction = np.zeros(self.level_count, dtype=np.float64)
        both = (old_count > 0) & active
        correction[both] = (
            np.square(delta[both])
            * old_count[both]
            * other.count[both]
            / combined_count[both]
        )

        self.mean[active] += (
            delta[active] * other.count[active] / combined_count[active]
        )
        self.m2[active] += other.m2[active] + correction[active]
        self.count = combined_count
        self.minimum[active] = np.minimum(
            self.minimum[active], other.minimum[active]
        )
        self.maximum[active] = np.maximum(
            self.maximum[active], other.maximum[active]
        )
        return self

    def variance(self, *, ddof: int = 0) -> np.ndarray:
        """Return per-level variance, using population variance by default."""

        if ddof < 0:
            raise ValueError("ddof must be non-negative")
        denominator = self.count - ddof
        result = np.full(self.level_count, np.nan, dtype=np.float64)
        valid = denominator > 0
        result[valid] = self.m2[valid] / denominator[valid]
        # Small negative values can arise only from floating-point roundoff.
        result[valid] = np.maximum(result[valid], 0.0)
        return result

    def std(self, *, ddof: int = 0) -> np.ndarray:
        return np.sqrt(self.variance(ddof=ddof))

    def to_dict(self, *, heights_km: np.ndarray | None = None) -> dict[str, Any]:
        """Return strict JSON-compatible population statistics."""

        self._validate_state()
        observed = self.count > 0
        mean = np.where(observed, self.mean, np.nan)
        minimum = np.where(observed, self.minimum, np.nan)
        maximum = np.where(observed, self.maximum, np.nan)
        result: dict[str, Any] = {
            "level_count": self.level_count,
            "count": _json_values(self.count),
            "mean": _json_values(mean),
            "std": _json_values(self.std(ddof=0)),
            "minimum": _json_values(minimum),
            "maximum": _json_values(maximum),
            "total_count": int(self.count.sum()),
            "empty_level_count": int((self.count == 0).sum()),
        }
        if heights_km is not None:
            heights = np.asarray(heights_km, dtype=np.float64)
            if heights.shape != (self.level_count,):
                raise ValueError(
                    f"heights_km must have shape {(self.level_count,)}, got "
                    f"{heights.shape}"
                )
            result["heights_km"] = _json_values(heights)
        return result


@dataclass(frozen=True)
class PerLevelStandardizer:
    """Apply stored mean/std vectors along the final height axis."""

    mean: np.ndarray
    std: np.ndarray
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float64)
        std = np.asarray(self.std, dtype=np.float64)
        if mean.ndim != 1 or std.shape != mean.shape:
            raise ValueError("mean and std must be one-dimensional arrays of equal shape")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", std)

    @classmethod
    def from_dict(cls, statistics: Mapping[str, Any]) -> "PerLevelStandardizer":
        def array(key: str) -> np.ndarray:
            return np.asarray(
                [np.nan if value is None else value for value in statistics[key]],
                dtype=np.float64,
            )

        return cls(mean=array("mean"), std=array("std"))

    def transform(
        self,
        values: Any,
        *,
        valid_mask: np.ndarray | None = None,
        fill_value: float | None = None,
        dtype: str | np.dtype = np.float32,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Standardize valid cells and return ``(values, effective_mask)``.

        Levels without fitted statistics, constant levels, and input NaNs are
        removed from the effective mask. They remain NaN unless ``fill_value``
        is explicitly provided. This preserves missingness until the caller has
        a separate mask channel ready for a neural network.
        """

        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 0 or array.shape[-1] != self.mean.size:
            raise ValueError(
                f"values must end with {self.mean.size} levels, got {array.shape}"
            )
        effective = np.isfinite(array)
        if valid_mask is not None:
            supplied = np.asarray(valid_mask, dtype=bool)
            if supplied.shape != array.shape:
                raise ValueError("valid_mask and values must have identical shapes")
            effective &= supplied
        fitted_levels = (
            np.isfinite(self.mean) & np.isfinite(self.std) & (self.std > self.epsilon)
        )
        shape = (1,) * (array.ndim - 1) + (-1,)
        effective &= fitted_levels.reshape(shape)
        output = np.full(array.shape, np.nan, dtype=np.float64)
        # Use a neutral scale for levels that will be masked out, preventing
        # divide-by-zero warnings from polluting training logs.
        safe_mean = np.where(np.isfinite(self.mean), self.mean, 0.0)
        safe_std = np.where(fitted_levels, self.std, 1.0)
        normalized = (array - safe_mean.reshape(shape)) / safe_std.reshape(shape)
        output[effective] = normalized[effective]
        if fill_value is not None:
            output[~effective] = fill_value
        return output.astype(dtype, copy=False), effective

    def inverse(
        self, values: Any, *, dtype: str | np.dtype = np.float32
    ) -> np.ndarray:
        """Invert standardization while preserving NaN values."""

        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 0 or array.shape[-1] != self.mean.size:
            raise ValueError(
                f"values must end with {self.mean.size} levels, got {array.shape}"
            )
        shape = (1,) * (array.ndim - 1) + (-1,)
        return (array * self.std.reshape(shape) + self.mean.reshape(shape)).astype(
            dtype, copy=False
        )


def log1p_rain(
    rain_rate: Any, *, valid_mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Transform non-negative rain rate with ``log(1 + R)`` and preserve mask."""

    rain = np.asarray(rain_rate, dtype=np.float64)
    valid = np.isfinite(rain) & (rain >= 0.0)
    if valid_mask is not None:
        supplied = np.asarray(valid_mask, dtype=bool)
        if supplied.shape != rain.shape:
            raise ValueError("valid_mask and rain_rate must have identical shapes")
        valid &= supplied
    transformed = np.full(rain.shape, np.nan, dtype=np.float32)
    transformed[valid] = np.log1p(rain[valid]).astype(np.float32)
    return transformed, valid


def inverse_log1p_rain(
    transformed: Any, *, clip_min: float | None = 0.0
) -> np.ndarray:
    """Invert ``log1p_rain``; optionally clamp negative model predictions."""

    values = np.asarray(transformed, dtype=np.float64)
    rain = np.expm1(values)
    if clip_min is not None:
        rain = np.maximum(rain, clip_min)
    return rain.astype(np.float32, copy=False)


def clip_below_threshold(
    values: Any,
    *,
    threshold: float,
    replacement: float = 0.0,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Replace finite values below a configured threshold, preserving missingness."""

    if not np.isfinite(threshold) or not np.isfinite(replacement):
        raise ValueError("threshold and replacement must be finite")
    output = np.asarray(values, dtype=np.float32).copy()
    valid = np.isfinite(output)
    if valid_mask is not None:
        supplied = np.asarray(valid_mask, dtype=bool)
        if supplied.shape != output.shape:
            raise ValueError("valid_mask and values must have identical shapes")
        valid &= supplied
    output[valid & (output < threshold)] = replacement
    return output


def fill_missing_with_mask(
    values: Any, *, fill_value: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Fill NaN/Inf only after returning their original observation mask."""

    if not np.isfinite(fill_value):
        raise ValueError("fill_value must be finite")
    array = np.asarray(values, dtype=np.float32)
    observed = np.isfinite(array)
    filled = np.where(observed, array, fill_value).astype(np.float32, copy=False)
    return filled, observed
