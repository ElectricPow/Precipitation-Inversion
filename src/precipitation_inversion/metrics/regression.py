"""Streaming masked regression metrics in log and physical rain-rate space.

The accumulators consume tensors shaped ``(B,1,D,H,Z)`` but work with any
identically shaped numeric arrays. Only values selected by the boolean mask
contribute, so halo, clutter, missing cells, and horizontal padding are ignored.
Sufficient statistics are accumulated in float64 without retaining predictions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
import torch.distributed as distributed


def _as_tensor(value: Any, *, device: torch.device | None = None) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    return tensor if device is None else tensor.to(device)


def _broadcast_mask(mask: Any, reference: torch.Tensor) -> torch.Tensor:
    tensor = _as_tensor(mask, device=reference.device)
    if tensor.dtype != torch.bool:
        raise TypeError("metric mask must have boolean dtype")
    try:
        return torch.broadcast_to(tensor, reference.shape)
    except RuntimeError as error:
        raise ValueError(
            f"mask shape {tuple(tensor.shape)} cannot broadcast to "
            f"{tuple(reference.shape)}"
        ) from error


@dataclass
class RegressionAccumulator:
    """Sufficient statistics for masked MAE, RMSE, bias, R², and correlation."""

    count: int = 0
    sum_abs_error: float = 0.0
    sum_squared_error: float = 0.0
    sum_error: float = 0.0
    sum_prediction: float = 0.0
    sum_target: float = 0.0
    sum_prediction_squared: float = 0.0
    sum_target_squared: float = 0.0
    sum_product: float = 0.0

    def reset(self) -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, 0 if name == "count" else 0.0)

    def update(self, prediction: Any, target: Any, mask: Any) -> None:
        """Accumulate selected values without changing their original shape."""

        predicted = _as_tensor(prediction)
        observed = _as_tensor(target, device=predicted.device)
        if predicted.shape != observed.shape:
            raise ValueError(
                f"prediction/target shapes differ: {tuple(predicted.shape)} != "
                f"{tuple(observed.shape)}"
            )
        if not torch.is_floating_point(predicted):
            predicted = predicted.float()
        if not torch.is_floating_point(observed):
            observed = observed.float()
        selected = _broadcast_mask(mask, predicted)
        if not bool(selected.any()):
            return

        # Before=(B,1,D,H,Z); boolean selection -> flat (N_valid,). Float64
        # sums avoid accuracy loss across millions of precipitation voxels.
        predicted_values = predicted[selected].double()
        observed_values = observed[selected].double()
        if not bool(
            torch.isfinite(predicted_values).all()
            and torch.isfinite(observed_values).all()
        ):
            raise ValueError("selected metric values must be finite")
        error = predicted_values - observed_values
        self.count += int(predicted_values.numel())
        self.sum_abs_error += float(error.abs().sum().item())
        self.sum_squared_error += float(error.square().sum().item())
        self.sum_error += float(error.sum().item())
        self.sum_prediction += float(predicted_values.sum().item())
        self.sum_target += float(observed_values.sum().item())
        self.sum_prediction_squared += float(predicted_values.square().sum().item())
        self.sum_target_squared += float(observed_values.square().sum().item())
        self.sum_product += float((predicted_values * observed_values).sum().item())

    def merge(self, other: "RegressionAccumulator") -> None:
        """Merge statistics produced by another worker or data subset."""

        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def _values(self, *, synchronize: bool) -> list[float]:
        values = [
            float(self.count),
            self.sum_abs_error,
            self.sum_squared_error,
            self.sum_error,
            self.sum_prediction,
            self.sum_target,
            self.sum_prediction_squared,
            self.sum_target_squared,
            self.sum_product,
        ]
        active = (
            synchronize
            and distributed.is_available()
            and distributed.is_initialized()
        )
        if not active:
            return values
        backend = str(distributed.get_backend()).lower()
        device = (
            torch.device("cuda", torch.cuda.current_device())
            if "nccl" in backend
            else torch.device("cpu")
        )
        tensor = torch.tensor(values, dtype=torch.float64, device=device)
        distributed.all_reduce(tensor, op=distributed.ReduceOp.SUM)
        return tensor.cpu().tolist()

    def compute(self, *, synchronize: bool = False) -> dict[str, float | int]:
        """Compute metrics; optionally all-reduce temporary statistics for DDP."""

        values = self._values(synchronize=synchronize)
        count = int(round(values[0]))
        if count == 0:
            return {
                "count": 0,
                "mae": math.nan,
                "rmse": math.nan,
                "bias": math.nan,
                "r2": math.nan,
                "pearson_r": math.nan,
            }
        (
            _,
            sum_abs_error,
            sum_squared_error,
            sum_error,
            sum_prediction,
            sum_target,
            sum_prediction_squared,
            sum_target_squared,
            sum_product,
        ) = values
        target_centered = sum_target_squared - sum_target * sum_target / count
        prediction_centered = (
            sum_prediction_squared - sum_prediction * sum_prediction / count
        )
        covariance = sum_product - sum_prediction * sum_target / count
        denominator = math.sqrt(max(prediction_centered, 0.0) * max(target_centered, 0.0))
        pearson = covariance / denominator if denominator > 0.0 else math.nan
        r2 = 1.0 - sum_squared_error / target_centered if target_centered > 0.0 else math.nan
        return {
            "count": count,
            "mae": sum_abs_error / count,
            "rmse": math.sqrt(sum_squared_error / count),
            "bias": sum_error / count,
            "r2": r2,
            "pearson_r": pearson,
        }


def _threshold_label(value: float) -> str:
    return f"{value:g}".replace(".", "p")


class PrecipitationRegressionMetrics:
    """Track log-space, physical-space, and target-intensity-bin metrics.

    Physical predictions are obtained as ``expm1(clamp(prediction_log, min=0))``.
    Target bins are half-open except the final bin, for example thresholds
    ``(1,5,10,30)`` produce ``lt_1``, ``1_to_5``, ..., and ``ge_30``.
    """

    def __init__(self, thresholds_mm_h: Sequence[float] = (1, 5, 10, 30)) -> None:
        thresholds = tuple(float(value) for value in thresholds_mm_h)
        if any(not math.isfinite(value) or value <= 0 for value in thresholds):
            raise ValueError("rain thresholds must be finite and positive")
        if any(right <= left for left, right in zip(thresholds, thresholds[1:])):
            raise ValueError("rain thresholds must be strictly increasing")
        self.thresholds = thresholds
        self.log = RegressionAccumulator()
        self.rain = RegressionAccumulator()
        labels = [f"lt_{_threshold_label(thresholds[0])}"] if thresholds else ["all"]
        labels.extend(
            f"{_threshold_label(left)}_to_{_threshold_label(right)}"
            for left, right in zip(thresholds, thresholds[1:])
        )
        if thresholds:
            labels.append(f"ge_{_threshold_label(thresholds[-1])}")
        self.bin_labels = tuple(labels)
        self.bins = {label: RegressionAccumulator() for label in self.bin_labels}

    def reset(self) -> None:
        self.log.reset()
        self.rain.reset()
        for accumulator in self.bins.values():
            accumulator.reset()

    def _update_bins(
        self,
        prediction_rain: torch.Tensor,
        target_rain: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        if not self.thresholds:
            self.bins["all"].update(prediction_rain, target_rain, mask)
            return
        lower: float | None = None
        for label, upper in zip(self.bin_labels[:-1], self.thresholds):
            bin_mask = mask & (target_rain < upper)
            if lower is not None:
                bin_mask &= target_rain >= lower
            self.bins[label].update(prediction_rain, target_rain, bin_mask)
            lower = upper
        self.bins[self.bin_labels[-1]].update(
            prediction_rain,
            target_rain,
            mask & (target_rain >= self.thresholds[-1]),
        )

    def update_log(
        self, prediction_log: Any, target_log: Any, mask: Any
    ) -> None:
        """Update from model/target tensors shaped ``(B,1,D,H,Z)``."""

        prediction = _as_tensor(prediction_log)
        target = _as_tensor(target_log, device=prediction.device)
        if prediction.shape != target.shape:
            raise ValueError("prediction_log and target_log shapes differ")
        selected = _broadcast_mask(mask, prediction)
        self.log.update(prediction, target, selected)
        # Shape remains (B,1,D,H,Z); values change from log1p units to mm/h.
        prediction_rain = torch.expm1(prediction.float().clamp_min(0.0))
        target_rain = torch.expm1(target.float().clamp_min(0.0))
        self.rain.update(prediction_rain, target_rain, selected)
        self._update_bins(prediction_rain, target_rain, selected)

    def update_rain(self, prediction_rain: Any, target_rain: Any, mask: Any) -> None:
        """Update from already inverted non-negative rain rates in mm/h."""

        prediction = _as_tensor(prediction_rain).float().clamp_min(0.0)
        target = _as_tensor(target_rain, device=prediction.device).float()
        if prediction.shape != target.shape:
            raise ValueError("prediction_rain and target_rain shapes differ")
        selected = _broadcast_mask(mask, prediction)
        self.rain.update(prediction, target, selected)
        self.log.update(torch.log1p(prediction), torch.log1p(target), selected)
        self._update_bins(prediction, target, selected)

    def compute(self, *, synchronize: bool = False) -> dict[str, Any]:
        return {
            "log": self.log.compute(synchronize=synchronize),
            "rain": {
                "all": self.rain.compute(synchronize=synchronize),
                "target_bins_mm_h": {
                    label: accumulator.compute(synchronize=synchronize)
                    for label, accumulator in self.bins.items()
                },
            },
        }
