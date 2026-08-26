"""Streaming masked regression metrics in log and physical rain-rate space.

The accumulators consume tensors shaped ``(B,1,D,H,Z)`` but work with any
identically shaped numeric arrays. Only values selected by the boolean mask
contribute, so halo, clutter, missing cells, and horizontal padding are ignored.
Sufficient statistics are accumulated in float64 without retaining predictions.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

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


def _broadcast_auxiliary(
    value: Any,
    reference: torch.Tensor,
    *,
    name: str,
) -> torch.Tensor:
    """Broadcast voxel-, profile-, or height-level metadata to ``reference``.

    The model tensors normally have shape ``(B,1,D,H,Z)``.  Evaluation
    metadata is intentionally allowed to remain compact:

    * heights may be ``(Z,)`` or ``(B,Z)``;
    * CFB distance may already be voxel-wise, or be broadcastable;
    * ``typePrecip`` may be profile-wise ``(B,D,H)``.

    Expanding these views does not allocate a second dense five-dimensional
    array.  This keeps stratified evaluation practical for complete orbits.
    """

    tensor = _as_tensor(value, device=reference.device)
    candidates = [tensor]

    # One shared physical height vector: (Z,) -> (1,...,1,Z).
    if tensor.ndim == 1 and tensor.shape[0] == reference.shape[-1]:
        candidates.append(
            tensor.reshape((1,) * (reference.ndim - 1) + (tensor.shape[0],))
        )

    # One height vector per batch item: (B,Z) -> (B,1,...,1,Z).
    if (
        reference.ndim >= 2
        and tensor.ndim == 2
        and tensor.shape == (reference.shape[0], reference.shape[-1])
    ):
        candidates.append(
            tensor.reshape(
                (tensor.shape[0],)
                + (1,) * (reference.ndim - 2)
                + (tensor.shape[1],)
            )
        )

    # Profile metadata for an orbit: (...,D,H) -> (...,D,H,1).
    if tensor.shape == reference.shape[:-1]:
        candidates.append(tensor.unsqueeze(-1))

    # The channel dimension is absent from common dataset metadata:
    # (B,D,H) -> (B,1,D,H,1), or (B,D,H,Z) -> (B,1,D,H,Z).
    if reference.ndim >= 3 and reference.shape[1] == 1:
        without_channel = reference.shape[:1] + reference.shape[2:]
        without_channel_and_height = reference.shape[:1] + reference.shape[2:-1]
        if tensor.shape == without_channel:
            candidates.append(tensor.unsqueeze(1))
        if tensor.shape == without_channel_and_height:
            candidates.append(tensor.unsqueeze(1).unsqueeze(-1))
        if tensor.shape == without_channel_and_height + (1,):
            candidates.append(tensor.unsqueeze(1))

    for candidate in candidates:
        try:
            return torch.broadcast_to(candidate, reference.shape)
        except RuntimeError:
            continue
    raise ValueError(
        f"{name} shape {tuple(tensor.shape)} cannot broadcast to metric shape "
        f"{tuple(reference.shape)}"
    )


def _metrics_from_statistics(values: Sequence[float]) -> dict[str, float | int]:
    """Convert the nine additive regression statistics into public metrics."""

    count = int(round(float(values[0])))
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
    ) = (float(value) for value in values)
    target_centered = sum_target_squared - sum_target * sum_target / count
    prediction_centered = (
        sum_prediction_squared - sum_prediction * sum_prediction / count
    )
    covariance = sum_product - sum_prediction * sum_target / count
    denominator = math.sqrt(
        max(prediction_centered, 0.0) * max(target_centered, 0.0)
    )
    pearson = covariance / denominator if denominator > 0.0 else math.nan
    r2 = (
        1.0 - sum_squared_error / target_centered
        if target_centered > 0.0
        else math.nan
    )
    return {
        "count": count,
        "mae": sum_abs_error / count,
        "rmse": math.sqrt(sum_squared_error / count),
        "bias": sum_error / count,
        "r2": r2,
        "pearson_r": pearson,
    }


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

        return _metrics_from_statistics(self._values(synchronize=synchronize))


@dataclass
class _GradientSummaryAccumulator:
    """Regression statistics plus diagnostics specific to signed gradients.

    The standard regression accumulator measures agreement between predicted
    and target ``dR/dz``.  The additional absolute-gradient sums expose a
    common failure mode that correlation alone misses: a vertically smoothed
    prediction can retain the correct ordering while its gradient amplitude is
    much too small.  Sign agreement is evaluated only where the target
    gradient magnitude exceeds ``sign_epsilon`` so numerical near-zero slopes
    do not dominate the percentage.
    """

    sign_epsilon: float = 0.1
    sum_abs_prediction: float = 0.0
    sum_abs_target: float = 0.0
    sign_agreement_count: int = 0
    sign_evaluated_count: int = 0

    def __post_init__(self) -> None:
        if not math.isfinite(self.sign_epsilon) or self.sign_epsilon < 0.0:
            raise ValueError("sign_epsilon must be finite and non-negative")
        self.regression = RegressionAccumulator()

    def reset(self) -> None:
        self.regression.reset()
        self.sum_abs_prediction = 0.0
        self.sum_abs_target = 0.0
        self.sign_agreement_count = 0
        self.sign_evaluated_count = 0

    def update(self, prediction: Any, target: Any, mask: Any) -> None:
        predicted = _as_tensor(prediction)
        observed = _as_tensor(target, device=predicted.device)
        if predicted.shape != observed.shape:
            raise ValueError("prediction/target gradient shapes differ")
        selected = _broadcast_mask(mask, predicted)
        self.regression.update(predicted, observed, selected)
        if not bool(selected.any()):
            return
        predicted_values = predicted[selected].double()
        observed_values = observed[selected].double()
        # RegressionAccumulator already validates finiteness. Keep these sums
        # in float64 as millions of adjacent-height pairs may be accumulated.
        self.sum_abs_prediction += float(predicted_values.abs().sum().item())
        self.sum_abs_target += float(observed_values.abs().sum().item())
        directional = observed_values.abs() >= self.sign_epsilon
        self.sign_evaluated_count += int(directional.sum().item())
        if bool(directional.any()):
            agreement = torch.sign(predicted_values[directional]) == torch.sign(
                observed_values[directional]
            )
            self.sign_agreement_count += int(agreement.sum().item())

    def merge(self, other: "_GradientSummaryAccumulator") -> None:
        if self.sign_epsilon != other.sign_epsilon:
            raise ValueError("cannot merge gradient summaries with different epsilon")
        self.regression.merge(other.regression)
        self.sum_abs_prediction += other.sum_abs_prediction
        self.sum_abs_target += other.sum_abs_target
        self.sign_agreement_count += other.sign_agreement_count
        self.sign_evaluated_count += other.sign_evaluated_count

    def _extra_values(self, *, synchronize: bool) -> list[float]:
        values = [
            self.sum_abs_prediction,
            self.sum_abs_target,
            float(self.sign_agreement_count),
            float(self.sign_evaluated_count),
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
        result = self.regression.compute(synchronize=synchronize)
        (
            sum_abs_prediction,
            sum_abs_target,
            sign_agreement_count,
            sign_evaluated_count,
        ) = self._extra_values(synchronize=synchronize)
        count = int(result["count"])
        return {
            **result,
            "mean_abs_prediction": (
                sum_abs_prediction / count if count else math.nan
            ),
            "mean_abs_target": sum_abs_target / count if count else math.nan,
            "mean_abs_gradient_ratio": (
                sum_abs_prediction / sum_abs_target
                if sum_abs_target > 0.0
                else math.nan
            ),
            "sign_epsilon": self.sign_epsilon,
            "sign_evaluated_count": int(round(sign_evaluated_count)),
            "sign_agreement_fraction": (
                sign_agreement_count / sign_evaluated_count
                if sign_evaluated_count > 0.0
                else math.nan
            ),
        }


class GroupedRegressionAccumulator:
    """Mergeable regression statistics for a fixed collection of groups.

    ``group_index`` contains integers in ``[0, group_count)`` and may have any
    shape broadcastable to the prediction.  ``-1`` means that the selected
    voxel has no valid group metadata and is omitted from this particular
    stratification.  All groups are synchronized in one DDP ``all_reduce``.
    """

    _STATISTIC_COUNT = 9

    def __init__(self, labels: Sequence[str]) -> None:
        normalized = tuple(str(label) for label in labels)
        if not normalized or any(not label for label in normalized):
            raise ValueError("group labels must be non-empty")
        if len(set(normalized)) != len(normalized):
            raise ValueError("group labels must be unique")
        self.labels = normalized
        # Rows are groups. Columns follow RegressionAccumulator's additive
        # statistics: N, |e|, e², e, pred, target, pred², target², pred*target.
        self.statistics = np.zeros(
            (len(self.labels), self._STATISTIC_COUNT), dtype=np.float64
        )

    def reset(self) -> None:
        self.statistics.fill(0.0)

    def update(
        self,
        prediction: Any,
        target: Any,
        mask: Any,
        group_index: Any,
    ) -> None:
        predicted = _as_tensor(prediction)
        observed = _as_tensor(target, device=predicted.device)
        if predicted.shape != observed.shape:
            raise ValueError(
                f"prediction/target shapes differ: {tuple(predicted.shape)} != "
                f"{tuple(observed.shape)}"
            )
        predicted = (
            predicted.float() if not torch.is_floating_point(predicted) else predicted
        )
        observed = (
            observed.float() if not torch.is_floating_point(observed) else observed
        )
        selected = _broadcast_mask(mask, predicted)
        groups = _broadcast_auxiliary(
            group_index, predicted, name="group_index"
        ).to(dtype=torch.int64)
        selected &= (groups >= 0) & (groups < len(self.labels))
        if not bool(selected.any()):
            return

        # Before=(B,1,D,H,Z); boolean selection -> three flat vectors (N,).
        predicted_values = predicted[selected].double()
        observed_values = observed[selected].double()
        selected_groups = groups[selected]
        if not bool(
            torch.isfinite(predicted_values).all()
            and torch.isfinite(observed_values).all()
        ):
            raise ValueError("selected metric values must be finite")
        error = predicted_values - observed_values
        values = (
            torch.ones_like(error),
            error.abs(),
            error.square(),
            error,
            predicted_values,
            observed_values,
            predicted_values.square(),
            observed_values.square(),
            predicted_values * observed_values,
        )
        batch_statistics = torch.zeros(
            (len(self.labels), self._STATISTIC_COUNT),
            dtype=torch.float64,
            device=predicted.device,
        )
        for column, value in enumerate(values):
            batch_statistics[:, column].scatter_add_(0, selected_groups, value)
        self.statistics += batch_statistics.cpu().numpy()

    def merge(self, other: "GroupedRegressionAccumulator") -> None:
        if self.labels != other.labels:
            raise ValueError("cannot merge grouped accumulators with different labels")
        self.statistics += other.statistics

    def _values(self, *, synchronize: bool) -> np.ndarray:
        values = self.statistics.copy()
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
        tensor = torch.as_tensor(values, dtype=torch.float64, device=device)
        distributed.all_reduce(tensor, op=distributed.ReduceOp.SUM)
        return tensor.cpu().numpy()

    def compute(
        self, *, synchronize: bool = False
    ) -> dict[str, dict[str, float | int]]:
        values = self._values(synchronize=synchronize)
        return {
            label: _metrics_from_statistics(values[index])
            for index, label in enumerate(self.labels)
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


class FilewisePrecipitationMetrics:
    """Per-file metrics, orbit-macro averages, and reproducible bootstrap CIs.

    File/orbit labels are fixed at construction, while ``file_id`` supplied to
    :meth:`update_log` or :meth:`update_rain` identifies the source file for
    each batch item.  Sufficient statistics are accumulated per file without
    retaining voxel predictions. In DDP, all file rows are synchronized in one
    collective before macro statistics are computed.

    The reported macro average gives every non-empty orbit equal weight. This
    deliberately complements the existing micro metric, where every valid
    voxel has equal weight and long or precipitation-rich orbits dominate.
    Bootstrap intervals resample whole files with replacement, preserving the
    spatial dependence among voxels from the same orbit.
    """

    _METRIC_NAMES = ("mae", "rmse", "bias", "r2", "pearson_r")

    def __init__(
        self,
        file_labels: Sequence[str],
        *,
        bootstrap_seed: int = 2026,
        bootstrap_replicates: int = 2000,
        confidence_level: float = 0.95,
    ) -> None:
        if bootstrap_seed < 0:
            raise ValueError("bootstrap_seed must be non-negative")
        if bootstrap_replicates <= 0:
            raise ValueError("bootstrap_replicates must be positive")
        if not math.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
            raise ValueError("confidence_level must lie strictly between 0 and 1")
        self.file_labels = tuple(str(value) for value in file_labels)
        self.accumulator = GroupedRegressionAccumulator(self.file_labels)
        self.bootstrap_seed = int(bootstrap_seed)
        self.bootstrap_replicates = int(bootstrap_replicates)
        self.confidence_level = float(confidence_level)

    def reset(self) -> None:
        self.accumulator.reset()

    def _file_groups(self, file_id: Any, reference: torch.Tensor) -> torch.Tensor:
        values = _as_tensor(file_id, device=reference.device)
        if values.ndim == 1 and values.shape[0] == reference.shape[0]:
            values = values.reshape(
                (values.shape[0],) + (1,) * (reference.ndim - 1)
            )
        else:
            values = _broadcast_auxiliary(values, reference, name="file_id")
        if torch.is_floating_point(values):
            if bool((~torch.isfinite(values) | (values != torch.floor(values))).any()):
                raise ValueError("file_id values must be finite integers")
        groups = values.to(dtype=torch.int64)
        if bool(((groups < 0) | (groups >= len(self.file_labels))).any()):
            raise ValueError("file_id lies outside the configured file labels")
        return torch.broadcast_to(groups, reference.shape)

    def update_log(
        self,
        prediction_log: Any,
        target_log: Any,
        mask: Any,
        *,
        file_id: Any,
    ) -> None:
        prediction = _as_tensor(prediction_log)
        target = _as_tensor(target_log, device=prediction.device)
        if prediction.shape != target.shape:
            raise ValueError("prediction_log and target_log shapes differ")
        prediction_rain = torch.expm1(prediction.float().clamp_min(0.0))
        target_rain = torch.expm1(target.float().clamp_min(0.0))
        self.accumulator.update(
            prediction_rain,
            target_rain,
            mask,
            self._file_groups(file_id, prediction_rain),
        )

    def update_rain(
        self,
        prediction_rain: Any,
        target_rain: Any,
        mask: Any,
        *,
        file_id: Any,
    ) -> None:
        prediction = _as_tensor(prediction_rain).float().clamp_min(0.0)
        target = _as_tensor(target_rain, device=prediction.device).float()
        if prediction.shape != target.shape:
            raise ValueError("prediction_rain and target_rain shapes differ")
        self.accumulator.update(
            prediction,
            target,
            mask,
            self._file_groups(file_id, prediction),
        )

    def update_values(
        self,
        prediction: Any,
        target: Any,
        mask: Any,
        *,
        file_id: Any,
    ) -> None:
        """Update arbitrary signed physical values without rain-rate clamping.

        This is used by the physical ``dR/dz`` diagnostic, whose values may be
        positive or negative.  The filewise aggregation and whole-orbit
        bootstrap are otherwise identical to those used for rain rate.
        """

        predicted = _as_tensor(prediction)
        observed = _as_tensor(target, device=predicted.device)
        if predicted.shape != observed.shape:
            raise ValueError("prediction and target shapes differ")
        self.accumulator.update(
            predicted,
            observed,
            mask,
            self._file_groups(file_id, predicted),
        )

    def _macro_and_bootstrap(
        self, per_file: Mapping[str, Mapping[str, float | int]]
    ) -> tuple[dict[str, float], dict[str, Any]]:
        nonempty_labels = [
            label for label in self.file_labels if int(per_file[label]["count"]) > 0
        ]
        values = np.asarray(
            [
                [float(per_file[label][name]) for name in self._METRIC_NAMES]
                for label in nonempty_labels
            ],
            dtype=np.float64,
        ).reshape((-1, len(self._METRIC_NAMES)))
        macro: dict[str, float] = {}
        for column, name in enumerate(self._METRIC_NAMES):
            finite = values[:, column][np.isfinite(values[:, column])]
            macro[name] = float(finite.mean()) if finite.size else math.nan

        intervals: dict[str, dict[str, float | int]] = {}
        if not nonempty_labels:
            for name in self._METRIC_NAMES:
                intervals[name] = {
                    "low": math.nan,
                    "high": math.nan,
                    "valid_replicates": 0,
                }
        else:
            rng = np.random.default_rng(self.bootstrap_seed)
            sampled_indices = rng.integers(
                0,
                len(nonempty_labels),
                size=(self.bootstrap_replicates, len(nonempty_labels)),
            )
            alpha = (1.0 - self.confidence_level) / 2.0
            for column, name in enumerate(self._METRIC_NAMES):
                sampled = values[sampled_indices, column]
                finite = np.isfinite(sampled)
                denominator = finite.sum(axis=1)
                replicate_values = np.divide(
                    np.where(finite, sampled, 0.0).sum(axis=1),
                    denominator,
                    out=np.full(self.bootstrap_replicates, np.nan),
                    where=denominator > 0,
                )
                valid = replicate_values[np.isfinite(replicate_values)]
                if valid.size:
                    low, high = np.quantile(valid, (alpha, 1.0 - alpha))
                else:
                    low = high = math.nan
                intervals[name] = {
                    "low": float(low),
                    "high": float(high),
                    "valid_replicates": int(valid.size),
                }
        return macro, {
            "method": "percentile bootstrap resampling whole files with replacement",
            "seed": self.bootstrap_seed,
            "replicates": self.bootstrap_replicates,
            "confidence_level": self.confidence_level,
            "confidence_interval": intervals,
        }

    def compute(self, *, synchronize: bool = False) -> dict[str, Any]:
        per_file = self.accumulator.compute(synchronize=synchronize)
        macro, bootstrap = self._macro_and_bootstrap(per_file)
        nonempty_count = sum(
            int(values["count"]) > 0 for values in per_file.values()
        )
        return {
            "definition": (
                "unweighted arithmetic mean of each non-empty file/orbit metric; "
                "bootstrap sampling unit is one complete file/orbit"
            ),
            "file_count_total": len(self.file_labels),
            "file_count_nonempty": nonempty_count,
            "valid_voxel_count": sum(
                int(values["count"]) for values in per_file.values()
            ),
            "macro_average": macro,
            "bootstrap": bootstrap,
            "per_file": per_file,
        }


def _number_label(value: float) -> str:
    """Return a compact JSON-key-safe representation of a finite number."""

    prefix = "m" if value < 0 else ""
    magnitude = f"{abs(value):g}".replace(".", "p")
    return prefix + magnitude


def _strictly_increasing_values(
    values: Sequence[float], *, name: str, minimum_length: int
) -> tuple[float, ...]:
    normalized = tuple(float(value) for value in values)
    if len(normalized) < minimum_length:
        raise ValueError(f"{name} must contain at least {minimum_length} values")
    if any(not math.isfinite(value) for value in normalized):
        raise ValueError(f"{name} must contain only finite values")
    if any(right <= left for left, right in zip(normalized, normalized[1:])):
        raise ValueError(f"{name} must be strictly increasing")
    return normalized


def _unbounded_interval_labels(
    thresholds: Sequence[float], *, suffix: str
) -> tuple[str, ...]:
    values = tuple(thresholds)
    labels = [f"lt_{_number_label(values[0])}{suffix}"]
    labels.extend(
        f"{_number_label(left)}_to_{_number_label(right)}{suffix}"
        for left, right in zip(values, values[1:])
    )
    labels.append(f"ge_{_number_label(values[-1])}{suffix}")
    return tuple(labels)


def _bounded_interval_labels(
    edges: Sequence[float], *, suffix: str
) -> tuple[str, ...]:
    return tuple(
        f"{_number_label(left)}_to_{_number_label(right)}{suffix}"
        for left, right in zip(edges, edges[1:])
    )


def _unbounded_group_index(
    values: torch.Tensor, thresholds: Sequence[float]
) -> torch.Tensor:
    boundaries = torch.tensor(
        thresholds, dtype=values.dtype, device=values.device
    )
    groups = torch.bucketize(values.contiguous(), boundaries, right=True)
    return torch.where(torch.isfinite(values), groups, torch.full_like(groups, -1))


class StratifiedPrecipitationMetrics:
    """Streaming physical-space metrics stratified by diagnostic metadata.

    The class reports four complementary views of the same evaluation mask:

    * each exact physical height, or configured absolute-height bands;
    * signed distance to CFB (``z - z_cfb``; negative means below CFB);
    * absolute height crossed with target rain-rate intensity;
    * DPR ``typePrecip`` profile class.

    Predictions and targets are never retained.  Each view uses a
    :class:`GroupedRegressionAccumulator`, so separate batches, processes, and
    complete orbit subsets can be merged from additive float64 statistics.
    Empty groups remain in the output with ``count=0`` and NaN-valued metrics.

    Parameters
    ----------
    height_levels_km:
        Exact ascending dataset heights. Passing all 60 values produces one
        group per native level. Mutually exclusive with ``height_bin_edges_km``.
    height_bin_edges_km:
        Optional ascending edges for wider absolute-height bands. Values below
        the first or at/above the last edge are excluded from height views.
    cfb_distance_edges_km:
        Thresholds for unbounded signed-distance groups. The default creates
        ``<-1``, ``[-1,0)``, ``[0,0.5)``, ``[0.5,2)``, and ``>=2 km`` groups.
    intensity_thresholds_mm_h:
        Target rain-rate thresholds for the height-by-intensity table.
    """

    _PRECIPITATION_TYPE_LABELS = (
        "stratiform",
        "convective",
        "other",
        "no_precipitation",
        "unclassified",
    )

    def __init__(
        self,
        height_levels_km: Sequence[float] | None = None,
        *,
        height_bin_edges_km: Sequence[float] | None = None,
        cfb_distance_edges_km: Sequence[float] = (-1.0, 0.0, 0.5, 2.0),
        intensity_thresholds_mm_h: Sequence[float] = (1.0, 5.0, 10.0, 30.0),
    ) -> None:
        if (height_levels_km is None) == (height_bin_edges_km is None):
            raise ValueError(
                "provide exactly one of height_levels_km or height_bin_edges_km"
            )
        if height_levels_km is not None:
            self.height_mode = "exact_levels"
            self.height_values = _strictly_increasing_values(
                height_levels_km, name="height_levels_km", minimum_length=1
            )
            self.height_labels = tuple(
                f"z_{_number_label(value)}_km" for value in self.height_values
            )
        else:
            self.height_mode = "bins"
            self.height_values = _strictly_increasing_values(
                height_bin_edges_km if height_bin_edges_km is not None else (),
                name="height_bin_edges_km",
                minimum_length=2,
            )
            self.height_labels = _bounded_interval_labels(
                self.height_values, suffix="_km"
            )

        self.cfb_distance_edges_km = _strictly_increasing_values(
            cfb_distance_edges_km,
            name="cfb_distance_edges_km",
            minimum_length=1,
        )
        self.cfb_distance_labels = _unbounded_interval_labels(
            self.cfb_distance_edges_km, suffix="_km"
        )
        self.intensity_thresholds_mm_h = _strictly_increasing_values(
            intensity_thresholds_mm_h,
            name="intensity_thresholds_mm_h",
            minimum_length=1,
        )
        if self.intensity_thresholds_mm_h[0] <= 0.0:
            raise ValueError("intensity_thresholds_mm_h must be positive")
        self.intensity_labels = _unbounded_interval_labels(
            self.intensity_thresholds_mm_h, suffix="_mm_h"
        )

        self.by_height = GroupedRegressionAccumulator(self.height_labels)
        self.by_cfb_distance = GroupedRegressionAccumulator(
            self.cfb_distance_labels
        )
        crossed_labels = tuple(
            f"{height_label}__{intensity_label}"
            for height_label in self.height_labels
            for intensity_label in self.intensity_labels
        )
        self.by_height_and_intensity = GroupedRegressionAccumulator(crossed_labels)
        self.by_precipitation_type = GroupedRegressionAccumulator(
            self._PRECIPITATION_TYPE_LABELS
        )

    def reset(self) -> None:
        self.by_height.reset()
        self.by_cfb_distance.reset()
        self.by_height_and_intensity.reset()
        self.by_precipitation_type.reset()

    def _height_group_index(self, height_km: torch.Tensor) -> torch.Tensor:
        values = height_km.float()
        reference = torch.tensor(
            self.height_values, dtype=values.dtype, device=values.device
        )
        if self.height_mode == "bins":
            # Interior boundaries produce n_edges-1 groups. The left endpoint
            # is inclusive, the right endpoint exclusive.
            groups = torch.bucketize(
                values.contiguous(), reference[1:-1], right=True
            )
            valid = (
                torch.isfinite(values)
                & (values >= reference[0])
                & (values < reference[-1])
            )
            return torch.where(valid, groups, torch.full_like(groups, -1))

        # Match exact native levels without constructing (...,Z,level_count).
        insertion = torch.searchsorted(reference, values.contiguous())
        right_index = insertion.clamp(max=reference.numel() - 1)
        left_index = (insertion - 1).clamp(min=0)
        right_distance = (values - reference[right_index]).abs()
        left_distance = (values - reference[left_index]).abs()
        choose_right = right_distance < left_distance
        groups = torch.where(choose_right, right_index, left_index)
        closest_distance = torch.where(choose_right, right_distance, left_distance)
        if reference.numel() > 1:
            tolerance = max(
                1e-5,
                float(torch.diff(reference).min().item()) * 1e-3,
            )
        else:
            tolerance = 1e-5
        valid = torch.isfinite(values) & (closest_distance <= tolerance)
        return torch.where(valid, groups, torch.full_like(groups, -1))

    @staticmethod
    def _precipitation_type_group_index(values: torch.Tensor) -> torch.Tensor:
        # typePrecip is a profile class: 1=stratiform, 2=convective, 3=other,
        # -1111=no precipitation. Any other finite code is kept auditable in
        # an explicit unclassified group; NaN/fill metadata is excluded.
        groups = torch.full(values.shape, -1, dtype=torch.int64, device=values.device)
        groups[values == 1] = 0
        groups[values == 2] = 1
        groups[values == 3] = 2
        groups[values == -1111] = 3
        groups[torch.isfinite(values) & (groups < 0)] = 4
        return groups

    def _update_rain_tensors(
        self,
        prediction_rain: torch.Tensor,
        target_rain: torch.Tensor,
        mask: Any,
        *,
        height_km: Any,
        cfb_distance_km: Any | None,
        precipitation_type: Any | None,
    ) -> None:
        if prediction_rain.shape != target_rain.shape:
            raise ValueError("prediction_rain and target_rain shapes differ")
        selected = _broadcast_mask(mask, prediction_rain)
        if bool(selected.any()) and not bool(
            torch.isfinite(prediction_rain[selected]).all()
            and torch.isfinite(target_rain[selected]).all()
        ):
            raise ValueError("selected metric values must be finite")

        expanded_height = _broadcast_auxiliary(
            height_km, prediction_rain, name="height_km"
        ).float()
        height_groups = self._height_group_index(expanded_height)
        self.by_height.update(
            prediction_rain, target_rain, selected, height_groups
        )

        intensity_groups = _unbounded_group_index(
            target_rain, self.intensity_thresholds_mm_h
        )
        crossed_groups = torch.where(
            height_groups >= 0,
            height_groups * len(self.intensity_labels) + intensity_groups,
            torch.full_like(height_groups, -1),
        )
        crossed_groups[intensity_groups < 0] = -1
        self.by_height_and_intensity.update(
            prediction_rain, target_rain, selected, crossed_groups
        )

        if cfb_distance_km is not None:
            expanded_distance = _broadcast_auxiliary(
                cfb_distance_km, prediction_rain, name="cfb_distance_km"
            ).float()
            distance_groups = _unbounded_group_index(
                expanded_distance, self.cfb_distance_edges_km
            )
            self.by_cfb_distance.update(
                prediction_rain, target_rain, selected, distance_groups
            )

        if precipitation_type is not None:
            expanded_type = _broadcast_auxiliary(
                precipitation_type, prediction_rain, name="precipitation_type"
            ).float()
            type_groups = self._precipitation_type_group_index(expanded_type)
            self.by_precipitation_type.update(
                prediction_rain, target_rain, selected, type_groups
            )

    def update_log(
        self,
        prediction_log: Any,
        target_log: Any,
        mask: Any,
        *,
        height_km: Any,
        cfb_distance_km: Any | None = None,
        precipitation_type: Any | None = None,
    ) -> None:
        """Update from ``log1p`` tensors, normally shaped ``(B,1,D,H,Z)``."""

        prediction = _as_tensor(prediction_log)
        target = _as_tensor(target_log, device=prediction.device)
        if prediction.shape != target.shape:
            raise ValueError("prediction_log and target_log shapes differ")
        # Shape is unchanged; only values are inverted from log1p to mm/h.
        prediction_rain = torch.expm1(prediction.float().clamp_min(0.0))
        target_rain = torch.expm1(target.float().clamp_min(0.0))
        self._update_rain_tensors(
            prediction_rain,
            target_rain,
            mask,
            height_km=height_km,
            cfb_distance_km=cfb_distance_km,
            precipitation_type=precipitation_type,
        )

    def update_rain(
        self,
        prediction_rain: Any,
        target_rain: Any,
        mask: Any,
        *,
        height_km: Any,
        cfb_distance_km: Any | None = None,
        precipitation_type: Any | None = None,
    ) -> None:
        """Update from physical rain-rate tensors in mm/h."""

        prediction = _as_tensor(prediction_rain).float().clamp_min(0.0)
        target = _as_tensor(target_rain, device=prediction.device).float()
        self._update_rain_tensors(
            prediction,
            target,
            mask,
            height_km=height_km,
            cfb_distance_km=cfb_distance_km,
            precipitation_type=precipitation_type,
        )

    def merge(self, other: "StratifiedPrecipitationMetrics") -> None:
        """Merge another batch/orbit/process accumulator with identical bins."""

        if (
            self.height_mode != other.height_mode
            or self.height_values != other.height_values
            or self.cfb_distance_edges_km != other.cfb_distance_edges_km
            or self.intensity_thresholds_mm_h != other.intensity_thresholds_mm_h
        ):
            raise ValueError("cannot merge stratified metrics with different bins")
        self.by_height.merge(other.by_height)
        self.by_cfb_distance.merge(other.by_cfb_distance)
        self.by_height_and_intensity.merge(other.by_height_and_intensity)
        self.by_precipitation_type.merge(other.by_precipitation_type)

    def compute(self, *, synchronize: bool = False) -> dict[str, Any]:
        height = self.by_height.compute(synchronize=synchronize)
        crossed_flat = self.by_height_and_intensity.compute(
            synchronize=synchronize
        )
        crossed = {
            height_label: {
                intensity_label: crossed_flat[
                    f"{height_label}__{intensity_label}"
                ]
                for intensity_label in self.intensity_labels
            }
            for height_label in self.height_labels
        }
        return {
            "definitions": {
                "height_mode": self.height_mode,
                "cfb_distance_km": (
                    "height_km - CFB boundary height; negative values are below CFB"
                ),
                "precipitation_type_codes": {
                    "stratiform": 1,
                    "convective": 2,
                    "other": 3,
                    "no_precipitation": -1111,
                },
            },
            "by_height_km": height,
            "by_cfb_distance_km": self.by_cfb_distance.compute(
                synchronize=synchronize
            ),
            "by_height_and_intensity_mm_h": crossed,
            "by_precipitation_type": self.by_precipitation_type.compute(
                synchronize=synchronize
            ),
        }


def physical_vertical_rain_gradient(
    prediction_rain: Any,
    target_rain: Any,
    mask: Any,
    *,
    height_km: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute adjacent-level physical ``dR/dz`` on strictly valid pairs.

    Parameters follow the model convention ``(B,1,D,H,Z)`` but any identical
    tensor shapes with height on the final axis are accepted. ``height_km`` may
    be a compact ``(Z,)`` vector or any broadcastable height tensor.

    Returns
    -------
    prediction_drdz, target_drdz, pair_mask, midpoint_height_km
        All returned tensors have the input shape with the final height axis
        shortened from ``Z`` to ``Z-1``. The gradient uses the standard upward
        sign convention ``(R[k+1]-R[k])/(z[k+1]-z[k])`` and therefore has units
        ``mm h^-1 km^-1``. A pair is selected only when *both* endpoint voxels
        are selected by the incoming reliable mask.
    """

    prediction = _as_tensor(prediction_rain)
    target = _as_tensor(target_rain, device=prediction.device)
    if prediction.shape != target.shape:
        raise ValueError("prediction_rain and target_rain shapes differ")
    if prediction.ndim < 1 or prediction.shape[-1] < 2:
        raise ValueError("rain tensors need at least two height levels")
    if not torch.is_floating_point(prediction):
        prediction = prediction.float()
    if not torch.is_floating_point(target):
        target = target.float()
    selected = _broadcast_mask(mask, prediction)
    expanded_height = _broadcast_auxiliary(
        height_km, prediction, name="height_km"
    ).to(dtype=prediction.dtype)

    # Before=(...,Z), after=(...,Z-1). The output location represents the
    # physical midpoint between two adjacent native DPR height levels.
    height_delta = expanded_height[..., 1:] - expanded_height[..., :-1]
    pair_mask = selected[..., 1:] & selected[..., :-1]
    invalid_selected_spacing = pair_mask & (
        ~torch.isfinite(height_delta) | (height_delta <= 0.0)
    )
    if bool(invalid_selected_spacing.any()):
        raise ValueError(
            "selected adjacent height levels must be finite and strictly ascending"
        )
    prediction_drdz = (
        prediction[..., 1:] - prediction[..., :-1]
    ) / height_delta
    target_drdz = (target[..., 1:] - target[..., :-1]) / height_delta
    midpoint_height = 0.5 * (
        expanded_height[..., 1:] + expanded_height[..., :-1]
    )
    return prediction_drdz, target_drdz, pair_mask, midpoint_height


class PhysicalRainGradientMetrics:
    """Streaming physical ``dR/dz`` metrics on reliable adjacent-height pairs.

    The primary metric support is intentionally derived from the same reliable
    voxel mask as rain-rate model selection. Optional weak CFB labels, missing
    voxels, and padding cannot enter unless both endpoints are explicitly
    selected upstream. This keeps E0/N/I/W comparisons on identical support.

    Besides overall MAE/RMSE/Bias/R²/Pearson ``r``, the class reports gradient
    amplitude and sign diagnostics, midpoint-height/CFB/type/intensity groups,
    and optional whole-file macro/bootstrap statistics.
    """

    def __init__(
        self,
        height_levels_km: Sequence[float],
        *,
        cfb_distance_edges_km: Sequence[float] = (-1.0, 0.0, 0.5, 2.0),
        intensity_thresholds_mm_h: Sequence[float] = (1.0, 5.0, 10.0, 30.0),
        sign_epsilon_mm_h_km: float = 0.1,
        file_labels: Sequence[str] | None = None,
        bootstrap_seed: int = 2026,
        bootstrap_replicates: int = 2000,
        confidence_level: float = 0.95,
    ) -> None:
        self.height_levels_km = _strictly_increasing_values(
            height_levels_km, name="height_levels_km", minimum_length=2
        )
        self.midpoint_heights_km = tuple(
            0.5 * (lower + upper)
            for lower, upper in zip(
                self.height_levels_km, self.height_levels_km[1:]
            )
        )
        self.height_labels = tuple(
            f"z_{_number_label(value)}_km" for value in self.midpoint_heights_km
        )
        self.cfb_distance_edges_km = _strictly_increasing_values(
            cfb_distance_edges_km,
            name="cfb_distance_edges_km",
            minimum_length=1,
        )
        self.cfb_distance_labels = _unbounded_interval_labels(
            self.cfb_distance_edges_km, suffix="_km"
        )
        self.intensity_thresholds_mm_h = _strictly_increasing_values(
            intensity_thresholds_mm_h,
            name="intensity_thresholds_mm_h",
            minimum_length=1,
        )
        if self.intensity_thresholds_mm_h[0] <= 0.0:
            raise ValueError("intensity_thresholds_mm_h must be positive")
        self.intensity_labels = _unbounded_interval_labels(
            self.intensity_thresholds_mm_h, suffix="_mm_h"
        )
        self.sign_epsilon_mm_h_km = float(sign_epsilon_mm_h_km)
        self.all = _GradientSummaryAccumulator(self.sign_epsilon_mm_h_km)
        self.by_height = GroupedRegressionAccumulator(self.height_labels)
        self.by_cfb_distance = GroupedRegressionAccumulator(
            self.cfb_distance_labels
        )
        self.by_intensity = GroupedRegressionAccumulator(self.intensity_labels)
        self.by_precipitation_type = GroupedRegressionAccumulator(
            StratifiedPrecipitationMetrics._PRECIPITATION_TYPE_LABELS
        )
        self.filewise = (
            FilewisePrecipitationMetrics(
                file_labels,
                bootstrap_seed=bootstrap_seed,
                bootstrap_replicates=bootstrap_replicates,
                confidence_level=confidence_level,
            )
            if file_labels is not None
            else None
        )
        self._support_hasher = hashlib.sha256()
        self._support_fingerprint_mergeable = True

    def reset(self) -> None:
        self.all.reset()
        self.by_height.reset()
        self.by_cfb_distance.reset()
        self.by_intensity.reset()
        self.by_precipitation_type.reset()
        if self.filewise is not None:
            self.filewise.reset()
        self._support_hasher = hashlib.sha256()
        self._support_fingerprint_mergeable = True

    def _midpoint_height_groups(self, height: torch.Tensor) -> torch.Tensor:
        reference = torch.tensor(
            self.midpoint_heights_km, dtype=height.dtype, device=height.device
        )
        insertion = torch.searchsorted(reference, height.contiguous())
        right_index = insertion.clamp(max=reference.numel() - 1)
        left_index = (insertion - 1).clamp(min=0)
        right_distance = (height - reference[right_index]).abs()
        left_distance = (height - reference[left_index]).abs()
        choose_right = right_distance < left_distance
        groups = torch.where(choose_right, right_index, left_index)
        closest = torch.where(choose_right, right_distance, left_distance)
        tolerance = max(
            1e-5,
            float(torch.diff(reference).min().item()) * 1e-3
            if reference.numel() > 1
            else 1e-5,
        )
        valid = torch.isfinite(height) & (closest <= tolerance)
        return torch.where(valid, groups, torch.full_like(groups, -1))

    def _update_rain_tensors(
        self,
        prediction_rain: torch.Tensor,
        target_rain: torch.Tensor,
        mask: Any,
        *,
        height_km: Any,
        cfb_distance_km: Any | None,
        precipitation_type: Any | None,
        file_id: Any | None,
    ) -> None:
        selected = _broadcast_mask(mask, prediction_rain)
        expanded_height = _broadcast_auxiliary(
            height_km, prediction_rain, name="height_km"
        ).float()
        expected_height = torch.tensor(
            self.height_levels_km,
            dtype=expanded_height.dtype,
            device=expanded_height.device,
        ).reshape((1,) * (prediction_rain.ndim - 1) + (-1,))
        expected_height = torch.broadcast_to(expected_height, prediction_rain.shape)
        if not bool(
            torch.isfinite(expanded_height).all()
            and torch.allclose(
                expanded_height, expected_height, rtol=0.0, atol=1e-6
            )
        ):
            raise ValueError(
                "height_km differs from the fixed physical dR/dz height grid"
            )
        if bool(selected.any()):
            if not bool(
                torch.isfinite(prediction_rain[selected]).all()
                and torch.isfinite(target_rain[selected]).all()
            ):
                raise ValueError("selected rain-rate values must be finite")
            if bool((target_rain[selected] < 0.0).any()):
                raise ValueError("selected target rain rates must be non-negative")
        (
            prediction_drdz,
            target_drdz,
            pair_mask,
            midpoint_height,
        ) = physical_vertical_rain_gradient(
            prediction_rain,
            target_rain,
            selected,
            height_km=expanded_height,
        )
        # Post-training evaluation is single-process and visits patches in a
        # deterministic order. Hashing the complete boolean support makes the
        # cross-model comparison stricter than checking counts alone. A merged
        # or multi-rank stream is marked unavailable because hash concatenation
        # is order-dependent and cannot be reduced additively.
        self._support_hasher.update(
            pair_mask.detach().to(device="cpu").contiguous().numpy().tobytes()
        )
        self.all.update(prediction_drdz, target_drdz, pair_mask)
        self.by_height.update(
            prediction_drdz,
            target_drdz,
            pair_mask,
            self._midpoint_height_groups(midpoint_height),
        )

        # Pair-mean target rain separates small physical gradients in weak rain
        # from equally sized gradients embedded in intense convective profiles.
        target_pair_mean = 0.5 * (
            target_rain[..., 1:] + target_rain[..., :-1]
        )
        intensity_groups = _unbounded_group_index(
            target_pair_mean, self.intensity_thresholds_mm_h
        )
        self.by_intensity.update(
            prediction_drdz,
            target_drdz,
            pair_mask,
            intensity_groups,
        )

        if cfb_distance_km is not None:
            expanded_distance = _broadcast_auxiliary(
                cfb_distance_km, prediction_rain, name="cfb_distance_km"
            ).float()
            midpoint_distance = 0.5 * (
                expanded_distance[..., 1:] + expanded_distance[..., :-1]
            )
            distance_groups = _unbounded_group_index(
                midpoint_distance, self.cfb_distance_edges_km
            )
            self.by_cfb_distance.update(
                prediction_drdz,
                target_drdz,
                pair_mask,
                distance_groups,
            )

        if precipitation_type is not None:
            expanded_type = _broadcast_auxiliary(
                precipitation_type, prediction_rain, name="precipitation_type"
            ).float()
            type_groups = StratifiedPrecipitationMetrics._precipitation_type_group_index(
                expanded_type[..., :-1]
            )
            self.by_precipitation_type.update(
                prediction_drdz,
                target_drdz,
                pair_mask,
                type_groups,
            )

        if self.filewise is not None:
            if file_id is None:
                raise KeyError("physical dR/dz filewise metrics require file_id")
            self.filewise.update_values(
                prediction_drdz,
                target_drdz,
                pair_mask,
                file_id=file_id,
            )

    def update_log(
        self,
        prediction_log: Any,
        target_log: Any,
        mask: Any,
        *,
        height_km: Any,
        cfb_distance_km: Any | None = None,
        precipitation_type: Any | None = None,
        file_id: Any | None = None,
    ) -> None:
        """Update from model/target ``log1p`` tensors shaped ``(...,Z)``."""

        prediction = _as_tensor(prediction_log)
        target = _as_tensor(target_log, device=prediction.device)
        if prediction.shape != target.shape:
            raise ValueError("prediction_log and target_log shapes differ")
        # Shape remains (...,Z); only units change from log1p to physical mm/h.
        prediction_rain = torch.expm1(prediction.float().clamp_min(0.0))
        target_rain = torch.expm1(target.float().clamp_min(0.0))
        self._update_rain_tensors(
            prediction_rain,
            target_rain,
            mask,
            height_km=height_km,
            cfb_distance_km=cfb_distance_km,
            precipitation_type=precipitation_type,
            file_id=file_id,
        )

    def update_rain(
        self,
        prediction_rain: Any,
        target_rain: Any,
        mask: Any,
        *,
        height_km: Any,
        cfb_distance_km: Any | None = None,
        precipitation_type: Any | None = None,
        file_id: Any | None = None,
    ) -> None:
        """Update from physical rain rates in ``mm h^-1``."""

        prediction = _as_tensor(prediction_rain).float().clamp_min(0.0)
        target = _as_tensor(target_rain, device=prediction.device).float()
        if prediction.shape != target.shape:
            raise ValueError("prediction_rain and target_rain shapes differ")
        self._update_rain_tensors(
            prediction,
            target,
            mask,
            height_km=height_km,
            cfb_distance_km=cfb_distance_km,
            precipitation_type=precipitation_type,
            file_id=file_id,
        )

    def merge(self, other: "PhysicalRainGradientMetrics") -> None:
        if (
            self.height_levels_km != other.height_levels_km
            or self.cfb_distance_edges_km != other.cfb_distance_edges_km
            or self.intensity_thresholds_mm_h
            != other.intensity_thresholds_mm_h
            or self.sign_epsilon_mm_h_km != other.sign_epsilon_mm_h_km
            or (self.filewise is None) != (other.filewise is None)
        ):
            raise ValueError("cannot merge physical dR/dz metrics with different setup")
        self.all.merge(other.all)
        self.by_height.merge(other.by_height)
        self.by_cfb_distance.merge(other.by_cfb_distance)
        self.by_intensity.merge(other.by_intensity)
        self.by_precipitation_type.merge(other.by_precipitation_type)
        if self.filewise is not None and other.filewise is not None:
            if self.filewise.file_labels != other.filewise.file_labels:
                raise ValueError("cannot merge dR/dz metrics with different files")
            self.filewise.accumulator.merge(other.filewise.accumulator)
        self._support_fingerprint_mergeable = False

    def compute(self, *, synchronize: bool = False) -> dict[str, Any]:
        distributed_stream = (
            synchronize
            and distributed.is_available()
            and distributed.is_initialized()
            and distributed.get_world_size() > 1
        )
        result: dict[str, Any] = {
            "definitions": {
                "quantity": "physical vertical rain-rate gradient dR/dz",
                "units": "mm h^-1 km^-1",
                "scope": (
                    "conditional gradients inside consecutive reliable positive "
                    "DPR-rain voxels; valid zero-rain pairs and rain occurrence "
                    "boundaries are not represented"
                ),
                "sign_convention": (
                    "z increases upward; positive means rain rate is larger aloft"
                ),
                "finite_difference": (
                    "(R[k+1]-R[k])/(z[k+1]-z[k]) at adjacent-level midpoint"
                ),
                "pair_mask": (
                    "both adjacent endpoints belong to the reliable primary "
                    "evaluation mask; weak-CFB, missing and padding endpoints "
                    "are excluded"
                ),
                "height_axis": (
                    f"{len(self.height_levels_km)} native levels produce "
                    f"{len(self.midpoint_heights_km)} gradient levels"
                ),
                "height_levels_km": list(self.height_levels_km),
                "cfb_distance_edges_km": list(self.cfb_distance_edges_km),
                "pair_mean_intensity_thresholds_mm_h": list(
                    self.intensity_thresholds_mm_h
                ),
                "sign_epsilon_mm_h_km": self.sign_epsilon_mm_h_km,
            },
            "support": {
                "sha256": (
                    self._support_hasher.hexdigest()
                    if self._support_fingerprint_mergeable
                    and not distributed_stream
                    else None
                ),
                "definition": (
                    "SHA-256 of deterministic batch-order reliable pair-mask "
                    "bytes; unavailable after merge or multi-rank synchronization"
                ),
            },
            "all": self.all.compute(synchronize=synchronize),
            "by_midpoint_height_km": self.by_height.compute(
                synchronize=synchronize
            ),
            "by_midpoint_cfb_distance_km": self.by_cfb_distance.compute(
                synchronize=synchronize
            ),
            "by_pair_mean_target_intensity_mm_h": self.by_intensity.compute(
                synchronize=synchronize
            ),
            "by_precipitation_type": self.by_precipitation_type.compute(
                synchronize=synchronize
            ),
        }
        if self.filewise is not None:
            result["filewise"] = self.filewise.compute(
                synchronize=synchronize
            )
        return result
