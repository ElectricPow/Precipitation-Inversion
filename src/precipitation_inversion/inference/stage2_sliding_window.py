"""Complete-orbit reconstruction and support-threshold selection for Stage 2.

The model consumes overlapping context windows, while only non-overlapping
central cores are retained.  Every reconstructed array uses source geometry
``(nscan,nray,z)``; horizontal padding and halo predictions are discarded.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from precipitation_inversion.inference.sliding_window import (
    CoreWindowReconstructor,
    _model_device,
)
from precipitation_inversion.metrics.stage2_reflectivity import (
    SupportConfusionAccumulator,
)
from precipitation_inversion.models.stage2_unet3d import (
    stage2_predictions_from_output,
)
from precipitation_inversion.models.stage3_direct import (
    stage3_d0_predictions_from_output,
)


try:
    import torch
except (ImportError, OSError):  # pragma: no cover - NumPy threshold utilities
    torch = None


@dataclass(frozen=True)
class Stage2OrbitPrediction:
    """Four dense model products, each shaped ``(nscan,nray,z)``."""

    support_logits: np.ndarray
    support_probability: np.ndarray
    reflectivity_standardized: np.ndarray
    reflectivity_dbz: np.ndarray


@dataclass(frozen=True)
class Stage3D0OrbitPrediction(Stage2OrbitPrediction):
    """D0 complete-orbit physical heads plus ungated rain intensity.

    ``rain_log1p`` and ``rain_rate`` are dense ``(nscan,nray,z)`` arrays.
    Neither field is thresholded by predicted support here, so callers can
    compare an oracle-support diagnostic and the deployable predicted-support
    route from exactly the same continuous rain output.
    """

    rain_log1p: np.ndarray
    rain_rate: np.ndarray


@dataclass(frozen=True)
class ThresholdSelection:
    """Validation-only support threshold search result."""

    threshold: float
    objective: str
    objective_value: float
    candidates: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SupportThresholdSweep:
    """Streaming validation-only CSI/F1 threshold search."""

    def __init__(self, candidates: Sequence[float], *, objective: str = "csi") -> None:
        if objective not in {"csi", "f1"}:
            raise ValueError("threshold objective must be 'csi' or 'f1'")
        self.thresholds = sorted({float(value) for value in candidates})
        if not self.thresholds or any(
            not math.isfinite(value) or not 0.0 < value < 1.0
            for value in self.thresholds
        ):
            raise ValueError("threshold candidates must be finite values in (0,1)")
        self.objective = objective
        self.accumulators = [
            SupportConfusionAccumulator() for _ in self.thresholds
        ]

    def update(
        self,
        probability: np.ndarray,
        target: np.ndarray,
        domain: np.ndarray,
    ) -> None:
        values = np.asarray(probability, dtype=np.float64)
        truth = np.asarray(target)
        selected = np.asarray(domain)
        if truth.dtype != np.bool_ or selected.dtype != np.bool_:
            raise TypeError("threshold targets and domains must be boolean")
        if values.shape != truth.shape or values.shape != selected.shape:
            raise ValueError("threshold arrays must share a shape")
        if np.any(selected & ~np.isfinite(values)):
            raise ValueError("selected support probabilities must be finite")
        for threshold, accumulator in zip(self.thresholds, self.accumulators):
            accumulator.update(values >= threshold, truth, selected)

    def compute(self) -> ThresholdSelection:
        rows = [
            {"threshold": threshold, **accumulator.compute()}
            for threshold, accumulator in zip(
                self.thresholds, self.accumulators
            )
        ]
        finite_rows = [
            row
            for row in rows
            if math.isfinite(float(row[self.objective]))
        ]
        if not finite_rows:
            raise ValueError("threshold objective is undefined for every candidate")
        best = max(
            finite_rows,
            key=lambda row: (
                float(row[self.objective]),
                -abs(float(row["threshold"]) - 0.5),
                -float(row["threshold"]),
            ),
        )
        return ThresholdSelection(
            threshold=float(best["threshold"]),
            objective=self.objective,
            objective_value=float(best[self.objective]),
            candidates=rows,
        )


def _source_shape(dataset: Any, file_id: int) -> tuple[int, int, int]:
    if not hasattr(dataset, "files") or not hasattr(dataset, "file_index_range"):
        raise TypeError("dataset must expose files and file_index_range")
    if file_id < 0 or file_id >= len(dataset.files):
        raise IndexError(file_id)
    entry = dataset.files[file_id]
    return int(entry["nscan"]), int(entry["nray"]), int(entry["z_size"])


def _dpr_statistics(dataset: Any) -> tuple[np.ndarray, np.ndarray]:
    standardizer = getattr(dataset, "dpr_standardizer", None)
    if standardizer is None:
        raise TypeError("dataset must expose dpr_standardizer")
    mean = np.asarray(standardizer.mean, dtype=np.float32)
    std = np.asarray(standardizer.std, dtype=np.float32)
    if mean.shape != std.shape or mean.ndim != 1:
        raise ValueError("DPR normalization vectors must be one-dimensional")
    if not (np.all(np.isfinite(mean)) and np.all(np.isfinite(std)) and np.all(std > 0)):
        raise ValueError("invalid DPR normalization statistics")
    return mean, std


def predict_stage2_full_orbit(
    model: Any,
    dataset: Any,
    file_id: int,
    *,
    device: str | Any | None = None,
    batch_size: int = 1,
    num_workers: int = 0,
    use_amp: bool = True,
) -> Stage2OrbitPrediction:
    """Predict and reconstruct one complete Stage-2 orbit.

    Model windows are ``(B,C,Dp,Hp,Z)`` using the channels recorded in the
    checkpoint configuration. Both heads return
    ``(B,1,Dp,Hp,Z)``. The central scan core and first ``nray`` physical rays
    are copied into two ``(nscan,nray,z)`` orbit arrays.
    """

    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for model inference")
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    original_shape = _source_shape(dataset, file_id)
    logical_indices = list(dataset.file_index_range(file_id))
    if not logical_indices:
        raise ValueError(f"file_id={file_id} has no patch records")
    resolved_device = _model_device(model, device)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, logical_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=resolved_device.type == "cuda",
    )
    support_reconstructor = CoreWindowReconstructor(
        original_shape,
        halo_size=int(dataset.halo_size),
        channels=1,
    )
    reflectivity_reconstructor = CoreWindowReconstructor(
        original_shape,
        halo_size=int(dataset.halo_size),
        channels=1,
    )
    model.to(resolved_device)
    was_training = bool(model.training)
    model.eval()
    try:
        with torch.inference_mode():
            for batch in loader:
                # inputs: (B,C,Dp,Hp,Z) -> both outputs: (B,1,Dp,Hp,Z).
                inputs = batch["inputs"].to(resolved_device, non_blocking=True)
                with torch.autocast(
                    device_type=resolved_device.type,
                    enabled=use_amp and resolved_device.type == "cuda",
                ):
                    support_logits, reflectivity = stage2_predictions_from_output(
                        model(inputs)
                    )
                if support_logits.shape != (inputs.shape[0], 1, *inputs.shape[2:]):
                    raise ValueError("Stage-2 output shape must match the input grid")
                if not bool(torch.isfinite(support_logits).all() and torch.isfinite(reflectivity).all()):
                    raise FloatingPointError("Stage-2 orbit predictions must be finite")
                support_cpu = support_logits.float().cpu()
                reflectivity_cpu = reflectivity.float().cpu()
                for position in range(inputs.shape[0]):
                    arguments = {
                        "core_start": int(batch["core_start"][position]),
                        "core_length": int(batch["core_length"][position]),
                    }
                    support_reconstructor.add(support_cpu[position], **arguments)
                    reflectivity_reconstructor.add(
                        reflectivity_cpu[position], **arguments
                    )
    finally:
        model.train(was_training)

    logits = support_reconstructor.finalize()[0]
    standardized = reflectivity_reconstructor.finalize()[0]
    mean, std = _dpr_statistics(dataset)
    physical = standardized * std.reshape((1, 1, -1)) + mean.reshape((1, 1, -1))
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))
    return Stage2OrbitPrediction(
        support_logits=logits.astype(np.float32, copy=False),
        support_probability=probability.astype(np.float32, copy=False),
        reflectivity_standardized=standardized.astype(np.float32, copy=False),
        reflectivity_dbz=physical.astype(np.float32, copy=False),
    )


def predict_stage3_d0_full_orbit(
    model: Any,
    dataset: Any,
    file_id: int,
    *,
    device: str | Any | None = None,
    batch_size: int = 1,
    num_workers: int = 0,
    use_amp: bool = True,
) -> Stage3D0OrbitPrediction:
    """Reconstruct all three D0 heads on one complete source orbit."""

    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for model inference")
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    original_shape = _source_shape(dataset, file_id)
    logical_indices = list(dataset.file_index_range(file_id))
    if not logical_indices:
        raise ValueError(f"file_id={file_id} has no patch records")
    resolved_device = _model_device(model, device)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, logical_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=resolved_device.type == "cuda",
    )
    reconstructors = {
        name: CoreWindowReconstructor(
            original_shape, halo_size=int(dataset.halo_size), channels=1
        )
        for name in ("support", "reflectivity", "rain")
    }
    model.to(resolved_device)
    was_training = bool(model.training)
    model.eval()
    try:
        with torch.inference_mode():
            for batch in loader:
                # GR inputs (B,C,Dp,Hp,Z) -> three (B,1,Dp,Hp,Z) heads.
                inputs = batch["inputs"].to(resolved_device, non_blocking=True)
                with torch.autocast(
                    device_type=resolved_device.type,
                    enabled=use_amp and resolved_device.type == "cuda",
                ):
                    rain, support, reflectivity = stage3_d0_predictions_from_output(
                        model(inputs)
                    )
                outputs = {
                    "support": support.float().cpu(),
                    "reflectivity": reflectivity.float().cpu(),
                    "rain": rain.float().cpu(),
                }
                if any(value.shape != (inputs.shape[0], 1, *inputs.shape[2:]) for value in outputs.values()):
                    raise ValueError("D0 output shapes must match the input grid")
                if any(not bool(torch.isfinite(value).all()) for value in outputs.values()):
                    raise FloatingPointError("D0 orbit predictions must be finite")
                for position in range(inputs.shape[0]):
                    arguments = {
                        "core_start": int(batch["core_start"][position]),
                        "core_length": int(batch["core_length"][position]),
                    }
                    for name, value in outputs.items():
                        reconstructors[name].add(value[position], **arguments)
    finally:
        model.train(was_training)

    logits = reconstructors["support"].finalize()[0]
    standardized = reconstructors["reflectivity"].finalize()[0]
    rain_log = reconstructors["rain"].finalize()[0]
    mean, std = _dpr_statistics(dataset)
    physical = standardized * std.reshape((1, 1, -1)) + mean.reshape((1, 1, -1))
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))
    rain_rate = np.expm1(np.maximum(rain_log, 0.0))
    return Stage3D0OrbitPrediction(
        support_logits=logits.astype(np.float32, copy=False),
        support_probability=probability.astype(np.float32, copy=False),
        reflectivity_standardized=standardized.astype(np.float32, copy=False),
        reflectivity_dbz=physical.astype(np.float32, copy=False),
        rain_log1p=rain_log.astype(np.float32, copy=False),
        rain_rate=rain_rate.astype(np.float32, copy=False),
    )


def reconstruct_stage2_fields(
    dataset: Any,
    file_id: int,
    field_names: Sequence[str],
) -> dict[str, np.ndarray]:
    """Reconstruct selected single-channel Dataset fields in source geometry.

    This is used for labels and diagnostic masks only.  It never feeds DPR or
    ``pre_dpr`` information back into the model.
    """

    if not field_names:
        raise ValueError("at least one field name is required")
    original_shape = _source_shape(dataset, file_id)
    reconstructors: dict[str, CoreWindowReconstructor] = {}
    dtypes: dict[str, np.dtype] = {}
    for index in dataset.file_index_range(file_id):
        sample = dataset[index]
        for name in field_names:
            if name not in sample:
                raise KeyError(f"Stage-2 sample missing field {name!r}")
            value = sample[name]
            array = (
                value.detach().cpu().numpy()
                if torch is not None and isinstance(value, torch.Tensor)
                else np.asarray(value)
            )
            if array.ndim != 4 or array.shape[0] != 1:
                raise ValueError(f"{name} must have shape (1,D,H,Z)")
            if name not in reconstructors:
                dtypes[name] = array.dtype
                reconstructors[name] = CoreWindowReconstructor(
                    original_shape,
                    halo_size=int(dataset.halo_size),
                    channels=1,
                    dtype=array.dtype,
                )
            reconstructors[name].add(
                array,
                core_start=int(sample["core_start"]),
                core_length=int(sample["core_length"]),
            )
    return {
        name: reconstructor.finalize()[0].astype(dtypes[name], copy=False)
        for name, reconstructor in reconstructors.items()
    }


def reconstruct_stage2_targets(dataset: Any, file_id: int) -> dict[str, np.ndarray]:
    """Return complete physical DPR target plus task and diagnostic masks."""

    names = (
        "target_dbz",
        "target_support",
        "support_loss_mask",
        "regression_mask",
        "overlap_mask",
        "dpr_only_mask",
        "gr_only_mask",
        "neither_mask",
        "gap_proxy_mask",
        "outside_proxy_mask",
        "below_cfb_target_mask",
        "gr_value_mask",
        "gr_native_available",
        "dpr_sparse_anchor_mask",
    )
    fields = reconstruct_stage2_fields(dataset, file_id, names)
    mean, std = _dpr_statistics(dataset)
    fields["target_dbz_standardized"] = fields.pop("target_dbz").astype(np.float32)
    fields["target_dbz"] = (
        fields["target_dbz_standardized"] * std.reshape((1, 1, -1))
        + mean.reshape((1, 1, -1))
    ).astype(np.float32)
    fields["target_support"] = fields["target_support"] >= 0.5
    for name in names[2:]:
        fields[name] = fields[name].astype(bool, copy=False)
    return fields


def select_support_threshold(
    probabilities: Sequence[np.ndarray] | np.ndarray,
    targets: Sequence[np.ndarray] | np.ndarray,
    domains: Sequence[np.ndarray] | np.ndarray,
    *,
    candidates: Sequence[float],
    objective: str = "csi",
) -> ThresholdSelection:
    """Select one support threshold on validation data only.

    Ties are resolved by choosing the candidate closest to 0.5, then the lower
    candidate.  Supported objectives are CSI and F1.
    """

    probability_items = [probabilities] if isinstance(probabilities, np.ndarray) else list(probabilities)
    target_items = [targets] if isinstance(targets, np.ndarray) else list(targets)
    domain_items = [domains] if isinstance(domains, np.ndarray) else list(domains)
    if not probability_items or not (
        len(probability_items) == len(target_items) == len(domain_items)
    ):
        raise ValueError("probability, target, and domain sequences must align")
    sweep = SupportThresholdSweep(candidates, objective=objective)
    for probability, target, domain in zip(
        probability_items, target_items, domain_items
    ):
        sweep.update(probability, target, domain)
    return sweep.compute()
