"""Complete-orbit reconstruction for the R1-O value-only control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from precipitation_inversion.inference.sliding_window import (
    CoreWindowReconstructor,
    _model_device,
)
from precipitation_inversion.inference.stage2_sliding_window import (
    _dpr_statistics,
    _source_shape,
)
from precipitation_inversion.models.stage2_completion_unet3d import (
    stage2_completion_prediction_from_output,
)

try:
    import torch
except (ImportError, OSError):  # pragma: no cover
    torch = None


@dataclass(frozen=True)
class Stage2CompletionOrbitPrediction:
    """Dense R1-O products, each shaped ``(nscan,nray,z)``."""

    reflectivity_standardized: np.ndarray
    reflectivity_dbz: np.ndarray


def predict_stage2_completion_full_orbit(
    model: Any,
    dataset: Any,
    file_id: int,
    *,
    device: str | Any | None = None,
    batch_size: int = 1,
    num_workers: int = 0,
    use_amp: bool = True,
) -> Stage2CompletionOrbitPrediction:
    """Discard Patch halo/padding and rebuild one source-geometry orbit."""

    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for R1-O inference")
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    original_shape = _source_shape(dataset, file_id)
    indices = list(dataset.file_index_range(file_id))
    if not indices:
        raise ValueError(f"file_id={file_id} has no patch records")
    resolved_device = _model_device(model, device)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=resolved_device.type == "cuda",
    )
    reconstructor = CoreWindowReconstructor(
        original_shape, halo_size=int(dataset.halo_size), channels=1
    )
    model.to(resolved_device)
    was_training = bool(model.training)
    model.eval()
    try:
        with torch.inference_mode():
            for batch in loader:
                # (B,4,Dp,Hp,Z) -> (B,1,Dp,Hp,Z); only unique cores survive.
                inputs = batch["inputs"].to(resolved_device, non_blocking=True)
                with torch.autocast(
                    device_type=resolved_device.type,
                    enabled=use_amp and resolved_device.type == "cuda",
                ):
                    prediction = stage2_completion_prediction_from_output(model(inputs))
                expected = (inputs.shape[0], 1, *inputs.shape[2:])
                if tuple(prediction.shape) != expected:
                    raise ValueError(f"R1-O output must have shape {expected}")
                if not bool(torch.isfinite(prediction).all()):
                    raise FloatingPointError("R1-O orbit predictions must be finite")
                prediction = prediction.float().cpu()
                for position in range(inputs.shape[0]):
                    reconstructor.add(
                        prediction[position],
                        core_start=int(batch["core_start"][position]),
                        core_length=int(batch["core_length"][position]),
                    )
    finally:
        model.train(was_training)

    standardized = reconstructor.finalize()[0]
    mean, std = _dpr_statistics(dataset)
    physical = (
        standardized * std.reshape((1, 1, -1))
        + mean.reshape((1, 1, -1))
    )
    return Stage2CompletionOrbitPrediction(
        reflectivity_standardized=standardized.astype(np.float32, copy=False),
        reflectivity_dbz=physical.astype(np.float32, copy=False),
    )


__all__ = [
    "Stage2CompletionOrbitPrediction",
    "predict_stage2_completion_full_orbit",
]
