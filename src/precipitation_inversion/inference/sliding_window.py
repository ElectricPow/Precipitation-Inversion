"""Reconstruct variable-length orbits from non-overlapping output cores.

The model sees overlapping context windows, but only the central core of each
prediction is retained. Core intervals are non-overlapping and exactly cover
``nscan``, so reconstruction uses direct assignment rather than weighted
blending and cannot double-count a voxel.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


try:
    import torch
except (ImportError, OSError):  # pragma: no cover - reconstruction supports NumPy
    torch = None


def _numpy(value: Any) -> np.ndarray:
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class CoreWindowReconstructor:
    """Incrementally assemble core predictions into one complete orbit.

    Parameters
    ----------
    original_shape:
        Source ``(nscan, nray, z)`` before context and U-Net padding.
    halo_size:
        Number of context scans preceding the core in every model window.
    channels:
        Number of prediction channels. Stage one uses one rain-rate channel.
    """

    def __init__(
        self,
        original_shape: Sequence[int],
        *,
        halo_size: int,
        channels: int = 1,
        dtype: str | np.dtype = np.float32,
    ) -> None:
        if len(original_shape) != 3:
            raise ValueError("original_shape must be (nscan,nray,z)")
        self.original_shape = tuple(int(value) for value in original_shape)
        if any(value <= 0 for value in self.original_shape):
            raise ValueError("all original_shape dimensions must be positive")
        if halo_size < 0 or channels <= 0:
            raise ValueError("halo_size must be non-negative and channels positive")
        self.halo_size = int(halo_size)
        self.channels = int(channels)
        # Complete output: (C,nscan,nray,z), initialized as zero rain.
        self.output = np.zeros(
            (self.channels, *self.original_shape), dtype=np.dtype(dtype)
        )
        # Coverage is scan-level because every core spans all physical rays/heights.
        self.covered_scans = np.zeros(self.original_shape[0], dtype=bool)

    def add(
        self,
        prediction: Any,
        *,
        core_start: int,
        core_length: int,
        output_mask: Any | None = None,
    ) -> None:
        """Crop one ``(C,Dp,Hp,Wp)`` window and place its central core."""

        values = _numpy(prediction)
        if values.ndim == 3:
            values = values[np.newaxis, ...]
        if values.ndim != 4 or values.shape[0] != self.channels:
            raise ValueError(
                f"prediction must have shape ({self.channels},D,H,W), got {values.shape}"
            )
        start = int(core_start)
        length = int(core_length)
        stop = start + length
        if start < 0 or length <= 0 or stop > self.original_shape[0]:
            raise ValueError("core interval lies outside the original orbit")
        if self.covered_scans[start:stop].any():
            raise ValueError(f"core interval [{start},{stop}) overlaps earlier output")

        nray, z_size = self.original_shape[1:]
        core_slice = slice(self.halo_size, self.halo_size + length)
        if (
            values.shape[1] < core_slice.stop
            or values.shape[2] < nray
            or values.shape[3] < z_size
        ):
            raise ValueError("prediction is too small for requested physical core")
        # Before=(C,Dp,Hp,Wp); after crop=(C,core_length,nray,z).
        core_values = values[:, core_slice, :nray, :z_size]

        if output_mask is not None:
            mask = _numpy(output_mask)
            if mask.ndim == 3:
                mask = mask[np.newaxis, ...]
            if mask.ndim != 4 or mask.shape[0] not in (1, self.channels):
                raise ValueError("output_mask must have shape (1,D,H,W) or (C,D,H,W)")
            if (
                mask.shape[1] < core_slice.stop
                or mask.shape[2] < nray
                or mask.shape[3] < z_size
            ):
                raise ValueError("output_mask is too small for requested core")
            # Before=(1 or C,Dp,Hp,Wp); crop broadcasts to (C,core_length,nray,z).
            core_mask = mask[:, core_slice, :nray, :z_size].astype(bool)
            core_values = np.where(core_mask, core_values, 0.0)

        self.output[:, start:stop, :, :] = core_values.astype(
            self.output.dtype, copy=False
        )
        self.covered_scans[start:stop] = True

    def finalize(self, *, require_complete: bool = True) -> np.ndarray:
        """Return ``(C,nscan,nray,z)`` and optionally require exact coverage."""

        if require_complete and not self.covered_scans.all():
            missing = np.flatnonzero(~self.covered_scans)
            raise ValueError(
                f"reconstruction is incomplete; first missing scan={int(missing[0])}"
            )
        return self.output.copy()


def stitch_core_predictions(
    predictions: Sequence[Any],
    *,
    core_starts: Sequence[int],
    core_lengths: Sequence[int],
    original_shape: Sequence[int],
    halo_size: int,
    output_masks: Sequence[Any] | None = None,
) -> np.ndarray:
    """Assemble a sequence of patch outputs into ``(C,nscan,nray,z)``."""

    if not (len(predictions) == len(core_starts) == len(core_lengths)):
        raise ValueError("predictions, starts, and lengths must have equal size")
    if not predictions:
        raise ValueError("at least one prediction is required")
    if output_masks is not None and len(output_masks) != len(predictions):
        raise ValueError("output_masks must align with predictions")
    first = _numpy(predictions[0])
    channels = 1 if first.ndim == 3 else int(first.shape[0])
    reconstructor = CoreWindowReconstructor(
        original_shape, halo_size=halo_size, channels=channels
    )
    for position, prediction in enumerate(predictions):
        reconstructor.add(
            prediction,
            core_start=int(core_starts[position]),
            core_length=int(core_lengths[position]),
            output_mask=None if output_masks is None else output_masks[position],
        )
    return reconstructor.finalize()


def _model_device(model: Any, requested: str | Any | None) -> Any:
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for model inference")
    if requested is not None:
        return torch.device(requested)
    try:
        return next(model.parameters()).device
    except StopIteration:
        try:
            return next(model.buffers()).device
        except StopIteration:
            return torch.device("cpu")


def predict_full_orbit(
    model: Any,
    dataset: Any,
    file_id: int,
    *,
    device: str | Any | None = None,
    batch_size: int = 1,
    num_workers: int = 0,
    use_amp: bool = True,
) -> np.ndarray:
    """Run all patches for one file and return rain rate ``(nscan,nray,z)``.

    The model must map ``inputs=(B,C,Dp,Hp,Z)`` to log-rain predictions with
    shape ``(B,1,Dp,Hp,Wp)``. Predictions are converted by ``expm1``, masked to
    DPR-observed support, cropped to each core, and concatenated along nscan.
    """

    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for model inference")
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    if getattr(dataset, "positive_only", False):
        raise ValueError("full-orbit inference requires positive_only=False")
    if file_id < 0 or file_id >= len(dataset.files):
        raise IndexError(file_id)

    logical_indices = list(dataset.file_index_range(file_id))
    if not logical_indices:
        raise ValueError(f"file_id={file_id} has no patch records")
    resolved_device = _model_device(model, device)
    subset = torch.utils.data.Subset(dataset, logical_indices)
    loader = torch.utils.data.DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=resolved_device.type == "cuda",
    )
    model.to(resolved_device)
    was_training = bool(model.training)
    model.eval()
    source_entry = dataset.source_files[file_id]
    reconstructor = CoreWindowReconstructor(
        (
            int(source_entry["nscan"]),
            int(source_entry["nray"]),
            int(source_entry["z_size"]),
        ),
        halo_size=int(dataset.halo_size),
        channels=1,
    )

    try:
        with torch.inference_mode():
            for batch in loader:
                # Before=(B,C,Dp,Hp,Z); model output=(B,1,Dp,Hp,Z).
                inputs = batch["inputs"].to(resolved_device, non_blocking=True)
                autocast_enabled = use_amp and resolved_device.type == "cuda"
                with torch.autocast(
                    device_type=resolved_device.type,
                    enabled=autocast_enabled,
                ):
                    prediction_log = model(inputs)
                if not isinstance(prediction_log, torch.Tensor):
                    raise TypeError("model must return a PyTorch Tensor")
                if (
                    prediction_log.ndim != 5
                    or prediction_log.shape[0] != inputs.shape[0]
                    or prediction_log.shape[1] != 1
                    or prediction_log.shape[2:] != inputs.shape[2:]
                ):
                    raise ValueError(
                        "model output must have shape (B,1,Dp,Hp,Wp) matching inputs"
                    )
                # log1p rain -> physical mm/h, maintaining (B,1,Dp,Hp,Wp).
                rain = torch.expm1(prediction_log.float().clamp_min(0.0)).cpu()
                masks = batch["output_mask"]
                for position in range(rain.shape[0]):
                    reconstructor.add(
                        rain[position],
                        core_start=int(batch["core_start"][position]),
                        core_length=int(batch["core_length"][position]),
                        output_mask=masks[position],
                    )
    finally:
        model.train(was_training)

    # Stage one has one output channel: (1,nscan,nray,z) -> (nscan,nray,z).
    return reconstructor.finalize()[0]
