"""Frozen Stage-2 -> Stage-1 complete-orbit cascade inference.

Stage 2 predicts physical DPR reflectivity and a DPR-support probability from
sparse GR.  Stage 1 was trained with the three-channel contract

``[DPR dBZ standardized per height, DPR-valid mask, scaled height]``.

This module is the single conversion point between those contracts.  It never
uses precipitation labels to construct model inputs.  Source arrays and
returned orbit fields use ``(nscan,nray,z)``; model windows use
``(B,C,Dp,Hp,z)``.  Only scan/ray are padded and the 60 physical height levels
are retained unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Sequence

import numpy as np

from precipitation_inversion.data.patch_dataset import (
    ceil_to_multiple,
    core_starts,
)
from precipitation_inversion.data.transforms import PerLevelStandardizer
from precipitation_inversion.inference.sliding_window import (
    CoreWindowReconstructor,
    _model_device,
)
from precipitation_inversion.models.multitask_unet3d import (
    rain_prediction_from_output,
)


try:
    import torch
except (ImportError, OSError):  # pragma: no cover - validation remains importable
    torch = None


@dataclass(frozen=True)
class Stage1CascadeWindow:
    """One fixed Stage-1 input window and its output-support mask.

    ``inputs`` has shape ``(C,Dp,Hp,z)`` and ``output_mask`` has shape
    ``(1,Dp,Hp,z)``.  The central non-overlapping core begins at local scan
    index ``halo_size``; halo and high-end horizontal padding provide context
    but are discarded during reconstruction.
    """

    inputs: np.ndarray
    output_mask: np.ndarray
    core_start: int
    core_length: int


@dataclass(frozen=True)
class Stage1CascadePrediction:
    """Complete physical Stage-1 products, each ``(nscan,nray,z)``."""

    rain_rate_mm_h: np.ndarray
    input_support: np.ndarray
    output_support: np.ndarray


def _validate_orbit_fields(
    reflectivity_dbz: np.ndarray,
    reflectivity_valid: np.ndarray,
    heights_km: np.ndarray,
    cfb_clutter: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(reflectivity_dbz, dtype=np.float32)
    valid = np.asarray(reflectivity_valid)
    z = np.asarray(heights_km, dtype=np.float32)
    if values.ndim != 3 or any(size <= 0 for size in values.shape):
        raise ValueError("reflectivity_dbz must have non-empty (nscan,nray,z) shape")
    if valid.shape != values.shape or valid.dtype != np.bool_:
        raise TypeError("reflectivity_valid must be a boolean array matching dBZ")
    if z.shape != (values.shape[-1],):
        raise ValueError("heights_km must match the final dBZ axis")
    if not np.all(np.isfinite(z)) or not np.all(np.diff(z) > 0.0):
        raise ValueError("heights_km must be finite and strictly increasing")
    if np.any(valid & ~np.isfinite(values)):
        raise ValueError("every selected reflectivity value must be finite")
    if cfb_clutter is None:
        clutter = np.zeros(values.shape, dtype=bool)
    else:
        clutter = np.asarray(cfb_clutter)
        if clutter.shape != values.shape or clutter.dtype != np.bool_:
            raise TypeError("cfb_clutter must be boolean and match the dBZ shape")
    return values, valid, z, clutter


def _extract_scan_window(
    source: np.ndarray,
    *,
    core_start: int,
    input_size: int,
    halo_size: int,
    fill_value: float | bool,
) -> np.ndarray:
    """Extract ``(nscan,nray,z)`` into fixed ``(input_size,nray,z)``."""

    window_start = core_start - halo_size
    window_stop = window_start + input_size
    source_start = max(window_start, 0)
    source_stop = min(window_stop, source.shape[0])
    destination_start = source_start - window_start
    destination_stop = destination_start + source_stop - source_start
    output = np.full(
        (input_size, source.shape[1], source.shape[2]),
        fill_value,
        dtype=source.dtype,
    )
    output[destination_start:destination_stop] = source[source_start:source_stop]
    return output


def _geometry_window(
    shape: tuple[int, int, int],
    *,
    core_start: int,
    input_size: int,
    halo_size: int,
) -> np.ndarray:
    """Return true only where a window refers to a physical source voxel."""

    nscan, nray, z_size = shape
    window_start = core_start - halo_size
    source_start = max(window_start, 0)
    source_stop = min(window_start + input_size, nscan)
    destination_start = source_start - window_start
    geometry = np.zeros((input_size, nray, z_size), dtype=bool)
    geometry[
        destination_start : destination_start + source_stop - source_start
    ] = True
    return geometry


def _pad_horizontal(
    array: np.ndarray,
    padded_shape: tuple[int, int, int],
    *,
    fill_value: float | bool,
) -> np.ndarray:
    if array.ndim != 3 or any(
        current > target for current, target in zip(array.shape, padded_shape)
    ):
        raise ValueError("array cannot be padded to the requested shape")
    return np.pad(
        array,
        tuple((0, target - current) for current, target in zip(array.shape, padded_shape)),
        mode="constant",
        constant_values=fill_value,
    )


def _cfb_relative_fields(
    cfb_clutter: np.ndarray,
    heights_km: np.ndarray,
    cfb_index: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Derive a signed CFB distance from the supplied voxel clutter mask.

    When the original two-dimensional CFB index is available it is authoritative,
    including the valid boundary ``cfb=0``.  Falling back to the contiguous
    clutter-prefix length is sufficient for the current baseline model and
    weak-layer output support, but a signed-distance input requires explicit
    CFB indices so an unknown profile cannot be confused with ``cfb=0``.
    """

    clutter = np.asarray(cfb_clutter, dtype=bool)
    clutter_count = clutter.sum(axis=-1, dtype=np.int64)
    if cfb_index is None:
        valid_profile = clutter.any(axis=-1)
        boundary_index = np.clip(clutter_count, 0, heights_km.size - 1)
    else:
        raw = np.asarray(cfb_index)
        if raw.shape != clutter.shape[:2]:
            raise ValueError("cfb_index must have shape (nscan,nray)")
        finite = np.isfinite(raw)
        integer = np.where(finite, raw, 0.0).astype(np.int64)
        valid_profile = (
            finite
            & (raw == integer)
            & (integer >= 0)
            & (integer < heights_km.size)
        )
        boundary_index = np.where(valid_profile, integer, 0)
    boundary_height = heights_km[boundary_index]
    distance = heights_km.reshape(1, 1, -1) - boundary_height[..., None]
    distance = distance.astype(np.float32, copy=False)
    distance[~valid_profile, :] = np.nan
    relative_level = (
        np.arange(heights_km.size, dtype=np.int64).reshape(1, 1, -1)
        - boundary_index[..., None]
    )
    relative_level = np.broadcast_to(relative_level, clutter.shape).copy()
    relative_level[~valid_profile, :] = np.iinfo(np.int64).min
    return distance, relative_level


def iter_stage1_cascade_windows(
    reflectivity_dbz: np.ndarray,
    reflectivity_valid: np.ndarray,
    *,
    heights_km: np.ndarray,
    standardizer: PerLevelStandardizer,
    cfb_clutter: np.ndarray | None = None,
    cfb_index: np.ndarray | None = None,
    core_size: int = 32,
    halo_size: int = 16,
    horizontal_multiple: int = 16,
    cfb_input_mode: str = "baseline",
    cfb_distance_scale_km: float = 2.0,
    weak_cfb_layer_weights: Sequence[float] = (),
) -> Iterator[Stage1CascadeWindow]:
    """Yield model-ready windows without using a precipitation label.

    Transformations for each source field are explicitly:

    * physical dBZ ``(nscan,nray,z)`` -> per-height standardized dBZ;
    * boolean support -> float mask channel;
    * ``z`` -> ``[-1,1]`` height channel;
    * stack -> ``(C,input_size,nray,z)``;
    * constant horizontal padding -> ``(C,Dp,Hp,z)``.

    Invalid/missing standardized dBZ and horizontal padding receive neutral
    zero only after their separate validity mask has been constructed.
    """

    values, valid, z, clutter = _validate_orbit_fields(
        reflectivity_dbz, reflectivity_valid, heights_km, cfb_clutter
    )
    if core_size <= 0 or halo_size < 0 or horizontal_multiple <= 0:
        raise ValueError("core_size/multiple must be positive and halo non-negative")
    if standardizer.mean.shape != z.shape or standardizer.std.shape != z.shape:
        raise ValueError("Stage-1 standardizer and orbit height axes differ")
    mode = str(cfb_input_mode).strip().lower()
    if mode not in {"baseline", "mask_below_cfb", "signed_distance"}:
        raise ValueError("unsupported Stage-1 cfb_input_mode")
    if not np.isfinite(cfb_distance_scale_km) or cfb_distance_scale_km <= 0.0:
        raise ValueError("cfb_distance_scale_km must be finite and positive")
    weak_weights = tuple(float(value) for value in weak_cfb_layer_weights)
    if any(not np.isfinite(value) or value < 0.0 for value in weak_weights):
        raise ValueError("weak CFB layer weights must be finite and non-negative")

    nscan, nray, z_size = values.shape
    input_size = core_size + 2 * halo_size
    padded_shape = (
        ceil_to_multiple(input_size, horizontal_multiple),
        ceil_to_multiple(nray, horizontal_multiple),
        z_size,
    )
    z_min, z_max = float(z[0]), float(z[-1])
    height_levels = 2.0 * (z - z_min) / (z_max - z_min) - 1.0
    if mode == "signed_distance" and cfb_index is None:
        raise ValueError("signed-distance Stage-1 input requires cfb_index")
    cfb_distance, relative_level = _cfb_relative_fields(clutter, z, cfb_index)

    for start in core_starts(nscan, core_size):
        length = min(core_size, nscan - start)
        geometry = _geometry_window(
            values.shape,
            core_start=start,
            input_size=input_size,
            halo_size=halo_size,
        )
        dbz_window = _extract_scan_window(
            values,
            core_start=start,
            input_size=input_size,
            halo_size=halo_size,
            fill_value=np.nan,
        )
        valid_window = _extract_scan_window(
            valid,
            core_start=start,
            input_size=input_size,
            halo_size=halo_size,
            fill_value=False,
        )
        clutter_window = _extract_scan_window(
            clutter,
            core_start=start,
            input_size=input_size,
            halo_size=halo_size,
            fill_value=False,
        )
        distance_window = _extract_scan_window(
            cfb_distance,
            core_start=start,
            input_size=input_size,
            halo_size=halo_size,
            fill_value=np.nan,
        )
        relative_window = _extract_scan_window(
            relative_level,
            core_start=start,
            input_size=input_size,
            halo_size=halo_size,
            fill_value=np.iinfo(np.int64).min,
        )

        transform_valid = valid_window & geometry
        if mode == "mask_below_cfb":
            transform_valid &= ~clutter_window
        standardized, effective = standardizer.transform(
            dbz_window,
            valid_mask=transform_valid,
            fill_value=0.0,
            dtype=np.float32,
        )
        # Stage-1 inference support must depend only on reflectivity/CFB, never
        # on pre_dpr. Reliable output excludes CFB-cluttered bins. Optional weak
        # layers add only explicitly configured valid bins immediately below it.
        output_support = effective & ~clutter_window
        if weak_weights:
            for offset, weight in enumerate(weak_weights, start=1):
                if weight > 0.0:
                    output_support |= (
                        valid_window & geometry & (relative_window == -offset)
                    )

        height = np.broadcast_to(
            height_levels.reshape(1, 1, -1), geometry.shape
        ).astype(np.float32, copy=True)
        height[~geometry] = 0.0
        channels = [standardized, effective.astype(np.float32), height]
        if mode == "signed_distance":
            signed = np.clip(
                distance_window / float(cfb_distance_scale_km), -1.0, 1.0
            ).astype(np.float32)
            signed[~np.isfinite(signed) | ~geometry] = 0.0
            channels.append(signed)

        # Before=(C,input_size,nray,z); after=(C,Dp,Hp,z).
        inputs = np.stack(
            [
                _pad_horizontal(channel, padded_shape, fill_value=0.0)
                for channel in channels
            ],
            axis=0,
        ).astype(np.float32, copy=False)
        mask = _pad_horizontal(
            output_support, padded_shape, fill_value=False
        )[None, ...]
        yield Stage1CascadeWindow(
            inputs=np.ascontiguousarray(inputs),
            output_mask=np.ascontiguousarray(mask),
            core_start=int(start),
            core_length=int(length),
        )


def predict_stage1_from_reflectivity_orbit(
    model: Any,
    reflectivity_dbz: np.ndarray,
    reflectivity_valid: np.ndarray,
    *,
    heights_km: np.ndarray,
    standardizer: PerLevelStandardizer,
    cfb_clutter: np.ndarray | None = None,
    cfb_index: np.ndarray | None = None,
    core_size: int = 32,
    halo_size: int = 16,
    horizontal_multiple: int = 16,
    cfb_input_mode: str = "baseline",
    cfb_distance_scale_km: float = 2.0,
    weak_cfb_layer_weights: Sequence[float] = (),
    device: str | Any | None = None,
    batch_size: int = 1,
    use_amp: bool = True,
) -> Stage1CascadePrediction:
    """Run frozen Stage 1 on one arbitrary reflectivity/support orbit.

    Rain is reconstructed in physical ``mm h^-1``.  Cells outside Stage-1's
    reflectivity/CFB output support are exactly zero, which means a missed
    Stage-2 support voxel remains a real zero prediction during final rain
    evaluation instead of being removed from the metric mask.
    """

    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for cascade inference")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    values, valid, z, clutter = _validate_orbit_fields(
        reflectivity_dbz, reflectivity_valid, heights_km, cfb_clutter
    )
    windows = list(
        iter_stage1_cascade_windows(
            values,
            valid,
            heights_km=z,
            standardizer=standardizer,
            cfb_clutter=clutter,
            cfb_index=cfb_index,
            core_size=core_size,
            halo_size=halo_size,
            horizontal_multiple=horizontal_multiple,
            cfb_input_mode=cfb_input_mode,
            cfb_distance_scale_km=cfb_distance_scale_km,
            weak_cfb_layer_weights=weak_cfb_layer_weights,
        )
    )
    if not windows:
        raise ValueError("orbit produced no Stage-1 windows")
    resolved_device = _model_device(model, device)
    reconstructor = CoreWindowReconstructor(
        values.shape, halo_size=halo_size, channels=1
    )
    support_reconstructor = CoreWindowReconstructor(
        values.shape, halo_size=halo_size, channels=1, dtype=bool
    )
    model.to(resolved_device)
    was_training = bool(model.training)
    model.eval()
    try:
        with torch.inference_mode():
            for batch_start in range(0, len(windows), batch_size):
                batch = windows[batch_start : batch_start + batch_size]
                # NumPy list of (C,Dp,Hp,Z) -> tensor (B,C,Dp,Hp,Z).
                inputs = torch.from_numpy(
                    np.stack([window.inputs for window in batch], axis=0)
                ).to(resolved_device, non_blocking=True)
                with torch.autocast(
                    device_type=resolved_device.type,
                    enabled=use_amp and resolved_device.type == "cuda",
                ):
                    prediction_log = rain_prediction_from_output(model(inputs))
                if prediction_log.shape != (inputs.shape[0], 1, *inputs.shape[2:]):
                    raise ValueError(
                        "Stage-1 cascade output must be (B,1,Dp,Hp,Z)"
                    )
                if not bool(torch.isfinite(prediction_log).all()):
                    raise FloatingPointError("Stage-1 cascade prediction is non-finite")
                rain = torch.expm1(prediction_log.float().clamp_min(0.0)).cpu()
                for position, window in enumerate(batch):
                    reconstructor.add(
                        rain[position],
                        core_start=window.core_start,
                        core_length=window.core_length,
                        output_mask=window.output_mask,
                    )
                    support_reconstructor.add(
                        window.output_mask,
                        core_start=window.core_start,
                        core_length=window.core_length,
                    )
    finally:
        model.train(was_training)

    output_support = support_reconstructor.finalize()[0].astype(bool)
    mode = str(cfb_input_mode).strip().lower()
    input_support = valid & (~clutter if mode == "mask_below_cfb" else True)
    return Stage1CascadePrediction(
        rain_rate_mm_h=reconstructor.finalize()[0].astype(np.float32, copy=False),
        input_support=input_support.astype(bool, copy=False),
        output_support=output_support,
    )


__all__ = [
    "Stage1CascadePrediction",
    "Stage1CascadeWindow",
    "iter_stage1_cascade_windows",
    "predict_stage1_from_reflectivity_orbit",
]
