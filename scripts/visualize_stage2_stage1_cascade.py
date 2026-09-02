#!/usr/bin/env python3
"""Render directly comparable final-rain reports from cascade orbit bundles.

The input is ``orbit_manifest.json`` written by
``evaluate_stage2_stage1_cascade.py``.  Inference and plotting are deliberately
separate: expensive complete-orbit predictions are saved once, while height,
resolution, selected methods, and figure layout can be changed repeatedly.

Every method receives the same physical height, A-B scan, rain color scale,
error scale, evaluation masks, and scatter subsampling.  The layout follows
``plot_nc_sample_diagnostics.py``: geolocated horizontal maps, A-B vertical
sections, long-tail distributions, correlation, per-height statistics, a
Markdown figure index, PNG files, and one multi-page PDF per orbit.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm, Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from plot_nc_sample_diagnostics import (  # noqa: E402
    FigureWriter,
    RAIN_THRESHOLDS,
    cumulative_distance_km,
    select_profile_row,
)
from scripts.visualize_stage1_test_predictions import (  # noqa: E402
    paired_metrics,
    vertical_statistics,
)


EXPECTED_FORMAT = "stage2_stage1_cascade_orbit_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--height-km", type=float, default=2.0)
    parser.add_argument("--max-points", type=int, default=200_000)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--count", type=int)
    parser.add_argument("--modes", nargs="+", help="Optional mode slugs to plot.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _rain_cmap():
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_under("white")
    cmap.set_bad("#bdbdbd")
    return cmap


def _rain_for_plot(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return np.where(np.isfinite(array) & (array == 0.0), 0.05, array)


def _shared_rain_norm(
    target: np.ndarray,
    predictions: Sequence[np.ndarray],
    target_mask: np.ndarray,
    prediction_masks: Sequence[np.ndarray],
) -> LogNorm:
    pieces = [target[target_mask & np.isfinite(target) & (target > 0.0)]]
    pieces.extend(
        prediction[mask & np.isfinite(prediction) & (prediction > 0.0)]
        for prediction, mask in zip(predictions, prediction_masks)
    )
    nonempty = [piece for piece in pieces if piece.size]
    values = np.concatenate(nonempty) if nonempty else np.empty(0)
    upper = max(20.0, float(np.percentile(values, 99.5))) if values.size else 20.0
    return LogNorm(vmin=0.1, vmax=upper)


def _shared_error_limit(
    target: np.ndarray,
    predictions: Sequence[np.ndarray],
    evaluation_mask: np.ndarray,
) -> float:
    values = [
        np.abs(prediction[evaluation_mask] - target[evaluation_mask])
        for prediction in predictions
    ]
    nonempty = [value[np.isfinite(value)] for value in values if value.size]
    combined = np.concatenate(nonempty) if nonempty else np.empty(0)
    return max(1.0, float(np.percentile(combined, 99.0))) if combined.size else 1.0


def compute_shared_geographic_extent(
    lon: np.ndarray,
    lat: np.ndarray,
    qc_footprint: np.ndarray,
    *,
    padding_fraction: float = 0.02,
) -> tuple[float, float, float, float]:
    """Return one ``(lon_min,lon_max,lat_min,lat_max)`` for every map panel.

    The extent is derived from the common two-dimensional QC footprint rather
    than from each method's output support.  Consequently a conservative or
    spatially shifted prediction cannot silently zoom its own panel.
    """

    x = np.asarray(lon, dtype=np.float64)
    y = np.asarray(lat, dtype=np.float64)
    footprint = np.asarray(qc_footprint)
    if x.ndim != 2 or y.shape != x.shape:
        raise ValueError("lon and lat must share a two-dimensional shape")
    if footprint.shape != x.shape or footprint.dtype != np.bool_:
        raise TypeError("qc_footprint must be boolean and match lon/lat")
    if not np.isfinite(padding_fraction) or padding_fraction < 0.0:
        raise ValueError("padding_fraction must be finite and non-negative")
    valid = footprint & np.isfinite(x) & np.isfinite(y)
    if not np.any(valid):
        valid = np.isfinite(x) & np.isfinite(y)
    if not np.any(valid):
        raise ValueError("no finite geolocation exists for a common map extent")
    x_min, x_max = float(x[valid].min()), float(x[valid].max())
    y_min, y_max = float(y[valid].min()), float(y[valid].max())
    # Degenerate synthetic/narrow swaths still receive a visible finite range.
    x_span = max(x_max - x_min, 1e-3)
    y_span = max(y_max - y_min, 1e-3)
    return (
        x_min - padding_fraction * x_span,
        x_max + padding_fraction * x_span,
        y_min - padding_fraction * y_span,
        y_max + padding_fraction * y_span,
    )


def _support_contour(
    ax,
    lon: np.ndarray,
    lat: np.ndarray,
    support: np.ndarray | None,
    *,
    color: str,
    linestyle: str,
    label: str,
) -> Line2D | None:
    """Overlay a geolocated support boundary and return a legend handle."""

    if support is None:
        return None
    mask = np.asarray(support)
    if mask.shape != lon.shape or mask.dtype != np.bool_:
        raise TypeError("support contour must be boolean and match lon/lat")
    coordinate_valid = np.isfinite(lon) & np.isfinite(lat)
    present = mask & coordinate_valid
    if not np.any(present):
        return None
    if np.all(coordinate_valid):
        # A contour needs both sides of the 0.5 level.  For all-true masks the
        # swath boundary itself is already represented by the QC footprint.
        if np.any(mask) and np.any(~mask):
            ax.contour(
                lon,
                lat,
                mask.astype(np.float32),
                levels=[0.5],
                colors=[color],
                linestyles=[linestyle],
                linewidths=[1.25],
                zorder=5,
            )
    else:
        # Rare invalid geolocation points prevent Matplotlib's curvilinear
        # contour.  Plot only the one-cell inner boundary as a robust fallback.
        interior = mask.copy()
        for axis_index in (0, 1):
            interior &= np.roll(mask, 1, axis=axis_index)
            interior &= np.roll(mask, -1, axis=axis_index)
        boundary = present & ~interior
        ax.plot(
            lon[boundary],
            lat[boundary],
            linestyle="none",
            marker=".",
            markersize=1.5,
            color=color,
            zorder=5,
        )
    return Line2D([0], [0], color=color, linestyle=linestyle, linewidth=1.4, label=label)


def _add_swath(
    ax,
    lon: np.ndarray,
    lat: np.ndarray,
    values: np.ndarray,
    title: str,
    *,
    cmap,
    norm,
    qc_footprint: np.ndarray,
    geographic_extent: tuple[float, float, float, float],
    true_support: np.ndarray | None = None,
    predicted_support: np.ndarray | None = None,
):
    coordinate_valid = np.isfinite(lon) & np.isfinite(lat)
    footprint = np.asarray(qc_footprint)
    if footprint.shape != lon.shape or footprint.dtype != np.bool_:
        raise TypeError("qc_footprint must be boolean and match lon/lat")
    background = coordinate_valid & footprint
    # Gray points explicitly expose the common scorable orbit footprint.  Data
    # values are then drawn on top; valid zero rain remains white via cmap.set_under.
    ax.scatter(
        lon[background],
        lat[background],
        c="#d0d0d0",
        s=4,
        edgecolors="none",
        rasterized=True,
        zorder=0,
    )
    valid = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(values)
    artist = ax.scatter(
        lon[valid],
        lat[valid],
        c=values[valid],
        s=3,
        cmap=cmap,
        norm=norm,
        edgecolors="none",
        rasterized=True,
        zorder=2,
    )
    handles = [
        handle
        for handle in (
            _support_contour(
                ax,
                lon,
                lat,
                true_support,
                color="#00b4d8",
                linestyle="--",
                label="True DPR support",
            ),
            _support_contour(
                ax,
                lon,
                lat,
                predicted_support,
                color="#ff7f0e",
                linestyle="-",
                label="Method input support",
            ),
        )
        if handle is not None
    ]
    if handles:
        ax.legend(handles=handles, loc="lower right", fontsize=7, framealpha=0.85)
    ax.set_facecolor("#eeeeee")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(geographic_extent[0], geographic_extent[1])
    ax.set_ylim(geographic_extent[2], geographic_extent[3])
    ax.set_title(title)
    return artist


def _add_section(
    ax,
    values: np.ndarray,
    z: np.ndarray,
    distance: np.ndarray,
    title: str,
    *,
    cmap,
    norm,
    true_support: np.ndarray | None = None,
    predicted_support: np.ndarray | None = None,
):
    artist = ax.pcolormesh(
        distance,
        z,
        np.ma.masked_invalid(values),
        shading="auto",
        cmap=cmap,
        norm=norm,
        rasterized=True,
    )
    ax.set_facecolor("#bdbdbd")
    ax.set_xlabel("Distance along A-B (km)")
    ax.set_ylabel("Height (km)")
    ax.set_title(title)
    handles: list[Line2D] = []
    for support, color, linestyle, label in (
        (true_support, "#00b4d8", "--", "True DPR support"),
        (predicted_support, "#ff7f0e", "-", "Method input support"),
    ):
        if support is None:
            continue
        mask = np.asarray(support)
        if mask.shape != values.shape or mask.dtype != np.bool_:
            raise TypeError("section support must be boolean and match values")
        if np.any(mask) and np.any(~mask):
            ax.contour(
                distance,
                z,
                mask.astype(np.float32),
                levels=[0.5],
                colors=[color],
                linestyles=[linestyle],
                linewidths=[1.0],
            )
            handles.append(
                Line2D([0], [0], color=color, linestyle=linestyle, label=label)
            )
    if handles:
        ax.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.85)
    return artist


def _display(value: Any, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    numeric = float(value)
    return f"{numeric:.{digits}f}" if math.isfinite(numeric) else "N/A"


def _subsample_pair(
    x: np.ndarray, y: np.ndarray, maximum: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    if x.size <= maximum:
        return x, y
    index = rng.choice(x.size, maximum, replace=False)
    return x[index], y[index]


def plot_mode_vs_target(
    *,
    target: np.ndarray,
    prediction: np.ndarray,
    qc_label_mask: np.ndarray,
    reliable_positive_mask: np.ndarray,
    true_dpr_support: np.ndarray,
    input_support: np.ndarray,
    output_support: np.ndarray,
    z: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    sample_id: str,
    mode_name: str,
    height_index: int,
    ab_scan: int,
    rain_norm: LogNorm,
    error_limit: float,
    qc_footprint: np.ndarray,
    geographic_extent: tuple[float, float, float, float],
    max_points: int,
    rng: np.random.Generator,
) -> tuple[plt.Figure, dict[str, Any]]:
    """Create one identical 3x3 target/prediction diagnostic layout."""

    target_plot = np.where(qc_label_mask, target, np.nan)
    prediction_plot = np.where(output_support, prediction, np.nan)
    error_plot = np.where(qc_label_mask, prediction - target, np.nan)
    cmap = _rain_cmap()
    fig, axes = plt.subplots(3, 3, figsize=(20, 17))
    level = float(z[height_index])
    target_artist = _add_swath(
        axes[0, 0],
        lon,
        lat,
        _rain_for_plot(target_plot[..., height_index]),
        f"Satellite pre_dpr at {level:.3f} km",
        cmap=cmap,
        norm=rain_norm,
        qc_footprint=qc_footprint,
        geographic_extent=geographic_extent,
        true_support=true_dpr_support[..., height_index],
    )
    prediction_artist = _add_swath(
        axes[0, 1],
        lon,
        lat,
        _rain_for_plot(prediction_plot[..., height_index]),
        f"{mode_name} at {level:.3f} km",
        cmap=cmap,
        norm=rain_norm,
        qc_footprint=qc_footprint,
        geographic_extent=geographic_extent,
        true_support=true_dpr_support[..., height_index],
        predicted_support=input_support[..., height_index],
    )
    error_artist = _add_swath(
        axes[0, 2],
        lon,
        lat,
        error_plot[..., height_index],
        "Prediction - satellite pre_dpr",
        cmap="coolwarm",
        norm=Normalize(-error_limit, error_limit),
        qc_footprint=qc_footprint,
        geographic_extent=geographic_extent,
        true_support=true_dpr_support[..., height_index],
        predicted_support=input_support[..., height_index],
    )
    for axis in axes[0]:
        valid_line = np.isfinite(lat[ab_scan]) & np.isfinite(lon[ab_scan])
        axis.plot(
            lon[ab_scan, valid_line],
            lat[ab_scan, valid_line],
            color="white",
            linewidth=1.2,
        )
        axis.text(0.02, 0.98, "A", transform=axis.transAxes, va="top", color="white", weight="bold")
        axis.text(0.95, 0.98, "B", transform=axis.transAxes, va="top", color="white", weight="bold")
    fig.colorbar(target_artist, ax=axes[0, 0], shrink=0.78, label="Rain rate (mm/h)")
    fig.colorbar(prediction_artist, ax=axes[0, 1], shrink=0.78, label="Rain rate (mm/h)")
    fig.colorbar(error_artist, ax=axes[0, 2], shrink=0.78, label="Error (mm/h)")

    line_valid = np.isfinite(lat[ab_scan]) & np.isfinite(lon[ab_scan])
    distance = cumulative_distance_km(lat[ab_scan, line_valid], lon[ab_scan, line_valid])
    sections = (
        (
            _rain_for_plot(target_plot[ab_scan, line_valid, :].T),
            "Satellite pre_dpr A-B section",
            cmap,
            rain_norm,
            true_dpr_support[ab_scan, line_valid, :].T,
            None,
        ),
        (
            _rain_for_plot(prediction_plot[ab_scan, line_valid, :].T),
            f"{mode_name} A-B section",
            cmap,
            rain_norm,
            true_dpr_support[ab_scan, line_valid, :].T,
            input_support[ab_scan, line_valid, :].T,
        ),
        (
            error_plot[ab_scan, line_valid, :].T,
            "A-B error section",
            "coolwarm",
            Normalize(-error_limit, error_limit),
            true_dpr_support[ab_scan, line_valid, :].T,
            input_support[ab_scan, line_valid, :].T,
        ),
    )
    for axis, (
        values,
        title,
        section_cmap,
        section_norm,
        true_section_support,
        predicted_section_support,
    ) in zip(axes[1], sections):
        artist = _add_section(
            axis,
            values,
            z,
            distance,
            f"{title}; scan={ab_scan}",
            cmap=section_cmap,
            norm=section_norm,
            true_support=true_section_support,
            predicted_support=predicted_section_support,
        )
        fig.colorbar(artist, ax=axis, shrink=0.78, label="mm/h")

    positive_metrics = paired_metrics(target, prediction, reliable_positive_mask)
    label_metrics = paired_metrics(target, prediction, qc_label_mask)
    x = target[reliable_positive_mask]
    y = prediction[reliable_positive_mask]
    x, y = _subsample_pair(x, y, max_points, rng)
    positive_pair = (x > 0.0) | (y > 0.0)
    if np.any(positive_pair):
        axes[2, 0].hexbin(
            np.log1p(x[positive_pair]),
            np.log1p(y[positive_pair]),
            gridsize=65,
            bins="log",
            mincnt=1,
            cmap="viridis",
        )
        limit = max(
            float(np.log1p(x[positive_pair]).max()),
            float(np.log1p(y[positive_pair]).max()),
            1e-6,
        )
        axes[2, 0].plot([0, limit], [0, limit], "r--", linewidth=1)
        axes[2, 0].set_xlim(0, limit)
        axes[2, 0].set_ylim(0, limit)
    axes[2, 0].set_xlabel("Satellite log1p(rain rate)")
    axes[2, 0].set_ylabel("Prediction log1p(rain rate)")
    axes[2, 0].set_title(
        "Reliable positive-rain correlation\n"
        f"n={positive_metrics['count']:,}, RMSE={_display(positive_metrics['rmse'])}, "
        f"r={_display(positive_metrics['pearson_r'])}"
    )

    target_positive = target[reliable_positive_mask]
    predicted_positive = prediction[reliable_positive_mask]
    upper = max(
        float(target_positive.max()) if target_positive.size else 1.0,
        float(predicted_positive.max()) if predicted_positive.size else 1.0,
        1.0,
    )
    bins = np.geomspace(0.01, upper, 70)
    axes[2, 1].hist(
        target_positive[target_positive > 0.0],
        bins=bins,
        histtype="step",
        linewidth=2,
        label="Satellite pre_dpr",
    )
    axes[2, 1].hist(
        predicted_positive[predicted_positive > 0.0],
        bins=bins,
        histtype="step",
        linewidth=2,
        label=mode_name,
    )
    axes[2, 1].set_xscale("log")
    axes[2, 1].set_yscale("log")
    axes[2, 1].set_xlabel("Positive precipitation rate (mm/h)")
    axes[2, 1].set_ylabel("Voxel count")
    axes[2, 1].set_title("Positive-rain long-tail distribution")
    axes[2, 1].legend(fontsize=8)

    rmse, bias, correlation = vertical_statistics(
        target, prediction, reliable_positive_mask
    )
    axes[2, 2].plot(rmse, z, label="RMSE", color="#e76f51")
    axes[2, 2].plot(bias, z, label="bias", color="#457b9d")
    axes[2, 2].axvline(0.0, color="black", linewidth=0.7)
    axes[2, 2].set_xlabel("Error (mm/h)")
    axes[2, 2].set_ylabel("Height (km)")
    axes[2, 2].set_title("Reliable positive-rain error by height")
    correlation_axis = axes[2, 2].twiny()
    correlation_axis.plot(correlation, z, label="Pearson r", color="#2a9d8f")
    correlation_axis.set_xlabel("Pearson r")
    correlation_axis.set_xlim(-0.1, 1.0)
    axes[2, 2].legend(loc="lower left")
    correlation_axis.legend(loc="lower right")
    axes[2, 2].grid(alpha=0.25)

    fig.suptitle(
        f"Two-stage complete-orbit rain diagnostic: {sample_id}\n{mode_name}\n"
        f"positive RMSE={_display(positive_metrics['rmse'])}, "
        f"r={_display(positive_metrics['pearson_r'])}; "
        f"QC-label-domain RMSE={_display(label_metrics['rmse'])}",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig, {
        "mode_name": mode_name,
        "reliable_positive_metrics": positive_metrics,
        "qc_label_domain_metrics": label_metrics,
        "vertical_rmse": rmse.tolist(),
        "vertical_bias": bias.tolist(),
        "vertical_pearson_r": correlation.tolist(),
        "output_support_count": int(output_support.sum()),
        "geographic_extent": list(geographic_extent),
    }


def plot_all_methods_overview(
    *,
    target: np.ndarray,
    predictions: Sequence[np.ndarray],
    input_supports: Sequence[np.ndarray],
    output_supports: Sequence[np.ndarray],
    true_dpr_support: np.ndarray,
    mode_names: Sequence[str],
    qc_label_mask: np.ndarray,
    z: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    height_index: int,
    rain_norm: LogNorm,
    qc_footprint: np.ndarray,
    geographic_extent: tuple[float, float, float, float],
) -> plt.Figure:
    """Show target and every method with one shared scale at one height."""

    panels = [
        (
            np.where(qc_label_mask, target, np.nan),
            "Satellite pre_dpr",
            None,
        )
    ] + [
        (np.where(output_mask, prediction, np.nan), name, input_mask)
        for prediction, input_mask, output_mask, name in zip(
            predictions, input_supports, output_supports, mode_names
        )
    ]
    columns = 3
    rows = int(math.ceil(len(panels) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(19, 5.6 * rows), squeeze=False)
    artist = None
    for axis, (values, title, method_support) in zip(axes.flat, panels):
        artist = _add_swath(
            axis,
            lon,
            lat,
            _rain_for_plot(values[..., height_index]),
            title,
            cmap=_rain_cmap(),
            norm=rain_norm,
            qc_footprint=qc_footprint,
            geographic_extent=geographic_extent,
            true_support=true_dpr_support[..., height_index],
            predicted_support=(
                None if method_support is None else method_support[..., height_index]
            ),
        )
    for axis in axes.flat[len(panels) :]:
        axis.axis("off")
    if artist is not None:
        fig.colorbar(artist, ax=list(axes.flat[: len(panels)]), shrink=0.75, label="Rain rate (mm/h)")
    fig.suptitle(
        f"All cascade methods at the same height and scale: z={z[height_index]:.3f} km",
        fontsize=15,
    )
    fig.subplots_adjust(top=0.92, wspace=0.28, hspace=0.30)
    return fig


def plot_all_methods_long_tail(
    *,
    target: np.ndarray,
    predictions: Sequence[np.ndarray],
    mode_names: Sequence[str],
    reliable_positive_mask: np.ndarray,
) -> plt.Figure:
    """Match the source diagnostic's histogram/CCDF/exceedance emphasis."""

    reference = target[reliable_positive_mask]
    estimates = [prediction[reliable_positive_mask] for prediction in predictions]
    maximum = max(
        [float(reference.max()) if reference.size else 1.0]
        + [float(values.max()) if values.size else 1.0 for values in estimates]
        + [1.0]
    )
    bins = np.geomspace(0.01, maximum, 70)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes[0, 0].hist(reference[reference > 0.0], bins=bins, histtype="step", linewidth=2.4, label="Satellite pre_dpr")
    for values, name in zip(estimates, mode_names):
        axes[0, 0].hist(values[values > 0.0], bins=bins, histtype="step", linewidth=1.5, label=name)
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlabel("Positive precipitation rate (mm/h)")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title("Positive-rain long-tail distributions")
    axes[0, 0].legend(fontsize=7)

    for values, name, width in [(reference, "Satellite pre_dpr", 2.4)] + [
        (values, name, 1.5) for values, name in zip(estimates, mode_names)
    ]:
        positive = np.sort(values[values > 0.0])
        if positive.size:
            exceedance = 1.0 - np.arange(positive.size) / positive.size
            axes[0, 1].plot(positive, exceedance, linewidth=width, label=name)
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xlabel("Precipitation threshold (mm/h)")
    axes[0, 1].set_ylabel("P(R >= threshold | reliable positive target)")
    axes[0, 1].set_title("Complementary cumulative distributions")
    axes[0, 1].grid(alpha=0.25, which="both")

    labels = ["Satellite"] + list(mode_names)
    series = [reference] + estimates
    x = np.arange(RAIN_THRESHOLDS.size)
    width = 0.8 / len(series)
    for index, (values, label) in enumerate(zip(series, labels)):
        counts = np.asarray([(values > threshold).sum() for threshold in RAIN_THRESHOLDS])
        axes[1, 0].bar(x - 0.4 + width / 2 + index * width, counts, width, label=label)
    axes[1, 0].set_xticks(x, [f">{value:g}" for value in RAIN_THRESHOLDS], rotation=30)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_ylabel("Voxel count")
    axes[1, 0].set_title("Strong-rain exceedance counts")
    axes[1, 0].legend(fontsize=6)

    positions = np.arange(len(mode_names))
    metrics = [paired_metrics(target, prediction, reliable_positive_mask) for prediction in predictions]
    axes[1, 1].bar(positions - 0.2, [item["mae"] for item in metrics], 0.4, label="MAE")
    axes[1, 1].bar(positions + 0.2, [item["rmse"] for item in metrics], 0.4, label="RMSE")
    axes[1, 1].set_xticks(positions, mode_names, rotation=20, ha="right")
    axes[1, 1].set_ylabel("mm/h")
    axes[1, 1].set_title("Reliable positive-rain errors")
    axes[1, 1].legend()
    correlation_axis = axes[1, 1].twinx()
    correlation_axis.plot(positions, [item["pearson_r"] for item in metrics], "o-", color="#2a9d8f", label="Pearson r")
    correlation_axis.set_ylabel("Pearson r")
    correlation_axis.set_ylim(-0.1, 1.0)
    correlation_axis.legend(loc="upper right")

    fig.suptitle("Satellite precipitation tail and all cascade outputs", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def visualize_cascade_orbit_bundle(
    metadata_path: Path,
    *,
    output_dir: Path,
    height_km: float = 2.0,
    max_points: int = 200_000,
    dpi: int = 150,
    selected_modes: Sequence[str] | None = None,
    seed: int = 2026,
) -> dict[str, Any]:
    """Render one saved orbit; public for focused synthetic tests."""

    if max_points <= 0 or dpi <= 0:
        raise ValueError("max_points and dpi must be positive")
    metadata = _load_json(metadata_path)
    if metadata.get("format") != EXPECTED_FORMAT:
        raise ValueError("unsupported cascade orbit format")
    fields_path = metadata_path.parent / metadata["fields_file"]
    with np.load(fields_path, allow_pickle=False) as archive:
        fields = {name: archive[name] for name in archive.files}
    z = fields["heights_km"]
    target = fields["target_rain_mm_h"]
    qc_mask = fields["qc_label_mask"].astype(bool)
    positive_mask = fields["reliable_positive_mask"].astype(bool)
    lat, lon = fields["lat"], fields["lon"]
    modes = list(metadata["modes"])
    if selected_modes is not None:
        requested = list(dict.fromkeys(selected_modes))
        available = {item["slug"] for item in modes}
        missing = sorted(set(requested).difference(available))
        if missing:
            raise KeyError(f"unknown cascade modes: {missing}")
        modes = [item for item in modes if item["slug"] in requested]
    if not modes:
        raise ValueError("no cascade modes selected for visualization")
    predictions = [fields[item["rain_field"]] for item in modes]
    input_supports = [fields[item["input_support_field"]].astype(bool) for item in modes]
    output_supports = [fields[item["output_support_field"]].astype(bool) for item in modes]
    true_dpr_support = fields.get("true_dpr_support")
    if true_dpr_support is None:
        # Backward-compatible fallback for orbit bundles generated before C0.
        # Prefer the explicit true-DPR route; older synthetic fixtures often
        # named that route simply ``oracle``.
        oracle_mode = next(
            (
                item
                for item in metadata["modes"]
                if item.get("input_kind") in {"true_dpr_oracle", "true_dpr_true_support"}
                or item.get("slug") in {"dpr_oracle", "oracle"}
            ),
            None,
        )
        if oracle_mode is None:
            raise KeyError(
                "orbit bundle has no true_dpr_support or identifiable DPR-oracle mode"
            )
        true_dpr_support = fields[oracle_mode["input_support_field"]]
    true_dpr_support = np.asarray(true_dpr_support).astype(bool)
    shape = target.shape
    if target.ndim != 3 or any(
        value.shape != shape
        for value in predictions + input_supports + output_supports + [true_dpr_support]
    ):
        raise ValueError("target, predictions and supports must share (nscan,nray,z)")
    if qc_mask.shape != shape or positive_mask.shape != shape:
        raise ValueError("saved evaluation masks do not match the orbit")
    if lat.shape != shape[:2] or lon.shape != shape[:2] or z.shape != (shape[-1],):
        raise ValueError("saved coordinate shapes are inconsistent")

    height_index = int(np.argmin(np.abs(z - height_km)))
    ab_scan = select_profile_row(target, height_index)
    mode_names = [str(item["display_name"]) for item in modes]
    rain_norm = _shared_rain_norm(target, predictions, qc_mask, output_supports)
    error_limit = _shared_error_limit(target, predictions, qc_mask)
    qc_footprint = np.any(qc_mask, axis=-1)
    geographic_extent = compute_shared_geographic_extent(
        lon, lat, qc_footprint
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = FigureWriter(output_dir, dpi)
    records: dict[str, Any] = {}
    try:
        writer.save(
            plot_all_methods_overview(
                target=target,
                predictions=predictions,
                input_supports=input_supports,
                output_supports=output_supports,
                true_dpr_support=true_dpr_support,
                mode_names=mode_names,
                qc_label_mask=qc_mask,
                z=z,
                lat=lat,
                lon=lon,
                height_index=height_index,
                rain_norm=rain_norm,
                qc_footprint=qc_footprint,
                geographic_extent=geographic_extent,
            ),
            "comparisons/00_all_methods_overview.png",
            "All final-rain fields at one identical height and color scale.",
        )
        for position, (mode, prediction, input_support, output_support) in enumerate(
            zip(modes, predictions, input_supports, output_supports), start=1
        ):
            figure, result = plot_mode_vs_target(
                target=target,
                prediction=prediction,
                qc_label_mask=qc_mask,
                reliable_positive_mask=positive_mask,
                true_dpr_support=true_dpr_support,
                input_support=input_support,
                output_support=output_support,
                z=z,
                lat=lat,
                lon=lon,
                sample_id=str(metadata["sample_id"]),
                mode_name=str(mode["display_name"]),
                height_index=height_index,
                ab_scan=ab_scan,
                rain_norm=rain_norm,
                error_limit=error_limit,
                qc_footprint=qc_footprint,
                geographic_extent=geographic_extent,
                max_points=max_points,
                rng=np.random.default_rng(seed + position),
            )
            relative = f"comparisons/{position:02d}_{mode['slug']}_vs_pre_dpr.png"
            writer.save(
                figure,
                relative,
                "Final orbit prediction versus satellite pre_dpr using the shared layout.",
            )
            records[str(mode["slug"])] = {"figure": relative, **result}
        writer.save(
            plot_all_methods_long_tail(
                target=target,
                predictions=predictions,
                mode_names=mode_names,
                reliable_positive_mask=positive_mask,
            ),
            "comparisons/90_all_methods_rain_tail.png",
            "Satellite and cascade positive-rain distributions, CCDF, exceedance counts and metrics.",
        )
    finally:
        writer.close()

    result = {
        "format": EXPECTED_FORMAT,
        "sample_id": metadata["sample_id"],
        "source_file": metadata["source_file"],
        "height_index": height_index,
        "height_km": float(z[height_index]),
        "ab_scan_index": ab_scan,
        "shared_rain_vmin_mm_h": float(rain_norm.vmin),
        "shared_rain_vmax_mm_h": float(rain_norm.vmax),
        "shared_error_limit_mm_h": error_limit,
        "shared_geographic_extent_lon_lat": list(geographic_extent),
        "qc_footprint_count": int(qc_footprint.sum()),
        "true_dpr_support_count": int(true_dpr_support.sum()),
        "modes": records,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(_json_safe(result), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# 双阶段串联轨道诊断图索引",
        "",
        f"- 样本：`{metadata['source_file']}`",
        f"- 水平切片：`z[{height_index}]={z[height_index]:.3f} km`",
        f"- 所有方法共用 A–B 扫描行：`{ab_scan}`",
        f"- 所有方法共用降水色标：`{rain_norm.vmin:g}–{rain_norm.vmax:.3f} mm/h`",
        f"- 所有方法共用误差范围：`±{error_limit:.3f} mm/h`",
        (
            "- 所有水平面板共用经纬度范围："
            f"`lon={geographic_extent[0]:.4f}–{geographic_extent[1]:.4f}, "
            f"lat={geographic_extent[2]:.4f}–{geographic_extent[3]:.4f}`"
        ),
        "- 浅灰点表示公共QC轨道底图；白色表示有效零降水。",
        "- 青色虚线表示真实DPR support，橙色实线表示当前方法送入Stage 1的support。",
        "",
        "## 图表",
        "",
    ]
    for relative, description in writer.entries:
        lines.extend(
            [
                f"### `{relative}`",
                "",
                description,
                "",
                f"![{relative}]({relative})",
                "",
            ]
        )
    (output_dir / "figure_manifest.md").write_text("\n".join(lines), encoding="utf-8")
    (output_dir / ".complete").write_text(
        f"sample_id={metadata['sample_id']}\nfigures={len(writer.entries)}\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    args = parse_args()
    if args.dpi <= 0 or args.max_points <= 0:
        raise ValueError("dpi and max-points must be positive")
    if args.count is not None and args.count <= 0:
        raise ValueError("count must be positive")
    input_dir = args.input_dir.expanduser().resolve()
    manifest = _load_json(input_dir / "orbit_manifest.json")
    if manifest.get("format") != EXPECTED_FORMAT:
        raise ValueError("unsupported cascade orbit manifest format")
    records = list(manifest["orbits"])
    if args.count is not None:
        records = records[: args.count]
    output_root = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else input_dir / "visualizations"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    for position, record in enumerate(records, start=1):
        destination = output_root / str(record["sample_id"])
        if destination.exists() and (destination / ".complete").is_file() and not args.overwrite:
            print(f"[skip {position}/{len(records)}] {record['sample_id']}", flush=True)
            continue
        result = visualize_cascade_orbit_bundle(
            input_dir / record["metadata"],
            output_dir=destination,
            height_km=args.height_km,
            max_points=args.max_points,
            dpi=args.dpi,
            selected_modes=args.modes,
            seed=2026 + int(record["file_id"]),
        )
        summaries.append(result)
        print(f"[visualize {position}/{len(records)}] {record['sample_id']}", flush=True)
    payload = {
        "source_manifest": str(input_dir / "orbit_manifest.json"),
        "height_km_requested": args.height_km,
        "selected_modes": args.modes,
        "processed_orbit_count": len(summaries),
        "orbits": summaries,
    }
    (output_root / "summary.json").write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Cascade visual reports -> {output_root}", flush=True)


if __name__ == "__main__":
    main()
