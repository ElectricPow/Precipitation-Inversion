#!/usr/bin/env python3
"""Create reproducible visual diagnostic reports for GR-DPR NetCDF samples.

The script writes per-variable figures, thematic comparison figures, a CSV table,
a Markdown figure index, and one multi-page PDF. Source NetCDF files are read-only.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm, Normalize
from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import PdfPages
from netCDF4 import Dataset

from zrh_nc_to_rain import load_zrh_parameters


PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = Path(
    "/storage/GR_DPR_3D/GRToDPRRes_V07_Pct_V1.2.1_sw_260412"
)
DEFAULT_MODEL_PATH = PROJECT_DIR / "ZRH_37refine.pth"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs" / "nc_sample_diagnostics"

RAIN_THRESHOLDS = np.asarray([0.1, 0.5, 1, 2, 5, 10, 20, 30, 50])
TYPE_LABELS = {-1111: "No precipitation", 1: "Stratiform", 2: "Convective", 3: "Other"}
CATEGORICAL_VARIABLES = {"cfb", "binRealSurface", "typePrecip", "flagPrecip"}
EXTERNAL_UNITS = {
    "p": "hPa",
    "t": "K",
    "q": "kg/kg",
    "lat": "degree_north",
    "lon": "degree_east",
    "nsrr_dpr": "mm/h",
    "srr_dpr": "mm/h",
    "cfb": "z index",
    "binRealSurface": "native DPR bin index",
    "typePrecip": "category",
    "flagPrecip": "category",
    "scan_id": "index",
}
SECTION_NAMES = (
    "overview",
    "variables",
    "gr_interp",
    "gr_dpr",
    "strong_window",
    "cfb",
    "precip_types",
    "rain_tail",
    "zrh",
)


@dataclass(frozen=True)
class VariableInfo:
    name: str
    dimensions: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: str
    units: str
    long_name: str


@dataclass
class SampleData:
    path: Path
    dimensions: dict[str, int]
    variables: dict[str, np.ndarray]
    info: dict[str, VariableInfo]

    @property
    def z(self) -> np.ndarray:
        return self.variables["z"]

    @property
    def lat(self) -> np.ndarray:
        return self.variables["lat"]

    @property
    def lon(self) -> np.ndarray:
        return self.variables["lon"]


class FigureWriter:
    """Save every figure as a PNG and append it to one PDF."""

    def __init__(self, output_dir: Path, dpi: int) -> None:
        self.output_dir = output_dir
        self.dpi = dpi
        self.pdf = PdfPages(output_dir / "diagnostics.pdf")
        self.entries: list[tuple[str, str]] = []

    def save(self, fig: Figure, relative_path: str, description: str) -> None:
        output_path = self.output_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        self.pdf.savefig(fig, bbox_inches="tight")
        self.entries.append((relative_path, description))
        plt.close(fig)
        print(f"OK figure: {relative_path}")

    def close(self) -> None:
        self.pdf.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot per-variable and thematic diagnostics for GR-DPR NetCDF samples."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DATASET_DIR,
        help="Directory containing NetCDF samples (default: current GR-DPR dataset).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=20,
        help="Number of files taken from the sorted input directory (default: 20).",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        help="Analyze exactly one file instead of selecting files from --input-dir.",
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--height-km", type=float, default=2.0)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--max-points", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument(
        "--sections",
        nargs="+",
        choices=("all",) + SECTION_NAMES,
        default=("all",),
        help="Figure groups to create (default: all).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate non-empty sample result directories; otherwise they are skipped.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first failed sample instead of continuing the batch.",
    )
    return parser.parse_args()


def normalized_array(variable, name: str) -> np.ndarray:
    """Read one variable as float64 while preserving meaningful category codes."""
    array = np.ma.filled(variable[:], np.nan).astype(np.float64, copy=False)
    if name == "typePrecip":
        array[array == -9999] = np.nan
    elif name not in CATEGORICAL_VARIABLES:
        array[array < -9990] = np.nan
    return array


def load_sample(path: Path) -> SampleData:
    if not path.is_file():
        raise FileNotFoundError(f"NetCDF sample not found: {path}")
    variables: dict[str, np.ndarray] = {}
    info: dict[str, VariableInfo] = {}
    with Dataset(path, "r") as dataset:
        dimensions = {name: len(dim) for name, dim in dataset.dimensions.items()}
        for name, variable in dataset.variables.items():
            variables[name] = normalized_array(variable, name)
            units = str(getattr(variable, "units", EXTERNAL_UNITS.get(name, "—")))
            info[name] = VariableInfo(
                name=name,
                dimensions=tuple(variable.dimensions),
                shape=tuple(variable.shape),
                dtype=str(variable.dtype),
                units=units,
                long_name=str(getattr(variable, "long_name", name)),
            )
    required = {"z", "lat", "lon", "dbz_gr_sparse", "dbz_gr_interp", "dbz_dpr", "pre_dpr"}
    missing = sorted(required.difference(variables))
    if missing:
        raise KeyError(f"Sample is missing required variables: {missing}")
    return SampleData(path=path, dimensions=dimensions, variables=variables, info=info)


def nearest_height_index(z: np.ndarray, target_km: float) -> int:
    valid = np.isfinite(z)
    if not valid.any():
        raise ValueError("The z coordinate contains no finite values")
    candidates = np.flatnonzero(valid)
    return int(candidates[np.argmin(np.abs(z[valid] - target_km))])


def finite_values(array: np.ndarray) -> np.ndarray:
    return array[np.isfinite(array)]


def sampled(values: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    values = np.asarray(values)
    if values.size <= max_points:
        return values
    indices = rng.choice(values.size, size=max_points, replace=False)
    return values[indices]


def variable_statistics(data: SampleData) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name, array in data.variables.items():
        values = finite_values(array)
        total = int(array.size)
        valid = int(values.size)
        row: dict[str, object] = {
            "variable": name,
            "dimensions": ",".join(data.info[name].dimensions),
            "shape": "×".join(str(size) for size in array.shape),
            "dtype": data.info[name].dtype,
            "units": data.info[name].units,
            "total": total,
            "valid": valid,
            "missing": total - valid,
            "valid_ratio": valid / total if total else np.nan,
            "zero_count": int(np.count_nonzero(values == 0)),
            "positive_count": int(np.count_nonzero(values > 0)),
        }
        if valid:
            quantiles = np.percentile(values, [0, 1, 25, 50, 75, 99, 100])
            row.update(
                {
                    "min": quantiles[0],
                    "q01": quantiles[1],
                    "q25": quantiles[2],
                    "median": quantiles[3],
                    "q75": quantiles[4],
                    "q99": quantiles[5],
                    "max": quantiles[6],
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                }
            )
        else:
            row.update({key: np.nan for key in ("min", "q01", "q25", "median", "q75", "q99", "max", "mean", "std")})
        rows.append(row)
    return rows


def write_statistics_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    fieldnames = list(rows[0])
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def select_profile_row(pre_dpr: np.ndarray, height_index: int) -> int:
    rain = pre_dpr[:, :, height_index]
    score = (np.isfinite(rain) & (rain > 0.1)).sum(axis=1)
    if score.max() > 0:
        return int(np.argmax(score))
    return int(np.argmax(np.isfinite(rain).sum(axis=1)))


def cumulative_distance_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    radius_km = 6371.0
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    dlat = np.diff(lat_rad)
    dlon = np.diff(lon_rad)
    value = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat_rad[:-1]) * np.cos(lat_rad[1:]) * np.sin(dlon / 2) ** 2
    )
    segments = 2 * radius_km * np.arcsin(np.sqrt(np.clip(value, 0, 1)))
    return np.r_[0.0, np.cumsum(segments)]


def robust_limits(values: np.ndarray, lower: float = 1, upper: float = 99) -> tuple[float, float]:
    finite = finite_values(values)
    if not finite.size:
        return 0.0, 1.0
    vmin, vmax = np.percentile(finite, [lower, upper])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        center = float(vmin) if np.isfinite(vmin) else 0.0
        return center - 0.5, center + 0.5
    return float(vmin), float(vmax)


def is_rain_variable(name: str) -> bool:
    return name in {"pre_dpr", "nsrr_dpr", "srr_dpr"} or name.startswith("rain_rate")


def is_dbz_variable(name: str) -> bool:
    return name.startswith("dbz_")


def color_style(name: str, values: np.ndarray):
    if is_dbz_variable(name):
        return "turbo", Normalize(vmin=0, vmax=50)
    if is_rain_variable(name):
        positive = values[np.isfinite(values) & (values > 0)]
        vmax = max(20.0, float(np.percentile(positive, 99.5))) if positive.size else 20.0
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_under("white")
        cmap.set_bad("#bdbdbd")
        return cmap, LogNorm(vmin=0.1, vmax=vmax)
    vmin, vmax = robust_limits(values)
    return "viridis", Normalize(vmin=vmin, vmax=vmax)


def rain_for_plot(values: np.ndarray, name: str) -> np.ndarray:
    if not is_rain_variable(name):
        return values
    return np.where(np.isfinite(values) & (values == 0), 0.05, values)


def add_swath_map(
    ax,
    data: SampleData,
    values: np.ndarray,
    name: str,
    title: str,
    size: float = 2.0,
):
    valid = np.isfinite(values) & np.isfinite(data.lat) & np.isfinite(data.lon)
    cmap, norm = color_style(name, values)
    shown = rain_for_plot(values, name)
    artist = ax.scatter(
        data.lon[valid], data.lat[valid], c=shown[valid], s=size,
        cmap=cmap, norm=norm, edgecolors="none", rasterized=True,
    )
    ax.set_facecolor("#eeeeee")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    return artist


def add_cross_section(
    ax,
    data: SampleData,
    values_3d: np.ndarray,
    name: str,
    profile_row: int,
    title: str,
):
    coordinate_ok = np.isfinite(data.lat[profile_row]) & np.isfinite(data.lon[profile_row])
    distance = cumulative_distance_km(
        data.lat[profile_row, coordinate_ok], data.lon[profile_row, coordinate_ok]
    )
    profile = values_3d[profile_row, coordinate_ok, :].T
    cmap, norm = color_style(name, values_3d)
    shown = rain_for_plot(profile, name)
    artist = ax.pcolormesh(
        distance, data.z, np.ma.masked_invalid(shown), cmap=cmap, norm=norm,
        shading="auto", rasterized=True,
    )
    ax.set_facecolor("#bdbdbd")
    ax.set_xlabel("Distance along A-B (km)")
    ax.set_ylabel("Height (km)")
    ax.set_title(title)
    return artist


def add_histogram(
    ax, values: np.ndarray, name: str, rng: np.random.Generator, max_points: int
) -> None:
    finite = finite_values(values)
    if not finite.size:
        ax.text(0.5, 0.5, "No valid values", ha="center", va="center")
        return
    if is_rain_variable(name):
        positive = sampled(finite[finite > 0], max_points, rng)
        if positive.size:
            lo = max(float(positive.min()), 1e-3)
            hi = max(float(positive.max()), lo * 1.01)
            bins = np.geomspace(lo, hi, 50)
            ax.hist(positive, bins=bins, color="#277da1", alpha=0.85)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Positive precipitation rate (mm/h)")
        zero_ratio = np.mean(finite == 0)
        ax.set_title(f"Positive-value distribution; zero={zero_ratio:.2%}")
    elif name in CATEGORICAL_VARIABLES:
        unique, counts = np.unique(finite, return_counts=True)
        order = np.argsort(counts)[::-1][:30]
        unique, counts = unique[order], counts[order]
        ax.bar(np.arange(unique.size), counts, color="#577590")
        ax.set_xticks(np.arange(unique.size), [f"{value:g}" for value in unique], rotation=60)
        ax.set_yscale("log" if counts.max() / max(counts.min(), 1) > 100 else "linear")
        ax.set_title("Category counts")
    else:
        selected = sampled(finite, max_points, rng)
        ax.hist(selected, bins=60, color="#277da1", alpha=0.85)
        ax.set_title("Value distribution")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.2)


def format_number(value: float) -> str:
    if not np.isfinite(value):
        return "—"
    if abs(value) >= 1000 or (0 < abs(value) < 0.001):
        return f"{value:.3e}"
    return f"{value:.4g}"


def statistics_text(name: str, values: np.ndarray, info: VariableInfo) -> str:
    finite = finite_values(values)
    lines = [
        f"name: {name}",
        f"long_name: {info.long_name}",
        f"dimensions: {info.dimensions}",
        f"shape: {info.shape}",
        f"dtype: {info.dtype}",
        f"units: {info.units}",
        f"valid: {finite.size:,}/{values.size:,} ({finite.size / values.size:.2%})",
    ]
    if finite.size:
        q = np.percentile(finite, [0, 1, 25, 50, 75, 99, 100])
        lines.extend(
            [
                f"min / max: {format_number(q[0])} / {format_number(q[-1])}",
                f"P01 / P50 / P99: {format_number(q[1])} / {format_number(q[3])} / {format_number(q[5])}",
                f"mean ± std: {format_number(float(finite.mean()))} ± {format_number(float(finite.std()))}",
                "sample values: " + ", ".join(format_number(v) for v in finite[:8]),
            ]
        )
    return "\n".join(lines)


def plot_dataset_overview(data: SampleData, rows: list[dict[str, object]]) -> Figure:
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), gridspec_kw={"width_ratios": [1.6, 1]})
    names = [str(row["variable"]) for row in rows]
    ratios = np.asarray([float(row["valid_ratio"]) for row in rows])
    colors = ["#43aa8b" if ratio >= 0.95 else "#f9c74f" if ratio >= 0.1 else "#f94144" for ratio in ratios]
    y = np.arange(len(names))
    axes[0].barh(y, ratios * 100, color=colors)
    axes[0].set_yticks(y, names)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 100)
    axes[0].set_xlabel("Valid values (%)")
    axes[0].set_title("Valid-data coverage by variable")
    axes[0].grid(axis="x", alpha=0.25)
    for index, ratio in enumerate(ratios):
        axes[0].text(min(ratio * 100 + 1, 96), index, f"{ratio:.2%}", va="center", fontsize=8)

    axes[1].axis("off")
    variable_groups = {
        "Coordinates": ["z", "lat", "lon", "time", "scan_id"],
        "GR": [name for name in names if "_gr_" in name],
        "DPR": ["dbz_dpr", "pre_dpr", "nsrr_dpr", "srr_dpr"],
        "Quality/classes": ["cfb", "binRealSurface", "typePrecip", "flagPrecip"],
        "Meteorology": ["p", "t", "q"],
    }
    text = [
        f"File: {data.path.name}",
        f"Dimensions: {data.dimensions}",
        f"Variables: {len(names)}",
        "",
    ]
    for group, members in variable_groups.items():
        text.append(f"{group} ({len(members)}):")
        text.append("  " + ", ".join(members))
        text.append("")
    axes[1].text(0, 1, "\n".join(text), va="top", family="monospace", fontsize=10)
    fig.suptitle("GR-DPR NetCDF sample overview", fontsize=15)
    fig.tight_layout()
    return fig


def plot_1d_variable(data: SampleData, name: str, rng, max_points: int) -> Figure:
    values = data.variables[name]
    info = data.info[name]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes[0, 0].plot(np.arange(values.size), values, color="#277da1", linewidth=1.2)
    axes[0, 0].set_xlabel("Array index")
    axes[0, 0].set_ylabel(info.units)
    axes[0, 0].set_title("Actual values by index")
    axes[0, 0].grid(alpha=0.25)
    add_histogram(axes[0, 1], values, name, rng, max_points)
    finite_mask = np.isfinite(values)
    axes[1, 0].bar(["valid", "missing"], [finite_mask.sum(), (~finite_mask).sum()], color=["#43aa8b", "#bdbdbd"])
    axes[1, 0].set_title("Validity counts")
    axes[1, 1].axis("off")
    axes[1, 1].text(0, 1, statistics_text(name, values, info), va="top", family="monospace")
    fig.suptitle(f"Variable diagnostic: {name}", fontsize=15)
    fig.tight_layout()
    return fig


def categorical_style(name: str, values: np.ndarray):
    finite = finite_values(values)
    unique = np.unique(finite)
    if name == "typePrecip":
        unique = np.asarray([-1111, 1, 2, 3], dtype=float)
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, max(unique.size, 1)))
    cmap = ListedColormap(colors)
    boundaries = np.r_[unique - 0.5, unique[-1] + 0.5] if unique.size else np.asarray([-0.5, 0.5])
    return unique, cmap, BoundaryNorm(boundaries, cmap.N)


def plot_2d_variable(data: SampleData, name: str, rng, max_points: int) -> Figure:
    values = data.variables[name]
    info = data.info[name]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    if name in CATEGORICAL_VARIABLES:
        unique, cmap, norm = categorical_style(name, values)
        valid = np.isfinite(values) & np.isfinite(data.lat) & np.isfinite(data.lon)
        artist = axes[0, 0].scatter(
            data.lon[valid], data.lat[valid], c=values[valid], s=2, cmap=cmap,
            norm=norm, edgecolors="none", rasterized=True,
        )
        cbar = fig.colorbar(artist, ax=axes[0, 0], shrink=0.8)
        cbar.set_ticks(unique)
        if name == "typePrecip":
            cbar.set_ticklabels([TYPE_LABELS.get(int(value), f"{value:g}") for value in unique])
        axes[0, 0].set_title("Spatial categories")
        axes[0, 0].set_xlabel("Longitude (°E)")
        axes[0, 0].set_ylabel("Latitude (°N)")
        axes[0, 0].set_aspect("equal", adjustable="box")
    else:
        artist = add_swath_map(axes[0, 0], data, values, name, "Spatial field")
        fig.colorbar(artist, ax=axes[0, 0], shrink=0.8, label=info.units)
    add_histogram(axes[0, 1], values, name, rng, max_points)
    center_row = values.shape[0] // 2
    axes[1, 0].plot(np.arange(values.shape[1]), values[center_row], marker=".", linewidth=1)
    axes[1, 0].set_xlabel("Ray index")
    axes[1, 0].set_ylabel(info.units)
    axes[1, 0].set_title(f"Actual values at scan={center_row}")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 1].axis("off")
    axes[1, 1].text(0, 1, statistics_text(name, values, info), va="top", family="monospace")
    fig.suptitle(f"Variable diagnostic: {name}", fontsize=15)
    fig.tight_layout()
    return fig


def plot_3d_variable(
    data: SampleData, name: str, height_index: int, profile_row: int, rng, max_points: int
) -> Figure:
    values = data.variables[name]
    info = data.info[name]
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    layer = values[:, :, height_index]
    artist = add_swath_map(
        axes[0, 0], data, layer, name,
        f"Horizontal field at z={data.z[height_index]:.3f} km",
    )
    fig.colorbar(artist, ax=axes[0, 0], shrink=0.8, label=info.units)
    section_artist = add_cross_section(
        axes[0, 1], data, values, name, profile_row,
        f"A-B cross-section at scan={profile_row}",
    )
    axes[0, 1].axhline(data.z[height_index], color="white", linestyle="--", linewidth=1)
    fig.colorbar(section_artist, ax=axes[0, 1], shrink=0.8, label=info.units)
    add_histogram(axes[1, 0], values, name, rng, max_points)

    valid_fraction = np.isfinite(values).mean(axis=(0, 1)) * 100
    axes[1, 1].plot(valid_fraction, data.z, color="#f94144", label="valid %")
    axes[1, 1].set_xlabel("Valid values (%)")
    axes[1, 1].set_ylabel("Height (km)")
    axes[1, 1].set_xlim(0, 100)
    axes[1, 1].grid(alpha=0.25)
    value_axis = axes[1, 1].twiny()
    medians = np.full(data.z.shape, np.nan)
    q10 = np.full(data.z.shape, np.nan)
    q90 = np.full(data.z.shape, np.nan)
    for index in range(values.shape[-1]):
        layer_values = finite_values(values[:, :, index])
        if layer_values.size:
            q10[index], medians[index], q90[index] = np.percentile(layer_values, [10, 50, 90])
    value_axis.plot(medians, data.z, color="#277da1", label="median")
    value_axis.fill_betweenx(data.z, q10, q90, color="#277da1", alpha=0.18, label="P10-P90")
    value_axis.set_xlabel(f"{info.units}: median and P10-P90")
    axes[1, 1].set_title("Vertical coverage and value profile")
    axes[1, 1].legend(loc="lower left", fontsize=8)
    value_axis.legend(loc="lower right", fontsize=8)
    fig.suptitle(f"Variable diagnostic: {name}\n{statistics_text(name, values, info).splitlines()[5]}", fontsize=15)
    fig.tight_layout()
    return fig


def plot_all_variables(
    data: SampleData, writer: FigureWriter, height_index: int, profile_row: int,
    rng: np.random.Generator, max_points: int,
) -> None:
    for name, values in data.variables.items():
        if values.ndim == 1:
            fig = plot_1d_variable(data, name, rng, max_points)
        elif values.ndim == 2:
            fig = plot_2d_variable(data, name, rng, max_points)
        elif values.ndim == 3:
            fig = plot_3d_variable(data, name, height_index, profile_row, rng, max_points)
        else:
            print(f"SKIP unsupported variable rank: {name} {values.shape}")
            continue
        writer.save(fig, f"variables/{name}.png", f"Per-variable diagnostic for `{name}`.")


def paired_metrics(reference: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(reference) & np.isfinite(estimate)
    if not valid.any():
        return {"count": 0, "bias": np.nan, "mae": np.nan, "rmse": np.nan, "corr": np.nan}
    ref = reference[valid]
    est = estimate[valid]
    diff = est - ref
    corr = np.corrcoef(est, ref)[0, 1] if ref.size > 1 and ref.std() > 0 and est.std() > 0 else np.nan
    return {
        "count": int(ref.size),
        "bias": float(diff.mean()),
        "mae": float(np.abs(diff).mean()),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "corr": float(corr),
    }


def metrics_label(metrics: dict[str, float]) -> str:
    return (
        f"n={int(metrics['count']):,}\n"
        f"bias={metrics['bias']:.3f}\nMAE={metrics['mae']:.3f}\n"
        f"RMSE={metrics['rmse']:.3f}\nr={metrics['corr']:.3f}"
    )


def plot_gr_sparse_vs_interp(data: SampleData, height_index: int, rng, max_points: int) -> Figure:
    sparse = data.variables["dbz_gr_sparse"]
    interp = data.variables["dbz_gr_interp"]
    sparse_layer = sparse[:, :, height_index]
    interp_layer = interp[:, :, height_index]
    added = ~np.isfinite(sparse_layer) & np.isfinite(interp_layer)
    fig, axes = plt.subplots(2, 3, figsize=(19, 11))
    for ax, values, title in (
        (axes[0, 0], sparse_layer, "Sparse GR reflectivity"),
        (axes[0, 1], interp_layer, "Interpolated GR reflectivity"),
    ):
        artist = add_swath_map(ax, data, values, "dbz_gr_sparse", f"{title}\nz={data.z[height_index]:.3f} km")
        fig.colorbar(artist, ax=ax, shrink=0.8, label="dBZ")
    added_artist = axes[0, 2].scatter(
        data.lon[added], data.lat[added], c="#f94144", s=3, edgecolors="none", rasterized=True
    )
    axes[0, 2].set_title(f"Newly filled cells: {added.sum():,}")
    axes[0, 2].set_xlabel("Longitude (°E)")
    axes[0, 2].set_ylabel("Latitude (°N)")
    axes[0, 2].set_aspect("equal", adjustable="box")
    del added_artist

    axes[1, 0].plot(np.isfinite(sparse).mean(axis=(0, 1)) * 100, data.z, label="sparse")
    axes[1, 0].plot(np.isfinite(interp).mean(axis=(0, 1)) * 100, data.z, label="interp")
    axes[1, 0].set_xlabel("Valid values (%)")
    axes[1, 0].set_ylabel("Height (km)")
    axes[1, 0].set_title("Vertical coverage")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.25)

    all_added = interp[~np.isfinite(sparse) & np.isfinite(interp)]
    axes[1, 1].hist(sampled(all_added, max_points, rng), bins=60, color="#f8961e")
    axes[1, 1].set_xlabel("Newly filled reflectivity (dBZ)")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].set_title(f"Inserted values in 3-D: {all_added.size:,}")

    common = np.isfinite(sparse) & np.isfinite(interp)
    x = sparse[common]
    y = interp[common]
    if x.size > max_points:
        indices = rng.choice(x.size, max_points, replace=False)
        x, y = x[indices], y[indices]
    axes[1, 2].hexbin(x, y, gridsize=60, bins="log", cmap="viridis", mincnt=1)
    limits = robust_limits(np.r_[x, y])
    axes[1, 2].plot(limits, limits, "r--", linewidth=1)
    axes[1, 2].set_xlim(limits)
    axes[1, 2].set_ylim(limits)
    axes[1, 2].set_xlabel("Sparse GR (dBZ)")
    axes[1, 2].set_ylabel("Interpolated GR (dBZ)")
    axes[1, 2].set_title("Values at originally observed cells")
    fig.suptitle("What does GR interpolation fill?", fontsize=16)
    fig.tight_layout()
    return fig


def plot_gr_vs_dpr(data: SampleData, height_index: int, rng, max_points: int) -> Figure:
    gr = data.variables["dbz_gr_sparse"]
    dpr = data.variables["dbz_dpr"]
    gr_layer = gr[:, :, height_index]
    dpr_layer = dpr[:, :, height_index]
    common_layer = np.isfinite(gr_layer) & np.isfinite(dpr_layer)
    difference = np.where(common_layer, gr_layer - dpr_layer, np.nan)
    fig, axes = plt.subplots(2, 3, figsize=(19, 11))
    for ax, values, title in (
        (axes[0, 0], gr_layer, "Sparse GR"),
        (axes[0, 1], dpr_layer, "DPR Ku"),
    ):
        artist = add_swath_map(ax, data, values, "dbz_dpr", f"{title}\nz={data.z[height_index]:.3f} km")
        fig.colorbar(artist, ax=ax, shrink=0.8, label="dBZ")
    valid = np.isfinite(difference)
    diff_artist = axes[0, 2].scatter(
        data.lon[valid], data.lat[valid], c=difference[valid], s=3,
        cmap="coolwarm", norm=Normalize(-15, 15), edgecolors="none", rasterized=True,
    )
    axes[0, 2].set_title("GR − DPR at common cells")
    axes[0, 2].set_xlabel("Longitude (°E)")
    axes[0, 2].set_ylabel("Latitude (°N)")
    axes[0, 2].set_aspect("equal", adjustable="box")
    fig.colorbar(diff_artist, ax=axes[0, 2], shrink=0.8, label="dBZ")

    common = np.isfinite(gr) & np.isfinite(dpr)
    x, y = dpr[common], gr[common]
    if x.size > max_points:
        indices = rng.choice(x.size, max_points, replace=False)
        x, y = x[indices], y[indices]
    axes[1, 0].hexbin(x, y, gridsize=65, bins="log", cmap="viridis", mincnt=1)
    axes[1, 0].plot([0, 60], [0, 60], "r--", linewidth=1)
    axes[1, 0].set_xlim(0, 60)
    axes[1, 0].set_ylim(0, 60)
    axes[1, 0].set_xlabel("DPR reflectivity (dBZ)")
    axes[1, 0].set_ylabel("GR reflectivity (dBZ)")
    metrics = paired_metrics(dpr, gr)
    axes[1, 0].text(0.03, 0.97, metrics_label(metrics), transform=axes[1, 0].transAxes, va="top", bbox={"facecolor": "white", "alpha": 0.8})
    axes[1, 0].set_title("All-height common cells")

    bias, rmse, corr = [], [], []
    for index in range(data.z.size):
        values = paired_metrics(dpr[:, :, index], gr[:, :, index])
        bias.append(values["bias"])
        rmse.append(values["rmse"])
        corr.append(values["corr"])
    axes[1, 1].plot(bias, data.z, label="bias")
    axes[1, 1].plot(rmse, data.z, label="RMSE")
    axes[1, 1].axvline(0, color="black", linewidth=0.8)
    axes[1, 1].set_xlabel("dBZ")
    axes[1, 1].set_ylabel("Height (km)")
    axes[1, 1].set_title("Error by height")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.25)
    axes[1, 2].plot(corr, data.z, color="#43aa8b")
    axes[1, 2].set_xlim(-0.1, 1)
    axes[1, 2].set_xlabel("Pearson correlation")
    axes[1, 2].set_ylabel("Height (km)")
    axes[1, 2].set_title("Correlation by height")
    axes[1, 2].grid(alpha=0.25)
    fig.suptitle("Ground-radar and satellite-DPR reflectivity differences", fontsize=16)
    fig.tight_layout()
    return fig


def annotated_window(ax, values: np.ndarray, title: str, cmap, norm, row_start: int, col_start: int):
    masked = np.ma.masked_invalid(values)
    cmap_obj = plt.get_cmap(cmap).copy() if isinstance(cmap, str) else cmap.copy()
    cmap_obj.set_bad("#bdbdbd")
    image = ax.imshow(masked, cmap=cmap_obj, norm=norm, origin="upper", aspect="equal")
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            label = "—" if not np.isfinite(values[row, col]) else f"{values[row, col]:.1f}"
            ax.text(col, row, label, ha="center", va="center", fontsize=7,
                    color="white" if np.isfinite(values[row, col]) else "black")
    ax.set_xticks(np.arange(values.shape[1]), np.arange(col_start, col_start + values.shape[1]))
    ax.set_yticks(np.arange(values.shape[0]), np.arange(row_start, row_start + values.shape[0]))
    ax.set_xlabel("ray")
    ax.set_ylabel("scan")
    ax.set_title(title)
    return image


def plot_strong_rain_window(data: SampleData, height_index: int) -> Figure:
    rain = data.variables["pre_dpr"][:, :, height_index]
    sparse = data.variables["dbz_gr_sparse"][:, :, height_index]
    candidates = np.isfinite(rain) & np.isfinite(sparse)
    if not candidates.any():
        candidates = np.isfinite(rain)
    score = np.where(candidates, rain, -np.inf)
    center_row, center_col = np.unravel_index(np.argmax(score), score.shape)
    row_start = max(0, min(center_row - 3, rain.shape[0] - 7))
    col_start = max(0, min(center_col - 3, rain.shape[1] - 7))
    row_slice = slice(row_start, row_start + 7)
    col_slice = slice(col_start, col_start + 7)
    fields = (
        ("dbz_gr_sparse", "Sparse GR reflectivity", "turbo", Normalize(0, 50)),
        ("dbz_gr_interp", "Interpolated GR reflectivity", "turbo", Normalize(0, 50)),
        ("dbz_dpr", "DPR reflectivity", "turbo", Normalize(0, 50)),
        ("pre_dpr", "DPR precipitation rate", "viridis", Normalize(0, max(20, float(np.nanmax(rain[row_slice, col_slice]))))),
    )
    fig, axes = plt.subplots(2, 2, figsize=(13, 12))
    for ax, (name, title, cmap, norm) in zip(axes.flat, fields):
        values = data.variables[name][row_slice, col_slice, height_index]
        artist = annotated_window(ax, values, title, cmap, norm, row_start, col_start)
        fig.colorbar(artist, ax=ax, shrink=0.75, label=data.info[name].units)
    lat = data.lat[center_row, center_col]
    lon = data.lon[center_row, center_col]
    fig.suptitle(
        f"7×7 window around strongest DPR rain with direct GR\n"
        f"scan={center_row}, ray={center_col}, z={data.z[height_index]:.3f} km, "
        f"lat={lat:.4f}, lon={lon:.4f}, rain={rain[center_row, center_col]:.2f} mm/h",
        fontsize=14,
    )
    fig.tight_layout()
    return fig


def plot_cfb_clutter(data: SampleData, profile_row: int, rng, max_points: int) -> Figure:
    cfb = data.variables["cfb"]
    pre = data.variables["pre_dpr"]
    cfb_indices = np.where(np.isfinite(cfb), np.clip(cfb.astype(int), 0, data.z.size - 1), 0)
    cfb_height = np.where(np.isfinite(cfb), data.z[cfb_indices], np.nan)
    z_indices = np.arange(data.z.size).reshape(1, 1, -1)
    below = np.isfinite(cfb)[..., None] & (z_indices < cfb_indices[..., None])
    above = np.isfinite(cfb)[..., None] & ~below
    below_values = pre[below & np.isfinite(pre)]
    above_values = pre[above & np.isfinite(pre)]
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    artist = add_swath_map(axes[0, 0], data, cfb_height, "cfb", "Clutter-free-bottom height")
    fig.colorbar(artist, ax=axes[0, 0], shrink=0.8, label="km")
    axes[0, 1].hist(finite_values(cfb_height), bins=np.arange(0, 4.6, 0.25), color="#f8961e")
    axes[0, 1].set_xlabel("CFB height (km)")
    axes[0, 1].set_ylabel("Profiles")
    axes[0, 1].set_title("CFB height distribution")

    section = add_cross_section(axes[1, 0], data, pre, "pre_dpr", profile_row, "DPR rain with CFB boundary")
    line_ok = np.isfinite(data.lat[profile_row]) & np.isfinite(data.lon[profile_row])
    distance = cumulative_distance_km(data.lat[profile_row, line_ok], data.lon[profile_row, line_ok])
    axes[1, 0].plot(distance, cfb_height[profile_row, line_ok], color="red", linewidth=2, label="CFB")
    axes[1, 0].legend()
    fig.colorbar(section, ax=axes[1, 0], shrink=0.8, label="mm/h")

    below_positive = sampled(below_values[below_values > 0], max_points, rng)
    above_positive = sampled(above_values[above_values > 0], max_points, rng)
    if below_positive.size:
        axes[1, 1].hist(below_positive, bins=np.geomspace(max(below_positive.min(), 1e-3), below_positive.max(), 45), histtype="step", linewidth=2, label=f"below CFB (n={below_values.size:,})")
    if above_positive.size:
        axes[1, 1].hist(above_positive, bins=np.geomspace(max(above_positive.min(), 1e-3), above_positive.max(), 45), histtype="step", linewidth=2, label=f"at/above CFB (n={above_values.size:,})")
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("Positive DPR precipitation rate (mm/h)")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].set_title("Saved values below and above the clutter boundary")
    handles, labels = axes[1, 1].get_legend_handles_labels()
    if handles:
        axes[1, 1].legend(handles, labels, fontsize=8)
    fig.suptitle("Clutter-free-bottom diagnostics", fontsize=16)
    fig.tight_layout()
    return fig


def plot_precipitation_types(data: SampleData) -> Figure:
    types = data.variables["typePrecip"]
    pre = data.variables["pre_dpr"]
    codes = [-1111, 1, 2, 3]
    colors = ["#bdbdbd", "#277da1", "#f94144", "#f9c74f"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    counts = [int(np.count_nonzero(types == code)) for code in codes]
    labels = [TYPE_LABELS[code] for code in codes]
    axes[0, 0].bar(labels, counts, color=colors)
    axes[0, 0].tick_params(axis="x", rotation=20)
    axes[0, 0].set_ylabel("Horizontal profiles")
    axes[0, 0].set_title("Precipitation-type counts")

    zero_ratios = []
    positive_distributions = []
    positive_labels = []
    for code, label in zip(codes, labels):
        values = pre[types == code]
        finite = finite_values(values)
        zero_ratios.append(np.mean(finite == 0) * 100 if finite.size else np.nan)
        positive = finite[finite > 0]
        if positive.size:
            positive_distributions.append(positive)
            positive_labels.append(label)
    axes[0, 1].bar(labels, zero_ratios, color=colors)
    axes[0, 1].tick_params(axis="x", rotation=20)
    axes[0, 1].set_ylim(0, 105)
    axes[0, 1].set_ylabel("Zero values (%)")
    axes[0, 1].set_title("Zero-rain fraction within each type")

    if positive_distributions:
        axes[1, 0].boxplot(
            positive_distributions,
            labels=positive_labels,
            showfliers=False,
            whis=(5, 95),
        )
        axes[1, 0].set_yscale("log")
    else:
        axes[1, 0].text(0.5, 0.5, "No positive precipitation", ha="center", va="center")
    axes[1, 0].tick_params(axis="x", rotation=20)
    axes[1, 0].set_ylabel("Positive precipitation rate (mm/h)")
    axes[1, 0].set_title("Positive-rain distributions (P05-P95 whiskers)")

    for code, label, color in zip(codes[1:], labels[1:], colors[1:]):
        values = pre[types == code]
        median = np.full(data.z.shape, np.nan)
        p90 = np.full(data.z.shape, np.nan)
        for index in range(data.z.size):
            positive = values[:, index]
            positive = positive[np.isfinite(positive) & (positive > 0)]
            if positive.size:
                median[index], p90[index] = np.percentile(positive, [50, 90])
        axes[1, 1].plot(median, data.z, color=color, label=f"{label} median")
        axes[1, 1].plot(p90, data.z, color=color, linestyle="--", label=f"{label} P90")
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_xlabel("Positive precipitation rate (mm/h)")
    axes[1, 1].set_ylabel("Height (km)")
    axes[1, 1].set_title("Positive-rain profiles by precipitation type")
    axes[1, 1].legend(fontsize=7)
    axes[1, 1].grid(alpha=0.2)
    fig.suptitle("Stratiform, convective, other, and no-precipitation samples", fontsize=16)
    fig.tight_layout()
    return fig


def plot_pre_dpr_long_tail(data: SampleData) -> Figure:
    rain = data.variables["pre_dpr"]
    finite = finite_values(rain)
    positive = finite[finite > 0]
    total = rain.size
    missing = total - finite.size
    zeros = np.count_nonzero(finite == 0)
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    axes[0, 0].bar(
        ["missing", "valid zero", "positive"], [missing, zeros, positive.size],
        color=["#bdbdbd", "#90be6d", "#f94144"],
    )
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylabel("3-D cells (log scale)")
    axes[0, 0].set_title("Missing, zero, and positive labels")
    for index, count in enumerate((missing, zeros, positive.size)):
        # Text at y=0 is infinitely far away on a logarithmic axis and makes
        # bbox_inches="tight" attempt to create an enormous image.
        text_y = max(int(count), 1)
        axes[0, 0].text(
            index, text_y, f"{count:,}\n{count/total:.3%}",
            ha="center", va="bottom", fontsize=8,
        )

    if positive.size:
        bins = np.geomspace(max(positive.min(), 1e-3), positive.max(), 65)
        axes[0, 1].hist(positive, bins=bins, color="#277da1")
        axes[0, 1].set_xscale("log")
        axes[0, 1].set_yscale("log")
    else:
        axes[0, 1].text(0.5, 0.5, "No positive precipitation", ha="center", va="center")
    axes[0, 1].set_xlabel("Positive precipitation rate (mm/h)")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title("Positive-rain long tail")

    if positive.size:
        sorted_positive = np.sort(positive)
        exceedance = 1 - np.arange(sorted_positive.size) / sorted_positive.size
        axes[1, 0].plot(sorted_positive, exceedance, color="#f94144")
        axes[1, 0].set_xscale("log")
        axes[1, 0].set_yscale("log")
    else:
        axes[1, 0].text(0.5, 0.5, "No positive precipitation", ha="center", va="center")
    axes[1, 0].set_xlabel("Precipitation threshold (mm/h)")
    axes[1, 0].set_ylabel("P(R ≥ threshold | R > 0)")
    axes[1, 0].set_title("Complementary cumulative distribution")
    axes[1, 0].grid(alpha=0.25, which="both")

    counts = np.asarray([(finite > threshold).sum() for threshold in RAIN_THRESHOLDS])
    axes[1, 1].bar([f">{value:g}" for value in RAIN_THRESHOLDS], counts, color="#f8961e")
    if counts.max() > 0:
        axes[1, 1].set_yscale("log")
    axes[1, 1].tick_params(axis="x", rotation=35)
    axes[1, 1].set_ylabel("Valid 3-D cells")
    axes[1, 1].set_title("Exceedance counts")
    for index, count in enumerate(counts):
        ratio = count / finite.size if finite.size else np.nan
        text_y = max(int(count), 1) if counts.max() > 0 else float(count)
        axes[1, 1].text(
            index, text_y, f"{count:,}\n{ratio:.4%}",
            ha="center", va="bottom", fontsize=7, rotation=20,
        )
    fig.suptitle("DPR precipitation-rate zero inflation and strong-rain tail", fontsize=16)
    fig.tight_layout()
    return fig


def zrh_to_rain(dbz: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    valid = np.isfinite(dbz) & (dbz >= 0) & (dbz < 70)
    rain = np.exp(np.where(valid, dbz, 0.0) * weight + bias)
    rain[~valid] = np.nan
    return rain


def plot_zrh_baseline(
    data: SampleData, height_index: int, weight: np.ndarray, bias: np.ndarray,
    rng, max_points: int,
) -> Figure:
    sparse_rain = zrh_to_rain(data.variables["dbz_gr_sparse"], weight, bias)
    interp_rain = zrh_to_rain(data.variables["dbz_gr_interp"], weight, bias)
    target = data.variables["pre_dpr"]
    fields = (
        (sparse_rain[:, :, height_index], "ZRH from sparse GR"),
        (interp_rain[:, :, height_index], "ZRH from interpolated GR"),
        (target[:, :, height_index], "DPR reference precipitation"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(19, 11))
    for ax, (values, title) in zip(axes[0], fields):
        artist = add_swath_map(ax, data, values, "pre_dpr", f"{title}\nz={data.z[height_index]:.3f} km")
        fig.colorbar(artist, ax=ax, shrink=0.8, label="mm/h")
    for ax, estimate, title in (
        (axes[1, 0], sparse_rain, "Sparse ZRH vs DPR"),
        (axes[1, 1], interp_rain, "Interpolated ZRH vs DPR"),
    ):
        valid = np.isfinite(estimate) & np.isfinite(target)
        x, y = target[valid], estimate[valid]
        if x.size > max_points:
            indices = rng.choice(x.size, max_points, replace=False)
            x, y = x[indices], y[indices]
        positive = (x > 0.05) & (y > 0.05)
        ax.hexbin(x[positive], y[positive], gridsize=60, bins="log", cmap="viridis", mincnt=1, xscale="log", yscale="log")
        ax.plot([0.05, 100], [0.05, 100], "r--", linewidth=1)
        ax.set_xlim(0.05, 100)
        ax.set_ylim(0.05, 100)
        ax.set_xlabel("DPR precipitation rate (mm/h)")
        ax.set_ylabel("ZRH precipitation rate (mm/h)")
        ax.set_title(title)
        ax.text(0.03, 0.97, metrics_label(paired_metrics(target, estimate)), transform=ax.transAxes, va="top", bbox={"facecolor": "white", "alpha": 0.8})
    layer_diff = interp_rain[:, :, height_index] - target[:, :, height_index]
    valid = np.isfinite(layer_diff) & np.isfinite(data.lat) & np.isfinite(data.lon)
    artist = axes[1, 2].scatter(
        data.lon[valid], data.lat[valid], c=layer_diff[valid], s=3,
        cmap="coolwarm", norm=Normalize(-10, 10), edgecolors="none", rasterized=True,
    )
    axes[1, 2].set_aspect("equal", adjustable="box")
    axes[1, 2].set_xlabel("Longitude (°E)")
    axes[1, 2].set_ylabel("Latitude (°N)")
    axes[1, 2].set_title("Interpolated ZRH − DPR at 2 km")
    fig.colorbar(artist, ax=axes[1, 2], shrink=0.8, label="mm/h")
    fig.suptitle("Height-dependent ZRH baselines compared with DPR reference rain", fontsize=16)
    fig.tight_layout()
    return fig


def write_manifest(
    output_path: Path, data: SampleData, height_index: int, profile_row: int,
    entries: Iterable[tuple[str, str]],
) -> None:
    lines = [
        "# 单样本诊断图索引",
        "",
        f"- 样本：`{data.path}`",
        f"- 水平切片：目标高度对应实际 `z[{height_index}]={data.z[height_index]:.3f} km`",
        f"- A–B剖面扫描行：`{profile_row}`",
        "- 灰色通常表示缺测；降水图中的白色表示有效零降水。",
        "",
        "## 图表",
        "",
    ]
    for relative_path, description in entries:
        lines.extend([f"### `{relative_path}`", "", description, "", f"![{relative_path}]({relative_path})", ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def select_input_files(
    input_dir: Path, input_file: Path | None, count: int
) -> list[Path]:
    """Return one explicit file or the first count files in sorted order."""
    if input_file is not None:
        if not input_file.is_file():
            raise FileNotFoundError(f"NetCDF sample not found: {input_file}")
        return [input_file]
    if count <= 0:
        raise ValueError("--count must be positive")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")
    source_files = sorted(input_dir.glob("*.nc"))
    if len(source_files) < count:
        raise ValueError(
            f"Requested {count} files, but only found {len(source_files)} NetCDF files."
        )
    return source_files[:count]


def prepare_output_dir(
    root: Path, sample_path: Path, overwrite: bool
) -> Path | None:
    output_dir = root / sample_path.stem
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        # figure_manifest.md is only written after every requested figure and
        # the PDF have completed. Its absence identifies an interrupted run.
        if (output_dir / ".complete").is_file() or (
            output_dir / "figure_manifest.md"
        ).is_file():
            return None
        print(f"RESTART incomplete sample output: {sample_path.name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def process_sample(
    sample_path: Path,
    args: argparse.Namespace,
    selected_sections: set[str],
    sample_index: int,
) -> bool:
    """Generate all requested outputs for one file and release it before the next."""
    output_dir = prepare_output_dir(args.output_dir, sample_path, args.overwrite)
    if output_dir is None:
        print(f"SKIP existing sample output: {sample_path.name}")
        return False
    print(f"\nPROCESS [{sample_index + 1}] {sample_path.name}")
    data = load_sample(sample_path)
    height_index = nearest_height_index(data.z, args.height_km)
    profile_row = select_profile_row(data.variables["pre_dpr"], height_index)
    rows = variable_statistics(data)
    write_statistics_csv(rows, output_dir / "statistics.csv")
    rng = np.random.default_rng(args.seed + sample_index)
    writer = FigureWriter(output_dir, args.dpi)
    try:
        tasks: list[tuple[str, str, str, Callable[[], Figure]]] = [
            ("overview", "00_dataset_overview.png", "All variables, shapes, sources, and valid-data coverage.", lambda: plot_dataset_overview(data, rows)),
            ("gr_interp", "comparisons/01_gr_sparse_vs_interp.png", "GR coverage added by interpolation and values retained at observed cells.", lambda: plot_gr_sparse_vs_interp(data, height_index, rng, args.max_points)),
            ("gr_dpr", "comparisons/02_gr_vs_dpr.png", "Matched ground-radar and DPR reflectivity differences in space and height.", lambda: plot_gr_vs_dpr(data, height_index, rng, args.max_points)),
            ("strong_window", "comparisons/03_strong_rain_window.png", "Actual values in a 7×7 window around the strongest DPR rain cell with direct GR.", lambda: plot_strong_rain_window(data, height_index)),
            ("cfb", "comparisons/04_cfb_clutter.png", "Clutter-free-bottom heights and saved DPR values below the boundary.", lambda: plot_cfb_clutter(data, profile_row, rng, args.max_points)),
            ("precip_types", "comparisons/05_precipitation_types.png", "Stratiform, convective, other, and no-precipitation label distributions.", lambda: plot_precipitation_types(data)),
            ("rain_tail", "comparisons/06_pre_dpr_long_tail.png", "Zero inflation, positive-rain distribution, CCDF, and strong-rain exceedance counts.", lambda: plot_pre_dpr_long_tail(data)),
        ]
        for section, relative_path, description, callback in tasks:
            if section in selected_sections:
                writer.save(callback(), relative_path, description)
        if "variables" in selected_sections:
            plot_all_variables(data, writer, height_index, profile_row, rng, args.max_points)
        if "zrh" in selected_sections:
            if not args.model_path.is_file():
                raise FileNotFoundError(f"ZRH model not found: {args.model_path}")
            weight, bias = load_zrh_parameters(args.model_path)
            writer.save(
                plot_zrh_baseline(data, height_index, weight, bias, rng, args.max_points),
                "comparisons/07_zrh_baseline_vs_dpr.png",
                "Height-dependent ZRH rain from sparse/interpolated GR compared with DPR rain.",
            )
    finally:
        writer.close()
    write_manifest(
        output_dir / "figure_manifest.md", data, height_index, profile_row, writer.entries
    )
    (output_dir / ".complete").write_text(
        f"figures={len(writer.entries)}\nsource={sample_path}\n", encoding="utf-8"
    )
    print(f"Statistics: {output_dir / 'statistics.csv'}")
    print(f"Figure index: {output_dir / 'figure_manifest.md'}")
    print(f"PDF report: {output_dir / 'diagnostics.pdf'}")
    print(f"Completed {len(writer.entries)} figures in: {output_dir}")
    return True


def main() -> None:
    args = parse_args()
    if args.dpi <= 0 or args.max_points <= 0:
        raise ValueError("--dpi and --max-points must be positive")
    source_files = select_input_files(args.input_dir, args.input_file, args.count)
    selected_sections = set(SECTION_NAMES if "all" in args.sections else args.sections)
    if "zrh" in selected_sections and not args.model_path.is_file():
        raise FileNotFoundError(f"ZRH model not found: {args.model_path}")
    print(f"Selected files: {len(source_files)}")
    for source_path in source_files:
        print(f"  {source_path.name}")
    completed = 0
    failures: list[tuple[Path, str]] = []
    for sample_index, sample_path in enumerate(source_files):
        try:
            completed += int(
                process_sample(sample_path, args, selected_sections, sample_index)
            )
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            failures.append((sample_path, message))
            print(f"ERROR sample {sample_path.name}: {message}")
            if args.fail_fast:
                raise
        finally:
            plt.close("all")
    print(
        f"\nBatch complete: selected={len(source_files)}, "
        f"generated={completed}, "
        f"skipped={len(source_files) - completed - len(failures)}, "
        f"failed={len(failures)}"
    )
    if failures:
        print("Failed samples:")
        for sample_path, message in failures:
            print(f"  {sample_path.name}: {message}")
        raise RuntimeError(f"{len(failures)} sample(s) failed; see the summary above")


if __name__ == "__main__":
    main()
