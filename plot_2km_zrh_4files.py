#!/usr/bin/env python3
"""Plot 2 km maps and A-B transects for the first GR-DPR NetCDF files.

Each figure uses a 3 x 2 panel layout:

    dbz_gr_sparse             | dbz_gr_interp
    rain_rate_zrh_gr_sparse   | rain_rate_zrh_gr_interp
    dbz_dpr                   | pre_dpr

The two ZRH rain-rate fields are calculated in memory from ZRH_37refine.pth;
the input NetCDF files are not modified.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, Normalize
from netCDF4 import Dataset

from zrh_nc_to_rain import load_zrh_parameters


PROJECT_DIR = Path(__file__).resolve().parent
DATASET_NAME = "GRToDPRRes_V07_Pct_V1.2.1_sw_260412"
DEFAULT_INPUT_DIR = Path("/storage/GR_DPR_3D") / DATASET_NAME
DEFAULT_MODEL_PATH = PROJECT_DIR / "ZRH_37refine.pth"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs" / "plots_2km_zrh"

DBZ_MIN = 0.0
DBZ_MAX = 70.0
RAIN_TICKS = [0.10, 0.18, 0.33, 0.59, 1.06, 1.91, 3.45, 6.22, 11.22, 20.24]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot 3 x 2 GR, ZRH-rain, and DPR comparisons for NetCDF files."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--count",
        type=int,
        default=4,
        help="Number of files to plot (taken from the sorted list; default: 4).",
    )
    return parser.parse_args()


def read_float(variable) -> np.ndarray:
    """Read NetCDF data as float64 and normalize masked/legacy missing values."""
    array = np.ma.filled(variable[:], np.nan).astype(np.float64, copy=False)
    array[array < -9990] = np.nan
    return array


def zrh_to_rain(
    dbz: np.ndarray, weight: np.ndarray, bias: np.ndarray
) -> np.ndarray:
    """Apply R = exp(dBZ * weight[z] + bias[z]) to valid GR reflectivity."""
    if dbz.ndim != 3 or dbz.shape[-1] != weight.shape[-1]:
        raise ValueError(
            f"Expected dBZ shape (nscan, nray, {weight.shape[-1]}), got {dbz.shape}."
        )

    valid = np.isfinite(dbz) & (dbz >= DBZ_MIN) & (dbz < DBZ_MAX)
    safe_dbz = np.where(valid, dbz, 0.0)
    rain = np.exp(safe_dbz * weight + bias)
    rain[~valid] = np.nan
    return rain


def select_profile_row(pre_dpr: np.ndarray, height_index: int) -> int:
    """Use the scan row containing the most DPR precipitation at the selected height."""
    pre_2km = pre_dpr[:, :, height_index]
    precip_mask = np.isfinite(pre_2km) & (pre_2km > 0.1)
    row_score = precip_mask.sum(axis=1)
    if row_score.max() > 0:
        return int(np.argmax(row_score))
    return int(np.argmax(np.isfinite(pre_2km).sum(axis=1)))


def cumulative_distance_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Calculate cumulative great-circle distance along the selected A-B line."""
    earth_radius_km = 6371.0
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    dlat = np.diff(lat_rad)
    dlon = np.diff(lon_rad)
    haversine_a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat_rad[:-1]) * np.cos(lat_rad[1:]) * np.sin(dlon / 2) ** 2
    )
    segment_km = 2 * earth_radius_km * np.arcsin(np.sqrt(np.clip(haversine_a, 0, 1)))
    return np.r_[0.0, np.cumsum(segment_km)]
    #(nray,)


def panel_style(unit: str) -> tuple[Normalize, str, list[float]]:
    if unit == "dBZ":
        return Normalize(vmin=10, vmax=50), "nipy_spectral", list(np.arange(10, 51, 5))
    return LogNorm(vmin=RAIN_TICKS[0], vmax=RAIN_TICKS[-1]), "jet", RAIN_TICKS


def plot_panel(
    fig,
    cell,
    variable_name: str,
    display_name: str,
    unit: str,
    data_3d: dict[str, np.ndarray],
    z: np.ndarray,
    height_index: int,
    lat: np.ndarray,
    lon: np.ndarray,
    profile_row: int,
    line_ok: np.ndarray,
    line_lat: np.ndarray,
    line_lon: np.ndarray,
    distance_km: np.ndarray,
) -> None:
    """Draw one map/cross-section/colorbar panel in the same style as plot_2km.py."""
    inner = cell.subgridspec(1, 3, width_ratios=[1.15, 1.0, 0.08], wspace=0.10)
    ax_map = fig.add_subplot(inner[0])
    ax_profile = fig.add_subplot(inner[1])
    ax_cbar = fig.add_subplot(inner[2])

    norm, cmap, ticks = panel_style(unit)
    arr_2km = data_3d[variable_name][:, :, height_index]
    valid = np.isfinite(arr_2km)
    if unit == "mm/h":
        valid &= arr_2km > RAIN_TICKS[0]

    ax_map.scatter(
        lon[valid],
        lat[valid],
        c=arr_2km[valid],
        s=2.0,
        cmap=cmap,
        norm=norm,
        edgecolors="none",
        rasterized=True,
    )
    # The first and last rays trace the two edges of the satellite swath.
    for edge_index in (0, -1):
        edge_ok = np.isfinite(lat[:, edge_index]) & np.isfinite(lon[:, edge_index])
        edge_lon = np.where(edge_ok, lon[:, edge_index], np.nan)
        edge_lat = np.where(edge_ok, lat[:, edge_index], np.nan)
        ax_map.plot(edge_lon, edge_lat, color="black", linewidth=2.0, zorder=4)
        ax_map.plot(edge_lon, edge_lat, color="cyan", linewidth=0.9, zorder=5)
    ax_map.plot(line_lon, line_lat, color="black", linewidth=2.5, zorder=5)
    ax_map.plot(line_lon, line_lat, color="white", linewidth=1.2, zorder=6)
    ax_map.scatter(
        [line_lon[0], line_lon[-1]],
        [line_lat[0], line_lat[-1]],
        c=["white", "yellow"],
        edgecolors="black",
        s=20,
        zorder=7,
    )
    ax_map.text(line_lon[0], line_lat[0], " A", color="white", fontsize=8, weight="bold")
    ax_map.text(line_lon[-1], line_lat[-1], " B", color="yellow", fontsize=8, weight="bold")
    ax_map.set_xlim(np.nanmin(lon) - 0.2, np.nanmax(lon) + 0.2)
    ax_map.set_ylim(np.nanmin(lat) - 0.2, np.nanmax(lat) + 0.2)
    ax_map.set_aspect("equal", adjustable="box")
    ax_map.set_xlabel("Longitude (deg E)")
    ax_map.set_ylabel("Latitude (deg N)")
    ax_map.set_title(
        f"{variable_name}\n{display_name} at {z[height_index]:.3f} km",
        fontsize=9,
    )

    profile = np.ma.masked_invalid(data_3d[variable_name][profile_row, line_ok, :].T)
    if unit == "mm/h":
        profile = np.ma.masked_less_equal(profile, RAIN_TICKS[0])
    ax_profile.pcolormesh(
        distance_km,
        z,
        profile,
        cmap=cmap,
        norm=norm,
        shading="auto",
        rasterized=True,
    )
    ax_profile.axhline(z[height_index], color="white", linewidth=1.0, linestyle="--")
    ax_profile.set_xlabel("Distance along A-B (km)")
    ax_profile.set_ylabel("Height (km)")
    ax_profile.set_ylim(np.nanmin(z), np.nanmax(z))
    ax_profile.set_title(f"{variable_name}\nA-B vertical cross-section", fontsize=9)

    color_mapper = ScalarMappable(norm=norm, cmap=cmap)
    color_mapper.set_array([])
    colorbar = fig.colorbar(color_mapper, cax=ax_cbar, ticks=ticks)
    colorbar.set_label(unit)
    if unit == "mm/h":
        colorbar.set_ticklabels([f"{tick:.2f}" for tick in ticks])
    else:
        colorbar.set_ticklabels([f"{int(tick)}" for tick in ticks])


def plot_file(
    source_path: Path,
    output_path: Path,
    weight: np.ndarray,
    bias: np.ndarray,
) -> None:
    with Dataset(source_path, "r") as dataset:
        required = ("z", "lat", "lon", "dbz_gr_sparse", "dbz_gr_interp", "dbz_dpr", "pre_dpr")
        missing = [name for name in required if name not in dataset.variables]
        if missing:
            raise KeyError(f"{source_path.name} is missing variables: {missing}")

        z = np.asarray(dataset.variables["z"][:], dtype=np.float64)
        lat = read_float(dataset.variables["lat"])
        lon = read_float(dataset.variables["lon"])
        data_3d = {
            "dbz_gr_sparse": read_float(dataset.variables["dbz_gr_sparse"]),
            "dbz_gr_interp": read_float(dataset.variables["dbz_gr_interp"]),
            "dbz_dpr": read_float(dataset.variables["dbz_dpr"]),
            "pre_dpr": read_float(dataset.variables["pre_dpr"]),
        }

    data_3d["rain_rate_zrh_gr_sparse"] = zrh_to_rain(
        data_3d["dbz_gr_sparse"], weight, bias
    )
    data_3d["rain_rate_zrh_gr_interp"] = zrh_to_rain(
        data_3d["dbz_gr_interp"], weight, bias
    )
    if len(z) != weight.shape[-1]:
        raise ValueError(f"{source_path.name} has {len(z)} altitude layers, expected 60.")

    height_index = int(np.argmin(np.abs(z - 2.0)))
    profile_row = select_profile_row(data_3d["pre_dpr"], height_index)
    line_ok = np.isfinite(lat[profile_row]) & np.isfinite(lon[profile_row])
    line_lat = lat[profile_row, line_ok]
    line_lon = lon[profile_row, line_ok]
    if line_lon.size < 2:
        raise ValueError(f"{source_path.name} has fewer than two valid A-B coordinates.")
    distance_km = cumulative_distance_km(line_lat, line_lon)

    panel_definitions = (
        ("dbz_gr_sparse", "GR Reflectivity (sparse)", "dBZ"),
        ("dbz_gr_interp", "GR Reflectivity (interp)", "dBZ"),
        ("rain_rate_zrh_gr_sparse", "ZRH Rain Rate from GR sparse", "mm/h"),
        ("rain_rate_zrh_gr_interp", "ZRH Rain Rate from GR interp", "mm/h"),
        ("dbz_dpr", "DPR Reflectivity", "dBZ"),
        ("pre_dpr", "DPR Precipitation Rate", "mm/h"),
    )

    fig = plt.figure(figsize=(18, 18))
    outer = fig.add_gridspec(3, 2, wspace=0.20, hspace=0.36)
    fig.suptitle(
        f"GR Reflectivity, ZRH Rain Rate, and DPR Products at 2 km\n{source_path.name}",
        fontsize=14,
    )
    for cell, (variable_name, display_name, unit) in zip(outer, panel_definitions):
        plot_panel(
            fig,
            cell,
            variable_name,
            display_name,
            unit,
            data_3d,
            z,
            height_index,
            lat,
            lon,
            profile_row,
            line_ok,
            line_lat,
            line_lon,
            distance_km,
        )

    fig.subplots_adjust(top=0.94)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(
        f"OK {source_path.name} -> {output_path.name} | "
        f"z[{height_index}]={z[height_index]:.3f} km, profile row={profile_row}"
    )


def main() -> None:
    args = parse_args()
    if args.count <= 0:
        raise ValueError("--count must be positive")
    if not args.input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {args.input_dir}")
    if not args.model_path.is_file():
        raise FileNotFoundError(f"ZRH model not found: {args.model_path}")

    source_files = sorted(args.input_dir.glob("*.nc"))
    if len(source_files) < args.count:
        raise ValueError(
            f"Requested {args.count} files, but only found {len(source_files)} NetCDF files."
        )
    selected_files = source_files[: args.count]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    weight, bias = load_zrh_parameters(args.model_path)
    print("Selected files:")
    for source_path in selected_files:
        print(f"  {source_path.name}")

    for source_path in selected_files:
        output_path = args.output_dir / f"plot_2km_zrh_{source_path.stem}.png"
        plot_file(source_path, output_path, weight, bias)


if __name__ == "__main__":
    main()
