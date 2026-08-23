#!/usr/bin/env python3
"""Convert all supported radar reflectivity fields in DPR-aligned NetCDF files to rain rate.

The ZRH model stores one weight and one bias for each of the 60 altitude layers.
For every valid dBZ value, this script calculates::

    R = exp(dBZ * weight[z] + bias[z])

Input files are never modified.  Each output file copies the source content and adds
one ``rain_rate_zrh_*`` variable for every requested dBZ input variable present.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from zipfile import BadZipFile, ZipFile

import numpy as np
from netCDF4 import Dataset


PROJECT_DIR = Path(__file__).resolve().parent
DATASET_NAME = "GRToDPRRes_V07_Pct_V1.2.1_sw_260412"

# The dataset was migrated from /data and is mounted read-only on this server.
# Keep generated files in the project so running with no arguments is safe and
# does not attempt to modify the original dataset.
DEFAULT_INPUT_DIR = Path("/storage/GR_DPR_3D") / DATASET_NAME
DEFAULT_MODEL_PATH = PROJECT_DIR / "ZRH_37refine.pth"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs" / f"zrh_{DATASET_NAME}"

# All reflectivity fields in the current dataset.  Other fields, such as pressure,
# temperature, humidity, spectral width, and existing DPR rain products, are not
# inputs to this ZRH relation.
DEFAULT_DBZ_VARIABLES = (
    "dbz_gr_sparse",
    "dbz_gr_sparse_min",
    "dbz_gr_sparse_max",
    "dbz_gr_interp",
    "dbz_dpr",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert NetCDF radar reflectivity fields to ZRH rain-rate fields."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--variables",
        nargs="+",
        default=DEFAULT_DBZ_VARIABLES,
        metavar="DBZ_VAR",
        help="Reflectivity variables to convert (default: all supported dBZ fields).",
    )
    parser.add_argument(
        "--dbz-min",
        type=float,
        default=0.0,
        help="Inclusive lower dBZ limit for conversion (default: 0).",
    )
    parser.add_argument(
        "--dbz-max",
        type=float,
        default=70.0,
        help="Exclusive upper dBZ limit for conversion (default: 70).",
    )
    parser.add_argument(
        "--chunk-scans",
        type=int,
        default=128,
        help="Number of scan rows processed at once (default: 128).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output file. By default, existing outputs are skipped.",
    )
    return parser.parse_args()


def load_zrh_parameters(model_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the two float64 arrays from this project's small PyTorch checkpoint.

    The supplied ``ZRH_37refine.pth`` is a ZIP-format checkpoint containing only
    two contiguous DoubleStorage records: ``weight`` and ``bias``. Reading those
    records directly keeps this preprocessing script independent of the roughly
    192 MB PyTorch runtime. This loader intentionally validates that exact format
    instead of trying to support arbitrary PyTorch checkpoints.
    """
    if not model_path.is_file():
        raise FileNotFoundError(f"ZRH model not found: {model_path}")

    try:
        with ZipFile(model_path) as checkpoint:
            names = checkpoint.namelist()
            data_pickle = next(name for name in names if name.endswith("/data.pkl"))
            prefix = data_pickle.removesuffix("data.pkl")
            metadata = checkpoint.read(data_pickle)
            if b"weight" not in metadata or b"bias" not in metadata:
                raise ValueError("Checkpoint metadata does not contain weight and bias")

            byteorder = checkpoint.read(prefix + "byteorder").decode("ascii").strip()
            if byteorder not in {"little", "big"}:
                raise ValueError(f"Unsupported checkpoint byte order: {byteorder!r}")
            dtype = np.dtype("<f8" if byteorder == "little" else ">f8")
            weight = np.frombuffer(checkpoint.read(prefix + "data/0"), dtype=dtype)
            bias = np.frombuffer(checkpoint.read(prefix + "data/1"), dtype=dtype)
    except (BadZipFile, KeyError, StopIteration) as exc:
        raise ValueError(
            f"Unsupported ZRH checkpoint layout in {model_path}"
        ) from exc

    weight = weight.astype(np.float64, copy=False).reshape(1, 1, -1)
    bias = bias.astype(np.float64, copy=False).reshape(1, 1, -1)
    if weight.shape != bias.shape or weight.shape[-1] != 60:
        raise ValueError(
            "Expected matching ZRH weight and bias tensors with 60 altitude layers; "
            f"got weight={tuple(weight.shape)}, bias={tuple(bias.shape)}."
        )
    return weight, bias
    #[1,1,60], [1,1,60]


def derived_variable_name(source_name: str) -> str:
    """Map dbz_gr_interp to rain_rate_zrh_gr_interp, and similarly for other fields."""
    suffix = source_name.removeprefix("dbz_")
    return f"rain_rate_zrh_{suffix}"


def copy_dimensions(source: Dataset, target: Dataset) -> None:
    for name, dimension in source.dimensions.items():
        size = None if dimension.isunlimited() else len(dimension)
        target.createDimension(name, size)


def copy_variable(source_var, target: Dataset, chunk_scans: int) -> None:
    """Copy one variable while preserving attributes and avoiding whole-file reads."""
    fill_value = (
        source_var.getncattr("_FillValue")
        if "_FillValue" in source_var.ncattrs()
        else None
    )
    create_kwargs = {}
    if fill_value is not None:
        create_kwargs["fill_value"] = fill_value
    if source_var.ndim and np.dtype(source_var.dtype).kind in "iufb":
        create_kwargs.update(zlib=True, complevel=4, shuffle=True)

    target_var = target.createVariable(
        source_var.name,
        source_var.datatype,
        source_var.dimensions,
        **create_kwargs,
    )
    target_var.setncatts(
        {
            attr: source_var.getncattr(attr)
            for attr in source_var.ncattrs()
            if attr != "_FillValue"
        }
    )

    if source_var.ndim == 0:
        target_var.assignValue(source_var.getValue())
        return

    for start in range(0, source_var.shape[0], chunk_scans):
        stop = min(start + chunk_scans, source_var.shape[0])
        index = (slice(start, stop),) + (slice(None),) * (source_var.ndim - 1)
        target_var[index] = source_var[index]


def copy_source_content(source: Dataset, target: Dataset, chunk_scans: int) -> None:
    copy_dimensions(source, target)
    target.setncatts({attr: source.getncattr(attr) for attr in source.ncattrs()})
    for source_var in source.variables.values():
        copy_variable(source_var, target, chunk_scans)


def convert_variable(
    source_var,
    target: Dataset,
    weight: np.ndarray,
    bias: np.ndarray,
    dbz_min: float,
    dbz_max: float,
    chunk_scans: int,
    model_path: Path,
) -> tuple[str, int, int]:
    """Add one derived rain-rate field and return its name and valid-value counts."""
    if source_var.ndim != 3 or source_var.shape[-1] != weight.shape[-1]:
        raise ValueError(
            f"{source_var.name} must have shape (nscan, nray, {weight.shape[-1]}); "
            f"got {source_var.shape} with dimensions {source_var.dimensions}."
        )

    output_name = derived_variable_name(source_var.name)
    rain_var = target.createVariable(
        output_name,
        "f4",
        source_var.dimensions,
        fill_value=np.nan,
        zlib=True,
        complevel=4,
        shuffle=True,
    )
    rain_var.setncatts(
        {
            "units": "mm/h",
            "long_name": f"ZRH-derived precipitation rate from {source_var.name}",
            "source_variable": source_var.name,
            "zrh_formula": "R = exp(dBZ * weight[z] + bias[z])",
            "zrh_model": str(model_path),
            "valid_dbz_range": f"[{dbz_min}, {dbz_max}) dBZ",
            "coordinates": getattr(source_var, "coordinates", "lat lon"),
        }
    )

    valid_count = 0
    total_count = 0
    for start in range(0, source_var.shape[0], chunk_scans):
        stop = min(start + chunk_scans, source_var.shape[0])
        index = (slice(start, stop), slice(None), slice(None))
        dbz_np = np.ma.filled(source_var[index], np.nan).astype(np.float64, copy=False)
        valid = np.isfinite(dbz_np) & (dbz_np >= dbz_min) & (dbz_np < dbz_max)

        # Invalid source values are set to zero only for the temporary tensor.
        # They are restored to NaN in the persisted rain-rate field below.
        safe_dbz = np.where(valid, dbz_np, 0.0)
        rain_np = np.exp(safe_dbz * weight + bias)
        rain_np = rain_np.astype(np.float32, copy=False)
        rain_np[~valid] = np.nan

        rain_var[index] = rain_np
        valid_count += int(valid.sum())
        total_count += valid.size

    return output_name, valid_count, total_count


def process_file(
    source_path: Path,
    output_path: Path,
    variables: Iterable[str],
    weight: np.ndarray,
    bias: np.ndarray,
    dbz_min: float,
    dbz_max: float,
    chunk_scans: int,
    model_path: Path,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        print(f"SKIP existing output: {output_path}")
        return

    temporary_path = output_path.with_suffix(output_path.suffix + ".partial")
    if temporary_path.exists():
        raise FileExistsError(
            f"Temporary output already exists: {temporary_path}. Remove it after checking it."
        )

    try:
        with Dataset(source_path, "r") as source, Dataset(temporary_path, "w") as target:
            available = [name for name in variables if name in source.variables]
            if not available:
                raise KeyError(
                    f"None of the requested dBZ variables exist in {source_path.name}: "
                    f"{list(variables)}"
                )

            copy_source_content(source, target, chunk_scans)
            summaries = []
            for name in available:
                output_name, valid_count, total_count = convert_variable(
                    source.variables[name],
                    target,
                    weight,
                    bias,
                    dbz_min,
                    dbz_max,
                    chunk_scans,
                    model_path,
                )
                summaries.append(f"{output_name}: {valid_count}/{total_count} valid")

            target.setncattr("zrh_model", str(model_path))
            target.setncattr("zrh_formula", "R = exp(dBZ * weight[z] + bias[z])")
            target.setncattr("zrh_valid_dbz_range", f"[{dbz_min}, {dbz_max}) dBZ")
            target.setncattr(
                "history",
                f"{datetime.now(timezone.utc).isoformat()} created by zrh_nc_to_rain.py",
            )

        # A completed temporary file is renamed atomically into its final location.
        temporary_path.replace(output_path)
        print(f"OK {source_path.name} -> {output_path.name} | " + "; ".join(summaries))
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise


def main() -> None:
    args = parse_args()
    if args.chunk_scans <= 0:
        raise ValueError("--chunk-scans must be positive")
    if args.dbz_min >= args.dbz_max:
        raise ValueError("--dbz-min must be smaller than --dbz-max")
    if not args.input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {args.input_dir}")

    input_files = sorted(args.input_dir.glob("*.nc"))
    if not input_files:
        raise FileNotFoundError(f"No .nc files found in: {args.input_dir}")

    weight, bias = load_zrh_parameters(args.model_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model: {args.model_path}")
    print(f"ZRH parameters: weight/bias {tuple(weight.shape)}")
    print(f"Input files: {len(input_files)}")
    print(f"Output directory: {args.output_dir}")
    print(f"dBZ variables: {', '.join(args.variables)}")

    for source_path in input_files:
        process_file(
            source_path=source_path,
            output_path=args.output_dir / source_path.name,
            variables=args.variables,
            weight=weight,
            bias=bias,
            dbz_min=args.dbz_min,
            dbz_max=args.dbz_max,
            chunk_scans=args.chunk_scans,
            model_path=args.model_path,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
