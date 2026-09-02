#!/usr/bin/env python3
"""Build and visualize matched Stage-1/Stage-2/cascade/R1-O orbit products.

The six reference orbits are read from the sealed Stage-1 test-prediction
manifest.  Historical Stage-2 orbit bundles were produced from the validation
split and therefore cannot be joined to those files.  This script reconstructs
only the selected test orbits, caches one self-contained bundle per orbit, and
then creates comparable maps, A--B sections, intensity-bin diagnostics and
Stage-2 decomposition figures.

The resulting test figures are qualitative diagnostics.  They must not be used
to select the next model or tune a support threshold.  In particular, W1.25
always uses its validation-selected threshold and R1-O uses unavailable DPR
anchors plus true DPR support, so R1-O is explicitly non-deployable.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm, Normalize


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for value in (PROJECT_ROOT, SRC_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from plot_nc_sample_diagnostics import (  # noqa: E402
    FigureWriter,
    cumulative_distance_km,
    select_profile_row,
)
from precipitation_inversion.data.nc_reader import read_nc_sample  # noqa: E402
from precipitation_inversion.data.stage2_patch_dataset import (  # noqa: E402
    Stage2PatchDataset,
)
from precipitation_inversion.data.transforms import PerLevelStandardizer  # noqa: E402
from precipitation_inversion.inference.stage2_completion_sliding_window import (  # noqa: E402
    predict_stage2_completion_full_orbit,
)
from precipitation_inversion.inference.stage2_sliding_window import (  # noqa: E402
    predict_stage2_full_orbit,
)
from precipitation_inversion.inference.stage2_stage1_cascade import (  # noqa: E402
    predict_stage1_from_reflectivity_orbit,
)
from precipitation_inversion.training.engine import load_checkpoint  # noqa: E402
from scripts.evaluate_stage1_unet3d import build_model as build_stage1_model  # noqa: E402
from scripts.evaluate_stage2_stage1_cascade import (  # noqa: E402
    _cfb_distance_km,
    _load_json,
    _validate_index_alignment,
    build_r0_subtask_masks,
    project_path,
    resolve_device,
)
from scripts.evaluate_stage2_unet3d import load_threshold_file  # noqa: E402
from scripts.train_stage2_r1_oracle_sparse_value import (  # noqa: E402
    build_model as build_r1_model,
    validate_r1_config,
)
from scripts.train_stage2_unet3d import build_model as build_stage2_model  # noqa: E402
from scripts.visualize_stage2_stage1_cascade import (  # noqa: E402
    _add_section,
    _add_swath,
    _rain_cmap,
    _rain_for_plot,
    _shared_error_limit,
    _shared_rain_norm,
    compute_shared_geographic_extent,
    plot_all_methods_long_tail,
    plot_all_methods_overview,
    plot_mode_vs_target,
)


DEFAULT_STAGE1_DIR = (
    PROJECT_ROOT / "outputs" / "ablations" / "stage1_i_g002_t3d"
)
DEFAULT_STAGE2_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "stage2_ablations"
    / "four_channel_distance_intensity_w1p25"
)
DEFAULT_CASCADE_DIR = PROJECT_ROOT / "outputs" / "stage2_stage1_cascade"
DEFAULT_R1_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "stage2_rethink"
    / "r1_o_dpr_sparse_value"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "multistage_final_model_diagnostics" / "test_six_orbits"
)

BUNDLE_FORMAT = "multistage_final_model_orbit_v1"
SUMMARY_FORMAT = "multistage_final_model_visualization_v1"

# Every bin is defined by the satellite target, never by a model prediction.
# Intervals are left-closed and right-open except for the unbounded last bin.
RAIN_INTENSITY_BINS: tuple[tuple[str, str, float, float], ...] = (
    ("lt_1", "<1", -math.inf, 1.0),
    ("1_to_5", "1-5", 1.0, 5.0),
    ("5_to_10", "5-10", 5.0, 10.0),
    ("10_to_30", "10-30", 10.0, 30.0),
    ("ge_30", ">=30", 30.0, math.inf),
)

DBZ_INTENSITY_BINS: tuple[tuple[str, str, float, float], ...] = (
    ("lt_15", "<15", -math.inf, 15.0),
    ("15_to_25", "15-25", 15.0, 25.0),
    ("25_to_35", "25-35", 25.0, 35.0),
    ("ge_35", ">=35", 35.0, math.inf),
)


@dataclass(frozen=True)
class OrbitSelection:
    """One Stage-1-selected complete test orbit and its cached prediction."""

    selection_position: int
    file_id: int
    sample_id: str
    file_path: Path
    stage1_npz: Path
    ab_scan_index: int


RAIN_MODE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "slug": "stage1_dpr_oracle",
        "display_name": "Stage 1: true DPR dBZ",
        "rain_field": "rain__stage1_dpr_oracle",
        "input_support_field": "input_support__stage1_dpr_oracle",
        "output_support_field": "output_support__stage1_dpr_oracle",
        "deployable": False,
        "role": "sealed Stage-1 upper bound",
    },
    {
        "slug": "gr_interp_stage1",
        "display_name": "Senior GR interpolation + Stage 1",
        "rain_field": "rain__gr_interp_stage1",
        "input_support_field": "input_support__gr_interp_stage1",
        "output_support_field": "output_support__gr_interp_stage1",
        "deployable": True,
        "role": "historical interpolation route",
    },
    {
        "slug": "w1p25_oracle_support",
        "display_name": "W1.25 dBZ + true DPR support",
        "rain_field": "rain__w1p25_oracle_support",
        "input_support_field": "input_support__w1p25_oracle_support",
        "output_support_field": "output_support__w1p25_oracle_support",
        "deployable": False,
        "role": "Stage-2 value-error isolation",
    },
    {
        "slug": "w1p25_predicted_support",
        "display_name": "W1.25 deployable cascade",
        "rain_field": "rain__w1p25_predicted_support",
        "input_support_field": "input_support__w1p25_predicted_support",
        "output_support_field": "output_support__w1p25_predicted_support",
        "deployable": True,
        "role": "Stage-2 -> Stage-1 deployed interface",
    },
    {
        "slug": "r1_o_oracle_support",
        "display_name": "R1-O oracle sparse DPR + true support",
        "rain_field": "rain__r1_o_oracle_support",
        "input_support_field": "input_support__r1_o_oracle_support",
        "output_support_field": "output_support__r1_o_oracle_support",
        "deployable": False,
        "role": "non-deployable spatial-completion upper bound",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-dir", type=Path, default=DEFAULT_STAGE1_DIR)
    parser.add_argument("--stage2-dir", type=Path, default=DEFAULT_STAGE2_DIR)
    parser.add_argument("--cascade-dir", type=Path, default=DEFAULT_CASCADE_DIR)
    parser.add_argument("--r1-dir", type=Path, default=DEFAULT_R1_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stage1-batch-size", type=int, default=1)
    parser.add_argument("--stage2-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--height-km", type=float, default=2.0)
    parser.add_argument("--max-points", type=int, default=200_000)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Require existing cached bundles and skip every model inference.",
    )
    parser.add_argument(
        "--overwrite-cache",
        action="store_true",
        help="Recompute selected orbit bundles even when the cache exists.",
    )
    parser.add_argument(
        "--overwrite-plots",
        action="store_true",
        help="Regenerate completed per-orbit figures.",
    )
    return parser.parse_args()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
        suffix=".tmp", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(_json_safe(value), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def _atomic_npz(path: Path, fields: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".partial.npz")
    np.savez_compressed(temporary, **fields)
    temporary.replace(path)


def discover_stage1_orbits(predictions_dir: Path, count: int | None = 6) -> list[OrbitSelection]:
    """Read the exact Stage-1 qualitative test selection in manifest order."""

    directory = predictions_dir.expanduser().resolve()
    summary = _load_json(directory / "summary.json")
    samples = list(summary.get("samples", ()))
    if not samples:
        raise ValueError("Stage-1 prediction summary contains no samples")
    if count is not None:
        if count <= 0:
            raise ValueError("count must be positive")
        samples = samples[:count]
    selections: list[OrbitSelection] = []
    for item in samples:
        position = int(item["selection_position"])
        sample_dir = directory / f"sample_{position:02d}"
        npz = sample_dir / "prediction_and_target.npz"
        file_path = Path(item["file_path"]).expanduser().resolve()
        if not npz.is_file() or not file_path.is_file():
            raise FileNotFoundError(npz if not npz.is_file() else file_path)
        selections.append(
            OrbitSelection(
                selection_position=position,
                file_id=int(item["file_id"]),
                sample_id=str(item["sample_id"]),
                file_path=file_path,
                stage1_npz=npz,
                ab_scan_index=int(item["ab_scan_index"]),
            )
        )
    return selections


def target_intensity_masks(
    target: np.ndarray, mask: np.ndarray
) -> "OrderedDict[str, np.ndarray]":
    """Partition a common evaluation mask using satellite rain-rate bins."""

    values = np.asarray(target)
    domain = np.asarray(mask)
    if values.shape != domain.shape:
        raise ValueError("target and mask must have identical shapes")
    if domain.dtype != np.bool_:
        raise TypeError("mask must be boolean")
    finite = np.isfinite(values)
    result: "OrderedDict[str, np.ndarray]" = OrderedDict()
    for slug, _label, lower, upper in RAIN_INTENSITY_BINS:
        selected = domain & finite & (values >= lower) & (values < upper)
        result[slug] = selected
    if not np.array_equal(np.logical_or.reduce(tuple(result.values())), domain & finite):
        raise RuntimeError("rain-intensity bins do not partition the finite domain")
    return result


def _dbz_intensity_masks(
    target: np.ndarray, mask: np.ndarray
) -> "OrderedDict[str, np.ndarray]":
    values = np.asarray(target)
    domain = np.asarray(mask)
    result: "OrderedDict[str, np.ndarray]" = OrderedDict()
    for slug, _label, lower, upper in DBZ_INTENSITY_BINS:
        result[slug] = domain & np.isfinite(values) & (values >= lower) & (values < upper)
    return result


def compute_regression_metrics(
    target: np.ndarray, prediction: np.ndarray, mask: np.ndarray
) -> dict[str, Any]:
    """Finite paired regression metrics under one explicit shared mask."""

    truth = np.asarray(target, dtype=np.float64)
    estimate = np.asarray(prediction, dtype=np.float64)
    selected = np.asarray(mask)
    if truth.shape != estimate.shape or truth.shape != selected.shape:
        raise ValueError("target, prediction and mask shapes must match")
    if selected.dtype != np.bool_:
        raise TypeError("metric mask must be boolean")
    selected = selected & np.isfinite(truth) & np.isfinite(estimate)
    count = int(selected.sum())
    if count == 0:
        return {
            "count": 0, "mae": math.nan, "rmse": math.nan, "bias": math.nan,
            "r2": math.nan, "pearson_r": math.nan, "ccc": math.nan,
        }
    x, y = truth[selected], estimate[selected]
    error = y - x
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(np.square(error))))
    bias = float(np.mean(error))
    target_variance_sum = float(np.sum(np.square(x - x.mean())))
    r2 = (
        float(1.0 - np.sum(np.square(error)) / target_variance_sum)
        if target_variance_sum > 0.0 else math.nan
    )
    if count > 1 and float(np.std(x)) > 0.0 and float(np.std(y)) > 0.0:
        correlation = float(np.corrcoef(x, y)[0, 1])
    else:
        correlation = math.nan
    denominator = float(np.var(x) + np.var(y) + (x.mean() - y.mean()) ** 2)
    ccc = float(2.0 * np.cov(x, y, ddof=0)[0, 1] / denominator) if denominator > 0 else math.nan
    return {
        "count": count, "mae": mae, "rmse": rmse, "bias": bias,
        "r2": r2, "pearson_r": correlation, "ccc": ccc,
    }


def _binary_metrics(
    target: np.ndarray, prediction: np.ndarray, domain: np.ndarray
) -> dict[str, Any]:
    truth = np.asarray(target)
    estimate = np.asarray(prediction)
    selected = np.asarray(domain)
    if truth.shape != estimate.shape or truth.shape != selected.shape:
        raise ValueError("binary metric shapes must match")
    if any(value.dtype != np.bool_ for value in (truth, estimate, selected)):
        raise TypeError("binary target, prediction and domain must be boolean")
    tp = int(np.count_nonzero(selected & truth & estimate))
    fp = int(np.count_nonzero(selected & ~truth & estimate))
    fn = int(np.count_nonzero(selected & truth & ~estimate))
    tn = int(np.count_nonzero(selected & ~truth & ~estimate))
    safe = lambda numerator, denominator: numerator / denominator if denominator else math.nan
    return {
        "count": tp + fp + fn + tn,
        "true_positive": tp, "false_positive": fp,
        "false_negative": fn, "true_negative": tn,
        "precision": safe(tp, tp + fp), "recall": safe(tp, tp + fn),
        "false_alarm_ratio": safe(fp, tp + fp),
        "csi": safe(tp, tp + fp + fn),
    }


def _physical_drdz_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    endpoint_mask: np.ndarray,
    heights_km: np.ndarray,
) -> dict[str, Any]:
    """Evaluate adjacent-level physical dR/dz inside reliable endpoints."""

    z = np.asarray(heights_km, dtype=np.float64)
    dz = np.diff(z).reshape((1,) * (target.ndim - 1) + (-1,))
    pair = endpoint_mask[..., :-1] & endpoint_mask[..., 1:]
    target_gradient = np.diff(target, axis=-1) / dz
    prediction_gradient = np.diff(prediction, axis=-1) / dz
    result = compute_regression_metrics(target_gradient, prediction_gradient, pair)
    selected = pair & np.isfinite(target_gradient) & np.isfinite(prediction_gradient)
    if np.any(selected):
        target_abs = float(np.mean(np.abs(target_gradient[selected])))
        prediction_abs = float(np.mean(np.abs(prediction_gradient[selected])))
        result["mean_abs_gradient_ratio"] = (
            prediction_abs / target_abs if target_abs > 0.0 else math.nan
        )
        significant = selected & (
            (np.abs(target_gradient) >= 0.1) | (np.abs(prediction_gradient) >= 0.1)
        )
        result["sign_agreement_fraction"] = (
            float(np.mean(np.sign(target_gradient[significant]) == np.sign(prediction_gradient[significant])))
            if np.any(significant) else math.nan
        )
    else:
        result["mean_abs_gradient_ratio"] = math.nan
        result["sign_agreement_fraction"] = math.nan
    return result


def _load_source(path: Path) -> dict[str, np.ndarray]:
    sample = read_nc_sample(
        path,
        variables=(
            "z", "lat", "lon", "dbz_gr_sparse", "dbz_gr_interp",
            "dbz_dpr", "pre_dpr", "cfb", "typePrecip",
        ),
        dtype=np.float32,
        build_masks=True,
    )
    target = sample.variables["pre_dpr"]
    dpr = sample.masks["dpr_reflectivity_valid"]
    return {
        "heights_km": sample.variables["z"].astype(np.float32),
        "lat": sample.variables["lat"].astype(np.float32),
        "lon": sample.variables["lon"].astype(np.float32),
        "dbz_gr_sparse": sample.variables["dbz_gr_sparse"].astype(np.float32),
        "dbz_gr_interp": sample.variables["dbz_gr_interp"].astype(np.float32),
        "target_dbz": sample.variables["dbz_dpr"].astype(np.float32),
        "target_rain_mm_h": target.astype(np.float32),
        "cfb": sample.variables["cfb"].astype(np.float32),
        "precipitation_type": sample.variables["typePrecip"].astype(np.float32),
        "gr_value_mask": sample.masks["gr_sparse_observed"].astype(bool),
        "gr_interp_mask": sample.masks["gr_interp_observed"].astype(bool),
        "dpr_support": dpr.astype(bool),
        "cfb_clutter": sample.masks["cfb_clutter"].astype(bool),
        "stage2_support_domain": sample.masks["pre_valid_native"].astype(bool),
        "qc_label_mask": sample.masks["pre_valid_qc"].astype(bool),
        "reliable_positive_mask": (
            sample.masks["pre_positive_qc"] & dpr & np.isfinite(target)
        ).astype(bool),
        "cfb_distance_km": _cfb_distance_km(
            sample.variables["cfb"], sample.variables["z"]
        ),
    }


def _index_by_sample_id(dataset: Stage2PatchDataset) -> dict[str, int]:
    mapping = {str(entry["sample_id"]): index for index, entry in enumerate(dataset.files)}
    if len(mapping) != len(dataset.files):
        raise ValueError("Stage-2 index contains duplicate sample IDs")
    return mapping


def _cascade_options(
    stage1_config: Mapping[str, Any], stage1_index: Mapping[str, Any], device: torch.device,
    standardizer: PerLevelStandardizer, heights: np.ndarray, batch_size: int,
) -> dict[str, Any]:
    data = stage1_config["data"]
    return {
        "heights_km": heights,
        "standardizer": standardizer,
        "core_size": int(stage1_index["core_size"]),
        "halo_size": int(stage1_index["halo_size"]),
        "horizontal_multiple": int(stage1_index["horizontal_multiple"]),
        "cfb_input_mode": str(data.get("cfb_input_mode", "baseline")),
        "cfb_distance_scale_km": float(data.get("cfb_distance_scale_km", 2.0)),
        "weak_cfb_layer_weights": tuple(data.get("weak_cfb_layer_weights", ())),
        "device": device,
        "batch_size": batch_size,
        "use_amp": bool(stage1_config.get("training", {}).get("amp", True)),
    }


def _load_stage1_saved(selection: OrbitSelection) -> dict[str, np.ndarray]:
    with np.load(selection.stage1_npz, allow_pickle=False) as archive:
        result = {name: archive[name] for name in archive.files}
    required = {
        "prediction_rain_mm_h", "target_rain_mm_h", "evaluation_mask",
        "positive_target_mask", "heights_km",
    }
    missing = sorted(required.difference(result))
    if missing:
        raise KeyError(f"Stage-1 saved prediction is missing {missing}")
    return result


def _build_orbit_bundle(
    selection: OrbitSelection,
    *,
    stage1_model: torch.nn.Module,
    stage2_model: torch.nn.Module,
    r1_model: torch.nn.Module,
    stage2_dataset: Stage2PatchDataset,
    r1_dataset: Stage2PatchDataset,
    stage2_file_id: int,
    r1_file_id: int,
    stage1_options: Mapping[str, Any],
    support_threshold: float,
    device: torch.device,
    stage2_batch_size: int,
    num_workers: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Run the missing models and return one matched physical-orbit bundle."""

    source = _load_source(selection.file_path)
    saved_stage1 = _load_stage1_saved(selection)
    shape = source["target_rain_mm_h"].shape
    if saved_stage1["prediction_rain_mm_h"].shape != shape:
        raise ValueError("saved Stage-1 prediction and source orbit shapes differ")
    if not np.allclose(
        saved_stage1["target_rain_mm_h"], source["target_rain_mm_h"],
        rtol=0.0, atol=1e-6, equal_nan=True,
    ):
        raise ValueError("saved Stage-1 target differs from the source NetCDF")

    stage2_prediction = predict_stage2_full_orbit(
        stage2_model,
        stage2_dataset,
        stage2_file_id,
        device=device,
        batch_size=stage2_batch_size,
        num_workers=num_workers,
        use_amp=device.type == "cuda",
    )
    r1_prediction = predict_stage2_completion_full_orbit(
        r1_model,
        r1_dataset,
        r1_file_id,
        device=device,
        batch_size=stage2_batch_size,
        num_workers=num_workers,
        use_amp=device.type == "cuda",
    )
    predicted_support = stage2_prediction.support_probability >= support_threshold
    common = {
        **stage1_options,
        "cfb_clutter": source["cfb_clutter"],
        "cfb_index": source["cfb"],
    }
    gr_interp = predict_stage1_from_reflectivity_orbit(
        stage1_model, source["dbz_gr_interp"], source["gr_interp_mask"], **common
    )
    w_oracle = predict_stage1_from_reflectivity_orbit(
        stage1_model, stage2_prediction.reflectivity_dbz, source["dpr_support"], **common
    )
    w_deployed = predict_stage1_from_reflectivity_orbit(
        stage1_model, stage2_prediction.reflectivity_dbz, predicted_support, **common
    )
    r1_oracle = predict_stage1_from_reflectivity_orbit(
        stage1_model, r1_prediction.reflectivity_dbz, source["dpr_support"], **common
    )

    # Rebuild the exact R0 partitions from source masks.  All arrays retain
    # physical orbit shape (nscan,nray,z); no Patch halo or padding survives.
    r0_source = {
        "gr_sparse_valid": source["gr_value_mask"],
        "gr_interp_valid": source["gr_interp_mask"],
        "pre_valid_native_mask": source["stage2_support_domain"],
        "dpr_valid": source["dpr_support"],
        "dbz_dpr": source["target_dbz"],
    }
    r0_masks = build_r0_subtask_masks(r0_source)
    stage1_output_support = source["dpr_support"] & ~source["cfb_clutter"]
    fields: dict[str, np.ndarray] = {
        **source,
        "rain__stage1_dpr_oracle": saved_stage1["prediction_rain_mm_h"].astype(np.float32),
        "rain__gr_interp_stage1": gr_interp.rain_rate_mm_h,
        "rain__w1p25_oracle_support": w_oracle.rain_rate_mm_h,
        "rain__w1p25_predicted_support": w_deployed.rain_rate_mm_h,
        "rain__r1_o_oracle_support": r1_oracle.rain_rate_mm_h,
        "input_support__stage1_dpr_oracle": source["dpr_support"],
        "output_support__stage1_dpr_oracle": stage1_output_support,
        "input_support__gr_interp_stage1": gr_interp.input_support,
        "output_support__gr_interp_stage1": gr_interp.output_support,
        "input_support__w1p25_oracle_support": w_oracle.input_support,
        "output_support__w1p25_oracle_support": w_oracle.output_support,
        "input_support__w1p25_predicted_support": w_deployed.input_support,
        "output_support__w1p25_predicted_support": w_deployed.output_support,
        "input_support__r1_o_oracle_support": r1_oracle.input_support,
        "output_support__r1_o_oracle_support": r1_oracle.output_support,
        "stage2_w1p25_dbz": stage2_prediction.reflectivity_dbz,
        "stage2_w1p25_support_probability": stage2_prediction.support_probability,
        "stage2_r1_o_dbz": r1_prediction.reflectivity_dbz,
        "w1p25_predicted_support": predicted_support.astype(bool),
        "anchor_mask": r0_masks.q11_overlap.astype(bool),
        "gap_mask": r0_masks.dpr_only_gap.astype(bool),
        "outside_mask": r0_masks.dpr_only_outside.astype(bool),
        "q10_gr_only_mask": r0_masks.q10_gr_only.astype(bool),
        "q00_neither_mask": r0_masks.q00_neither.astype(bool),
    }
    # The original Stage-1 files use ``evaluation_mask`` for reliable positive
    # DPR-reflectivity/rain pairs despite the historical name.  Require exact
    # agreement before preserving it as provenance.
    if not np.array_equal(
        saved_stage1["evaluation_mask"].astype(bool), source["reliable_positive_mask"]
    ):
        raise ValueError("Stage-1 and current reliable-positive masks differ")
    validate_orbit_bundle(fields)
    metadata = {
        "format": BUNDLE_FORMAT,
        "split": "test",
        "test_set_accessed": True,
        "qualitative_diagnostics_only": True,
        "selection_position": selection.selection_position,
        "file_id": selection.file_id,
        "sample_id": selection.sample_id,
        "source_file": str(selection.file_path),
        "shape": list(shape),
        "ab_scan_index_from_stage1": selection.ab_scan_index,
        "support_threshold": support_threshold,
        "support_threshold_source": "validation-only W1.25 threshold file",
        "rain_modes": list(RAIN_MODE_SPECS),
    }
    return fields, metadata


def validate_orbit_bundle(fields: Mapping[str, np.ndarray]) -> tuple[int, int, int]:
    """Validate semantic dtypes and the shared physical `(scan,ray,z)` grid."""

    float3d = (
        "target_rain_mm_h", "target_dbz", "dbz_gr_sparse", "dbz_gr_interp",
        "stage2_w1p25_dbz", "stage2_w1p25_support_probability", "stage2_r1_o_dbz",
    )
    bool3d = (
        "reliable_positive_mask", "qc_label_mask", "dpr_support",
        "stage2_support_domain", "gr_value_mask", "gr_interp_mask", "anchor_mask",
        "gap_mask", "outside_mask", "w1p25_predicted_support",
    )
    missing = sorted(set(float3d + bool3d + ("heights_km", "lat", "lon")).difference(fields))
    if missing:
        raise KeyError(f"orbit bundle is missing fields: {missing}")
    shape = np.asarray(fields["target_rain_mm_h"]).shape
    if len(shape) != 3 or any(size <= 0 for size in shape):
        raise ValueError("orbit bundle fields must use non-empty (nscan,nray,z)")
    for name in float3d:
        value = np.asarray(fields[name])
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if not np.issubdtype(value.dtype, np.floating):
            raise TypeError(f"{name} must be floating")
    for name in bool3d:
        value = np.asarray(fields[name])
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if value.dtype != np.bool_:
            raise TypeError(f"{name} must be boolean")
    rain_fields = sorted(name for name in fields if name.startswith("rain__"))
    if not rain_fields:
        raise KeyError("orbit bundle must contain at least one rain__<mode> field")
    for rain_name in rain_fields:
        slug = rain_name.removeprefix("rain__")
        rain = np.asarray(fields[rain_name])
        if rain.shape != shape or not np.issubdtype(rain.dtype, np.floating):
            raise TypeError(f"{rain_name} must be floating and have shape {shape}")
        for prefix in ("input_support__", "output_support__"):
            support_name = prefix + slug
            if support_name not in fields:
                raise KeyError(f"orbit bundle is missing mode field {support_name}")
            support = np.asarray(fields[support_name])
            if support.shape != shape or support.dtype != np.bool_:
                raise TypeError(f"{support_name} must be boolean and have shape {shape}")
    z = np.asarray(fields["heights_km"])
    if z.shape != (shape[-1],) or not np.all(np.isfinite(z)) or not np.all(np.diff(z) > 0):
        raise ValueError("heights_km must be finite, increasing and match z")
    if np.asarray(fields["lat"]).shape != shape[:2] or np.asarray(fields["lon"]).shape != shape[:2]:
        raise ValueError("lat/lon must match the horizontal orbit grid")
    if np.any(fields["anchor_mask"] & (fields["gap_mask"] | fields["outside_mask"])):
        raise ValueError("anchor/gap/outside masks must be disjoint")
    return tuple(int(value) for value in shape)


def _validated_mode_fields(
    fields: Mapping[str, np.ndarray], mode_specs: Sequence[Mapping[str, Any]]
) -> list[tuple[Mapping[str, Any], np.ndarray, np.ndarray, np.ndarray]]:
    shape = np.asarray(fields["target_rain_mm_h"]).shape
    if not mode_specs:
        raise ValueError("at least one rain mode is required")
    result: list[tuple[Mapping[str, Any], np.ndarray, np.ndarray, np.ndarray]] = []
    seen: set[str] = set()
    for mode in mode_specs:
        slug = str(mode["slug"])
        if slug in seen:
            raise ValueError(f"duplicate rain mode slug: {slug}")
        seen.add(slug)
        rain_name = str(mode.get("rain_field", f"rain__{slug}"))
        input_name = str(mode.get("input_support_field", f"input_support__{slug}"))
        output_name = str(mode.get("output_support_field", f"output_support__{slug}"))
        missing = [name for name in (rain_name, input_name, output_name) if name not in fields]
        if missing:
            raise KeyError(f"rain mode {slug} is missing fields: {missing}")
        rain = np.asarray(fields[rain_name])
        input_support = np.asarray(fields[input_name])
        output_support = np.asarray(fields[output_name])
        if any(value.shape != shape for value in (rain, input_support, output_support)):
            raise ValueError(f"rain mode {slug} does not match orbit shape {shape}")
        if input_support.dtype != np.bool_ or output_support.dtype != np.bool_:
            raise TypeError(f"rain mode {slug} support fields must be boolean")
        result.append((mode, rain, input_support, output_support))
    return result


def plot_rain_intensity_analysis(
    fields: Mapping[str, np.ndarray],
    mode_specs: Sequence[Mapping[str, Any]],
    *,
    sample_id: str,
) -> tuple[plt.Figure, dict[str, Any]]:
    """Compare every cascade under identical satellite-defined rain bins."""

    target = np.asarray(fields["target_rain_mm_h"])
    domain = np.asarray(fields["reliable_positive_mask"])
    modes = _validated_mode_fields(fields, mode_specs)
    bin_masks = target_intensity_masks(target, domain)
    metrics: dict[str, Any] = {}
    for mode, prediction, _input_support, _output_support in modes:
        by_bin = {
            slug: compute_regression_metrics(target, prediction, selected)
            for slug, selected in bin_masks.items()
        }
        metrics[str(mode["slug"])] = {
            "display_name": str(mode["display_name"]),
            "deployable": bool(mode.get("deployable", False)),
            "overall": compute_regression_metrics(target, prediction, domain),
            "by_target_intensity": by_bin,
        }

    labels = [item[1] for item in RAIN_INTENSITY_BINS]
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.9, len(modes)))
    figure, axes = plt.subplots(2, 3, figsize=(20, 12), constrained_layout=True)
    counts = [int(bin_masks[item[0]].sum()) for item in RAIN_INTENSITY_BINS]
    axes[0, 0].bar(labels, counts, color="#457b9d")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Satellite target count by rain-rate bin")
    axes[0, 0].set_xlabel("Target rain rate (mm/h)")
    axes[0, 0].set_ylabel("Voxel count (log scale)")

    metric_panels = (
        ("mae", "MAE (mm/h)", axes[0, 1]),
        ("rmse", "RMSE (mm/h)", axes[0, 2]),
        ("bias", "Bias = prediction - target (mm/h)", axes[1, 0]),
        ("pearson_r", "Pearson r", axes[1, 1]),
        ("ccc", "CCC", axes[1, 2]),
    )
    x = np.arange(len(labels), dtype=np.float64)
    for metric_name, title, axis in metric_panels:
        for color, (mode, _prediction, _input, _output) in zip(colors, modes):
            slug = str(mode["slug"])
            values = [
                metrics[slug]["by_target_intensity"][bin_slug][metric_name]
                for bin_slug, _label, _lower, _upper in RAIN_INTENSITY_BINS
            ]
            axis.plot(
                x, values, "o-", linewidth=1.8, markersize=5,
                color=color, label=str(mode["display_name"]),
            )
        axis.set_xticks(x, labels)
        axis.set_title(title)
        axis.set_xlabel("Satellite target rain-rate bin (mm/h)")
        axis.grid(alpha=0.25)
        if metric_name == "bias":
            axis.axhline(0.0, color="black", linewidth=0.8)
        if metric_name in {"pearson_r", "ccc"}:
            axis.set_ylim(-0.1, 1.0)
    axes[0, 1].legend(fontsize=7)
    figure.suptitle(
        f"Target-defined rain intensity diagnostics: {sample_id}\n"
        "All methods use the same reliable-positive satellite voxels",
        fontsize=15,
    )
    return figure, metrics


def _subsample_pair(
    target: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
    maximum: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    selected = mask & np.isfinite(target) & np.isfinite(prediction)
    indices = np.flatnonzero(selected)
    if indices.size > maximum:
        indices = rng.choice(indices, size=maximum, replace=False)
    return target.ravel()[indices], prediction.ravel()[indices]


def _plane_image(
    axis: plt.Axes,
    values: np.ndarray,
    title: str,
    *,
    cmap: Any,
    norm: Any | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
):
    artist = axis.imshow(
        np.asarray(values).T,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    axis.set_title(title)
    axis.set_xlabel("scan")
    axis.set_ylabel("ray")
    return artist


def plot_stage2_audit(
    fields: Mapping[str, np.ndarray],
    *,
    sample_id: str,
    height_km: float,
    max_points: int,
    rng: np.random.Generator,
) -> tuple[plt.Figure, dict[str, Any]]:
    """Show W1.25 support plus R1-O anchor/gap/outside value recovery."""

    if max_points <= 0:
        raise ValueError("max_points must be positive")
    z = np.asarray(fields["heights_km"])
    level = int(np.argmin(np.abs(z - height_km)))
    target_dbz = np.asarray(fields["target_dbz"])
    w_dbz = np.asarray(fields["stage2_w1p25_dbz"])
    r1_dbz = np.asarray(fields["stage2_r1_o_dbz"])
    target_support = np.asarray(fields["dpr_support"])
    domain = np.asarray(fields["stage2_support_domain"])
    probability = np.asarray(fields["stage2_w1p25_support_probability"])
    predicted_support = np.asarray(fields["w1p25_predicted_support"])
    region_masks = OrderedDict(
        (
            ("anchor", np.asarray(fields["anchor_mask"])),
            ("gap", np.asarray(fields["gap_mask"])),
            ("outside", np.asarray(fields["outside_mask"])),
        )
    )
    region_metrics: dict[str, Any] = {}
    for name, selected in region_masks.items():
        selected = selected & target_support
        region_metrics[name] = {
            "count": int(selected.sum()),
            "w1p25": compute_regression_metrics(target_dbz, w_dbz, selected),
            "r1_o": compute_regression_metrics(target_dbz, r1_dbz, selected),
        }
    dbz_bins = _dbz_intensity_masks(target_dbz, target_support)
    intensity_metrics = {
        name: {
            "count": int(selected.sum()),
            "w1p25": compute_regression_metrics(target_dbz, w_dbz, selected),
            "r1_o": compute_regression_metrics(target_dbz, r1_dbz, selected),
        }
        for name, selected in dbz_bins.items()
    }
    support_metrics = _binary_metrics(target_support, predicted_support, domain)

    figure, axes = plt.subplots(4, 4, figsize=(22, 19), constrained_layout=True)
    planes = (
        (fields["gr_value_mask"][..., level], "Direct GR value mask", "gray", 0, 1),
        (target_support[..., level], "True DPR support", "gray", 0, 1),
        (probability[..., level], "W1.25 support probability", "viridis", 0, 1),
        (predicted_support[..., level], "W1.25 predicted support @ val threshold", "gray", 0, 1),
    )
    for axis, (values, title, cmap, lower, upper) in zip(axes[0], planes):
        image = _plane_image(
            axis, values, f"{title}\nz={z[level]:.3f} km", cmap=cmap, vmin=lower, vmax=upper
        )
        figure.colorbar(image, ax=axis, shrink=0.72)

    category = np.zeros(target_support.shape, dtype=np.int8)
    category[region_masks["anchor"]] = 1
    category[region_masks["gap"]] = 2
    category[region_masks["outside"]] = 3
    category_cmap = ListedColormap(["#f0f0f0", "#2a9d8f", "#e9c46a", "#e76f51"])
    image = _plane_image(
        axes[1, 0], category[..., level], "DPR-positive observability region",
        cmap=category_cmap, norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], 4),
    )
    colorbar = figure.colorbar(image, ax=axes[1, 0], shrink=0.72, ticks=[0, 1, 2, 3])
    colorbar.ax.set_yticklabels(["other", "anchor", "gap", "outside"])
    support_error = np.zeros(target_support.shape, dtype=np.int8)
    support_error[domain & ~target_support & predicted_support] = 1
    support_error[domain & target_support & ~predicted_support] = 2
    support_error[domain & target_support & predicted_support] = 3
    support_cmap = ListedColormap(["#eeeeee", "#e76f51", "#457b9d", "#2a9d8f"])
    image = _plane_image(
        axes[1, 1], support_error[..., level], "W1.25 support confusion",
        cmap=support_cmap, norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], 4),
    )
    colorbar = figure.colorbar(image, ax=axes[1, 1], shrink=0.72, ticks=[0, 1, 2, 3])
    colorbar.ax.set_yticklabels(["TN/out", "FP", "FN", "TP"])
    for axis, values, title in (
        (axes[1, 2], np.where(target_support[..., level], target_dbz[..., level], np.nan), "True DPR dBZ"),
        (axes[1, 3], np.where(target_support[..., level], r1_dbz[..., level], np.nan), "R1-O dBZ on true support"),
    ):
        image = _plane_image(axis, values, title, cmap="turbo", vmin=0, vmax=50)
        figure.colorbar(image, ax=axis, shrink=0.72, label="dBZ")

    names = list(region_masks)
    x = np.arange(len(names))
    for metric_name, title, axis in (
        ("rmse", "dBZ RMSE by observability region", axes[2, 0]),
        ("pearson_r", "dBZ Pearson r by observability region", axes[2, 1]),
        ("bias", "dBZ bias by observability region", axes[2, 2]),
    ):
        width = 0.35
        for offset, key, label, color in (
            (-width / 2, "w1p25", "W1.25", "#457b9d"),
            (width / 2, "r1_o", "R1-O", "#e76f51"),
        ):
            axis.bar(
                x + offset,
                [region_metrics[name][key][metric_name] for name in names],
                width,
                label=label,
                color=color,
            )
        axis.set_xticks(x, names)
        axis.set_title(title)
        axis.grid(alpha=0.25, axis="y")
        if metric_name == "bias":
            axis.axhline(0.0, color="black", linewidth=0.8)
        if metric_name == "pearson_r":
            axis.set_ylim(-0.1, 1.0)
    axes[2, 0].legend()
    positive_probability = probability[domain & target_support]
    negative_probability = probability[domain & ~target_support]
    if negative_probability.size:
        axes[2, 3].hist(negative_probability, bins=40, range=(0, 1), density=True, alpha=0.55, label="DPR negative")
    if positive_probability.size:
        axes[2, 3].hist(positive_probability, bins=40, range=(0, 1), density=True, alpha=0.55, label="DPR positive")
    axes[2, 3].axvline(0.8, color="black", linestyle="--", label="validation threshold")
    axes[2, 3].set_title(
        f"W1.25 support probability\nCSI={support_metrics['csi']:.3f}, recall={support_metrics['recall']:.3f}"
    )
    axes[2, 3].set_xlabel("support probability")
    axes[2, 3].legend(fontsize=8)

    dbz_labels = [item[1] for item in DBZ_INTENSITY_BINS]
    for metric_name, title, axis in (
        ("rmse", "dBZ RMSE by target-intensity bin", axes[3, 0]),
        ("pearson_r", "dBZ Pearson r by target-intensity bin", axes[3, 1]),
    ):
        for key, label, color in (("w1p25", "W1.25", "#457b9d"), ("r1_o", "R1-O", "#e76f51")):
            axis.plot(
                dbz_labels,
                [intensity_metrics[item[0]][key][metric_name] for item in DBZ_INTENSITY_BINS],
                "o-", label=label, color=color,
            )
        axis.set_title(title)
        axis.grid(alpha=0.25)
        if metric_name == "pearson_r":
            axis.set_ylim(-0.1, 1.0)
    axes[3, 0].legend()

    for axis, prediction, title, color in (
        (axes[3, 2], w_dbz, "W1.25 dBZ correlation", "#457b9d"),
        (axes[3, 3], r1_dbz, "R1-O dBZ correlation", "#e76f51"),
    ):
        x_values, y_values = _subsample_pair(
            target_dbz, prediction, target_support, max_points, rng
        )
        if x_values.size:
            axis.hexbin(x_values, y_values, gridsize=60, bins="log", mincnt=1, cmap="viridis")
            lower = min(float(x_values.min()), float(y_values.min()))
            upper = max(float(x_values.max()), float(y_values.max()))
            axis.plot([lower, upper], [lower, upper], "k--", linewidth=1)
        metric = compute_regression_metrics(target_dbz, prediction, target_support)
        axis.set_title(f"{title}\nRMSE={metric['rmse']:.3f}, r={metric['pearson_r']:.3f}")
        axis.set_xlabel("true DPR dBZ")
        axis.set_ylabel("predicted dBZ")

    figure.suptitle(
        f"Stage-2 decomposition audit: {sample_id}\n"
        "R1-O uses true DPR sparse anchors and true support (non-deployable)",
        fontsize=15,
    )
    return figure, {
        "height_index": level,
        "height_km": float(z[level]),
        "support": support_metrics,
        "regions": region_metrics,
        "intensity": intensity_metrics,
    }


def plot_reflectivity_comparison(
    fields: Mapping[str, np.ndarray],
    *,
    sample_id: str,
    height_km: float,
    ab_scan_index: int,
    max_points: int,
    rng: np.random.Generator,
) -> tuple[plt.Figure, dict[str, Any]]:
    """Compare sparse GR, true DPR, W1.25 and R1-O on one shared grid."""

    shape = np.asarray(fields["target_dbz"]).shape
    if not 0 <= ab_scan_index < shape[0]:
        raise ValueError("ab_scan_index is outside the orbit")
    z = np.asarray(fields["heights_km"])
    level = int(np.argmin(np.abs(z - height_km)))
    lat, lon = np.asarray(fields["lat"]), np.asarray(fields["lon"])
    target = np.asarray(fields["target_dbz"])
    support = np.asarray(fields["dpr_support"])
    gr_mask = np.asarray(fields["gr_value_mask"])
    w_prediction = np.asarray(fields["stage2_w1p25_dbz"])
    r1_prediction = np.asarray(fields["stage2_r1_o_dbz"])
    fields_and_masks = (
        (fields["dbz_gr_sparse"], gr_mask, "Sparse GR dBZ"),
        (target, support, "True DPR dBZ"),
        (w_prediction, support, "W1.25 dBZ on true support"),
        (r1_prediction, support, "R1-O dBZ on true support"),
    )
    qc_footprint = np.any(fields["stage2_support_domain"], axis=-1)
    geographic_extent = compute_shared_geographic_extent(lon, lat, qc_footprint)
    dbz_norm = Normalize(0.0, 50.0)
    error_values = np.concatenate(
        (
            np.abs(w_prediction[support] - target[support]),
            np.abs(r1_prediction[support] - target[support]),
        )
    )
    error_limit = max(5.0, float(np.percentile(error_values, 99.0))) if error_values.size else 15.0
    error_norm = Normalize(-error_limit, error_limit)
    figure, axes = plt.subplots(3, 4, figsize=(22, 16), constrained_layout=True)
    for axis, (values, mask, title) in zip(axes[0], fields_and_masks):
        plane = np.where(mask[..., level], np.asarray(values)[..., level], np.nan)
        image = _add_swath(
            axis, lon, lat, plane, f"{title}\nz={z[level]:.3f} km",
            cmap="turbo", norm=dbz_norm, qc_footprint=qc_footprint,
            geographic_extent=geographic_extent,
            true_support=support[..., level] if title != "Sparse GR dBZ" else None,
            predicted_support=gr_mask[..., level] if title == "Sparse GR dBZ" else None,
        )
        valid_line = np.isfinite(lat[ab_scan_index]) & np.isfinite(lon[ab_scan_index])
        axis.plot(lon[ab_scan_index, valid_line], lat[ab_scan_index, valid_line], color="white", linewidth=1.0)
        figure.colorbar(image, ax=axis, shrink=0.72, label="dBZ")

    line_valid = np.isfinite(lat[ab_scan_index]) & np.isfinite(lon[ab_scan_index])
    distance = cumulative_distance_km(
        lat[ab_scan_index, line_valid], lon[ab_scan_index, line_valid]
    )
    for axis, (values, mask, title) in zip(axes[1], fields_and_masks):
        section = np.where(
            mask[ab_scan_index, line_valid, :],
            np.asarray(values)[ab_scan_index, line_valid, :],
            np.nan,
        ).T
        image = _add_section(
            axis, section, z, distance,
            f"{title} A-B; scan={ab_scan_index}", cmap="turbo", norm=dbz_norm,
            true_support=(
                support[ab_scan_index, line_valid, :].T
                if title != "Sparse GR dBZ" else None
            ),
            predicted_support=(
                gr_mask[ab_scan_index, line_valid, :].T
                if title == "Sparse GR dBZ" else None
            ),
        )
        figure.colorbar(image, ax=axis, shrink=0.72, label="dBZ")

    for axis, prediction, title in (
        (axes[2, 0], w_prediction, "W1.25 error on true DPR support"),
        (axes[2, 1], r1_prediction, "R1-O error on true DPR support"),
    ):
        error = np.where(support[..., level], prediction[..., level] - target[..., level], np.nan)
        image = _add_swath(
            axis, lon, lat, error, title, cmap="coolwarm", norm=error_norm,
            qc_footprint=qc_footprint, geographic_extent=geographic_extent,
            true_support=support[..., level],
        )
        figure.colorbar(image, ax=axis, shrink=0.72, label="prediction - target (dBZ)")
    metric_result: dict[str, Any] = {}
    for axis, prediction, slug, title in (
        (axes[2, 2], w_prediction, "w1p25", "W1.25 all-support correlation"),
        (axes[2, 3], r1_prediction, "r1_o", "R1-O all-support correlation"),
    ):
        truth_values, predicted_values = _subsample_pair(
            target, prediction, support, max_points, rng
        )
        if truth_values.size:
            axis.hexbin(truth_values, predicted_values, gridsize=65, bins="log", mincnt=1, cmap="viridis")
            lower = min(float(truth_values.min()), float(predicted_values.min()))
            upper = max(float(truth_values.max()), float(predicted_values.max()))
            axis.plot([lower, upper], [lower, upper], "r--", linewidth=1)
        metric_result[slug] = compute_regression_metrics(target, prediction, support)
        axis.set_title(
            f"{title}\nRMSE={metric_result[slug]['rmse']:.3f}, "
            f"r={metric_result[slug]['pearson_r']:.3f}"
        )
        axis.set_xlabel("true DPR dBZ")
        axis.set_ylabel("predicted dBZ")
    figure.suptitle(
        f"Matched reflectivity fields and A-B sections: {sample_id}\n"
        "Every DPR comparison uses the same true-support mask and physical dBZ scale",
        fontsize=15,
    )
    return figure, {
        "height_index": level,
        "height_km": float(z[level]),
        "ab_scan_index": ab_scan_index,
        "error_limit_dbz": error_limit,
        "metrics": metric_result,
    }


def _vertical_regression(
    target: np.ndarray, prediction: np.ndarray, mask: np.ndarray
) -> dict[str, np.ndarray]:
    metrics = {name: np.full(target.shape[-1], np.nan, dtype=np.float64) for name in ("mae", "rmse", "bias", "pearson_r")}
    for level in range(target.shape[-1]):
        result = compute_regression_metrics(
            target[..., level], prediction[..., level], mask[..., level]
        )
        for name in metrics:
            metrics[name][level] = result[name]
    return metrics


def _cfad_fraction(values: np.ndarray, mask: np.ndarray, edges: np.ndarray) -> np.ndarray:
    result = np.zeros((values.shape[-1], edges.size - 1), dtype=np.float64)
    for level in range(values.shape[-1]):
        selected = values[..., level][mask[..., level] & np.isfinite(values[..., level])]
        if selected.size:
            counts, _ = np.histogram(selected, bins=edges)
            result[level] = counts / counts.sum()
    return result


def _coordinate_edges(centers: np.ndarray) -> np.ndarray:
    """Convert increasing bin centers to edges for ``pcolormesh``.

    A CFAD with ``H`` physical height levels has shape ``(H, N)`` and thus
    requires ``H + 1`` vertical edges.  Keeping this conversion explicit avoids
    silently shifting a profile by half a layer.
    """

    values = np.asarray(centers, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("coordinate centers must be a non-empty finite 1-D array")
    if values.size == 1:
        return np.asarray([values[0] - 0.5, values[0] + 0.5], dtype=np.float64)
    differences = np.diff(values)
    if np.any(differences <= 0.0):
        raise ValueError("coordinate centers must be strictly increasing")
    midpoints = (values[:-1] + values[1:]) / 2.0
    return np.concatenate(
        ([values[0] - differences[0] / 2.0], midpoints, [values[-1] + differences[-1] / 2.0])
    )


def plot_vertical_structure(
    fields: Mapping[str, np.ndarray],
    mode_specs: Sequence[Mapping[str, Any]],
    *,
    sample_id: str,
) -> tuple[plt.Figure, dict[str, Any]]:
    """Plot dBZ CFADs, height errors and final-rain physical dR/dz."""

    modes = _validated_mode_fields(fields, mode_specs)
    z = np.asarray(fields["heights_km"], dtype=np.float64)
    target_dbz = np.asarray(fields["target_dbz"])
    dpr_support = np.asarray(fields["dpr_support"])
    w_dbz = np.asarray(fields["stage2_w1p25_dbz"])
    r1_dbz = np.asarray(fields["stage2_r1_o_dbz"])
    dbz_edges = np.arange(-10.0, 65.0, 5.0)
    cfads = {
        "target": _cfad_fraction(target_dbz, dpr_support, dbz_edges),
        "w1p25": _cfad_fraction(w_dbz, dpr_support, dbz_edges),
        "r1_o": _cfad_fraction(r1_dbz, dpr_support, dbz_edges),
    }
    positive = np.concatenate([value[value > 0.0] for value in cfads.values()])
    norm = LogNorm(
        vmin=max(float(positive.min()), 1e-5) if positive.size else 1e-5,
        vmax=max(float(positive.max()), 1e-2) if positive.size else 1.0,
    )
    figure, axes = plt.subplots(2, 3, figsize=(20, 13), constrained_layout=True)
    height_edges = _coordinate_edges(z)
    for axis, key, title in (
        (axes[0, 0], "target", "True DPR CFAD"),
        (axes[0, 1], "w1p25", "W1.25 CFAD on true support"),
        (axes[0, 2], "r1_o", "R1-O CFAD on true support"),
    ):
        image = axis.pcolormesh(
            dbz_edges, height_edges, np.ma.masked_less_equal(cfads[key], 0.0),
            shading="auto", cmap="magma", norm=norm,
        )
        axis.set_title(title)
        axis.set_xlabel("Reflectivity (dBZ)")
        axis.set_ylabel("Height (km)")
        figure.colorbar(image, ax=axis, shrink=0.75, label="fraction at height")

    vertical_dbz = {
        "w1p25": _vertical_regression(target_dbz, w_dbz, dpr_support),
        "r1_o": _vertical_regression(target_dbz, r1_dbz, dpr_support),
    }
    for key, label, color in (("w1p25", "W1.25", "#457b9d"), ("r1_o", "R1-O", "#e76f51")):
        axes[1, 0].plot(vertical_dbz[key]["rmse"], z, label=f"{label} RMSE", color=color)
    axes[1, 0].set_title("dBZ RMSE by height")
    axes[1, 0].set_xlabel("RMSE (dBZ)")
    axes[1, 0].set_ylabel("Height (km)")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend()
    correlation_axis = axes[1, 0].twiny()
    for key, label, color in (("w1p25", "W1.25 r", "#264653"), ("r1_o", "R1-O r", "#f4a261")):
        correlation_axis.plot(vertical_dbz[key]["pearson_r"], z, linestyle="--", label=label, color=color)
    correlation_axis.set_xlim(-0.1, 1.0)
    correlation_axis.set_xlabel("Pearson r")

    target_rain = np.asarray(fields["target_rain_mm_h"])
    rain_mask = np.asarray(fields["reliable_positive_mask"])
    vertical_rain: dict[str, Any] = {}
    drdz: dict[str, Any] = {}
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.9, len(modes)))
    for color, (mode, prediction, _input, _output) in zip(colors, modes):
        slug = str(mode["slug"])
        vertical_rain[slug] = _vertical_regression(target_rain, prediction, rain_mask)
        drdz[slug] = _physical_drdz_metrics(target_rain, prediction, rain_mask, z)
        axes[1, 1].plot(
            vertical_rain[slug]["rmse"], z,
            label=str(mode["display_name"]), color=color,
        )
    axes[1, 1].set_title("Final-rain RMSE by height")
    axes[1, 1].set_xlabel("RMSE (mm/h)")
    axes[1, 1].set_ylabel("Height (km)")
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend(fontsize=7)

    x = np.arange(len(modes))
    names = [str(item[0]["display_name"]) for item in modes]
    axes[1, 2].bar(x - 0.18, [drdz[str(item[0]["slug"])]["pearson_r"] for item in modes], 0.36, label="dR/dz r")
    axes[1, 2].bar(x + 0.18, [drdz[str(item[0]["slug"])]["ccc"] for item in modes], 0.36, label="dR/dz CCC")
    axes[1, 2].set_xticks(x, names, rotation=22, ha="right")
    axes[1, 2].set_ylim(-0.1, 1.0)
    axes[1, 2].set_title("Physical vertical-gradient agreement")
    axes[1, 2].legend()
    axes[1, 2].grid(alpha=0.25, axis="y")
    figure.suptitle(
        f"Vertical structure and CFAD: {sample_id}\n"
        "R1-O support is oracle; its echo-top/base are not learned metrics",
        fontsize=15,
    )
    return figure, {
        "cfad_dbz_edges": dbz_edges.tolist(),
        "cfad": {key: value.tolist() for key, value in cfads.items()},
        "vertical_dbz": {
            key: {metric: value.tolist() for metric, value in metrics.items()}
            for key, metrics in vertical_dbz.items()
        },
        "vertical_rain": {
            key: {metric: value.tolist() for metric, value in metrics.items()}
            for key, metrics in vertical_rain.items()
        },
        "physical_drdz": drdz,
    }


def _load_bundle(directory: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    metadata = _load_json(directory / "metadata.json")
    if metadata.get("format") != BUNDLE_FORMAT:
        raise ValueError(f"unsupported multistage bundle: {directory}")
    with np.load(directory / "fields.npz", allow_pickle=False) as archive:
        fields = {name: archive[name] for name in archive.files}
    validate_orbit_bundle(fields)
    return fields, metadata


def render_orbit_report(
    bundle_directory: Path,
    *,
    output_directory: Path,
    height_km: float,
    max_points: int,
    dpi: int,
    overwrite: bool,
) -> dict[str, Any]:
    """Render one cached orbit into PNG pages, one PDF and a figure index."""

    complete = output_directory / ".complete"
    if complete.is_file() and not overwrite:
        return _load_json(output_directory / "metrics.json")
    fields, metadata = _load_bundle(bundle_directory)
    mode_specs = list(metadata.get("rain_modes", RAIN_MODE_SPECS))
    modes = _validated_mode_fields(fields, mode_specs)
    target = fields["target_rain_mm_h"]
    qc_mask = fields["qc_label_mask"]
    reliable = fields["reliable_positive_mask"]
    z = fields["heights_km"]
    lat, lon = fields["lat"], fields["lon"]
    true_support = fields["dpr_support"]
    predictions = [item[1] for item in modes]
    input_supports = [item[2] for item in modes]
    output_supports = [item[3] for item in modes]
    mode_names = [str(item[0]["display_name"]) for item in modes]
    height_index = int(np.argmin(np.abs(z - height_km)))
    ab_scan = int(metadata.get("ab_scan_index_from_stage1", -1))
    if not 0 <= ab_scan < target.shape[0]:
        ab_scan = select_profile_row(target, height_index)
    rain_norm = _shared_rain_norm(target, predictions, qc_mask, output_supports)
    error_limit = _shared_error_limit(target, predictions, qc_mask)
    qc_footprint = np.any(qc_mask, axis=-1)
    geographic_extent = compute_shared_geographic_extent(lon, lat, qc_footprint)
    output_directory.mkdir(parents=True, exist_ok=True)
    writer = FigureWriter(output_directory, dpi)
    result: dict[str, Any] = {
        "format": SUMMARY_FORMAT,
        "sample_id": metadata["sample_id"],
        "source_file": metadata["source_file"],
        "split": "test",
        "test_set_accessed": True,
        "qualitative_diagnostics_only": True,
        "height_index": height_index,
        "height_km": float(z[height_index]),
        "ab_scan_index": ab_scan,
        "rain_modes": {},
    }
    try:
        writer.save(
            plot_all_methods_overview(
                target=target,
                predictions=predictions,
                input_supports=input_supports,
                output_supports=output_supports,
                true_dpr_support=true_support,
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
            "rain/00_all_methods_same_height.png",
            "同一高度、经纬范围和降水色标下的卫星标签与全部阶段输出。",
        )
        for position, (mode, prediction, input_support, output_support) in enumerate(modes, start=1):
            figure, metrics = plot_mode_vs_target(
                target=target,
                prediction=prediction,
                qc_label_mask=qc_mask,
                reliable_positive_mask=reliable,
                true_dpr_support=true_support,
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
                rng=np.random.default_rng(2026 + int(metadata["file_id"]) + position),
            )
            relative = f"rain/{position:02d}_{mode['slug']}_vs_satellite.png"
            writer.save(
                figure,
                relative,
                "固定布局的水平场、A-B剖面、相关性、长尾分布和逐高度误差。",
            )
            result["rain_modes"][str(mode["slug"])] = metrics
        writer.save(
            plot_all_methods_long_tail(
                target=target,
                predictions=predictions,
                mode_names=mode_names,
                reliable_positive_mask=reliable,
            ),
            "rain/90_all_methods_tail.png",
            "全部阶段输出的降水长尾、CCDF、阈值超越数量和总体指标。",
        )
        intensity_figure, intensity_metrics = plot_rain_intensity_analysis(
            fields, mode_specs, sample_id=str(metadata["sample_id"])
        )
        writer.save(
            intensity_figure,
            "rain/91_target_intensity_bins.png",
            "按卫星标签定义的<1、1-5、5-10、10-30和>=30 mm/h分段指标。",
        )
        result["rain_intensity"] = intensity_metrics
        reflectivity_figure, reflectivity_metrics = plot_reflectivity_comparison(
            fields,
            sample_id=str(metadata["sample_id"]),
            height_km=height_km,
            ab_scan_index=ab_scan,
            max_points=max_points,
            rng=np.random.default_rng(4026 + int(metadata["file_id"])),
        )
        writer.save(
            reflectivity_figure,
            "stage2/00_reflectivity_fields_and_sections.png",
            "GR稀疏值、真实DPR、W1.25与R1-O反射率的统一平面、剖面和散点比较。",
        )
        result["reflectivity"] = reflectivity_metrics
        audit_figure, audit_metrics = plot_stage2_audit(
            fields,
            sample_id=str(metadata["sample_id"]),
            height_km=height_km,
            max_points=max_points,
            rng=np.random.default_rng(6026 + int(metadata["file_id"])),
        )
        writer.save(
            audit_figure,
            "stage2/01_support_and_decomposition_audit.png",
            "W1.25支持域以及anchor/gap/outside和强反射率分段审计。",
        )
        result["stage2_audit"] = audit_metrics
        vertical_figure, vertical_metrics = plot_vertical_structure(
            fields, mode_specs, sample_id=str(metadata["sample_id"])
        )
        writer.save(
            vertical_figure,
            "stage2/02_vertical_cfad_and_drdz.png",
            "DPR反射率CFAD、逐高度误差及最终降水物理dR/dz对比。",
        )
        result["vertical_structure"] = vertical_metrics
    finally:
        writer.close()

    result["shared_plot_contract"] = {
        "rain_vmin_mm_h": float(rain_norm.vmin),
        "rain_vmax_mm_h": float(rain_norm.vmax),
        "rain_error_limit_mm_h": error_limit,
        "geographic_extent": list(geographic_extent),
    }
    _atomic_json(output_directory / "metrics.json", result)
    lines = [
        "# 多阶段固定测试轨道诊断图索引",
        "",
        f"- 轨道：`{metadata['sample_id']}`",
        f"- 原始文件：`{metadata['source_file']}`",
        f"- 水平切片：`z[{height_index}]={z[height_index]:.3f} km`",
        f"- A-B扫描行：`{ab_scan}`",
        "- 数据划分：`test`；这里只做固定样本定性观察，不用于模型选择。",
        "- W1.25 support使用validation固定阈值；R1-O使用真实DPR稀疏锚点和真实support，不能部署。",
        "- 所有降水分段都由卫星目标定义，所有模型使用完全相同的评价体素。",
        "",
        "## 图表",
        "",
    ]
    for relative, description in writer.entries:
        lines.extend((f"### `{relative}`", "", description, "", f"![{relative}]({relative})", ""))
    (output_directory / "figure_manifest.md").write_text("\n".join(lines), encoding="utf-8")
    complete.write_text(
        f"sample_id={metadata['sample_id']}\nfigures={len(writer.entries)}\n",
        encoding="utf-8",
    )
    return result


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def _finite_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def plot_archived_cascade_performance(
    cascade_rows: Sequence[Mapping[str, Any]],
    r1_rows: Sequence[Mapping[str, Any]],
) -> plt.Figure:
    """Summarize the already-completed full-validation cascade experiments."""

    base = {str(row["mode"]): row for row in cascade_rows}
    r1 = {str(row["mode"]): row for row in r1_rows}
    selections = (
        ("DPR oracle", base["dpr_oracle"]),
        ("GR interp", base["gr_interp"]),
        ("W1.25 + true support", base["w1p25_oracle_mask"]),
        ("W1.25 deployable", base["w1p25_predicted_mask"]),
        ("R1-O + true support", r1["r1_o_oracle_support"]),
    )
    labels = [item[0] for item in selections]
    x = np.arange(len(labels))
    colors = ["#2a9d8f", "#8d99ae", "#457b9d", "#264653", "#e76f51"]
    figure, axes = plt.subplots(2, 2, figsize=(19, 12), constrained_layout=True)
    axes[0, 0].bar(x, [_finite_float(row["positive_rmse"]) for _, row in selections], color=colors)
    axes[0, 0].set_title("Complete-validation final-rain RMSE")
    axes[0, 0].set_ylabel("RMSE (mm/h; lower is better)")
    axes[0, 0].set_xticks(x, labels, rotation=22, ha="right")
    axes[0, 0].grid(alpha=0.25, axis="y")
    axes[0, 1].bar(x, [_finite_float(row["positive_pearson_r"]) for _, row in selections], color=colors)
    axes[0, 1].axhline(0.68, color="black", linestyle="--", label="senior reference ~0.68 (different protocol)")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_title("Complete-validation final-rain Pearson r")
    axes[0, 1].set_xticks(x, labels, rotation=22, ha="right")
    axes[0, 1].legend(fontsize=8)
    bin_keys = [item[0] for item in RAIN_INTENSITY_BINS]
    bin_labels = [item[1] for item in RAIN_INTENSITY_BINS]
    for (label, row), color in zip(selections, colors):
        axes[1, 0].plot(
            bin_labels,
            [_finite_float(row[f"positive_{key}_pearson_r"]) for key in bin_keys],
            "o-", label=label, color=color,
        )
    axes[1, 0].set_ylim(-0.1, 1)
    axes[1, 0].set_title("Pearson r by satellite target rain-rate bin")
    axes[1, 0].set_xlabel("Target rain rate (mm/h)")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend(fontsize=7)
    width = 0.38
    axes[1, 1].bar(
        x - width / 2, [_finite_float(row["drdz_pearson_r"]) for _, row in selections],
        width, label="dR/dz Pearson r", color="#457b9d",
    )
    axes[1, 1].bar(
        x + width / 2, [_finite_float(row["drdz_sign_agreement_fraction"]) for _, row in selections],
        width, label="gradient sign agreement", color="#e9c46a",
    )
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_title("Final-rain vertical-structure metrics")
    axes[1, 1].set_xticks(x, labels, rotation=22, ha="right")
    axes[1, 1].legend(fontsize=8)
    figure.suptitle(
        "Archived complete-validation cascade performance\n"
        "Oracle-support routes are diagnostic upper bounds, not deployable results",
        fontsize=15,
    )
    return figure


def plot_archived_stage2_regions(
    w_rows: Sequence[Mapping[str, Any]], r1_rows: Sequence[Mapping[str, Any]]
) -> plt.Figure:
    w = {str(row["region"]): row for row in w_rows}
    r1 = {str(row["region"]): row for row in r1_rows}
    regions = (
        ("Q11 anchor", "q11_direct_overlap"),
        ("Q01 gap", "dpr_gap_proxy"),
        ("Q01 outside", "dpr_outside_proxy"),
        (">=35 dBZ", "dpr_dbz_ge35"),
    )
    labels = [item[0] for item in regions]
    x = np.arange(len(labels))
    figure, axes = plt.subplots(2, 2, figsize=(17, 12), constrained_layout=True)
    for axis, metric, title in (
        (axes[0, 0], "mae", "dBZ MAE"),
        (axes[0, 1], "rmse", "dBZ RMSE"),
        (axes[1, 0], "bias", "dBZ bias (prediction-target)"),
        (axes[1, 1], "pearson_r", "dBZ Pearson r"),
    ):
        w_values = [_finite_float(w[key][f"reflectivity_on_target_support_{metric}_dbz" if metric in {"mae", "rmse", "bias"} else "reflectivity_on_target_support_pearson_r"]) for _, key in regions]
        r_values = [_finite_float(r1[key][f"{metric}_dbz" if metric in {"mae", "rmse", "bias"} else "pearson_r"]) for _, key in regions]
        width = 0.36
        axis.bar(x - width / 2, w_values, width, label="W1.25", color="#457b9d")
        axis.bar(x + width / 2, r_values, width, label="R1-O", color="#e76f51")
        axis.set_xticks(x, labels, rotation=15, ha="right")
        axis.set_title(title)
        axis.grid(alpha=0.25, axis="y")
        if metric == "bias":
            axis.axhline(0, color="black", linewidth=0.8)
        if metric == "pearson_r":
            axis.set_ylim(-0.1, 1)
    axes[0, 0].legend()
    figure.suptitle(
        "Stage-2 full-validation decomposition\n"
        "R1-O removes sensor-value mismatch but retains the sparse GR geometry",
        fontsize=15,
    )
    return figure


def plot_training_histories(
    stage1_rows: Sequence[Mapping[str, Any]],
    stage2_rows: Sequence[Mapping[str, Any]],
    r1_rows: Sequence[Mapping[str, Any]],
) -> plt.Figure:
    figure, axes = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)
    specifications = (
        (
            axes[0], stage1_rows, "Stage 1 I+G0.02+T3D",
            lambda row: row["val"]["metrics"]["rain"]["all"]["pearson_r"],
            lambda row: row["val"]["metrics"]["rain"]["all"]["rmse"],
            "rain",
        ),
        (
            axes[1], stage2_rows, "Stage 2 W1.25",
            lambda row: row["val"]["metrics"]["reflectivity_on_target_support"]["pearson_r"],
            lambda row: row["val"]["metrics"]["reflectivity_on_target_support"]["rmse_dbz"],
            "dBZ",
        ),
        (
            axes[2], r1_rows, "R1-O DPRSparseValue",
            lambda row: row["val"]["metrics"]["reflectivity_on_oracle_support"]["pearson_r"],
            lambda row: row["val"]["metrics"]["reflectivity_on_oracle_support"]["rmse_dbz"],
            "dBZ",
        ),
    )
    for axis, rows, title, r_getter, rmse_getter, unit in specifications:
        epoch = [int(row["epoch"]) + 1 for row in rows]
        correlation = [_finite_float(r_getter(row)) for row in rows]
        rmse = [_finite_float(rmse_getter(row)) for row in rows]
        axis.plot(epoch, correlation, color="#2a9d8f", label="validation Pearson r")
        axis.set_ylim(0, 1)
        axis.set_xlabel("epoch")
        axis.set_ylabel("Pearson r")
        axis.set_title(title)
        axis.grid(alpha=0.25)
        second = axis.twinx()
        second.plot(epoch, rmse, color="#e76f51", label=f"validation RMSE ({unit})")
        second.set_ylabel(f"RMSE ({unit})")
        handles = axis.get_lines() + second.get_lines()
        axis.legend(handles, [item.get_label() for item in handles], fontsize=8)
    figure.suptitle("Training histories from the selected final checkpoints", fontsize=15)
    return figure


def plot_archived_r1_physics(
    per_height_rows: Sequence[Mapping[str, Any]],
    cfad_rows: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Any],
) -> plt.Figure:
    z = np.asarray([_finite_float(row["height_km"]) for row in per_height_rows])
    rmse = np.asarray([_finite_float(row["rmse_dbz"]) for row in per_height_rows])
    correlation = np.asarray([_finite_float(row["pearson_r"]) for row in per_height_rows])
    height_indices = sorted({int(row["height_index"]) for row in cfad_rows})
    bin_indices = sorted({int(row["dbz_bin_index"]) for row in cfad_rows})
    target = np.zeros((len(height_indices), len(bin_indices)), dtype=np.float64)
    prediction = np.zeros_like(target)
    for row in cfad_rows:
        i, j = int(row["height_index"]), int(row["dbz_bin_index"])
        target[i, j] = _finite_float(row["target_fraction_at_height"])
        prediction[i, j] = _finite_float(row["prediction_fraction_at_height"])
    # ``cfad.csv`` contains one row per (height, dBZ-bin).  Taking the first
    # ``len(bin_indices)`` sorted rows would accidentally select the same bin
    # repeatedly (once for several heights).  Keep one representative row per
    # bin when reconstructing the physical dBZ edges.
    by_bin: dict[int, Mapping[str, Any]] = {}
    for row in cfad_rows:
        by_bin.setdefault(int(row["dbz_bin_index"]), row)
    ordered = [by_bin[index] for index in bin_indices]
    dbz_edges = np.asarray([_finite_float(row["dbz_lower"]) for row in ordered] + [_finite_float(ordered[-1]["dbz_upper"])])
    positive = np.concatenate((target[target > 0], prediction[prediction > 0]))
    norm = LogNorm(vmin=max(float(positive.min()), 1e-6), vmax=float(positive.max()))
    height_edges = _coordinate_edges(z)
    figure, axes = plt.subplots(2, 2, figsize=(17, 12), constrained_layout=True)
    axes[0, 0].plot(rmse, z, color="#e76f51", label="RMSE")
    axes[0, 0].set_xlabel("RMSE (dBZ)")
    axes[0, 0].set_ylabel("Height (km)")
    axes[0, 0].set_title("R1-O dBZ error by height")
    twin = axes[0, 0].twiny()
    twin.plot(correlation, z, color="#2a9d8f", label="Pearson r")
    twin.set_xlim(-0.1, 1)
    twin.set_xlabel("Pearson r")
    for axis, values, title in (
        (axes[0, 1], target, "True DPR full-validation CFAD"),
        (axes[1, 0], prediction, "R1-O full-validation CFAD"),
    ):
        image = axis.pcolormesh(
            dbz_edges,
            height_edges,
            np.ma.masked_less_equal(values, 0),
            shading="auto",
            cmap="magma",
            norm=norm,
        )
        axis.set_xlabel("dBZ")
        axis.set_ylabel("Height (km)")
        axis.set_title(title)
        figure.colorbar(image, ax=axis, shrink=0.75, label="fraction at height")
    nested = metrics["physical_diagnostics"]["nested_dbz_events"]
    labels = ["15", "25", "35"]
    x = np.arange(3)
    axes[1, 1].bar(x - 0.18, [nested[key]["support"]["csi"] for key in labels], 0.36, label="exact CSI")
    axes[1, 1].bar(x + 0.18, [nested[key]["fss"]["2"]["fss"] for key in labels], 0.36, label="FSS radius=2")
    axes[1, 1].set_xticks(x, [f">={key} dBZ" for key in labels])
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_title("Nested reflectivity-event structure")
    axes[1, 1].legend()
    figure.suptitle("R1-O complete-validation vertical and threshold diagnostics", fontsize=15)
    return figure


def render_archived_summary(
    *, stage1_dir: Path, stage2_dir: Path, cascade_dir: Path, r1_dir: Path,
    output_directory: Path, dpi: int,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    writer = FigureWriter(output_directory, dpi)
    cascade_root = cascade_dir / "validation_d_vs_w1p25"
    w_analysis = stage2_dir / "analysis" / "validation_candidates" / "reflectivity"
    r1_validation = r1_dir / "analysis" / "full_validation"
    r1_cascade = r1_dir / "analysis" / "frozen_stage1_cascade"
    try:
        writer.save(
            plot_archived_cascade_performance(
                _read_csv(cascade_root / "comparison.csv"),
                _read_csv(r1_cascade / "comparison.csv"),
            ),
            "00_complete_validation_cascade.png",
            "完整验证集上的最终降水、分降水段和dR/dz性能。",
        )
        writer.save(
            plot_archived_stage2_regions(
                _read_csv(w_analysis / "per_region.csv"),
                _read_csv(r1_validation / "per_region.csv"),
            ),
            "01_stage2_region_decomposition.png",
            "W1.25与R1-O在anchor、gap、outside和强回波区的完整验证集对照。",
        )
        writer.save(
            plot_training_histories(
                _read_jsonl(stage1_dir / "metrics.jsonl"),
                _read_jsonl(stage2_dir / "metrics.jsonl"),
                _read_jsonl(r1_dir / "metrics.jsonl"),
            ),
            "02_training_histories.png",
            "Stage 1、W1.25和R1-O的逐epoch验证指标。",
        )
        writer.save(
            plot_archived_r1_physics(
                _read_csv(r1_validation / "per_height.csv"),
                _read_csv(r1_validation / "cfad.csv"),
                _load_json(r1_validation / "metrics.json"),
            ),
            "03_r1_vertical_and_cfad.png",
            "R1-O逐高度、CFAD和15/25/35 dBZ嵌套事件表现。",
        )
    finally:
        writer.close()
    lines = [
        "# 已完成实验总体性能图",
        "",
        "这些图使用已经保存的完整validation结果，不从六条test轨重新选择模型或阈值。",
        "R1-O及带true support的路线均为不可部署诊断上限。",
        "",
    ]
    for relative, description in writer.entries:
        lines.extend((f"## `{relative}`", "", description, "", f"![{relative}]({relative})", ""))
    (output_directory / "figure_manifest.md").write_text("\n".join(lines), encoding="utf-8")


def plot_selected_orbit_performance(
    reports: Sequence[Mapping[str, Any]],
) -> tuple[plt.Figure, dict[str, Any]]:
    """Summarize the six fixed test orbits without pooling away orbit identity.

    These statistics are descriptive only: the test orbits are never used to
    choose a checkpoint, support threshold, loss weight, or model variant.
    """

    if not reports:
        raise ValueError("at least one orbit report is required")
    mode_specs = list(RAIN_MODE_SPECS)
    orbit_labels = [f"T{position + 1}" for position in range(len(reports))]
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.9, len(mode_specs)))
    figure, axes = plt.subplots(2, 2, figsize=(20, 12), constrained_layout=True)
    summary: dict[str, Any] = {"orbit_count": len(reports), "rain_modes": {}}

    for color, mode in zip(colors, mode_specs):
        slug = str(mode["slug"])
        per_orbit = [report["rain_intensity"][slug]["overall"] for report in reports]
        rmse = np.asarray([_finite_float(item["rmse"]) for item in per_orbit])
        correlation = np.asarray(
            [_finite_float(item["pearson_r"]) for item in per_orbit]
        )
        axes[0, 0].plot(orbit_labels, rmse, "o-", color=color, label=str(mode["display_name"]))
        axes[0, 1].plot(
            orbit_labels,
            correlation,
            "o-",
            color=color,
            label=str(mode["display_name"]),
        )
        intensity_r = np.asarray(
            [
                [
                    _finite_float(
                        report["rain_intensity"][slug]["by_target_intensity"][bin_slug][
                            "pearson_r"
                        ]
                    )
                    for bin_slug, _label, _lower, _upper in RAIN_INTENSITY_BINS
                ]
                for report in reports
            ],
            dtype=np.float64,
        )
        axes[1, 0].plot(
            [item[1] for item in RAIN_INTENSITY_BINS],
            np.nanmedian(intensity_r, axis=0),
            "o-",
            color=color,
            label=str(mode["display_name"]),
        )
        summary["rain_modes"][slug] = {
            "display_name": str(mode["display_name"]),
            "deployable": bool(mode.get("deployable", False)),
            "per_orbit_rmse": rmse.tolist(),
            "per_orbit_pearson_r": correlation.tolist(),
            "median_orbit_rmse": float(np.nanmedian(rmse)),
            "median_orbit_pearson_r": float(np.nanmedian(correlation)),
            "median_orbit_pearson_r_by_target_intensity": np.nanmedian(
                intensity_r, axis=0
            ).tolist(),
        }

    axes[0, 0].set_title("Final-rain RMSE on each fixed test orbit")
    axes[0, 0].set_ylabel("RMSE (mm/h; lower is better)")
    axes[0, 0].grid(alpha=0.25)
    axes[0, 1].set_title("Final-rain Pearson r on each fixed test orbit")
    axes[0, 1].set_ylabel("Pearson r")
    axes[0, 1].set_ylim(-0.1, 1.0)
    axes[0, 1].grid(alpha=0.25)
    axes[0, 1].legend(fontsize=7)
    axes[1, 0].set_title("Median per-orbit r by satellite target rain-rate bin")
    axes[1, 0].set_xlabel("Target rain rate (mm/h)")
    axes[1, 0].set_ylabel("Median Pearson r across fixed orbits")
    axes[1, 0].set_ylim(-0.1, 1.0)
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend(fontsize=7)

    support_csi = np.asarray(
        [_finite_float(report["stage2_audit"]["support"]["csi"]) for report in reports]
    )
    support_recall = np.asarray(
        [
            _finite_float(report["stage2_audit"]["support"]["recall"])
            for report in reports
        ]
    )
    x = np.arange(len(reports))
    axes[1, 1].bar(x - 0.18, support_csi, 0.36, label="CSI", color="#457b9d")
    axes[1, 1].bar(x + 0.18, support_recall, 0.36, label="Recall", color="#e9c46a")
    axes[1, 1].set_xticks(x, orbit_labels)
    axes[1, 1].set_ylim(0.0, 1.0)
    axes[1, 1].set_title("W1.25 support recovery @ validation threshold")
    axes[1, 1].grid(alpha=0.25, axis="y")
    axes[1, 1].legend()
    summary["w1p25_support"] = {
        "per_orbit_csi": support_csi.tolist(),
        "per_orbit_recall": support_recall.tolist(),
        "median_csi": float(np.nanmedian(support_csi)),
        "median_recall": float(np.nanmedian(support_recall)),
    }
    figure.suptitle(
        "Six fixed Stage-1 test orbits: matched multi-stage diagnostics\n"
        "Test results are descriptive; oracle-support/R1-O routes are not deployable",
        fontsize=15,
    )
    return figure, summary


def render_selected_orbit_summary(
    reports: Sequence[Mapping[str, Any]], *, output_directory: Path, dpi: int
) -> dict[str, Any]:
    output_directory.mkdir(parents=True, exist_ok=True)
    writer = FigureWriter(output_directory, dpi)
    try:
        figure, summary = plot_selected_orbit_performance(reports)
        writer.save(
            figure,
            "00_selected_test_orbits_performance.png",
            "六条固定测试轨道的逐轨降水性能、分降水段相关性及support恢复。",
        )
    finally:
        writer.close()
    result = {
        "format": SUMMARY_FORMAT,
        "split": "test",
        "test_set_accessed": True,
        "qualitative_diagnostics_only": True,
        **summary,
    }
    _atomic_json(output_directory / "metrics.json", result)
    lines = [
        "# 六条固定测试轨道汇总",
        "",
        "测试集仅用于封存模型的定性与最终描述，不用于选择模型或阈值。",
        "R1-O与true-support结果是不可部署的诊断上限。",
        "",
    ]
    for relative, description in writer.entries:
        lines.extend((f"## `{relative}`", "", description, "", f"![{relative}]({relative})", ""))
    (output_directory / "figure_manifest.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def _bundle_directory(output_dir: Path, selection: OrbitSelection) -> Path:
    return output_dir / "bundles" / f"sample_{selection.selection_position:02d}_{selection.sample_id}"


def _report_directory(output_dir: Path, selection: OrbitSelection) -> Path:
    return output_dir / "orbits" / f"sample_{selection.selection_position:02d}_{selection.sample_id}"


def _checkpoint_payload(path: Path, label: str) -> tuple[Path, dict[str, Any], Mapping[str, Any]]:
    source = path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    config = payload.get("config")
    if not isinstance(config, Mapping):
        raise ValueError(f"{label} checkpoint has no configuration: {source}")
    return source, payload, config


def _write_root_index(
    *, output_dir: Path, selections: Sequence[OrbitSelection], elapsed_note: str
) -> None:
    lines = [
        "# 多阶段最终模型：六条固定测试轨道可视化",
        "",
        "本目录把 Stage 1、W1.25 Stage 2、两阶段串联与 R1-O 空间恢复上限放到同一物理轨道、同一高度、同一A-B剖面和同一色标下比较。",
        "",
        "- 数据划分：`test`；仅作封存模型的定性观察与最终描述，不能据此调参。",
        "- `W1.25 deployable cascade` 才是完整可部署两阶段结果。",
        "- `W1.25 + true DPR support` 隔离数值场误差；support来自标签，不可部署。",
        "- `R1-O` 同时使用真实DPR稀疏锚点和真实support，是空间补全上限，不可部署。",
        "- support阈值固定来自W1.25验证集文件，未在这6条测试轨道重新选择。",
        f"- 运行状态：{elapsed_note}",
        "",
        "## 汇总图",
        "",
        "- [六条固定测试轨道汇总](selected_test_summary/figure_manifest.md)",
        "- [历史完整验证集汇总](archived_complete_validation/figure_manifest.md)",
        "",
        "## 每条轨道",
        "",
    ]
    for selection in selections:
        relative = _report_directory(output_dir, selection).relative_to(output_dir)
        lines.append(
            f"- T{selection.selection_position + 1}: `{selection.sample_id}` — "
            f"[图表索引]({relative}/figure_manifest.md) | "
            f"[合并PDF]({relative}/diagnostics.pdf)"
        )
    lines.extend(
        (
            "",
            "## 缓存",
            "",
            "`bundles/`保存每条轨道在物理 `(nscan,49,60)` 网格上的统一NPZ；"
            "后续可使用 `--plot-only` 重画而无需再次运行三个3D模型。",
            "",
        )
    )
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.stage1_batch_size <= 0 or args.stage2_batch_size <= 0:
        raise ValueError("inference batch sizes must be positive")
    if args.num_workers < 0 or args.max_points <= 0 or args.dpi <= 0:
        raise ValueError("num-workers must be non-negative; max-points/dpi must be positive")

    stage1_dir = args.stage1_dir.expanduser().resolve()
    stage2_dir = args.stage2_dir.expanduser().resolve()
    cascade_dir = args.cascade_dir.expanduser().resolve()
    r1_dir = args.r1_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selections = discover_stage1_orbits(
        stage1_dir / "analysis" / "test_predictions", count=args.count
    )
    bundle_directories = [_bundle_directory(output_dir, item) for item in selections]
    missing = [
        (selection, directory)
        for selection, directory in zip(selections, bundle_directories)
        if args.overwrite_cache
        or not (directory / "fields.npz").is_file()
        or not (directory / "metadata.json").is_file()
    ]
    if args.plot_only and missing:
        names = ", ".join(item.sample_id for item, _directory in missing)
        raise FileNotFoundError(f"--plot-only requires cached bundles; missing: {names}")

    provenance: dict[str, Any] = {}
    if missing:
        device = resolve_device(args.device)
        print(f"[setup] inference device={device}; missing bundles={len(missing)}", flush=True)
        stage1_path, stage1_payload, stage1_config = _checkpoint_payload(
            stage1_dir / "best.pt", "Stage 1"
        )
        stage2_path, stage2_payload, stage2_config = _checkpoint_payload(
            stage2_dir / "best_dbz.pt", "W1.25 Stage 2"
        )
        r1_path, r1_payload, r1_config = _checkpoint_payload(
            r1_dir / "best.pt", "R1-O"
        )
        validate_r1_config(r1_config)
        threshold_path = (
            stage2_dir
            / "analysis"
            / "validation_candidates"
            / "reflectivity"
            / "support_threshold.json"
        )
        support_threshold = load_threshold_file(threshold_path)

        stage1_data = stage1_config["data"]
        stage1_index_path = project_path(stage1_data["test_index"])
        stage1_index = _load_json(stage1_index_path)
        normalization = _load_json(project_path(stage1_data["normalization"]))
        standardizer = PerLevelStandardizer.from_dict(
            normalization["variables"]["dbz_dpr"]
        )
        heights = np.asarray(stage1_index["heights_km"], dtype=np.float32)

        stage2_data = stage2_config["data"]
        stage2_dataset = Stage2PatchDataset(
            project_path(stage2_data["test_index"]),
            project_path(stage2_data["normalization"]),
            cache_size=int(stage2_data.get("cache_size", 1)),
            input_channels=stage2_data.get("input_channels"),
        )
        r1_data = r1_config["data"]
        r1_dataset = Stage2PatchDataset(
            project_path(r1_data["test_index"]),
            project_path(r1_data["normalization"]),
            cache_size=int(r1_data.get("cache_size", 1)),
            input_channels=r1_data.get("input_channels"),
        )
        _validate_index_alignment(stage1_index, stage2_dataset.index_metadata, label="W1.25")
        _validate_index_alignment(stage1_index, r1_dataset.index_metadata, label="R1-O")
        if not np.allclose(stage2_dataset.z, heights, rtol=0.0, atol=1e-6):
            raise ValueError("W1.25 and Stage-1 height grids differ")
        if not np.allclose(r1_dataset.z, heights, rtol=0.0, atol=1e-6):
            raise ValueError("R1-O and Stage-1 height grids differ")
        stage2_ids = _index_by_sample_id(stage2_dataset)
        r1_ids = _index_by_sample_id(r1_dataset)

        stage1_model = build_stage1_model(stage1_config).to(device)
        stage2_model = build_stage2_model(stage2_config).to(device)
        r1_model = build_r1_model(r1_config).to(device)
        load_checkpoint(stage1_path, stage1_model, map_location=device)
        load_checkpoint(stage2_path, stage2_model, map_location=device)
        load_checkpoint(r1_path, r1_model, map_location=device)
        stage1_model.eval()
        stage2_model.eval()
        r1_model.eval()
        stage1_options = _cascade_options(
            stage1_config,
            stage1_index,
            device,
            standardizer,
            heights,
            args.stage1_batch_size,
        )
        provenance = {
            "stage1_checkpoint": str(stage1_path),
            "stage1_epoch": int(stage1_payload["epoch"]),
            "stage2_checkpoint": str(stage2_path),
            "stage2_epoch": int(stage2_payload["epoch"]),
            "r1_checkpoint": str(r1_path),
            "r1_epoch": int(r1_payload["epoch"]),
            "support_threshold_file": str(threshold_path),
            "support_threshold": support_threshold,
        }
        try:
            for position, (selection, bundle_dir) in enumerate(missing, start=1):
                if selection.sample_id not in stage2_ids or selection.sample_id not in r1_ids:
                    raise KeyError(f"selected sample is absent from Stage-2 indices: {selection.sample_id}")
                fields, metadata = _build_orbit_bundle(
                    selection,
                    stage1_model=stage1_model,
                    stage2_model=stage2_model,
                    r1_model=r1_model,
                    stage2_dataset=stage2_dataset,
                    r1_dataset=r1_dataset,
                    stage2_file_id=stage2_ids[selection.sample_id],
                    r1_file_id=r1_ids[selection.sample_id],
                    stage1_options=stage1_options,
                    support_threshold=support_threshold,
                    device=device,
                    stage2_batch_size=args.stage2_batch_size,
                    num_workers=args.num_workers,
                )
                metadata["provenance"] = provenance
                _atomic_npz(bundle_dir / "fields.npz", fields)
                _atomic_json(bundle_dir / "metadata.json", metadata)
                print(
                    f"[bundle {position}/{len(missing)}] {selection.sample_id} "
                    f"shape={tuple(fields['target_rain_mm_h'].shape)}",
                    flush=True,
                )
                stage2_dataset.clear_cache()
                r1_dataset.clear_cache()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        finally:
            del stage1_model, stage2_model, r1_model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print("[figures] rendering archived complete-validation summary", flush=True)
    render_archived_summary(
        stage1_dir=stage1_dir,
        stage2_dir=stage2_dir,
        cascade_dir=cascade_dir,
        r1_dir=r1_dir,
        output_directory=output_dir / "archived_complete_validation",
        dpi=args.dpi,
    )
    reports: list[dict[str, Any]] = []
    for position, (selection, bundle_dir) in enumerate(
        zip(selections, bundle_directories), start=1
    ):
        print(f"[figures {position}/{len(selections)}] {selection.sample_id}", flush=True)
        report = render_orbit_report(
            bundle_dir,
            output_directory=_report_directory(output_dir, selection),
            height_km=args.height_km,
            max_points=args.max_points,
            dpi=args.dpi,
            overwrite=args.overwrite_plots,
        )
        reports.append(report)
    render_selected_orbit_summary(
        reports, output_directory=output_dir / "selected_test_summary", dpi=args.dpi
    )
    # A plot-only resume does not reload checkpoint payloads.  Preserve the
    # exact checkpoint/epoch/threshold provenance stored in the first bundle
    # instead of replacing it with an empty object in the root summary.
    if not provenance and bundle_directories:
        cached_metadata = _load_json(bundle_directories[0] / "metadata.json")
        cached_provenance = cached_metadata.get("provenance", {})
        if isinstance(cached_provenance, Mapping):
            provenance = dict(cached_provenance)
    summary = {
        "format": SUMMARY_FORMAT,
        "split": "test",
        "test_set_accessed": True,
        "qualitative_diagnostics_only": True,
        "selected_orbit_count": len(selections),
        "selected_sample_ids": [item.sample_id for item in selections],
        "output_directory": str(output_dir),
        "bundle_directories": [str(item) for item in bundle_directories],
        "provenance": provenance,
    }
    _atomic_json(output_dir / "summary.json", summary)
    _write_root_index(
        output_dir=output_dir,
        selections=selections,
        elapsed_note=f"已完成{len(selections)}条轨道的统一推理与绘图",
    )
    print(f"OK multistage diagnostics: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
