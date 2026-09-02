#!/usr/bin/env python3
"""Evaluate frozen Stage-2 -> Stage-1 cascades on complete val/test orbits.

One invocation can compare multiple Stage-2 checkpoints with a strict 2x2
``reflectivity value x support`` interface audit plus the historical GR
interpolation baseline:

1. true DPR dBZ + true DPR support (sealed Stage-1 upper bound);
2. Stage-2 dBZ + true DPR support (oracle-mask value-error isolation);
3. true DPR dBZ + Stage-2 validation-thresholded support (support isolation);
4. Stage-2 dBZ + its validation-thresholded support (deployable cascade);
5. senior-project interpolated GR dBZ + its stored support + sealed Stage 1.

True DPR dBZ is undefined at a Stage-2 false-positive support voxel.  Route 3
therefore uses the Stage-1 per-height training mean at those voxels, which is
exactly neutral zero after Stage-1 standardization.  This explicit
counterfactual avoids borrowing Stage-2 dBZ while keeping the route runnable.

Precipitation labels are used only for final metrics and saved visual reports.
They never restrict a predicted support mask or a model input.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for value in (PROJECT_ROOT, SRC_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from precipitation_inversion.data.nc_reader import read_nc_sample  # noqa: E402
from precipitation_inversion.data.dataset import sha256_file  # noqa: E402
from precipitation_inversion.data.stage2_patch_dataset import (  # noqa: E402
    Stage2PatchDataset,
)
from precipitation_inversion.data.stage2_subtask_masks import (  # noqa: E402
    Stage2SubtaskMasks,
    build_stage2_gr_routing_masks,
    build_stage2_subtask_masks,
)
from precipitation_inversion.data.transforms import PerLevelStandardizer  # noqa: E402
from precipitation_inversion.inference.stage2_sliding_window import (  # noqa: E402
    predict_stage2_full_orbit,
)
from precipitation_inversion.inference.stage2_stage1_cascade import (  # noqa: E402
    Stage1CascadePrediction,
    predict_stage1_from_reflectivity_orbit,
)
from precipitation_inversion.inference.stage2_oracles import (  # noqa: E402
    build_regional_oracle_input,
)
from precipitation_inversion.metrics.regression import (  # noqa: E402
    FilewisePrecipitationMetrics,
    PhysicalRainGradientMetrics,
    PrecipitationRegressionMetrics,
    StratifiedPrecipitationMetrics,
)
from precipitation_inversion.metrics.stage2_reflectivity import (  # noqa: E402
    finite_metrics_for_json,
)
from precipitation_inversion.metrics.stage2_decomposition import (  # noqa: E402
    Stage2DecompositionDiagnostics,
)
from precipitation_inversion.training.engine import load_checkpoint  # noqa: E402
from scripts.evaluate_stage1_unet3d import build_model as build_stage1_model  # noqa: E402
from scripts.evaluate_stage2_unet3d import load_threshold_file  # noqa: E402
from scripts.train_stage2_unet3d import build_model as build_stage2_model  # noqa: E402


MODE_SLUG_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
CASCADE_ORBIT_FORMAT = "stage2_stage1_cascade_orbit_v1"
FACTORIAL_AUDIT_FORMAT = "stage3_c0_factorial_2x2_v1"
R0_ORACLE_AUDIT_FORMAT = "stage2_r0_decomposition_oracle_v1"
R0_DIAGNOSTICS_FORMAT = "stage2_r0_decomposition_diagnostics_v1"
R0_SAMPLE_ID_HASH_CONTRACT = "sha256(compact-utf8-json-ordered-sample-ids)"

# These value-oracle regions are deliberately not additive: ``strong_ge35``
# overlaps the three observability regions.  ``q01`` is retained as a useful
# aggregate while ``gap`` and ``outside`` form its disjoint decomposition.
R0_ORACLE_REGIONS: "OrderedDict[str, str]" = OrderedDict(
    (
        ("q11", "GR and DPR both have a physical reflectivity value"),
        ("q01", "DPR has echo but direct GR is missing"),
        ("gap", "Q01 and the GR-only interpolation proxy reaches the voxel"),
        ("outside", "Q01 and the GR-only interpolation proxy cannot reach it"),
        ("strong_ge35", "true DPR reflectivity is at least 35 dBZ"),
    )
)

# A support oracle must also include target-negative regions so it can measure
# the downstream effect of removing Q10/Q00 false alarms.  Q01 and strong_ge35
# intentionally overlap other entries and are reported as aggregate views,
# never as additive contributions.
R0_SUPPORT_ORACLE_REGIONS: "OrderedDict[str, str]" = OrderedDict(
    (
        ("q11", "GR and DPR both have a physical reflectivity value"),
        ("q01", "DPR has echo but direct GR is missing"),
        ("gap", "Q01 and the GR-only interpolation proxy reaches the voxel"),
        ("outside", "Q01 and the GR-only interpolation proxy cannot reach it"),
        ("q10", "GR has a physical value but DPR has no echo"),
        ("q00", "neither GR nor DPR has a physical echo value"),
        ("strong_ge35", "true DPR reflectivity is at least 35 dBZ"),
    )
)


@dataclass(frozen=True)
class Stage2RunSpec:
    label: str
    slug: str
    checkpoint: Path
    threshold_file: Path


def ordered_sample_ids_sha256(sample_ids: Sequence[str]) -> str:
    """Hash one ordered orbit identity list with an explicit stable contract."""

    values = [str(value) for value in sample_ids]
    payload = json.dumps(
        values,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass
class CascadeMetricBundle:
    """Streaming final-rain metrics for one controlled cascade mode."""

    positive: PrecipitationRegressionMetrics
    label_domain: PrecipitationRegressionMetrics
    filewise: FilewisePrecipitationMetrics
    stratified: StratifiedPrecipitationMetrics
    physical_drdz: PhysicalRainGradientMetrics

    @classmethod
    def create(
        cls,
        heights_km: np.ndarray,
        file_labels: Sequence[str],
        *,
        thresholds_mm_h: Sequence[float],
        cfb_distance_edges_km: Sequence[float],
        sign_epsilon: float,
        bootstrap_seed: int,
        bootstrap_replicates: int,
        bootstrap_confidence: float,
    ) -> "CascadeMetricBundle":
        return cls(
            positive=PrecipitationRegressionMetrics(thresholds_mm_h),
            label_domain=PrecipitationRegressionMetrics(thresholds_mm_h),
            filewise=FilewisePrecipitationMetrics(
                file_labels,
                bootstrap_seed=bootstrap_seed,
                bootstrap_replicates=bootstrap_replicates,
                confidence_level=bootstrap_confidence,
            ),
            stratified=StratifiedPrecipitationMetrics(
                heights_km.tolist(),
                cfb_distance_edges_km=cfb_distance_edges_km,
                intensity_thresholds_mm_h=thresholds_mm_h,
            ),
            physical_drdz=PhysicalRainGradientMetrics(
                heights_km.tolist(),
                cfb_distance_edges_km=cfb_distance_edges_km,
                intensity_thresholds_mm_h=thresholds_mm_h,
                sign_epsilon_mm_h_km=sign_epsilon,
                file_labels=file_labels,
                bootstrap_seed=bootstrap_seed,
                bootstrap_replicates=bootstrap_replicates,
                confidence_level=bootstrap_confidence,
            ),
        )

    def update(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
        *,
        reliable_positive_mask: np.ndarray,
        qc_label_mask: np.ndarray,
        heights_km: np.ndarray,
        cfb_distance_km: np.ndarray,
        precipitation_type: np.ndarray,
        file_id: int,
    ) -> None:
        """Update identical target supports for every competing input mode."""

        self.positive.update_rain(prediction, target, reliable_positive_mask)
        self.label_domain.update_rain(prediction, target, qc_label_mask)
        self.filewise.update_rain(
            prediction, target, reliable_positive_mask, file_id=file_id
        )
        self.stratified.update_rain(
            prediction,
            target,
            reliable_positive_mask,
            height_km=heights_km,
            cfb_distance_km=cfb_distance_km,
            precipitation_type=precipitation_type,
        )
        self.physical_drdz.update_rain(
            prediction,
            target,
            reliable_positive_mask,
            height_km=heights_km,
            cfb_distance_km=cfb_distance_km,
            precipitation_type=precipitation_type,
            file_id=file_id,
        )

    def compute(self) -> dict[str, Any]:
        return {
            "reliable_positive": self.positive.compute(),
            "qc_label_domain_including_zero": self.label_domain.compute(),
            "filewise_reliable_positive": self.filewise.compute(),
            "stratified_reliable_positive": self.stratified.compute(),
            "physical_drdz_reliable_positive": self.physical_drdz.compute(),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--stage2-run",
        action="append",
        nargs=3,
        metavar=("LABEL", "CHECKPOINT", "VAL_THRESHOLD_JSON"),
        default=[],
        help=(
            "Repeat for every Stage-2 model. Each model creates the other "
            "three cells needed with the common true-DPR upper bound for a "
            "strict value x support 2x2 audit."
        ),
    )
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stage1-batch-size", type=int, default=1)
    parser.add_argument("--stage2-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--save-orbits", type=int, default=6)
    parser.add_argument("--selection-seed", type=int, default=2026)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--no-dpr-oracle", action="store_true")
    parser.add_argument("--no-gr-interp", action="store_true")
    parser.add_argument(
        "--r0-decomposition-oracles",
        action="store_true",
        help=(
            "Enable validation-only Stage-2 R0 diagnostics: run separate regional "
            "value oracles (global true support) and support oracles (fixed Stage-2 "
            "dBZ), plus probability, nested-dBZ, echo-column, centroid, and CFAD "
            "metrics."
        ),
    )
    parser.add_argument(
        "--r0-dbz-thresholds",
        type=float,
        nargs="+",
        default=(15.0, 25.0, 35.0),
        metavar="DBZ",
    )
    parser.add_argument(
        "--r0-fss-radii",
        type=int,
        nargs="+",
        default=(0, 1, 2, 4),
        metavar="RADIUS",
    )
    parser.add_argument("--r0-strong-dbz-threshold", type=float, default=35.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif value == "cuda":
        value = "cuda:0"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def normalize_mode_slug(label: str) -> str:
    slug = re.sub(r"[^a-z0-9_-]+", "_", str(label).strip().lower()).strip("_-")
    if not slug or not MODE_SLUG_PATTERN.fullmatch(slug):
        raise ValueError(f"cannot construct a safe mode slug from {label!r}")
    return slug


def parse_stage2_run_specs(values: Sequence[Sequence[str]]) -> list[Stage2RunSpec]:
    result: list[Stage2RunSpec] = []
    slugs: set[str] = set()
    for raw in values:
        if len(raw) != 3:
            raise ValueError("each --stage2-run requires LABEL CHECKPOINT THRESHOLD")
        label, checkpoint_value, threshold_value = raw
        slug = normalize_mode_slug(label)
        if slug in slugs:
            raise ValueError(f"duplicate Stage-2 run slug: {slug}")
        checkpoint = Path(checkpoint_value).expanduser().resolve()
        threshold_file = Path(threshold_value).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Stage-2 checkpoint not found: {checkpoint}")
        if not threshold_file.is_file():
            raise FileNotFoundError(f"threshold JSON not found: {threshold_file}")
        # Validate now so a test-selected or malformed threshold cannot enter
        # a long formal cascade run.
        load_threshold_file(threshold_file)
        result.append(Stage2RunSpec(label.strip(), slug, checkpoint, threshold_file))
        slugs.add(slug)
    return result


def build_mode_definitions(
    run_specs: Sequence[Stage2RunSpec],
    *,
    include_dpr_oracle: bool = True,
    include_gr_interp: bool = True,
    include_r0_decomposition_oracles: bool = False,
) -> "OrderedDict[str, dict[str, Any]]":
    """Build one ordered, auditable definition for every cascade route.

    For each Stage-2 run the four factorial cells are named as follows:

    ``dpr_oracle``
        true DPR value + true DPR support (shared upper-bound cell);
    ``<run>_oracle_mask``
        predicted value + true support;
    ``<run>_true_dbz_predicted_mask``
        true value + predicted support, with neutral fill only where a false
        positive support has no corresponding true DPR value;
    ``<run>_predicted_mask``
        predicted value + predicted support (deployable cell).
    """

    modes: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
    if include_dpr_oracle:
        modes["dpr_oracle"] = {
            "display_name": "True DPR dBZ + true DPR support",
            "input_kind": "true_dpr_true_support",
            "deployable": False,
            "factorial_axes": {"reflectivity_value": "true", "support": "true"},
        }
    if include_gr_interp:
        modes["gr_interp"] = {
            "display_name": "Senior interpolated GR + sealed Stage 1",
            "input_kind": "gr_interpolated_baseline",
            "deployable": True,
        }
    for spec in run_specs:
        threshold = load_threshold_file(spec.threshold_file)
        common = {
            "stage2_label": spec.label,
            "stage2_checkpoint": str(spec.checkpoint),
            "stage2_threshold_file": str(spec.threshold_file),
            "stage2_support_threshold": threshold,
        }
        modes[f"{spec.slug}_oracle_mask"] = {
            "display_name": f"{spec.label}: Stage-2 dBZ + true DPR support",
            "input_kind": "stage2_dbz_true_support",
            "deployable": False,
            "factorial_axes": {
                "reflectivity_value": "predicted",
                "support": "true",
            },
            **common,
        }
        modes[f"{spec.slug}_true_dbz_predicted_mask"] = {
            "display_name": (
                f"{spec.label}: true DPR dBZ + predicted support "
                "(neutral false-positive fill)"
            ),
            "input_kind": "true_dpr_dbz_predicted_support",
            "deployable": False,
            "counterfactual": True,
            "false_positive_value_fill": (
                "Stage-1 train per-height dBZ mean; standardized value=0"
            ),
            "factorial_axes": {
                "reflectivity_value": "true",
                "support": "predicted",
            },
            **common,
        }
        modes[f"{spec.slug}_predicted_mask"] = {
            "display_name": f"{spec.label}: Stage-2 dBZ + predicted support",
            "input_kind": "stage2_dbz_predicted_support",
            "deployable": True,
            "factorial_axes": {
                "reflectivity_value": "predicted",
                "support": "predicted",
            },
            **common,
        }
        if include_r0_decomposition_oracles:
            for region, description in R0_ORACLE_REGIONS.items():
                modes[f"{spec.slug}_oracle_value_{region}"] = {
                    "display_name": (
                        f"{spec.label}: true DPR dBZ in {region}, "
                        "Stage-2 dBZ elsewhere + true support"
                    ),
                    "input_kind": "stage2_dbz_region_oracle_true_support",
                    "deployable": False,
                    "counterfactual": True,
                    "r0_oracle_region": region,
                    "r0_oracle_region_semantics": description,
                    "factorial_axes": {
                        "reflectivity_value": f"predicted_except_{region}",
                        "support": "true",
                    },
                    **common,
                }
            for region, description in R0_SUPPORT_ORACLE_REGIONS.items():
                modes[f"{spec.slug}_oracle_support_{region}"] = {
                    "display_name": (
                        f"{spec.label}: true DPR support in {region}, "
                        "Stage-2 support elsewhere + Stage-2 dBZ"
                    ),
                    "input_kind": "stage2_dbz_region_oracle_support",
                    "deployable": False,
                    "counterfactual": True,
                    "r0_oracle_component": "support",
                    "r0_oracle_region": region,
                    "r0_oracle_region_semantics": description,
                    "factorial_axes": {
                        "reflectivity_value": "predicted",
                        "support": f"predicted_except_{region}",
                    },
                    **common,
                }
    return modes


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
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(_json_safe(value), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for name in row:
            if name not in fields:
                fields.append(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(_json_safe(list(rows)))
    temporary.replace(path)


def _atomic_npz(path: Path, fields: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".partial.npz")
    np.savez_compressed(temporary, **fields)
    temporary.replace(path)


def _cfb_distance_km(cfb: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Convert profile CFB indices to voxel signed distance ``z-z_cfb``."""

    values = np.asarray(cfb)
    finite = np.isfinite(values)
    integer = np.where(finite, values, 0.0).astype(np.int64)
    valid = finite & (values == integer) & (integer >= 0) & (integer < z.size)
    safe = np.where(valid, integer, 0)
    boundary = z[safe]
    distance = z.reshape(1, 1, -1) - boundary[..., None]
    return np.where(valid[..., None], distance, np.nan).astype(np.float32)


def prepare_true_dpr_for_predicted_support(
    dbz_dpr: np.ndarray,
    true_dpr_support: np.ndarray,
    predicted_support: np.ndarray,
    standardizer: PerLevelStandardizer,
) -> np.ndarray:
    """Make the true-value/predicted-support counterfactual well defined.

    All arrays are physical-orbit fields with shape ``(nscan,nray,z)``.  A
    Stage-2 false positive has ``predicted_support=1`` but no true DPR dBZ;
    using the Stage-1 per-height training mean makes that selected input value
    exactly zero after standardization.  True observed DPR values are never
    changed and no Stage-2 predicted dBZ leaks into this factorial cell.
    """

    values = np.asarray(dbz_dpr, dtype=np.float32)
    truth = np.asarray(true_dpr_support)
    predicted = np.asarray(predicted_support)
    if values.ndim != 3:
        raise ValueError("dbz_dpr must have shape (nscan,nray,z)")
    if truth.shape != values.shape or predicted.shape != values.shape:
        raise ValueError("true and predicted support must match dbz_dpr")
    if truth.dtype != np.bool_ or predicted.dtype != np.bool_:
        raise TypeError("true and predicted support must be boolean")
    if standardizer.mean.shape != (values.shape[-1],):
        raise ValueError("Stage-1 standardizer does not match the height axis")
    if np.any(truth & ~np.isfinite(values)):
        raise ValueError("true DPR support selects a non-finite dBZ value")
    neutral = np.broadcast_to(
        np.asarray(standardizer.mean, dtype=np.float32).reshape(1, 1, -1),
        values.shape,
    )
    # Values outside true DPR support do not exist physically.  They receive
    # the height-dependent neutral value irrespective of whether Stage 2 later
    # selects them; the separate predicted-support channel retains occupancy.
    result = np.where(truth, values, neutral).astype(np.float32, copy=False)
    if np.any(predicted & ~np.isfinite(result)):
        raise ValueError("counterfactual selected reflectivity is not finite")
    return result


def build_r0_subtask_masks(
    source: Mapping[str, np.ndarray],
    *,
    strong_dbz_threshold: float = 35.0,
) -> Stage2SubtaskMasks:
    """Build validation-only R0 regions from one complete source orbit.

    Routing uses only GR-derived masks.  DPR reflectivity and the trustworthy
    label domain are added in the separate supervision object, making the
    input/label boundary explicit even inside an offline oracle audit.
    Every array keeps physical-orbit shape ``(nscan,nray,z)``.
    """

    required = (
        "gr_sparse_valid",
        "gr_interp_valid",
        "pre_valid_native_mask",
        "dpr_valid",
        "dbz_dpr",
    )
    missing = [name for name in required if name not in source]
    if missing:
        raise KeyError(f"R0 source is missing fields: {missing}")
    # Routing is a deployable GR-only quantity.  In a complete physical orbit
    # there is no padding geometry to remove, so omitting ``domain_mask`` makes
    # the routing domain all True.  In particular, neither pre_dpr validity nor
    # the CFB-derived QC mask is allowed to crop observed/near/far.
    routing = build_stage2_gr_routing_masks(
        np.asarray(source["gr_sparse_valid"]),
        np.asarray(source["gr_interp_valid"]),
    )
    # Stage-2 support/dBZ training includes native-valid bins below CFB.  R0
    # uses that same supervision domain.  ``qc_label_mask`` remains separate
    # and is used only by the final-rain evaluation bundle.
    label_domain = np.asarray(source["pre_valid_native_mask"])
    return build_stage2_subtask_masks(
        routing,
        label_domain,
        np.asarray(source["dpr_valid"]),
        np.asarray(source["dbz_dpr"]),
        strong_dbz_threshold=strong_dbz_threshold,
    )


def r0_oracle_region_masks(masks: Stage2SubtaskMasks) -> "OrderedDict[str, np.ndarray]":
    """Return the stable region names used by R0 value-oracle modes."""

    result: "OrderedDict[str, np.ndarray]" = OrderedDict(
        (
            ("q11", np.asarray(masks.q11_overlap)),
            ("q01", np.asarray(masks.q01_dpr_only)),
            ("gap", np.asarray(masks.dpr_only_gap)),
            ("outside", np.asarray(masks.dpr_only_outside)),
            ("strong_ge35", np.asarray(masks.strong_echo)),
        )
    )
    if tuple(result) != tuple(R0_ORACLE_REGIONS):
        raise RuntimeError("R0 oracle region names drifted from the public contract")
    return result


def r0_support_oracle_region_masks(
    masks: Stage2SubtaskMasks,
) -> "OrderedDict[str, np.ndarray]":
    """Return regions for one-factor-at-a-time support correction.

    Positive regions repair false negatives; Q10/Q00 repair false positives.
    Every mask remains supervision-only and is restricted to the native-valid
    Stage-2 label domain by :class:`Stage2SubtaskMasks`.
    """

    result: "OrderedDict[str, np.ndarray]" = OrderedDict(
        (
            ("q11", np.asarray(masks.q11_overlap)),
            ("q01", np.asarray(masks.q01_dpr_only)),
            ("gap", np.asarray(masks.dpr_only_gap)),
            ("outside", np.asarray(masks.dpr_only_outside)),
            ("q10", np.asarray(masks.q10_gr_only)),
            ("q00", np.asarray(masks.q00_neither)),
            ("strong_ge35", np.asarray(masks.strong_echo)),
        )
    )
    if tuple(result) != tuple(R0_SUPPORT_ORACLE_REGIONS):
        raise RuntimeError("R0 support-oracle names drifted from the public contract")
    return result


def replace_stage2_dbz_with_oracle_region(
    predicted_dbz: np.ndarray,
    true_dpr_dbz: np.ndarray,
    true_dpr_support: np.ndarray,
    replacement_mask: np.ndarray,
) -> np.ndarray:
    """Replace one target-defined region while leaving all other values fixed.

    This counterfactual is only used with *true DPR support* downstream.  The
    replacement mask must therefore be a subset of true support and cannot
    make an undefined DPR value enter Stage 1.  Inputs and result all have
    shape ``(nscan,nray,z)`` in physical dBZ.
    """

    result = build_regional_oracle_input(
        predicted_dbz,
        np.asarray(true_dpr_support),
        true_dpr_dbz,
        true_dpr_support,
        replacement_mask,
        component="value",
    )
    return result.reflectivity_dbz


def _load_source_fields(path: Path) -> dict[str, np.ndarray]:
    sample = read_nc_sample(
        path,
        variables=(
            "z",
            "lat",
            "lon",
            "dbz_gr_sparse",
            "dbz_gr_interp",
            "dbz_dpr",
            "pre_dpr",
            "cfb",
            "typePrecip",
        ),
        dtype=np.float32,
        build_masks=True,
    )
    target = sample.variables["pre_dpr"]
    dpr_valid = sample.masks["dpr_reflectivity_valid"]
    reliable_positive = (
        sample.masks["pre_positive_qc"] & dpr_valid & np.isfinite(target)
    )
    native_label = sample.masks["pre_valid_native"] & np.isfinite(target)
    qc_label = sample.masks["pre_valid_qc"] & np.isfinite(target)
    return {
        "z": sample.variables["z"],
        "lat": sample.variables["lat"],
        "lon": sample.variables["lon"],
        "gr_sparse_valid": sample.masks["gr_sparse_observed"],
        "dbz_gr_interp": sample.variables["dbz_gr_interp"],
        "gr_interp_valid": sample.masks["gr_interp_observed"],
        "dbz_dpr": sample.variables["dbz_dpr"],
        "dpr_valid": dpr_valid,
        "target_rain": target,
        "cfb": sample.variables["cfb"],
        "cfb_clutter": sample.masks["cfb_clutter"],
        "cfb_distance_km": _cfb_distance_km(
            sample.variables["cfb"], sample.variables["z"]
        ),
        "precipitation_type": sample.variables["typePrecip"],
        "reliable_positive_mask": reliable_positive,
        # Stage-2 support/value supervision and R0 decomposition domain.  It
        # preserves native-valid labels below CFB, matching Stage-2 training.
        "pre_valid_native_mask": native_label,
        # Final precipitation reporting domain; excludes CFB-below bins.
        "qc_label_mask": qc_label,
    }


def _validate_index_alignment(
    reference: Mapping[str, Any], candidate: Mapping[str, Any], *, label: str
) -> None:
    for key in ("core_size", "halo_size", "horizontal_multiple", "nray", "z_size"):
        if int(reference[key]) != int(candidate[key]):
            raise ValueError(f"{label} index differs in {key}")
    reference_files = reference["files"]
    candidate_files = candidate["files"]
    if len(reference_files) != len(candidate_files):
        raise ValueError(f"{label} index file count differs from Stage 1")
    for position, (left, right) in enumerate(zip(reference_files, candidate_files)):
        keys = ("sample_id", "nscan", "nray", "z_size")
        if any(left[key] != right[key] for key in keys):
            raise ValueError(f"{label} index differs at file_id={position}")


def _selected_orbit_ids(
    files: Sequence[Mapping[str, Any]],
    file_ids: Sequence[int],
    count: int,
    seed: int,
) -> list[int]:
    if count < 0:
        raise ValueError("save-orbits must be non-negative")
    if count == 0:
        return []
    eligible = [
        file_id
        for file_id in file_ids
        if int(files[file_id].get("positive_count", 0)) > 0
    ]
    if not eligible:
        eligible = list(file_ids)
    selected_count = min(count, len(eligible))
    selected = np.random.default_rng(seed).choice(
        np.asarray(eligible, dtype=np.int64), selected_count, replace=False
    )
    return sorted(int(value) for value in selected)


def _mode_row(slug: str, metadata: Mapping[str, Any], metrics: Mapping[str, Any]) -> dict[str, Any]:
    positive = metrics["reliable_positive"]["rain"]["all"]
    label_domain = metrics["qc_label_domain_including_zero"]["rain"]["all"]
    drdz = metrics["physical_drdz_reliable_positive"]["all"]
    bins = metrics["reliable_positive"]["rain"]["target_bins_mm_h"]
    row: dict[str, Any] = {
        "mode": slug,
        "display_name": metadata["display_name"],
        "input_kind": metadata["input_kind"],
    }
    for prefix, values in (
        ("positive", positive),
        ("label_domain", label_domain),
        ("drdz", drdz),
    ):
        for name, value in values.items():
            if isinstance(value, (int, float)):
                row[f"{prefix}_{name}"] = value
    for bin_name, values in bins.items():
        for metric_name in ("count", "mae", "rmse", "bias", "pearson_r", "ccc"):
            row[f"positive_{bin_name}_{metric_name}"] = values[metric_name]
    return row


_FACTORIAL_METRIC_DOMAINS: tuple[
    tuple[str, tuple[str, ...], tuple[str, ...]], ...
] = (
    (
        "reliable_positive_rain",
        ("reliable_positive", "rain", "all"),
        ("mae", "rmse", "bias", "r2", "pearson_r", "ccc"),
    ),
    (
        "qc_label_domain_rain",
        ("qc_label_domain_including_zero", "rain", "all"),
        ("mae", "rmse", "bias", "r2", "pearson_r", "ccc"),
    ),
    (
        "physical_drdz_reliable_positive",
        ("physical_drdz_reliable_positive", "all"),
        (
            "mae",
            "rmse",
            "bias",
            "r2",
            "pearson_r",
            "ccc",
            "mean_abs_gradient_ratio",
            "sign_agreement_fraction",
        ),
    ),
)


def _nested_mapping(
    value: Mapping[str, Any], path: Sequence[str]
) -> Mapping[str, Any]:
    current: Any = value
    for name in path:
        if not isinstance(current, Mapping) or name not in current:
            raise KeyError("missing metric path: " + "/".join(path))
        current = current[name]
    if not isinstance(current, Mapping):
        raise TypeError("metric path is not an object: " + "/".join(path))
    return current


def _preferred_direction(metric: str) -> str:
    if metric in {"pearson_r", "r2", "ccc", "sign_agreement_fraction"}:
        return "higher"
    if metric in {"mae", "rmse"}:
        return "lower"
    if metric == "mean_abs_gradient_ratio":
        return "closer_to_one"
    return "closer_to_zero"


def build_factorial_2x2_audit(
    run_specs: Sequence[Stage2RunSpec],
    computed: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize Stage-3 C0 value/support main effects and interaction.

    The cell abbreviations are ``TT`` (true value, true support), ``PT``
    (predicted value, true support), ``TP`` (true value, predicted support),
    and ``PP`` (predicted value, predicted support).  Differences are raw
    metric differences; their desirability depends on ``preferred_direction``.
    """

    runs: dict[str, Any] = {}
    for spec in run_specs:
        cell_modes = {
            "TT_true_value_true_support": "dpr_oracle",
            "PT_predicted_value_true_support": f"{spec.slug}_oracle_mask",
            "TP_true_value_predicted_support": (
                f"{spec.slug}_true_dbz_predicted_mask"
            ),
            "PP_predicted_value_predicted_support": f"{spec.slug}_predicted_mask",
        }
        missing = [slug for slug in cell_modes.values() if slug not in computed]
        if missing:
            raise KeyError(
                f"{spec.label} cannot form a complete 2x2 audit; missing {missing}"
            )
        run_metrics: dict[str, Any] = {}
        for domain, path, metric_names in _FACTORIAL_METRIC_DOMAINS:
            domain_result: dict[str, Any] = {}
            extracted = {
                cell: _nested_mapping(computed[slug], path)
                for cell, slug in cell_modes.items()
            }
            for metric in metric_names:
                values: dict[str, float | None] = {}
                for cell, source in extracted.items():
                    raw = source.get(metric)
                    values[cell] = (
                        float(raw)
                        if isinstance(raw, (int, float)) and math.isfinite(float(raw))
                        else None
                    )
                tt = values["TT_true_value_true_support"]
                pt = values["PT_predicted_value_true_support"]
                tp = values["TP_true_value_predicted_support"]
                pp = values["PP_predicted_value_predicted_support"]
                if any(value is None for value in (tt, pt, tp, pp)):
                    effects = {
                        "value_effect_at_true_support_PT_minus_TT": None,
                        "support_effect_at_true_value_TP_minus_TT": None,
                        "support_effect_at_predicted_value_PP_minus_PT": None,
                        "value_effect_at_predicted_support_PP_minus_TP": None,
                        "interaction_PP_minus_PT_minus_TP_plus_TT": None,
                    }
                else:
                    # The preceding guard narrows all four values to floats.
                    effects = {
                        "value_effect_at_true_support_PT_minus_TT": pt - tt,
                        "support_effect_at_true_value_TP_minus_TT": tp - tt,
                        "support_effect_at_predicted_value_PP_minus_PT": pp - pt,
                        "value_effect_at_predicted_support_PP_minus_TP": pp - tp,
                        "interaction_PP_minus_PT_minus_TP_plus_TT": pp - pt - tp + tt,
                    }
                domain_result[metric] = {
                    "preferred_direction": _preferred_direction(metric),
                    "cells": values,
                    "effects": effects,
                }
            run_metrics[domain] = domain_result
        runs[spec.slug] = {
            "label": spec.label,
            "cells": cell_modes,
            "metrics": run_metrics,
        }
    return {
        "format": FACTORIAL_AUDIT_FORMAT,
        "cell_axes": {
            "first_letter": "reflectivity value: T=true DPR, P=Stage-2 prediction",
            "second_letter": "support: T=true DPR, P=Stage-2 prediction",
        },
        "counterfactual_boundary": (
            "In TP false-positive support voxels, true DPR dBZ is undefined. "
            "The physical value is filled with the Stage-1 train per-height mean, "
            "which becomes standardized zero; the predicted support channel remains 1."
        ),
        "interpretation": (
            "Effects are raw metric differences, not additive percentages. A non-zero "
            "interaction means value and support errors are coupled."
        ),
        "runs": runs,
    }


def factorial_2x2_rows(audit: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Flatten :func:`build_factorial_2x2_audit` for spreadsheet review."""

    rows: list[dict[str, Any]] = []
    for run_slug, run in audit["runs"].items():
        for domain, metrics in run["metrics"].items():
            for metric, result in metrics.items():
                rows.append(
                    {
                        "stage2_run": run_slug,
                        "stage2_label": run["label"],
                        "domain": domain,
                        "metric": metric,
                        "preferred_direction": result["preferred_direction"],
                        **result["cells"],
                        **result["effects"],
                    }
                )
    return rows


def _metric_distance_from_oracle(
    metric: str, value: float, oracle: float
) -> float:
    """Map heterogeneous metrics to a non-negative oracle distance."""

    if metric in {"mae", "rmse"}:
        # For these error metrics the DPR-oracle value need not be zero because
        # sealed Stage 1 is itself imperfect.  Absolute separation is the
        # relevant remaining downstream gap.
        return abs(value - oracle)
    if metric in {"bias"}:
        return abs(value - oracle)
    if metric == "mean_abs_gradient_ratio":
        return abs(value - oracle)
    return abs(value - oracle)


def _build_r0_component_metric_audit(
    computed: Mapping[str, Mapping[str, Any]],
    *,
    baseline_slug: str,
    reference_slug: str,
    candidates: Mapping[str, str],
    reference_name: str,
) -> dict[str, Any]:
    """Build one value- or support-only downstream gap-closure table."""

    required_modes = {baseline_slug, reference_slug, *candidates.values()}
    missing = sorted(required_modes.difference(computed))
    if missing:
        raise KeyError(f"R0 decomposition is missing modes: {missing}")
    component_result: dict[str, Any] = {}
    for domain, path, metric_names in _FACTORIAL_METRIC_DOMAINS:
        baseline_values = _nested_mapping(computed[baseline_slug], path)
        reference_values = _nested_mapping(computed[reference_slug], path)
        domain_result: dict[str, Any] = {}
        for metric in metric_names:
            baseline_raw = baseline_values.get(metric)
            reference_raw = reference_values.get(metric)
            baseline = (
                float(baseline_raw)
                if isinstance(baseline_raw, (int, float))
                and math.isfinite(float(baseline_raw))
                else None
            )
            reference = (
                float(reference_raw)
                if isinstance(reference_raw, (int, float))
                and math.isfinite(float(reference_raw))
                else None
            )
            baseline_distance = (
                _metric_distance_from_oracle(metric, baseline, reference)
                if baseline is not None and reference is not None
                else None
            )
            region_results: dict[str, Any] = {}
            for region, slug in candidates.items():
                raw = _nested_mapping(computed[slug], path).get(metric)
                candidate = (
                    float(raw)
                    if isinstance(raw, (int, float)) and math.isfinite(float(raw))
                    else None
                )
                if baseline is None or reference is None or candidate is None:
                    candidate_distance = closed = delta = None
                else:
                    candidate_distance = _metric_distance_from_oracle(
                        metric, candidate, reference
                    )
                    closed = (
                        (baseline_distance - candidate_distance) / baseline_distance
                        if baseline_distance is not None and baseline_distance > 0.0
                        else None
                    )
                    delta = candidate - baseline
                region_results[region] = {
                    "mode": slug,
                    "candidate": candidate,
                    "candidate_minus_baseline": delta,
                    "candidate_distance_to_reference": candidate_distance,
                    "fraction_of_baseline_to_reference_gap_closed": closed,
                }
            domain_result[metric] = {
                "preferred_direction": _preferred_direction(metric),
                "baseline_mode": baseline_slug,
                "baseline": baseline,
                "reference_name": reference_name,
                "reference_mode": reference_slug,
                "reference": reference,
                "baseline_distance_to_reference": baseline_distance,
                "regions": region_results,
            }
        component_result[domain] = domain_result
    return component_result


def build_r0_decomposition_oracle_audit(
    run_specs: Sequence[Stage2RunSpec],
    computed: Mapping[str, Mapping[str, Any]],
    *,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Quantify how much final-rain error each perfect Stage-2 region closes.

    Value candidates use global true DPR support and isolate conditional dBZ;
    their baseline is ``<run>_oracle_mask`` and reference is ``dpr_oracle``.
    Support candidates keep Stage-2 dBZ fixed, correct occurrence in one
    region, and compare ``<run>_predicted_mask`` with the global-true-support
    ``<run>_oracle_mask`` reference.  This cleanly separates R3 from R4 while
    retaining all effects in final-rain units.
    """

    required_provenance = {
        "split",
        "file_count",
        "expected_file_count",
        "formal_complete_validation",
        "sample_ids",
        "sample_ids_sha256",
        "sample_id_hash_contract",
        "stage1",
        "stage2_runs",
    }
    missing_provenance = sorted(required_provenance.difference(provenance))
    if missing_provenance:
        raise ValueError(
            "R0 oracle provenance is missing fields: "
            + ", ".join(missing_provenance)
        )
    sample_ids = provenance["sample_ids"]
    if isinstance(sample_ids, (str, bytes)) or not isinstance(sample_ids, Sequence):
        raise TypeError("R0 provenance sample_ids must be an ordered sequence")
    normalized_sample_ids = [str(value) for value in sample_ids]
    if int(provenance["file_count"]) != len(normalized_sample_ids):
        raise ValueError("R0 provenance file_count differs from sample_ids")
    if provenance["sample_id_hash_contract"] != R0_SAMPLE_ID_HASH_CONTRACT:
        raise ValueError("R0 provenance uses an unsupported sample-ID hash contract")
    if provenance["sample_ids_sha256"] != ordered_sample_ids_sha256(
        normalized_sample_ids
    ):
        raise ValueError("R0 provenance sample_ids_sha256 is inconsistent")
    stage2_provenance = provenance["stage2_runs"]
    if not isinstance(stage2_provenance, Mapping) or set(stage2_provenance) != {
        spec.slug for spec in run_specs
    }:
        raise ValueError("R0 provenance Stage-2 runs differ from audited runs")

    if "dpr_oracle" not in computed:
        raise KeyError("R0 decomposition requires the common dpr_oracle mode")
    runs: dict[str, Any] = {}
    for spec in run_specs:
        value_baseline_slug = f"{spec.slug}_oracle_mask"
        value_candidates = {
            region: f"{spec.slug}_oracle_value_{region}"
            for region in R0_ORACLE_REGIONS
        }
        support_baseline_slug = f"{spec.slug}_predicted_mask"
        support_candidates = {
            region: f"{spec.slug}_oracle_support_{region}"
            for region in R0_SUPPORT_ORACLE_REGIONS
        }
        value_result = _build_r0_component_metric_audit(
            computed,
            baseline_slug=value_baseline_slug,
            reference_slug="dpr_oracle",
            candidates=value_candidates,
            reference_name="sealed_stage1_dpr_oracle",
        )
        support_result = _build_r0_component_metric_audit(
            computed,
            baseline_slug=support_baseline_slug,
            reference_slug=value_baseline_slug,
            candidates=support_candidates,
            reference_name="same_stage2_dbz_with_true_support",
        )
        runs[spec.slug] = {
            "label": spec.label,
            "value_oracle": {
                "baseline_mode": value_baseline_slug,
                "reference_mode": "dpr_oracle",
                "regions": dict(R0_ORACLE_REGIONS),
                "metrics": value_result,
            },
            "support_oracle": {
                "baseline_mode": support_baseline_slug,
                "reference_mode": value_baseline_slug,
                "regions": dict(R0_SUPPORT_ORACLE_REGIONS),
                "metrics": support_result,
            },
            # Compatibility alias for the initial R0 value-only schema.
            "metrics": value_result,
        }
    return {
        "format": R0_ORACLE_AUDIT_FORMAT,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": dict(provenance),
        "split_requirement": "complete validation split for formal conclusions",
        "component_controls": {
            "value": "true DPR support globally; only regional dBZ changes",
            "support": "Stage-2 dBZ globally; only regional support changes",
        },
        "non_additivity": (
            "q11/gap/outside/q10/q00 form a label-domain partition; q01 is the "
            "gap+outside aggregate and strong_ge35 overlaps positive regions. "
            "Closed-gap fractions must not be summed."
        ),
        "runs": runs,
    }


def r0_decomposition_oracle_rows(audit: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Flatten the R0 oracle-replacement audit for spreadsheet review."""

    rows: list[dict[str, Any]] = []
    for run_slug, run in audit["runs"].items():
        for component_name in ("value_oracle", "support_oracle"):
            component = run[component_name]
            for domain, metrics in component["metrics"].items():
                for metric, result in metrics.items():
                    for region, region_result in result["regions"].items():
                        rows.append(
                            {
                                "stage2_run": run_slug,
                                "stage2_label": run["label"],
                                "oracle_component": component_name.removesuffix(
                                    "_oracle"
                                ),
                                "domain": domain,
                                "metric": metric,
                                "preferred_direction": result[
                                    "preferred_direction"
                                ],
                                "oracle_region": region,
                                "baseline_mode": result["baseline_mode"],
                                "baseline": result["baseline"],
                                "reference_name": result["reference_name"],
                                "reference_mode": result["reference_mode"],
                                "reference": result["reference"],
                                "baseline_distance_to_reference": result[
                                    "baseline_distance_to_reference"
                                ],
                                **region_result,
                            }
                        )
    return rows


def _write_orbit_bundles(
    output_dir: Path,
    selected_ids: Sequence[int],
    stage1_files: Sequence[Mapping[str, Any]],
    saved: Mapping[int, Mapping[str, np.ndarray]],
    modes: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for position, file_id in enumerate(selected_ids):
        entry = stage1_files[file_id]
        sample_id = str(entry["sample_id"])
        relative_dir = Path("orbits") / f"{position:02d}_{sample_id}"
        directory = output_dir / relative_dir
        fields = dict(saved[file_id])
        _atomic_npz(directory / "fields.npz", fields)
        metadata = {
            "format": CASCADE_ORBIT_FORMAT,
            "file_id": file_id,
            "sample_id": sample_id,
            "file_name": entry["file_name"],
            "source_file": entry["file_path"],
            "shape": [int(entry["nscan"]), int(entry["nray"]), int(entry["z_size"])],
            "fields_file": "fields.npz",
            "modes": [
                {
                    "slug": slug,
                    **dict(values),
                    "rain_field": f"rain__{slug}",
                    "input_support_field": f"input_support__{slug}",
                    "output_support_field": f"output_support__{slug}",
                }
                for slug, values in modes.items()
            ],
        }
        _atomic_json(directory / "metadata.json", metadata)
        records.append(
            {
                "selection_position": position,
                "file_id": file_id,
                "sample_id": sample_id,
                "directory": str(relative_dir),
                "metadata": str(relative_dir / "metadata.json"),
            }
        )
    return records


def main() -> None:
    args = parse_args()
    if args.stage1_batch_size <= 0 or args.stage2_batch_size <= 0:
        raise ValueError("Stage-1/2 batch sizes must be positive")
    if args.num_workers < 0 or args.bootstrap_replicates <= 0:
        raise ValueError("num-workers must be non-negative and bootstrap positive")
    if not 0.0 < args.bootstrap_confidence < 1.0:
        raise ValueError("bootstrap-confidence must lie in (0,1)")
    if args.max_files is not None and args.max_files <= 0:
        raise ValueError("max-files must be positive")
    if (
        not args.r0_dbz_thresholds
        or any(not math.isfinite(value) for value in args.r0_dbz_thresholds)
        or tuple(sorted(set(args.r0_dbz_thresholds)))
        != tuple(args.r0_dbz_thresholds)
    ):
        raise ValueError("R0 dBZ thresholds must be finite, sorted, and unique")
    if (
        not args.r0_fss_radii
        or any(value < 0 for value in args.r0_fss_radii)
        or len(set(args.r0_fss_radii)) != len(args.r0_fss_radii)
    ):
        raise ValueError("R0 FSS radii must be unique non-negative integers")
    if not math.isfinite(args.r0_strong_dbz_threshold):
        raise ValueError("R0 strong dBZ threshold must be finite")
    run_specs = parse_stage2_run_specs(args.stage2_run)
    if not run_specs and args.no_dpr_oracle and args.no_gr_interp:
        raise ValueError("no cascade input mode was selected")
    if run_specs and args.no_dpr_oracle:
        raise ValueError(
            "Stage-3 C0 strict 2x2 auditing requires dpr_oracle; "
            "remove --no-dpr-oracle"
        )
    if args.r0_decomposition_oracles:
        if args.split != "val":
            raise ValueError("S2-R0 decomposition oracles may only use split=val")
        # The public R0 contract and output mode name are intentionally sealed
        # at 35 dBZ.  Silently accepting a different value while continuing to
        # write ``strong_ge35`` would make two audit directories incomparable.
        if not math.isclose(
            args.r0_strong_dbz_threshold,
            35.0,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "S2-R0 strong-tail oracle is fixed at 35 dBZ; "
                "do not override --r0-strong-dbz-threshold"
            )
        if not run_specs:
            raise ValueError("S2-R0 decomposition requires at least one Stage-2 run")
        if args.no_dpr_oracle:
            raise ValueError("S2-R0 decomposition requires the DPR oracle upper bound")

    output_dir = args.output_dir.expanduser().resolve()
    protected = (output_dir / "metrics.json", output_dir / "comparison.csv")
    if not args.overwrite and any(path.exists() for path in protected):
        raise FileExistsError("cascade output exists; pass --overwrite to replace it")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    stage1_path = args.stage1_checkpoint.expanduser().resolve()
    stage1_checkpoint = torch.load(stage1_path, map_location="cpu", weights_only=False)
    stage1_config = stage1_checkpoint.get("config")
    if not isinstance(stage1_config, Mapping):
        raise ValueError("Stage-1 checkpoint has no configuration")
    stage1_model = build_stage1_model(stage1_config).to(device)
    stage1_model.load_state_dict(stage1_checkpoint["model"])
    stage1_model.eval()
    stage1_data = stage1_config["data"]
    stage1_index_path = project_path(stage1_data[f"{args.split}_index"])
    stage1_index = _load_json(stage1_index_path)
    stage1_files = stage1_index["files"]
    file_ids = list(range(len(stage1_files)))
    if args.max_files is not None:
        file_ids = file_ids[: args.max_files]
    if not file_ids:
        raise ValueError("no complete orbits selected")
    selected_ids = _selected_orbit_ids(
        stage1_files, file_ids, args.save_orbits, args.selection_seed
    )

    normalization = _load_json(project_path(stage1_data["normalization"]))
    stage1_standardizer = PerLevelStandardizer.from_dict(
        normalization["variables"]["dbz_dpr"]
    )
    heights = np.asarray(stage1_index["heights_km"], dtype=np.float32)
    thresholds = tuple(float(value) for value in stage1_config["loss"]["thresholds_mm_h"])
    stratified_config = stage1_config.get("evaluation", {}).get("stratified", {})
    cfb_edges = tuple(
        float(value)
        for value in stratified_config.get("cfb_distance_edges_km", (-1, 0, 0.5, 2))
    )
    gradient_config = stage1_config.get("evaluation", {}).get("physical_drdz", {})
    sign_epsilon = float(gradient_config.get("sign_epsilon_mm_h_km", 0.1))
    file_labels = [str(stage1_files[index]["sample_id"]) for index in file_ids]
    if args.r0_decomposition_oracles and len(file_labels) != len(set(file_labels)):
        raise ValueError("S2-R0 requires unique ordered validation sample IDs")

    r0_formal_complete_validation = bool(
        args.r0_decomposition_oracles
        and args.split == "val"
        and len(file_ids) == len(stage1_files)
    )
    r0_stage2_run_provenance: dict[str, dict[str, Any]] = {}
    r0_base_provenance: dict[str, Any] | None = None
    if args.r0_decomposition_oracles:
        r0_base_provenance = {
            "split": args.split,
            "file_count": len(file_ids),
            "expected_file_count": len(stage1_files),
            "formal_complete_validation": r0_formal_complete_validation,
            "sample_ids": file_labels,
            "sample_ids_sha256": ordered_sample_ids_sha256(file_labels),
            "sample_id_hash_contract": R0_SAMPLE_ID_HASH_CONTRACT,
            "stage1": {
                "checkpoint_path": str(stage1_path),
                "checkpoint_sha256": sha256_file(stage1_path),
                "checkpoint_epoch": int(stage1_checkpoint["epoch"]),
                "index_path": str(stage1_index_path),
                "index_sha256": sha256_file(stage1_index_path),
            },
            # Filled after each Stage-2 checkpoint is loaded and verified.
            "stage2_runs": r0_stage2_run_provenance,
        }

    expected_stage1_channels = 4 if stage1_data.get("cfb_input_mode") == "signed_distance" else 3
    if int(stage1_config["model"]["in_channels"]) != expected_stage1_channels:
        raise ValueError("Stage-1 model channel count differs from cascade contract")

    modes = build_mode_definitions(
        run_specs,
        include_dpr_oracle=not args.no_dpr_oracle,
        include_gr_interp=not args.no_gr_interp,
        include_r0_decomposition_oracles=args.r0_decomposition_oracles,
    )

    bundles = {
        slug: CascadeMetricBundle.create(
            heights,
            file_labels,
            thresholds_mm_h=thresholds,
            cfb_distance_edges_km=cfb_edges,
            sign_epsilon=sign_epsilon,
            bootstrap_seed=args.bootstrap_seed,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_confidence=args.bootstrap_confidence,
        )
        for slug in modes
    }
    support_rows: list[dict[str, Any]] = []
    r0_diagnostics: dict[str, Stage2DecompositionDiagnostics] = {}
    saved: dict[int, dict[str, np.ndarray]] = {file_id: {} for file_id in selected_ids}

    def evaluate_mode(
        slug: str,
        file_id: int,
        source: Mapping[str, np.ndarray],
        cascade: Stage1CascadePrediction,
    ) -> None:
        target = source["target_rain"]
        bundles[slug].update(
            cascade.rain_rate_mm_h,
            target,
            reliable_positive_mask=source["reliable_positive_mask"],
            qc_label_mask=source["qc_label_mask"],
            heights_km=source["z"],
            cfb_distance_km=source["cfb_distance_km"],
            precipitation_type=source["precipitation_type"],
            file_id=file_ids.index(file_id),
        )
        entry = stage1_files[file_id]
        support_rows.append(
            {
                "mode": slug,
                "file_id": file_id,
                "sample_id": entry["sample_id"],
                "input_support_count": int(cascade.input_support.sum()),
                "output_support_count": int(cascade.output_support.sum()),
                "true_dpr_support_count": int(source["dpr_valid"].sum()),
                "reliable_positive_count": int(source["reliable_positive_mask"].sum()),
                "qc_label_count": int(source["qc_label_mask"].sum()),
            }
        )
        if file_id in saved:
            fields = saved[file_id]
            if not fields:
                fields.update(
                    {
                        "target_rain_mm_h": target.astype(np.float32),
                        "reliable_positive_mask": source["reliable_positive_mask"],
                        "qc_label_mask": source["qc_label_mask"],
                        "heights_km": source["z"].astype(np.float32),
                        "lat": source["lat"].astype(np.float32),
                        "lon": source["lon"].astype(np.float32),
                        "cfb": source["cfb"].astype(np.float32),
                        "precipitation_type": source["precipitation_type"].astype(np.float32),
                        # Common spatial references used by every visualization
                        # panel.  Shape: (nscan,nray,z).
                        "true_dpr_support": source["dpr_valid"].astype(bool),
                    }
                )
            fields[f"rain__{slug}"] = cascade.rain_rate_mm_h.astype(np.float32)
            fields[f"input_support__{slug}"] = cascade.input_support.astype(bool)
            fields[f"output_support__{slug}"] = cascade.output_support.astype(bool)

    stage1_inference_options = {
        "heights_km": heights,
        "standardizer": stage1_standardizer,
        "core_size": int(stage1_index["core_size"]),
        "halo_size": int(stage1_index["halo_size"]),
        "horizontal_multiple": int(stage1_index["horizontal_multiple"]),
        "cfb_input_mode": str(stage1_data.get("cfb_input_mode", "baseline")),
        "cfb_distance_scale_km": float(stage1_data.get("cfb_distance_scale_km", 2.0)),
        "weak_cfb_layer_weights": tuple(stage1_data.get("weak_cfb_layer_weights", ())),
        "device": device,
        "batch_size": args.stage1_batch_size,
        "use_amp": bool(stage1_config.get("training", {}).get("amp", True)),
    }

    # Input routes not requiring a Stage-2 checkpoint are evaluated together.
    if not args.no_dpr_oracle or not args.no_gr_interp:
        for position, file_id in enumerate(file_ids, start=1):
            entry = stage1_files[file_id]
            source = _load_source_fields(Path(entry["file_path"]))
            if not np.allclose(source["z"], heights, rtol=0.0, atol=1e-6):
                raise ValueError(f"height grid changed: {entry['file_name']}")
            common = {
                **stage1_inference_options,
                "cfb_clutter": source["cfb_clutter"],
                "cfb_index": source["cfb"],
            }
            if not args.no_dpr_oracle:
                cascade = predict_stage1_from_reflectivity_orbit(
                    stage1_model,
                    source["dbz_dpr"],
                    source["dpr_valid"],
                    **common,
                )
                evaluate_mode("dpr_oracle", file_id, source, cascade)
            if not args.no_gr_interp:
                cascade = predict_stage1_from_reflectivity_orbit(
                    stage1_model,
                    source["dbz_gr_interp"],
                    source["gr_interp_valid"],
                    **common,
                )
                evaluate_mode("gr_interp", file_id, source, cascade)
            print(
                f"[cascade baselines {position}/{len(file_ids)}] {entry['file_name']}",
                flush=True,
            )

    # Load only one Stage-2 model at a time so multiple comparisons do not
    # retain several large 3-D U-Nets in GPU memory.
    for run_position, spec in enumerate(run_specs, start=1):
        checkpoint = torch.load(spec.checkpoint, map_location="cpu", weights_only=False)
        config = checkpoint.get("config")
        if not isinstance(config, Mapping):
            raise ValueError(f"Stage-2 checkpoint has no config: {spec.checkpoint}")
        stage2_data = config["data"]
        stage2_index_path = project_path(stage2_data[f"{args.split}_index"])
        dataset = Stage2PatchDataset(
            stage2_index_path,
            project_path(stage2_data["normalization"]),
            cache_size=int(stage2_data.get("cache_size", 1)),
            input_channels=stage2_data.get("input_channels"),
        )
        _validate_index_alignment(stage1_index, dataset.index_metadata, label=spec.label)
        if not np.allclose(dataset.z, heights, rtol=0.0, atol=1e-6):
            raise ValueError(f"{spec.label} height grid differs from Stage 1")
        model = build_stage2_model(config).to(device)
        load_checkpoint(spec.checkpoint, model, map_location=device)
        model.eval()
        threshold = load_threshold_file(spec.threshold_file)
        if args.r0_decomposition_oracles:
            r0_stage2_run_provenance[spec.slug] = {
                "label": spec.label,
                "checkpoint_path": str(spec.checkpoint),
                "checkpoint_sha256": sha256_file(spec.checkpoint),
                "checkpoint_epoch": int(checkpoint["epoch"]),
                "support_threshold": threshold,
                "threshold_file_path": str(spec.threshold_file),
                "threshold_file_sha256": sha256_file(spec.threshold_file),
                "index_path": str(stage2_index_path),
                "index_sha256": sha256_file(stage2_index_path),
            }
            r0_diagnostics[spec.slug] = Stage2DecompositionDiagnostics.create(
                heights,
                thresholds_dbz=tuple(args.r0_dbz_thresholds),
                fss_radii=tuple(args.r0_fss_radii),
            )
        for position, file_id in enumerate(file_ids, start=1):
            entry = stage1_files[file_id]
            source = _load_source_fields(Path(entry["file_path"]))
            prediction = predict_stage2_full_orbit(
                model,
                dataset,
                file_id,
                device=device,
                batch_size=args.stage2_batch_size,
                num_workers=args.num_workers,
                use_amp=bool(config.get("training", {}).get("amp", True)),
            )
            predicted_support = prediction.support_probability >= threshold
            common = {
                **stage1_inference_options,
                "cfb_clutter": source["cfb_clutter"],
                "cfb_index": source["cfb"],
            }
            oracle = predict_stage1_from_reflectivity_orbit(
                stage1_model,
                prediction.reflectivity_dbz,
                source["dpr_valid"],
                **common,
            )
            r0_masks: Stage2SubtaskMasks | None = None
            if args.r0_decomposition_oracles:
                r0_masks = build_r0_subtask_masks(
                    source,
                    strong_dbz_threshold=args.r0_strong_dbz_threshold,
                )
                diagnostic_regions = {
                    "q11": r0_masks.q11_overlap,
                    "q10": r0_masks.q10_gr_only,
                    "q01": r0_masks.q01_dpr_only,
                    "q00": r0_masks.q00_neither,
                    "gap": r0_masks.dpr_only_gap,
                    "outside": r0_masks.dpr_only_outside,
                    "strong_ge35": r0_masks.strong_echo,
                    "route_observed": r0_masks.routing.observed,
                    "route_near": r0_masks.routing.near,
                    "route_far": r0_masks.routing.far,
                }
                r0_diagnostics[spec.slug].update(
                    prediction.support_probability,
                    prediction.reflectivity_dbz,
                    predicted_support,
                    source["dbz_dpr"],
                    source["dpr_valid"],
                    r0_masks.label_domain,
                    region_masks=diagnostic_regions,
                )
            # Counterfactual TP cell: observed DPR values wherever they exist,
            # and the Stage-1 neutral physical mean at Stage-2 false positives.
            # Shapes remain (nscan,nray,z) throughout this audit route.
            true_dbz_counterfactual = prepare_true_dpr_for_predicted_support(
                source["dbz_dpr"],
                source["dpr_valid"],
                predicted_support,
                stage1_standardizer,
            )
            true_dbz_predicted_mask = predict_stage1_from_reflectivity_orbit(
                stage1_model,
                true_dbz_counterfactual,
                predicted_support,
                **common,
            )
            deployed = predict_stage1_from_reflectivity_orbit(
                stage1_model,
                prediction.reflectivity_dbz,
                predicted_support,
                **common,
            )
            evaluate_mode(f"{spec.slug}_oracle_mask", file_id, source, oracle)
            evaluate_mode(
                f"{spec.slug}_true_dbz_predicted_mask",
                file_id,
                source,
                true_dbz_predicted_mask,
            )
            evaluate_mode(f"{spec.slug}_predicted_mask", file_id, source, deployed)
            if r0_masks is not None:
                for region, replacement_mask in r0_oracle_region_masks(
                    r0_masks
                ).items():
                    hybrid_dbz = replace_stage2_dbz_with_oracle_region(
                        prediction.reflectivity_dbz,
                        source["dbz_dpr"],
                        source["dpr_valid"],
                        replacement_mask,
                    )
                    hybrid = predict_stage1_from_reflectivity_orbit(
                        stage1_model,
                        hybrid_dbz,
                        source["dpr_valid"],
                        **common,
                    )
                    evaluate_mode(
                        f"{spec.slug}_oracle_value_{region}",
                        file_id,
                        source,
                        hybrid,
                    )
                # Support-only candidates keep the dense Stage-2 dBZ field
                # fixed and correct occurrence only inside one supervision
                # region.  This separates R3 support recovery from R4
                # conditional-value recovery in final-rain units.
                for region, replacement_mask in r0_support_oracle_region_masks(
                    r0_masks
                ).items():
                    oracle_input = build_regional_oracle_input(
                        prediction.reflectivity_dbz,
                        predicted_support,
                        source["dbz_dpr"],
                        source["dpr_valid"],
                        replacement_mask,
                        component="support",
                    )
                    hybrid = predict_stage1_from_reflectivity_orbit(
                        stage1_model,
                        oracle_input.reflectivity_dbz,
                        oracle_input.reflectivity_support,
                        **common,
                    )
                    evaluate_mode(
                        f"{spec.slug}_oracle_support_{region}",
                        file_id,
                        source,
                        hybrid,
                    )
            if file_id in saved:
                # Stage-2 diagnostic products make support boundaries and
                # threshold decisions independently inspectable downstream.
                fields = saved[file_id]
                fields[f"stage2_support_probability__{spec.slug}"] = (
                    prediction.support_probability.astype(np.float32)
                )
                fields[f"stage2_predicted_support__{spec.slug}"] = (
                    predicted_support.astype(bool)
                )
                fields[f"stage2_reflectivity_dbz__{spec.slug}"] = (
                    prediction.reflectivity_dbz.astype(np.float32)
                )
            print(
                f"[cascade {run_position}/{len(run_specs)} {position}/{len(file_ids)}] "
                f"{spec.label}: {entry['file_name']}",
                flush=True,
            )
        del model, dataset, checkpoint
        if device.type == "cuda":
            torch.cuda.empty_cache()

    computed = {slug: bundle.compute() for slug, bundle in bundles.items()}
    comparison_rows = [_mode_row(slug, modes[slug], computed[slug]) for slug in modes]
    factorial_audit = build_factorial_2x2_audit(run_specs, computed)
    factorial_rows = factorial_2x2_rows(factorial_audit)
    r0_oracle_audit = (
        build_r0_decomposition_oracle_audit(
            run_specs,
            computed,
            provenance=r0_base_provenance,
        )
        if args.r0_decomposition_oracles
        else None
    )
    r0_oracle_rows = (
        r0_decomposition_oracle_rows(r0_oracle_audit)
        if r0_oracle_audit is not None
        else []
    )
    r0_diagnostic_results = {
        slug: diagnostics.compute()
        for slug, diagnostics in sorted(r0_diagnostics.items())
    }
    r0_cfad_rows = [
        {"stage2_run": slug, **row}
        for slug, diagnostics in sorted(r0_diagnostics.items())
        for row in diagnostics.cfad.rows()
    ]
    orbit_records = _write_orbit_bundles(
        output_dir, selected_ids, stage1_files, saved, modes
    )
    summary = {
        "format_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "frozen_stage2_stage1_complete_orbit_cascade",
        "split": args.split,
        "file_count": len(file_ids),
        "stage1_checkpoint": str(stage1_path),
        "stage1_checkpoint_epoch": int(stage1_checkpoint["epoch"]),
        "stage1_input_contract": {
            "physical_to_model": (
                "physical dBZ -> Stage-1 train-only per-height standardization; "
                "invalid standardized cells=0 with a separate support channel"
            ),
            "tensor_shape": f"(B,{expected_stage1_channels},64,64,{heights.size})",
            "height_padding": 0,
            "precipitation_label_used_as_input": False,
        },
        "evaluation_masks": {
            "reliable_positive": (
                "pre_positive_qc AND true DPR reflectivity valid; common across modes; "
                "a Stage-2 miss remains prediction rain=0 and is still scored"
            ),
            "qc_label_domain_including_zero": (
                "pre_valid_qc including valid zero rain; common across modes"
            ),
            "predicted_support": (
                "support_probability >= validation-selected threshold over the full "
                "physical grid; never intersected with pre_dpr/occupancy_domain"
            ),
            "true_dbz_predicted_support_false_positives": (
                "true DPR dBZ is undefined there; fill physical dBZ with the "
                "Stage-1 train per-height mean so its standardized value is 0, "
                "while keeping predicted support=1"
            ),
        },
        "factorial_2x2_audit": {
            "format": FACTORIAL_AUDIT_FORMAT,
            "json": "factorial_2x2.json",
            "csv": "factorial_2x2.csv" if factorial_rows else None,
            "stage2_run_count": len(run_specs),
        },
        "r0_decomposition_audit": {
            "enabled": args.r0_decomposition_oracles,
            "oracle_format": (
                R0_ORACLE_AUDIT_FORMAT if args.r0_decomposition_oracles else None
            ),
            "diagnostics_format": (
                R0_DIAGNOSTICS_FORMAT if args.r0_decomposition_oracles else None
            ),
            "formal_complete_validation": r0_formal_complete_validation,
            "oracle_json": (
                "r0_decomposition_oracles.json"
                if args.r0_decomposition_oracles
                else None
            ),
            "oracle_csv": (
                "r0_decomposition_oracles.csv" if r0_oracle_rows else None
            ),
            "diagnostics_json": (
                "r0_decomposition_diagnostics.json"
                if args.r0_decomposition_oracles
                else None
            ),
            "cfad_csv": "r0_cfad.csv" if r0_cfad_rows else None,
            "dbz_thresholds": list(args.r0_dbz_thresholds),
            "fss_radii": list(args.r0_fss_radii),
            "strong_dbz_threshold": args.r0_strong_dbz_threshold,
        },
        "modes": modes,
        "metrics": finite_metrics_for_json(computed),
        "saved_orbit_selection": {
            "seed": args.selection_seed,
            "requested_count": args.save_orbits,
            "selected_file_ids": selected_ids,
            "orbits": orbit_records,
            "manifest": "orbit_manifest.json",
        },
    }
    orbit_manifest = {
        "format": CASCADE_ORBIT_FORMAT,
        "evaluation_summary": "metrics.json",
        "split": args.split,
        "modes": [{"slug": slug, **values} for slug, values in modes.items()],
        "orbits": orbit_records,
    }
    _atomic_json(output_dir / "metrics.json", summary)
    _atomic_json(output_dir / "orbit_manifest.json", orbit_manifest)
    _atomic_json(output_dir / "factorial_2x2.json", factorial_audit)
    if factorial_rows:
        _atomic_csv(output_dir / "factorial_2x2.csv", factorial_rows)
    if r0_oracle_audit is not None:
        _atomic_json(output_dir / "r0_decomposition_oracles.json", r0_oracle_audit)
        _atomic_json(
            output_dir / "r0_decomposition_diagnostics.json",
            {
                "format": R0_DIAGNOSTICS_FORMAT,
                "split": args.split,
                "file_count": len(file_ids),
                "formal_complete_validation": r0_formal_complete_validation,
                "oracle_provenance": r0_oracle_audit["provenance"],
                "runs": r0_diagnostic_results,
            },
        )
    if r0_oracle_rows:
        _atomic_csv(output_dir / "r0_decomposition_oracles.csv", r0_oracle_rows)
    if r0_cfad_rows:
        _atomic_csv(output_dir / "r0_cfad.csv", r0_cfad_rows)
    _atomic_csv(output_dir / "comparison.csv", comparison_rows)
    _atomic_csv(output_dir / "support_per_file.csv", support_rows)
    print(f"Cascade evaluation complete -> {output_dir}", flush=True)
    for row in comparison_rows:
        print(
            f"  {row['mode']}: RMSE={row['positive_rmse']:.4f}, "
            f"r={row['positive_pearson_r']:.4f}, "
            f"dR/dz RMSE={row['drdz_rmse']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
