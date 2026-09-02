#!/usr/bin/env python3
"""Backfill physical dR/dz for E0/N/I/W, plot it, then compare runs.

The historical full-validation JSON files predate the physical-gradient
implementation.  This entry point deliberately reruns the shared evaluator for
each ``best.pt`` instead of weakening comparison checks or editing old JSON by
hand.  A successfully completed run is reusable after an interruption.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMPARISON_OUTPUT = (
    PROJECT_ROOT / "outputs" / "ablations" / "stage1_drdz_comparison"
)
RUN_DIRECTORIES = (
    ("E0", PROJECT_ROOT / "outputs" / "ablations" / "stage1_e0_baseline"),
    ("N", PROJECT_ROOT / "outputs" / "ablations" / "stage1_e0_n_dbz_valid"),
    ("I", PROJECT_ROOT / "outputs" / "ablations" / "stage1_e0_n_i_intensity"),
    ("W", PROJECT_ROOT / "outputs" / "ablations" / "stage1_e0_n_w_weak_cfb"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        default="cuda:0",
        help=(
            "Logical evaluation device. With CUDA_VISIBLE_DEVICES=7, use "
            "cuda:0 (default: cuda:0)."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers per evaluation (default: 0).",
    )
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--comparison-output",
        type=Path,
        default=DEFAULT_COMPARISON_OUTPUT,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-evaluate even when a complete physical dR/dz report exists.",
    )
    return parser.parse_args()


def reusable_physical_drdz_report(path: Path) -> tuple[bool, str]:
    """Return whether ``path`` is a complete, exact-support validation report."""

    if not path.is_file():
        return False, "metrics.json does not exist"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return False, f"cannot read metrics.json: {error}"
    if not isinstance(value, Mapping) or value.get("split") != "val":
        return False, "report is not from the validation split"
    patch = value.get("patch_evaluation")
    if not isinstance(patch, Mapping):
        return False, "patch_evaluation is missing"
    coverage = patch.get("coverage")
    if not isinstance(coverage, Mapping) or coverage.get(
        "complete_patch_support"
    ) is not True:
        return False, "complete patch coverage is missing"
    metrics = patch.get("metrics")
    drdz = metrics.get("physical_drdz") if isinstance(metrics, Mapping) else None
    if not isinstance(drdz, Mapping):
        return False, "physical_drdz is missing"
    support = drdz.get("support")
    fingerprint = support.get("sha256") if isinstance(support, Mapping) else None
    if not isinstance(fingerprint, str) or not fingerprint:
        return False, "exact dR/dz support fingerprint is missing"
    return True, "complete physical dR/dz report"


def evaluation_command(
    checkpoint: Path,
    output: Path,
    *,
    device: str,
    num_workers: int,
    bootstrap_seed: int,
    bootstrap_replicates: int,
    bootstrap_confidence: float,
) -> list[str]:
    """Build the canonical full-validation command shared by all four runs."""

    return [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "evaluate_stage1_unet3d.py"),
        str(checkpoint),
        "--split",
        "val",
        "--stratified",
        "--device",
        device,
        "--num-workers",
        str(num_workers),
        "--full-orbits",
        "0",
        "--bootstrap-seed",
        str(bootstrap_seed),
        "--bootstrap-replicates",
        str(bootstrap_replicates),
        "--bootstrap-confidence",
        str(bootstrap_confidence),
        "--output",
        str(output),
    ]


def _run_logged(command: Sequence[str], log_path: Path, *, label: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{label}] running; detailed output -> {log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            list(command),
            cwd=PROJECT_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode == 0:
        return
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    tail = "\n".join(lines[-25:])
    raise RuntimeError(
        f"{label} evaluation failed with exit code {completed.returncode}; "
        f"see {log_path}\n--- log tail ---\n{tail}"
    )


def main() -> None:
    args = parse_args()
    if args.num_workers < 0:
        raise ValueError("num-workers must be non-negative")
    if args.bootstrap_seed < 0:
        raise ValueError("bootstrap-seed must be non-negative")
    if args.bootstrap_replicates <= 0:
        raise ValueError("bootstrap-replicates must be positive")
    if not 0.0 < args.bootstrap_confidence < 1.0:
        raise ValueError("bootstrap-confidence must lie between zero and one")
    if args.dpi <= 0:
        raise ValueError("dpi must be positive")

    metric_paths: dict[str, Path] = {}
    for label, run_directory in RUN_DIRECTORIES:
        checkpoint = run_directory / "best.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"{label} checkpoint does not exist: {checkpoint}")
        analysis = run_directory / "analysis" / "full_validation"
        metrics = analysis / "metrics.json"
        metric_paths[label] = metrics
        reusable, reason = reusable_physical_drdz_report(metrics)
        if reusable and not args.force:
            print(f"[{label}] reusing {metrics}", flush=True)
        else:
            print(f"[{label}] backfill required: {reason}", flush=True)
            command = evaluation_command(
                checkpoint,
                metrics,
                device=args.device,
                num_workers=args.num_workers,
                bootstrap_seed=args.bootstrap_seed,
                bootstrap_replicates=args.bootstrap_replicates,
                bootstrap_confidence=args.bootstrap_confidence,
            )
            _run_logged(command, analysis / "drdz_backfill.log", label=label)
            reusable, reason = reusable_physical_drdz_report(metrics)
            if not reusable:
                raise RuntimeError(
                    f"{label} evaluator exited successfully but produced an "
                    f"unusable report: {reason}"
                )

        subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "plot_stage1_stratified_metrics.py"),
                str(metrics),
                "--output-dir",
                str(analysis / "stratified"),
                "--dpi",
                str(args.dpi),
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )

    comparison = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "compare_stage1_drdz.py"),
    ]
    for label, _ in RUN_DIRECTORIES:
        comparison.extend(("--run", f"{label}={metric_paths[label]}"))
    comparison.extend(
        (
            "--baseline",
            "E0",
            "--bootstrap-seed",
            str(args.bootstrap_seed),
            "--bootstrap-replicates",
            str(args.bootstrap_replicates),
            "--confidence",
            str(args.bootstrap_confidence),
            "--output-dir",
            str(args.comparison_output.expanduser().resolve()),
            "--dpi",
            str(args.dpi),
        )
    )
    subprocess.run(comparison, cwd=PROJECT_ROOT, check=True)
    print(
        "Physical dR/dz backfill and comparison completed: "
        f"{args.comparison_output.expanduser().resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
