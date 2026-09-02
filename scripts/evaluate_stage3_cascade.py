#!/usr/bin/env python3
"""Evaluate C1 Stage1-only or C2 Stage2-only checkpoints with recorded sources."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE3_CHECKPOINT_FORMAT = "stage3_c1_oracle_stage1_only_v1"
STAGE3_C2_CHECKPOINT_FORMAT = "stage3_c2_oracle_stage2_only_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--threshold-file",
        type=Path,
        help="Override the recorded source threshold (required for formal C2 support audit).",
    )
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-orbits", type=int, default=6)
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_stage3_sources(checkpoint: str | Path) -> dict[str, str]:
    """Resolve the trainable checkpoint side and its recorded frozen source."""

    source = Path(checkpoint).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    stage3_format = payload.get("stage3_format")
    if stage3_format not in (STAGE3_CHECKPOINT_FORMAT, STAGE3_C2_CHECKPOINT_FORMAT):
        raise ValueError("checkpoint is not a supported C1-O/C2-O Stage-3 checkpoint")
    values = payload.get("stage3_sources")
    if not isinstance(values, Mapping):
        raise ValueError("Stage-3 checkpoint is missing frozen-source metadata")
    if stage3_format == STAGE3_CHECKPOINT_FORMAT:
        result = {
            "stage3_format": str(stage3_format),
            "stage1_checkpoint": str(source),
            "stage2_checkpoint": str(values["stage2_checkpoint"]),
            "stage2_threshold_file": str(values["stage2_threshold_file"]),
            "stage2_label": "C1Adapt-W1.25",
        }
    else:
        result = {
            "stage3_format": str(stage3_format),
            "stage1_checkpoint": str(values["stage1_checkpoint"]),
            "stage2_checkpoint": str(source),
            # This is the pre-adaptation threshold and is retained only as a
            # fallback. Formal C2 postprocessing supplies a newly selected val
            # threshold through --threshold-file.
            "stage2_threshold_file": str(values["stage2_threshold_file"]),
            "stage2_label": "C2TaskAware-W1.25",
        }
    for name in ("stage1_checkpoint", "stage2_checkpoint", "stage2_threshold_file"):
        value = result[name]
        if not Path(value).is_file():
            raise FileNotFoundError(f"resolved {name} no longer exists: {value}")
    return result


def _mode_slug(label: str) -> str:
    return re.sub(r"[^a-z0-9_-]+", "_", label.strip().lower()).strip("_-")


def build_evaluation_command(
    *,
    checkpoint: Path,
    output_dir: Path,
    split: str,
    device: str,
    save_orbits: int,
    max_files: int | None,
    overwrite: bool,
    threshold_file: Path | None = None,
) -> list[str]:
    """Build the common C0 evaluator invocation used for fair comparison."""

    if split not in ("val", "test"):
        raise ValueError("split must be val or test")
    if save_orbits < 0:
        raise ValueError("save_orbits must be non-negative")
    sources = load_stage3_sources(checkpoint)
    selected_threshold = (
        threshold_file.expanduser().resolve()
        if threshold_file is not None
        else Path(sources["stage2_threshold_file"]).resolve()
    )
    if not selected_threshold.is_file():
        raise FileNotFoundError(selected_threshold)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "evaluate_stage2_stage1_cascade.py"),
        "--stage1-checkpoint", sources["stage1_checkpoint"],
        "--stage2-run", sources["stage2_label"],
        sources["stage2_checkpoint"], str(selected_threshold),
        "--split", split,
        "--output-dir", str(output_dir.resolve()),
        "--device", device,
        "--save-orbits", str(save_orbits),
    ]
    if max_files is not None:
        if max_files <= 0:
            raise ValueError("max_files must be positive")
        command.extend(("--max-files", str(max_files)))
    if overwrite:
        command.append("--overwrite")
    return command


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    command = build_evaluation_command(
        checkpoint=checkpoint,
        output_dir=output_dir,
        split=args.split,
        device=args.device,
        save_orbits=args.save_orbits,
        max_files=args.max_files,
        overwrite=args.overwrite,
        threshold_file=args.threshold_file,
    )
    commands: list[Sequence[str]] = [command]
    if args.visualize and args.save_orbits > 0:
        sources = load_stage3_sources(checkpoint)
        oracle_mode = _mode_slug(sources["stage2_label"]) + "_oracle_mask"
        visualization = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "visualize_stage2_stage1_cascade.py"),
            "--input-dir", str(output_dir),
            "--output-dir", str(output_dir / "visualizations"),
            "--modes", "dpr_oracle", oracle_mode,
        ]
        if args.overwrite:
            visualization.append("--overwrite")
        commands.append(visualization)
    if args.dry_run:
        print(json.dumps(commands, ensure_ascii=False, indent=2))
        return
    for current in commands:
        print("[stage3-eval] running: " + " ".join(current), flush=True)
        subprocess.run(current, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
