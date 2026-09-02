#!/usr/bin/env python3
"""Evaluate type classification plus height shuffle/occlusion diagnostics."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_stage1_unet3d import build_model, project_path, resolve_device  # noqa: E402
from precipitation_inversion.data.patch_dataset import (  # noqa: E402
    Stage1PatchDataset,
    stage1_patch_dataset_kwargs,
)
from precipitation_inversion.data.precipitation_type import TYPE_NAMES  # noqa: E402
from precipitation_inversion.losses.masked_classification import (  # noqa: E402
    MaskedCrossEntropyLoss,
)
from precipitation_inversion.metrics.classification import (  # noqa: E402
    MulticlassConfusionMetrics,
)
from precipitation_inversion.models.multitask_unet3d import (  # noqa: E402
    Stage1MultiTaskUNet3D,
)
from precipitation_inversion.training.engine import move_batch_to_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def evaluate_variant(
    model: Stage1MultiTaskUNet3D,
    loader: torch.utils.data.DataLoader,
    criterion: MaskedCrossEntropyLoss,
    device: torch.device,
    *,
    permutation: torch.Tensor | None = None,
    occlusion: slice | None = None,
    max_batches: int | None = None,
) -> dict[str, Any]:
    metrics = MulticlassConfusionMetrics(TYPE_NAMES)
    loss_sum = 0.0
    count = 0
    with torch.inference_mode():
        for batch_index, source in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            batch = move_batch_to_device(source, device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                # Shared feature meaning: (B,16,64,64,60). Perturbations are
                # applied only before the type head and never alter rain output.
                features = model.forward_features(batch["inputs"])
                if occlusion is not None:
                    features = features.clone()
                    features[..., occlusion] = 0.0
                if permutation is not None and hasattr(model.type_head, "height_levels"):
                    logits = model.type_head(features, height_permutation=permutation)
                else:
                    if permutation is not None:
                        features = features.index_select(-1, permutation)
                    logits = model.type_head(features)
                loss = criterion(logits, batch["type_target"], batch["type_loss_mask"])
            current = int(batch["type_loss_mask"].sum().item())
            loss_sum += float(loss.item()) * current
            count += current
            metrics.update(logits, batch["type_target"], batch["type_loss_mask"])
    return {
        "cross_entropy": loss_sum / count if count else float("nan"),
        **metrics.compute(),
    }


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config")
    if not isinstance(config, dict) or not bool(config.get("type_task", {}).get("enabled", False)):
        raise ValueError("checkpoint is not a precipitation-type multitask model")
    device = resolve_device(args.device)
    model = build_model(config).to(device)
    if not isinstance(model, Stage1MultiTaskUNet3D):
        raise TypeError("configuration did not construct Stage1MultiTaskUNet3D")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    data = config["data"]
    dataset = Stage1PatchDataset(
        project_path(data[f"{args.split}_index"]),
        project_path(data["normalization"]),
        positive_only=False,
        cache_size=int(data["cache_size"]),
        **stage1_patch_dataset_kwargs(data, config["loss"]),
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(data["batch_size"]),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    weights = config["type_task"].get("resolved_class_weights")
    if weights is None:
        weights = config["type_task"].get("class_weights")
    if not isinstance(weights, list):
        raise ValueError("checkpoint does not contain resolved class weights")
    criterion = MaskedCrossEntropyLoss(weights).to(device)
    z_size = int(dataset.z.size)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    reverse = torch.arange(z_size - 1, -1, -1, device=device)
    random_order = torch.randperm(z_size, generator=generator).to(device)
    thirds = (0, z_size // 3, 2 * z_size // 3, z_size)
    result = {
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "type_head_kind": config["type_task"]["head"]["kind"],
        "normal": evaluate_variant(model, loader, criterion, device, max_batches=args.max_batches),
        "reverse_height": evaluate_variant(model, loader, criterion, device, permutation=reverse, max_batches=args.max_batches),
        "random_height": evaluate_variant(model, loader, criterion, device, permutation=random_order, max_batches=args.max_batches),
        "occlude_low_third": evaluate_variant(model, loader, criterion, device, occlusion=slice(thirds[0], thirds[1]), max_batches=args.max_batches),
        "occlude_middle_third": evaluate_variant(model, loader, criterion, device, occlusion=slice(thirds[1], thirds[2]), max_batches=args.max_batches),
        "occlude_high_third": evaluate_variant(model, loader, criterion, device, occlusion=slice(thirds[2], thirds[3]), max_batches=args.max_batches),
        "notes": {
            "shuffle": "permutes decoder-feature z order only for the type head",
            "occlusion": "sets one decoder-feature height third to zero only for the type head",
            "rain_feedback": False,
        },
    }
    safe = json_safe(result)
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else checkpoint_path.parent / "analysis" / f"type_head_{args.split}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(safe, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(safe, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
