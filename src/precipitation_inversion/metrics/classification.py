"""Dependency-free masked multiclass metrics with optional DDP reduction."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.distributed as distributed


class MulticlassConfusionMetrics:
    """Accumulate rows=true, columns=predicted confusion counts on CPU."""

    def __init__(self, class_names: Sequence[str]) -> None:
        self.class_names = tuple(str(value) for value in class_names)
        if len(self.class_names) < 2 or len(set(self.class_names)) != len(self.class_names):
            raise ValueError("class_names must contain at least two unique labels")
        self.confusion = torch.zeros(
            (len(self.class_names), len(self.class_names)), dtype=torch.int64
        )

    def reset(self) -> None:
        self.confusion.zero_()

    def update(
        self, logits_or_prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> None:
        if logits_or_prediction.ndim == target.ndim + 1:
            prediction = logits_or_prediction.argmax(dim=1)
        elif logits_or_prediction.shape == target.shape:
            prediction = logits_or_prediction
        else:
            raise ValueError("prediction must be class indices or (B,K,D,H) logits")
        if prediction.shape != target.shape or mask.shape != target.shape:
            raise ValueError("prediction, target and mask shapes differ")
        selected_target = target[mask].to(dtype=torch.int64)
        selected_prediction = prediction[mask].to(dtype=torch.int64)
        class_count = len(self.class_names)
        if selected_target.numel() == 0:
            return
        invalid = (
            (selected_target < 0)
            | (selected_target >= class_count)
            | (selected_prediction < 0)
            | (selected_prediction >= class_count)
        )
        if bool(invalid.any()):
            raise ValueError("selected class index is out of range")
        flat = selected_target * class_count + selected_prediction
        counts = torch.bincount(flat, minlength=class_count * class_count)
        self.confusion += counts.reshape(class_count, class_count).cpu()

    def compute(self, *, synchronize: bool = False) -> dict[str, Any]:
        confusion = self.confusion.clone()
        if synchronize and distributed.is_available() and distributed.is_initialized():
            backend = str(distributed.get_backend()).lower()
            device = (
                torch.device("cuda", torch.cuda.current_device())
                if "nccl" in backend
                else torch.device("cpu")
            )
            reduced = confusion.to(device)
            distributed.all_reduce(reduced, op=distributed.ReduceOp.SUM)
            confusion = reduced.cpu()
        values = confusion.to(dtype=torch.float64)
        support = values.sum(dim=1)
        predicted = values.sum(dim=0)
        true_positive = values.diag()
        supported = support > 0
        recall = torch.where(supported, true_positive / support, torch.nan)
        # Match the usual zero_division=0 convention: a supported class that
        # is never predicted has precision/F1 zero, not a silently omitted NaN.
        precision = torch.where(
            supported,
            torch.where(predicted > 0, true_positive / predicted, 0.0),
            torch.nan,
        )
        f1 = torch.where(
            supported,
            torch.where(
                (precision + recall) > 0,
                2.0 * precision * recall / (precision + recall),
                0.0,
            ),
            torch.nan,
        )
        count = int(support.sum().item())

        def finite_mean(vector: torch.Tensor) -> float:
            selected = vector[torch.isfinite(vector)]
            return float(selected.mean().item()) if selected.numel() else float("nan")

        per_class = {
            name: {
                "support": int(support[index].item()),
                "precision": float(precision[index].item()),
                "recall": float(recall[index].item()),
                "f1": float(f1[index].item()),
            }
            for index, name in enumerate(self.class_names)
        }
        return {
            "count": count,
            "accuracy": float(true_positive.sum().item() / count) if count else float("nan"),
            "balanced_accuracy": finite_mean(recall),
            "macro_precision": finite_mean(precision),
            "macro_recall": finite_mean(recall),
            "macro_f1": finite_mean(f1),
            "class_names": list(self.class_names),
            "confusion_matrix": confusion.tolist(),
            "per_class": per_class,
        }
