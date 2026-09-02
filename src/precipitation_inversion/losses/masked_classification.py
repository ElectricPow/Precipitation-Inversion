"""Masked profile-level losses for the DPR precipitation-type auxiliary task."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as functional
from torch import nn


class MaskedCrossEntropyLoss(nn.Module):
    """Cross entropy over explicitly valid ``(scan,ray)`` profiles only."""

    def __init__(self, class_weights: Sequence[float] | None = None) -> None:
        super().__init__()
        weights = (
            torch.empty(0, dtype=torch.float32)
            if class_weights is None
            else torch.as_tensor(tuple(class_weights), dtype=torch.float32)
        )
        if weights.ndim != 1 or (weights.numel() and bool((weights <= 0).any())):
            raise ValueError("class_weights must be a positive one-dimensional sequence")
        self.register_buffer("class_weights", weights)

    def forward(
        self, logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        if logits.ndim != 4:
            raise ValueError("logits must have shape (B,K,nscan,nray)")
        expected = (logits.shape[0], logits.shape[2], logits.shape[3])
        if target.shape != expected or mask.shape != expected:
            raise ValueError("target/mask must match logits spatial shape")
        if target.dtype != torch.long or mask.dtype != torch.bool:
            raise TypeError("target must be int64 and mask must be boolean")
        selected_target = target[mask]
        if selected_target.numel() == 0:
            return logits.sum() * 0.0
        if bool(((selected_target < 0) | (selected_target >= logits.shape[1])).any()):
            raise ValueError("selected class target lies outside logits classes")
        # (B,K,D,H) -> (B,D,H,K), then boolean selection -> (N,K).
        selected_logits = logits.permute(0, 2, 3, 1)[mask]
        weights = self.class_weights if self.class_weights.numel() else None
        return functional.cross_entropy(
            selected_logits, selected_target, weight=weights, reduction="mean"
        )

