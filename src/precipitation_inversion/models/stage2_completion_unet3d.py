"""Single-head Stage-2 value completion network for the R1-O upper bound.

This model intentionally has no support head.  It predicts standardized DPR
reflectivity everywhere and is supervised/evaluated only on the true DPR
support.  The non-deployable Oracle-DPRSparse Dataset supplies true DPR values
at the sparse GR observation geometry, isolating spatial completion from the
GR-to-DPR sensor-domain conversion.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from .unet3d import Stage1UNet3D


class Stage2CompletionUNet3D(Stage1UNet3D):
    """Height-preserving single-head 3D U-Net for standardized DPR dBZ.

    Input and output follow ``(B,C,D,H,Z)``.  With the formal R1-O contract,
    ``C=4`` and ``(D,H,Z)=(64,64,60)``.  Every encoder reduction uses
    ``(2,2,1)``, so the 60 physical height levels are never pooled or padded.
    """

    def __init__(
        self,
        *,
        in_channels: int = 4,
        base_channels: int = 16,
        channel_multipliers: Sequence[int] = (1, 2, 4, 8, 16),
        max_groups: int = 8,
        bottleneck_dropout: float = 0.1,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=1,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            max_groups=max_groups,
            bottleneck_dropout=bottleneck_dropout,
        )
        # Rename the Stage-1 rain head so checkpoints cannot be mistaken for a
        # precipitation model.  Zero standardized dBZ is the per-height train
        # mean and therefore a stable initial output.
        del self.output_head
        self.reflectivity_head = nn.Conv3d(
            self.channels[0], 1, kernel_size=1, bias=True
        )
        nn.init.normal_(self.reflectivity_head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.reflectivity_head.bias)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        # (B,C,D,H,Z) -> shared decoder (B,base,D,H,Z) -> (B,1,D,H,Z).
        features = self.forward_features(inputs)
        return {"reflectivity": self.reflectivity_head(features)}


def stage2_completion_prediction_from_output(output: Any) -> torch.Tensor:
    """Validate and extract one ``(B,1,D,H,Z)`` reflectivity tensor."""

    if not isinstance(output, Mapping):
        raise TypeError("Stage-2 completion output must be a mapping")
    reflectivity = output.get("reflectivity")
    if not isinstance(reflectivity, torch.Tensor):
        raise TypeError("Stage-2 completion output needs Tensor 'reflectivity'")
    if reflectivity.ndim != 5 or reflectivity.shape[1] != 1:
        raise ValueError(
            "Stage-2 completion reflectivity must have shape (B,1,D,H,Z)"
        )
    return reflectivity


__all__ = [
    "Stage2CompletionUNet3D",
    "stage2_completion_prediction_from_output",
]
