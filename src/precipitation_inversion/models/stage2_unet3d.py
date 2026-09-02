"""Dual-head height-preserving 3D U-Net for sparse-GR to dense-DPR mapping."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from .unet3d import Stage1UNet3D


class Stage2UNet3D(Stage1UNet3D):
    """Share an anisotropic 3D U-Net and predict support plus standardized dBZ.

    Input is ``(B,C,D,H,Z)``. The four-channel baseline uses standardized
    sparse-GR dBZ, GR value mask, GR native-available storage proxy and scaled
    height. Controlled ablations may use a configured subset such as the
    three-channel native variant. Four default encoder reductions change only
    ``D/H``; the complete height axis remains at 60.

    The model returns two independent dense tensors:

    * ``support_logits``: unbounded logits ``(B,1,D,H,Z)``;
    * ``reflectivity``: standardized DPR dBZ ``(B,1,D,H,Z)``.

    No sigmoid or support hard gate is applied inside the model.  This lets the
    reflectivity head learn from every valid DPR target even when the support
    head is initially wrong.
    """

    def __init__(
        self,
        *,
        in_channels: int = 4,
        base_channels: int = 16,
        channel_multipliers: Sequence[int] = (1, 2, 4, 8, 16),
        max_groups: int = 8,
        bottleneck_dropout: float = 0.1,
        support_prior_probability: float | None = 0.04,
    ) -> None:
        if in_channels <= 0:
            raise ValueError("Stage2UNet3D in_channels must be positive")
        if support_prior_probability is not None and (
            not math.isfinite(support_prior_probability)
            or not 0.0 < support_prior_probability < 1.0
        ):
            raise ValueError("support_prior_probability must be null or lie in (0,1)")
        super().__init__(
            in_channels=in_channels,
            out_channels=1,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            max_groups=max_groups,
            bottleneck_dropout=bottleneck_dropout,
        )
        # The parent constructs a single Stage-1 rain head. Remove it before
        # registering the two explicit Stage-2 heads so state_dict keys and task
        # semantics cannot be confused.
        del self.output_head
        self.support_prior_probability = support_prior_probability
        self.support_head = nn.Conv3d(self.channels[0], 1, kernel_size=1, bias=True)
        self.reflectivity_head = nn.Conv3d(
            self.channels[0], 1, kernel_size=1, bias=True
        )
        self.reset_stage2_heads()

    def reset_stage2_heads(self) -> None:
        """Initialize stable near-prior support and near-mean dBZ outputs."""

        nn.init.normal_(self.support_head.weight, mean=0.0, std=1e-3)
        support_bias = (
            0.0
            if self.support_prior_probability is None
            else math.log(
                self.support_prior_probability
                / (1.0 - self.support_prior_probability)
            )
        )
        nn.init.constant_(self.support_head.bias, support_bias)
        # Zero standardized dBZ is the train mean at each physical height.
        nn.init.normal_(self.reflectivity_head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.reflectivity_head.bias)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        # Shared decoder features: (B,base,D,H,Z), height unchanged.
        features = self.forward_features(inputs)
        return {
            "support_logits": self.support_head(features),
            "reflectivity": self.reflectivity_head(features),
        }


def stage2_predictions_from_output(
    output: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate and extract ``(support_logits, reflectivity)`` tensors."""

    if not isinstance(output, Mapping):
        raise TypeError("Stage-2 model output must be a mapping")
    support = output.get("support_logits")
    reflectivity = output.get("reflectivity")
    if not isinstance(support, torch.Tensor) or not isinstance(
        reflectivity, torch.Tensor
    ):
        raise TypeError(
            "Stage-2 output must contain Tensor values 'support_logits' and "
            "'reflectivity'"
        )
    if support.shape != reflectivity.shape or support.ndim != 5:
        raise ValueError(
            "Stage-2 support/reflectivity outputs must share shape (B,1,D,H,Z)"
        )
    if support.shape[1] != 1:
        raise ValueError("Stage-2 output tensors must have one channel")
    return support, reflectivity
