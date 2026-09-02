"""Height-preserving anisotropic 3D U-Net for DPR precipitation inversion."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from .blocks3d import (
    AnisotropicResidualBlock3D,
    HorizontalDownBlock3D,
    HorizontalUpBlock3D,
)


class Stage1UNet3D(nn.Module):
    """Map DPR reflectivity features to a dense log-rain-rate volume.

    Input convention is ``(B,C,nscan,nray,z)``. The default three channels are
    standardized DPR reflectivity, its validity mask, and scaled height. Four
    encoder stages downsample only ``nscan/nray`` using stride ``(2,2,1)``;
    therefore input and output retain the same number of physical height levels.

    The output head is linear. Training targets are ``log1p(pre_dpr)`` and the
    caller must apply ``loss_mask`` rather than treating zero-filled locations
    as observed zero rain.
    """

    horizontal_downsample_factor = 16

    def __init__(
        self,
        *,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 16,
        channel_multipliers: Sequence[int] = (1, 2, 4, 8, 16),
        max_groups: int = 8,
        bottleneck_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0 or base_channels <= 0:
            raise ValueError("input/output/base channel counts must be positive")
        if len(channel_multipliers) < 2:
            raise ValueError("at least two channel levels are required")
        if any(value <= 0 for value in channel_multipliers):
            raise ValueError("channel multipliers must be positive")
        if not 0.0 <= bottleneck_dropout < 1.0:
            raise ValueError("bottleneck_dropout must lie in [0,1)")

        channels = tuple(base_channels * value for value in channel_multipliers)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.channels = channels
        self.horizontal_downsample_factor = 2 ** (len(channels) - 1)

        self.stem = AnisotropicResidualBlock3D(
            in_channels, channels[0], max_groups=max_groups
        )
        self.encoder = nn.ModuleList(
            HorizontalDownBlock3D(
                channels[level - 1],
                channels[level],
                max_groups=max_groups,
                dropout=(bottleneck_dropout if level == len(channels) - 1 else 0.0),
            )
            for level in range(1, len(channels))
        )
        self.decoder = nn.ModuleList(
            HorizontalUpBlock3D(
                channels[level],
                channels[level - 1],
                channels[level - 1],
                max_groups=max_groups,
            )
            for level in range(len(channels) - 1, 0, -1)
        )
        self.output_head = nn.Conv3d(
            channels[0], out_channels, kernel_size=1, bias=True
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Use Kaiming initialization while keeping GroupNorm initially neutral."""

        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GroupNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        # A full Kaiming head can emit very large untrained log-rain values;
        # expm1 then produces meaningless physical metrics and may overflow AMP.
        # Small weights start predictions near 0 mm/h while retaining gradients.
        nn.init.normal_(self.output_head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.output_head.bias)

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the last decoder feature volume ``(B,base,D,H,Z)``.

        Exposing this tensor lets diagnostic probes and auxiliary heads reuse
        exactly the same backbone without changing the historical rain head.
        """

        if inputs.ndim != 5:
            raise ValueError("inputs must have shape (B,C,nscan,nray,z)")
        if inputs.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} input channels, got {inputs.shape[1]}"
            )
        factor = self.horizontal_downsample_factor
        if inputs.shape[2] % factor or inputs.shape[3] % factor:
            raise ValueError(
                f"nscan and nray must be divisible by {factor}, got "
                f"{tuple(inputs.shape[2:4])}"
            )

        # Encoder shapes for defaults:
        # (B,3,64,64,60) -> (B,16,64,64,60) -> ... -> (B,256,4,4,60).
        features = self.stem(inputs)
        skips = [features]
        for down in self.encoder:
            features = down(features)
            skips.append(features)

        # The deepest feature is already in ``features``. Each decoder stage
        # consumes the next shallower skip and doubles scan/ray only.
        for up, skip in zip(self.decoder, reversed(skips[:-1])):
            features = up(features, skip)

        return features

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(inputs)
        # Linear head: (B,base,64,64,60) -> log-rain (B,1,64,64,60).
        return self.output_head(features)
