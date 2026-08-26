"""Anisotropic 3D building blocks that never resample height.

Tensor convention is ``(B,C,D,H,Z)`` where ``D=nscan``, ``H=nray`` and
``Z=height``. Down/up blocks change only ``D`` and ``H``; the 60 physical
height levels remain aligned from input to output.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def compatible_group_count(channels: int, max_groups: int = 8) -> int:
    """Return the largest GroupNorm group count dividing ``channels``."""

    if channels <= 0 or max_groups <= 0:
        raise ValueError("channels and max_groups must be positive")
    for groups in range(min(channels, max_groups), 0, -1):
        if channels % groups == 0:
            return groups
    raise AssertionError("one always divides a positive channel count")


def _norm(channels: int, max_groups: int) -> nn.GroupNorm:
    return nn.GroupNorm(compatible_group_count(channels, max_groups), channels)


class AnisotropicResidualBlock3D(nn.Module):
    """Fuse horizontal context and vertical profiles without changing shape.

    The first convolution has kernel ``(3,3,1)`` and therefore mixes the two
    horizontal axes only. The second has kernel ``(1,1,5)`` and mixes five
    adjacent physical height levels while preserving their positions.

    Shape: ``(B,Cin,D,H,Z) -> (B,Cout,D,H,Z)``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        max_groups: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0,1)")

        self.horizontal = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(3, 3, 1),
            padding=(1, 1, 0),
            bias=False,
        )
        self.horizontal_norm = _norm(out_channels, max_groups)
        self.vertical = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=(1, 1, 5),
            padding=(0, 0, 2),
            bias=False,
        )
        self.vertical_norm = _norm(out_channels, max_groups)
        self.activation = nn.SiLU(inplace=True)
        self.dropout = nn.Dropout3d(dropout) if dropout > 0.0 else nn.Identity()
        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 5:
            raise ValueError("3D block input must have shape (B,C,D,H,Z)")
        residual = self.residual(inputs)
        features = self.horizontal(inputs)
        features = self.activation(self.horizontal_norm(features))
        features = self.vertical(features)
        features = self.dropout(self.vertical_norm(features))
        return self.activation(features + residual)


class HorizontalDownBlock3D(nn.Module):
    """Halve scan/ray resolution and retain height resolution exactly.

    Shape: ``(B,Cin,D,H,Z) -> (B,Cout,D/2,H/2,Z)`` for even ``D,H``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        max_groups: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.downsample = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(2, 2, 1),
            stride=(2, 2, 1),
            bias=False,
        )
        self.down_norm = _norm(out_channels, max_groups)
        self.activation = nn.SiLU(inplace=True)
        self.block = AnisotropicResidualBlock3D(
            out_channels,
            out_channels,
            max_groups=max_groups,
            dropout=dropout,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 5:
            raise ValueError("down-block input must have shape (B,C,D,H,Z)")
        if inputs.shape[2] < 2 or inputs.shape[3] < 2:
            raise ValueError("scan and ray dimensions must be at least two")
        # Before=(B,Cin,D,H,Z); stride (2,2,1) -> (B,Cout,D/2,H/2,Z).
        features = self.downsample(inputs)
        features = self.activation(self.down_norm(features))
        return self.block(features)


class HorizontalUpBlock3D(nn.Module):
    """Upsample scan/ray, concatenate an encoder skip, and retain height.

    ``inputs`` is the coarser decoder tensor and ``skip`` is the encoder tensor.
    Interpolation targets the exact skip shape, which also handles odd horizontal
    sizes without ever interpolating the height axis to a different length.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        *,
        max_groups: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if min(in_channels, skip_channels, out_channels) <= 0:
            raise ValueError("all channel counts must be positive")
        self.reduce = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.reduce_norm = _norm(out_channels, max_groups)
        self.activation = nn.SiLU(inplace=True)
        self.fuse = AnisotropicResidualBlock3D(
            out_channels + skip_channels,
            out_channels,
            max_groups=max_groups,
            dropout=dropout,
        )

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 5 or skip.ndim != 5:
            raise ValueError("up-block tensors must have shape (B,C,D,H,Z)")
        if inputs.shape[0] != skip.shape[0]:
            raise ValueError("decoder and skip batch sizes differ")
        if inputs.shape[-1] != skip.shape[-1]:
            raise ValueError("height must remain identical across skip connections")

        # Before=(B,Cin,D,H,Z); after=(B,Cin,skip_D,skip_H,Z).
        features = F.interpolate(
            inputs,
            size=skip.shape[2:],
            mode="trilinear",
            align_corners=False,
        )
        features = self.activation(self.reduce_norm(self.reduce(features)))
        # Concatenate channels: (B,Cout+skip_C,skip_D,skip_H,Z).
        features = torch.cat((features, skip), dim=1)
        return self.fuse(features)
