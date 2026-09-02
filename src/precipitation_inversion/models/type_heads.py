"""Profile-level precipitation-type heads for stage-one decoder features."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn


class _SeparableMorphologyBranch(nn.Sequential):
    """Depthwise spatial/vertical filtering followed by channel mixing."""

    def __init__(self, channels: int, kernel_size: tuple[int, int, int]) -> None:
        padding = tuple(value // 2 for value in kernel_size)
        super().__init__(
            nn.Conv3d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=channels,
                bias=False,
            ),
            nn.Conv3d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(min(4, channels), channels),
            nn.SiLU(inplace=True),
        )


class GlobalPoolTypeHead(nn.Module):
    """Cheap mean/max-over-height control that intentionally loses z order."""

    def __init__(self, in_channels: int, *, num_classes: int = 3) -> None:
        super().__init__()
        hidden = max(8, in_channels)
        self.classifier = nn.Sequential(
            nn.Conv2d(2 * in_channels, hidden, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, num_classes, kernel_size=1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 5:
            raise ValueError("features must have shape (B,C,nscan,nray,z)")
        # (B,C,D,H,Z) -> mean/max (B,C,D,H) -> concat (B,2C,D,H).
        pooled = torch.cat(
            (features.mean(dim=-1), features.amax(dim=-1)), dim=1
        )
        return self.classifier(pooled)


class Ordered3DMorphologyHead(nn.Module):
    """Learn horizontal/vertical morphology while preserving height order.

    For the production feature shape ``(B,16,64,64,60)`` the default path is
    ``16 -> 8`` channels, four factorized morphology branches, learned height
    reduction ``60 -> 30 -> 15``, then an ordered reshape
    ``(B,8,64,64,15) -> (B,120,64,64)`` before 2-D classification.
    """

    def __init__(
        self,
        in_channels: int,
        *,
        compressed_channels: int = 8,
        height_levels: int = 60,
        num_classes: int = 3,
        hidden_2d_channels: int = 32,
    ) -> None:
        super().__init__()
        if height_levels <= 0 or height_levels % 4:
            raise ValueError("height_levels must be positive and divisible by 4")
        if compressed_channels <= 0 or hidden_2d_channels <= 0:
            raise ValueError("head channel counts must be positive")
        self.height_levels = int(height_levels)
        self.compressed_channels = int(compressed_channels)
        self.project = nn.Sequential(
            nn.Conv3d(in_channels, compressed_channels, kernel_size=1, bias=False),
            nn.GroupNorm(min(4, compressed_channels), compressed_channels),
            nn.SiLU(inplace=True),
        )
        # Horizontal texture, vertical profile, and two directional x/z terms.
        self.branches = nn.ModuleList(
            _SeparableMorphologyBranch(compressed_channels, kernel)
            for kernel in ((3, 3, 1), (1, 1, 5), (3, 1, 5), (1, 3, 5))
        )
        self.fuse = nn.Sequential(
            nn.Conv3d(4 * compressed_channels, compressed_channels, 1, bias=False),
            nn.GroupNorm(min(4, compressed_channels), compressed_channels),
            nn.SiLU(inplace=True),
        )
        self.height_compressor = nn.Sequential(
            nn.Conv3d(
                compressed_channels,
                compressed_channels,
                kernel_size=(1, 1, 2),
                stride=(1, 1, 2),
                bias=False,
            ),
            nn.GroupNorm(min(4, compressed_channels), compressed_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(
                compressed_channels,
                compressed_channels,
                kernel_size=(1, 1, 2),
                stride=(1, 1, 2),
                bias=False,
            ),
            nn.GroupNorm(min(4, compressed_channels), compressed_channels),
            nn.SiLU(inplace=True),
        )
        ordered_channels = compressed_channels * (height_levels // 4)
        self.classifier = nn.Sequential(
            nn.Conv2d(ordered_channels, hidden_2d_channels, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_2d_channels, num_classes, 1),
        )

    def forward(
        self, features: torch.Tensor, *, height_permutation: torch.Tensor | None = None
    ) -> torch.Tensor:
        if features.ndim != 5:
            raise ValueError("features must have shape (B,C,nscan,nray,z)")
        if features.shape[-1] != self.height_levels:
            raise ValueError(
                f"expected {self.height_levels} height levels, got {features.shape[-1]}"
            )
        if height_permutation is not None:
            if height_permutation.ndim != 1 or height_permutation.numel() != self.height_levels:
                raise ValueError("height_permutation must contain every z index")
            features = features.index_select(-1, height_permutation.to(features.device))
        reduced = self.project(features)
        morphology = self.fuse(torch.cat([branch(reduced) for branch in self.branches], dim=1))
        reduced = reduced + morphology
        # (B,8,D,H,60) -> (B,8,D,H,15), preserving the ordered z-axis.
        reduced = self.height_compressor(reduced)
        batch, channels, nscan, nray, height = reduced.shape
        # Move z beside C, then fold them: (B,C,D,H,Z) -> (B,C,Z,D,H)
        # -> (B,C*Z,D,H). No averaging/max operation removes height order.
        ordered = reduced.permute(0, 1, 4, 2, 3).reshape(
            batch, channels * height, nscan, nray
        )
        return self.classifier(ordered)


def build_type_head(
    kind: str, in_channels: int, config: Mapping[str, Any] | None = None
) -> nn.Module:
    values = dict(config or {})
    normalized = kind.lower().replace("-", "_")
    if normalized in {"pool", "global_pool"}:
        return GlobalPoolTypeHead(
            in_channels, num_classes=int(values.get("num_classes", 3))
        )
    if normalized in {"ordered_3d", "ordered3d"}:
        return Ordered3DMorphologyHead(
            in_channels,
            compressed_channels=int(values.get("compressed_channels", 8)),
            height_levels=int(values.get("height_levels", 60)),
            num_classes=int(values.get("num_classes", 3)),
            hidden_2d_channels=int(values.get("hidden_2d_channels", 32)),
        )
    raise ValueError(f"unsupported precipitation type head: {kind!r}")

