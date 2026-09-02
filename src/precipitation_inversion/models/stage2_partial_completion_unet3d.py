"""Mask-aware Stage-2 completion model for ``S2-R1-P-PartialConv``.

Tensor convention is ``(B,C,D,H,Z)`` where ``D`` is along-track scan, ``H``
is cross-track ray and ``Z`` is the 60-level physical height axis.  Partial
convolution is applied only to the sparse DPR-anchor value branch.  The dense
anchor-distance and height channels use ordinary convolution and are fused
afterwards, so valid geometric context is never erased by the anchor mask.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .blocks3d import AnisotropicResidualBlock3D, compatible_group_count
from .unet3d import Stage1UNet3D


class PartialConv3D(nn.Module):
    """Three-dimensional partial convolution with a shared spatial mask.

    ``inputs`` has shape ``(B,Cin,D,H,Z)`` and ``valid_mask`` has shape
    ``(B,1,D,H,Z)``.  Invalid input values are first set to zero.  The ordinary
    convolution result is then multiplied by ``window_size / valid_count`` and
    is set to zero when a receptive field contains no valid values.  The
    returned propagated mask has shape ``(B,1,Dout,Hout,Zout)``.

    A single mask is intentionally shared by all feature channels: after the
    first value convolution, every derived channel has the same observation
    support.  Dense geometry is not an argument to this module.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: tuple[int, int, int],
        padding: tuple[int, int, int],
        stride: tuple[int, int, int] = (1, 1, 1),
        dilation: tuple[int, int, int] = (1, 1, 1),
        bias: bool = False,
        epsilon: float = 1.0e-8,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("partial-convolution channel counts must be positive")
        if len(kernel_size) != 3 or any(value <= 0 for value in kernel_size):
            raise ValueError("kernel_size must contain three positive values")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        self.convolution = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.kernel_size = tuple(int(value) for value in kernel_size)
        self.stride = tuple(int(value) for value in stride)
        self.padding = tuple(int(value) for value in padding)
        self.dilation = tuple(int(value) for value in dilation)
        self.epsilon = float(epsilon)
        window_size = int(self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2])
        self.window_size = window_size
        self.register_buffer(
            "mask_kernel",
            torch.ones((1, 1, *self.kernel_size), dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self, inputs: torch.Tensor, valid_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.ndim != 5 or valid_mask.ndim != 5:
            raise ValueError("partial convolution expects (B,C,D,H,Z) tensors")
        if valid_mask.shape[1] != 1:
            raise ValueError("partial-convolution mask must have one channel")
        if inputs.shape[0] != valid_mask.shape[0] or inputs.shape[2:] != valid_mask.shape[2:]:
            raise ValueError("partial-convolution input and mask geometry differ")
        if inputs.shape[1] != self.convolution.in_channels:
            raise ValueError(
                f"expected {self.convolution.in_channels} value channels, "
                f"got {inputs.shape[1]}"
            )
        if not torch.is_floating_point(inputs):
            raise TypeError("partial-convolution inputs must be floating-point")

        # Dataset masks contain exact 0/1 floats. Thresholding also makes this
        # layer robust to bool masks while preventing fractional validity from
        # changing the physical meaning of valid-neighbour counts.
        binary_mask = (valid_mask > 0.5).to(dtype=inputs.dtype)
        masked_inputs = inputs * binary_mask
        raw = self.convolution(masked_inputs)
        with torch.no_grad():
            mask_sum = F.conv3d(
                binary_mask,
                self.mask_kernel.to(dtype=binary_mask.dtype),
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
            )
            updated_mask = mask_sum > 0.0
            # ``mask_sum`` is an integer-valued count: every valid window is
            # at least 1.  Clamp to 1 instead of a tiny epsilon so FP16 cannot
            # underflow the denominator to zero; explicitly selecting zero for
            # empty windows also avoids the undefined ``0 * inf`` operation.
            safe_count = mask_sum.clamp_min(1.0)
            ratio = torch.where(
                updated_mask,
                self.window_size / safe_count,
                torch.zeros_like(mask_sum),
            )
        if self.convolution.bias is None:
            output = raw * ratio
        else:
            bias = self.convolution.bias.reshape(1, -1, 1, 1, 1)
            output = (raw - bias) * ratio + bias
            output = output * updated_mask.to(dtype=output.dtype)
        return output, updated_mask


class SparseValuePartialStem3D(nn.Module):
    """Anisotropic partial-convolution stem for sparse standardized dBZ.

    Shape changes::

        value: (B,1,D,H,Z), anchor mask: (B,1,D,H,Z)
          -> horizontal PConv (3,3,1): (B,C,D,H,Z)
          -> vertical   PConv (1,1,5): (B,C,D,H,Z)

    Neither convolution downsamples or pads the physical height count.
    """

    def __init__(self, channels: int, *, max_groups: int = 8) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("sparse stem channels must be positive")
        self.horizontal = PartialConv3D(
            1,
            channels,
            kernel_size=(3, 3, 1),
            padding=(1, 1, 0),
            bias=False,
        )
        self.horizontal_norm = nn.GroupNorm(
            compatible_group_count(channels, max_groups), channels
        )
        self.vertical = PartialConv3D(
            channels,
            channels,
            kernel_size=(1, 1, 5),
            padding=(0, 0, 2),
            bias=False,
        )
        self.vertical_norm = nn.GroupNorm(
            compatible_group_count(channels, max_groups), channels
        )
        self.residual = nn.Conv3d(1, channels, kernel_size=1, bias=False)
        self.activation = nn.SiLU(inplace=True)

    def forward(
        self, values: torch.Tensor, anchor_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        horizontal, propagated = self.horizontal(values, anchor_mask)
        horizontal = self.activation(self.horizontal_norm(horizontal))
        # GroupNorm aggregates over spatial locations and could otherwise turn
        # exact zeros outside the propagated support into a non-zero offset.
        horizontal = horizontal * propagated.to(dtype=horizontal.dtype)
        vertical, propagated = self.vertical(horizontal, propagated)
        vertical = self.vertical_norm(vertical)
        vertical = vertical * propagated.to(dtype=vertical.dtype)
        residual = self.residual(values * (anchor_mask > 0.5).to(values.dtype))
        result = self.activation(vertical + residual)
        result = result * propagated.to(dtype=result.dtype)
        return result, propagated


class Stage2PartialCompletionUNet3D(Stage1UNet3D):
    """R1-P single-head U-Net with separate sparse and dense input stems.

    Formal input order is fixed and is validated by the training configuration:

    ``[sparse_anchor_dBZ, anchor_mask, nearest_anchor_distance, height]``.

    Only the first two channels enter :class:`PartialConv3D`.  Distance and
    height have valid values throughout the padded patch and pass through an
    ordinary anisotropic residual block.  The two ``(B,base,D,H,Z)`` feature
    volumes are concatenated and fused back to ``base`` channels before the
    unchanged R1-O encoder/decoder.
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
        if in_channels != 4:
            raise ValueError("R1-P PartialConv requires exactly four ordered channels")
        super().__init__(
            in_channels=in_channels,
            out_channels=1,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            max_groups=max_groups,
            bottleneck_dropout=bottleneck_dropout,
        )
        del self.stem
        del self.output_head
        self.sparse_stem = SparseValuePartialStem3D(
            self.channels[0], max_groups=max_groups
        )
        self.geometry_stem = AnisotropicResidualBlock3D(
            2, self.channels[0], max_groups=max_groups
        )
        self.fusion = nn.Sequential(
            nn.Conv3d(self.channels[0] * 2, self.channels[0], kernel_size=1, bias=False),
            nn.GroupNorm(
                compatible_group_count(self.channels[0], max_groups), self.channels[0]
            ),
            nn.SiLU(inplace=True),
        )
        self.reflectivity_head = nn.Conv3d(
            self.channels[0], 1, kernel_size=1, bias=True
        )
        self._reset_new_modules()

    def _reset_new_modules(self) -> None:
        for root in (self.sparse_stem, self.geometry_stem, self.fusion):
            for module in root.modules():
                if isinstance(module, nn.Conv3d):
                    nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.GroupNorm):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.reflectivity_head.weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(self.reflectivity_head.bias)

    def forward_input_stem(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return fused base features and the propagated sparse-value mask."""

        if inputs.ndim != 5:
            raise ValueError("R1-P inputs must have shape (B,4,D,H,Z)")
        if inputs.shape[1] != self.in_channels:
            raise ValueError(f"R1-P expected four channels, got {inputs.shape[1]}")
        # (B,4,D,H,Z) -> sparse value/mask (B,1,D,H,Z) each.
        sparse_features, propagated_mask = self.sparse_stem(
            inputs[:, 0:1], inputs[:, 1:2]
        )
        # Dense distance/height are never multiplied by the sparse anchor mask.
        geometry_features = self.geometry_stem(inputs[:, 2:4])
        # (B,C,D,H,Z) + (B,C,D,H,Z) -> concat (B,2C,D,H,Z)
        # -> fused (B,C,D,H,Z).
        fused = self.fusion(torch.cat((sparse_features, geometry_features), dim=1))
        return fused, propagated_mask

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        factor = self.horizontal_downsample_factor
        if inputs.ndim != 5 or inputs.shape[1] != self.in_channels:
            raise ValueError("R1-P inputs must have shape (B,4,D,H,Z)")
        if inputs.shape[2] % factor or inputs.shape[3] % factor:
            raise ValueError(
                f"scan/ray must be divisible by {factor}, got {tuple(inputs.shape[2:4])}"
            )
        features, _propagated_mask = self.forward_input_stem(inputs)
        skips = [features]
        for down in self.encoder:
            features = down(features)
            skips.append(features)
        for up, skip in zip(self.decoder, reversed(skips[:-1])):
            features = up(features, skip)
        return features

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        # (B,4,D,H,Z) -> (B,base,D,H,Z) -> standardized dBZ (B,1,D,H,Z).
        return {"reflectivity": self.reflectivity_head(self.forward_features(inputs))}


__all__ = [
    "PartialConv3D",
    "SparseValuePartialStem3D",
    "Stage2PartialCompletionUNet3D",
]
