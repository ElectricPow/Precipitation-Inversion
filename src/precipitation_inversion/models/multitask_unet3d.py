"""Stage-one U-Net with a label-only precipitation-type auxiliary branch."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from .type_heads import build_type_head
from .unet3d import Stage1UNet3D


class Stage1MultiTaskUNet3D(Stage1UNet3D):
    """Return rain and type predictions from one shared decoder feature map."""

    def __init__(
        self,
        *,
        type_head_kind: str = "ordered_3d",
        type_head_config: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.type_head_kind = type_head_kind
        self.type_head = build_type_head(
            type_head_kind, self.channels[0], type_head_config
        )

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        # Shared decoder features=(B,base,D,H,Z). typePrecip is not an input.
        features = self.forward_features(inputs)
        return {
            "rain": self.output_head(features),       # (B,1,D,H,Z)
            "type_logits": self.type_head(features), # (B,3,D,H)
        }


def rain_prediction_from_output(output: Any) -> torch.Tensor:
    """Extract rain tensor while retaining compatibility with old checkpoints."""

    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, Mapping) and isinstance(output.get("rain"), torch.Tensor):
        return output["rain"]
    raise TypeError("model must return a rain Tensor or {'rain': Tensor, ...}")

