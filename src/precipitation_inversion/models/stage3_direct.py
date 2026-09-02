"""Deployable direct GR-to-rain multi-head model for Stage 3 D0.

The network keeps the Stage-2 height-preserving 3-D U-Net and its physical
support/dBZ heads, then adds a rain-intensity head on the shared final decoder
features.  Only GR-derived channels enter the model; DPR support, DPR dBZ and
rain are label-side supervision and are never concatenated to ``inputs``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from .stage2_unet3d import Stage2UNet3D


STAGE3_D0_RAIN_HEAD_ONLY = "rain_head_only"
STAGE3_D0_DECODER_AND_HEADS = "decoder_and_all_heads"
STAGE3_D0_TRAINABLE_SCOPES = frozenset(
    (STAGE3_D0_RAIN_HEAD_ONLY, STAGE3_D0_DECODER_AND_HEADS)
)


class Stage3DirectMultiHeadUNet3D(Stage2UNet3D):
    """Predict DPR support, DPR dBZ and rain from one shared GR backbone.

    Input and every dense output preserve the same physical grid::

        inputs:       (B,C,D,H,Z), normally (B,4,64,64,60)
        shared:       (B,F,D,H,Z)
        support:      (B,1,D,H,Z) unbounded logits
        reflectivity: (B,1,D,H,Z) standardized DPR dBZ
        rain:         (B,1,D,H,Z) predicted log1p(mm/h)

    The rain output is deliberately not multiplied by support inside the
    model.  Positive-rain intensity and occurrence retain independent losses;
    validation can report both the true-support diagnostic and the deployable
    validation-thresholded support route without introducing a hard gate into
    backpropagation.
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
        trainable_scope: str = STAGE3_D0_RAIN_HEAD_ONLY,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            max_groups=max_groups,
            bottleneck_dropout=bottleneck_dropout,
            support_prior_probability=support_prior_probability,
        )
        scope = str(trainable_scope).strip().lower()
        if scope not in STAGE3_D0_TRAINABLE_SCOPES:
            raise ValueError(
                "D0 trainable_scope must be one of "
                + ", ".join(sorted(STAGE3_D0_TRAINABLE_SCOPES))
            )
        self.trainable_scope = scope
        self.rain_head = nn.Conv3d(self.channels[0], 1, kernel_size=1, bias=True)
        # A small non-zero kernel lets a later decoder-unfreeze experiment
        # receive rain gradients on its first update. Bias zero represents
        # neutral log1p rain=0 before the new head learns anything.
        nn.init.normal_(self.rain_head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.rain_head.bias)
        self.configure_trainable_scope(scope)

    def configure_trainable_scope(self, scope: str) -> None:
        """Apply one preregistered D0 trainable-parameter contract."""

        normalized = str(scope).strip().lower()
        if normalized not in STAGE3_D0_TRAINABLE_SCOPES:
            raise ValueError("unsupported D0 trainable scope")
        self.trainable_scope = normalized
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in self.rain_head.parameters():
            parameter.requires_grad_(True)
        if normalized == STAGE3_D0_DECODER_AND_HEADS:
            # Multi-scale spatial recovery lives in the complete decoder, not
            # only in its final 1x1 heads. The encoder/bottleneck remain sealed
            # to preserve the W1.25 GR representation for this controlled step.
            for block in self.decoder:
                for parameter in block.parameters():
                    parameter.requires_grad_(True)
            for head in (self.support_head, self.reflectivity_head):
                for parameter in head.parameters():
                    parameter.requires_grad_(True)

    def train(self, mode: bool = True) -> "Stage3DirectMultiHeadUNet3D":
        """Keep sealed modules in eval mode while trainable D0 modules train."""

        super().train(mode)
        # Encoder and bottleneck are frozen for both registered scopes. This is
        # especially important for bottleneck dropout: a frozen feature probe
        # must see the same deterministic representation as its source model.
        self.stem.eval()
        for block in self.encoder:
            block.eval()
        if self.trainable_scope == STAGE3_D0_RAIN_HEAD_ONLY:
            for block in self.decoder:
                block.eval()
            self.support_head.eval()
            self.reflectivity_head.eval()
        return self

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        # (B,C,D,H,Z) -> shared (B,F,D,H,Z); D/H are down/up-sampled inside
        # the U-Net while all Z=60 height levels remain present at every stage.
        features = self.forward_features(inputs)
        return {
            "support_logits": self.support_head(features),
            "reflectivity": self.reflectivity_head(features),
            "rain": self.rain_head(features),
        }


def stage3_d0_predictions_from_output(
    output: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Validate and return ``(rain, support_logits, reflectivity)``."""

    if not isinstance(output, Mapping):
        raise TypeError("D0 model output must be a mapping")
    values = tuple(output.get(name) for name in ("rain", "support_logits", "reflectivity"))
    if not all(isinstance(value, torch.Tensor) for value in values):
        raise TypeError("D0 output must contain rain, support_logits and reflectivity tensors")
    rain, support, reflectivity = values
    assert isinstance(rain, torch.Tensor)
    assert isinstance(support, torch.Tensor)
    assert isinstance(reflectivity, torch.Tensor)
    if rain.ndim != 5 or rain.shape[1] != 1:
        raise ValueError("D0 rain output must have shape (B,1,D,H,Z)")
    if support.shape != rain.shape or reflectivity.shape != rain.shape:
        raise ValueError("all D0 outputs must share shape (B,1,D,H,Z)")
    return rain, support, reflectivity


def expected_d0_trainable_parameter_names(model: Stage3DirectMultiHeadUNet3D) -> set[str]:
    """Return the exact parameter-name set allowed by the configured scope."""

    expected = {name for name, _ in model.rain_head.named_parameters(prefix="rain_head")}
    if model.trainable_scope == STAGE3_D0_DECODER_AND_HEADS:
        expected.update(
            name for name, _ in model.decoder.named_parameters(prefix="decoder")
        )
        expected.update(
            name for name, _ in model.support_head.named_parameters(prefix="support_head")
        )
        expected.update(
            name
            for name, _ in model.reflectivity_head.named_parameters(
                prefix="reflectivity_head"
            )
        )
    return expected


def assert_d0_trainable_contract(model: nn.Module) -> None:
    """Fail fast if a D0 optimizer could update an unintended parameter."""

    candidate = model.module if hasattr(model, "module") else model
    if not isinstance(candidate, Stage3DirectMultiHeadUNet3D):
        raise TypeError("model must be Stage3DirectMultiHeadUNet3D (possibly DDP-wrapped)")
    actual = {name for name, value in candidate.named_parameters() if value.requires_grad}
    expected = expected_d0_trainable_parameter_names(candidate)
    if actual != expected:
        raise RuntimeError(
            f"D0 trainable scope mismatch; missing={sorted(expected-actual)}, "
            f"unexpected={sorted(actual-expected)}"
        )
    if not actual:
        raise RuntimeError("D0 has no trainable parameters")


def d0_shared_audit_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Return parameters shared by physical and rain tasks for gradient audit.

    Task-specific heads do not receive gradients from the other tasks and must
    not inflate the relative gradient norm. Decoder parameters are the only
    shared trainable representation in ``decoder_and_all_heads``.
    """

    candidate = model.module if hasattr(model, "module") else model
    if not isinstance(candidate, Stage3DirectMultiHeadUNet3D):
        raise TypeError("model must be a D0 multi-head network")
    if candidate.trainable_scope != STAGE3_D0_DECODER_AND_HEADS:
        raise ValueError("gradient-scale audit requires decoder_and_all_heads scope")
    return [value for value in candidate.decoder.parameters() if value.requires_grad]


def build_stage3_d0_model(
    model_config: Mapping[str, Any],
    *,
    trainable_scope: str = STAGE3_D0_RAIN_HEAD_ONLY,
) -> Stage3DirectMultiHeadUNet3D:
    """Build D0 from the same architecture mapping stored by Stage 2."""

    return Stage3DirectMultiHeadUNet3D(
        in_channels=int(model_config["in_channels"]),
        base_channels=int(model_config.get("base_channels", 16)),
        channel_multipliers=tuple(model_config.get("channel_multipliers", (1, 2, 4, 8, 16))),
        max_groups=int(model_config.get("max_groups", 8)),
        bottleneck_dropout=float(model_config.get("bottleneck_dropout", 0.1)),
        support_prior_probability=model_config.get("support_prior_probability", 0.04),
        trainable_scope=trainable_scope,
    )


def load_stage2_source_into_d0(
    model: Stage3DirectMultiHeadUNet3D,
    state_dict: Mapping[str, torch.Tensor],
) -> None:
    """Load W1.25 strictly except for the intentionally new rain head."""

    incompatibility = model.load_state_dict(state_dict, strict=False)
    expected_missing = {"rain_head.weight", "rain_head.bias"}
    if set(incompatibility.missing_keys) != expected_missing:
        raise ValueError(
            "D0 source load has unexpected missing keys: "
            + repr(sorted(incompatibility.missing_keys))
        )
    if incompatibility.unexpected_keys:
        raise ValueError(
            "D0 source load has unexpected keys: "
            + repr(sorted(incompatibility.unexpected_keys))
        )
