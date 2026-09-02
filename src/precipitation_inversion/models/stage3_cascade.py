"""Differentiable Stage-2 -> Stage-1 cascade wrappers.

The first Stage-3 experiment (C1-O) freezes Stage 2 and adapts Stage 1 while
using the *true* DPR reflectivity support.  The true support is deliberately
an oracle diagnostic: it isolates reflectivity-value/domain mismatch from the
separate support-reconstruction error and is not deployable at inference.

The second controlled experiment (C2-O) reverses the trainable side: the
sealed Stage-1 rain model is frozen, while the Stage-2 reflectivity head and
highest-resolution decoder block are optimized through both their original
physical objectives and the final rain objective.  Frozen Stage-1 parameters
still participate in autograd so rain gradients can reach Stage 2.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from .stage2_unet3d import Stage2UNet3D, stage2_predictions_from_output
from .unet3d import Stage1UNet3D


STAGE3_C2_TRAINABLE_SCOPE = "reflectivity_head_and_last_decoder"


class Stage3C1OracleCascade(nn.Module):
    """Freeze Stage 2 and train Stage 1 on predicted dBZ + true support.

    Packed input shape is ``(B,C2+2,D,H,Z)``:

    - ``[:C2]``: deployable Stage-2 GR channels;
    - ``[C2:C2+1]``: true DPR support (oracle, boolean semantics);
    - ``[C2+1:C2+2]``: scaled physical height.

    Stage-2 reflectivity is first converted from its standardized space to
    physical dBZ and then into Stage-1 standardized space.  Outside the true
    DPR support it is filled with standardized zero (the Stage-1 train mean at
    that height), after which the Stage-1 tensor has shape ``(B,3,D,H,Z)``.
    """

    def __init__(
        self,
        stage2_model: Stage2UNet3D,
        stage1_model: Stage1UNet3D,
        *,
        stage2_dbz_mean: Sequence[float],
        stage2_dbz_std: Sequence[float],
        stage1_dbz_mean: Sequence[float],
        stage1_dbz_std: Sequence[float],
        stage2_in_channels: int,
    ) -> None:
        super().__init__()
        if stage2_in_channels <= 0:
            raise ValueError("stage2_in_channels must be positive")
        if int(stage2_model.in_channels) != int(stage2_in_channels):
            raise ValueError("Stage-2 model and packed-input channel counts differ")
        if int(stage1_model.in_channels) != 3:
            raise ValueError("C1-O requires the three-channel Stage-1 interface")

        vectors = [
            torch.as_tensor(value, dtype=torch.float32)
            for value in (
                stage2_dbz_mean,
                stage2_dbz_std,
                stage1_dbz_mean,
                stage1_dbz_std,
            )
        ]
        if any(value.ndim != 1 for value in vectors):
            raise ValueError("normalization mean/std values must be one-dimensional")
        level_count = int(vectors[0].numel())
        if level_count <= 0 or any(value.numel() != level_count for value in vectors):
            raise ValueError("all normalization vectors must have equal nonzero length")
        if any(not bool(torch.isfinite(value).all()) for value in vectors):
            raise ValueError("normalization vectors must be finite at every height")
        if bool((vectors[1] <= 0).any()) or bool((vectors[3] <= 0).any()):
            raise ValueError("normalization standard deviations must be positive")

        self.stage2_model = stage2_model
        self.stage1_model = stage1_model
        self.stage2_in_channels = int(stage2_in_channels)
        # Broadcast (Z,) over prediction tensors (B,1,D,H,Z).
        shape = (1, 1, 1, 1, level_count)
        self.register_buffer("stage2_dbz_mean", vectors[0].reshape(shape))
        self.register_buffer("stage2_dbz_std", vectors[1].reshape(shape))
        self.register_buffer("stage1_dbz_mean", vectors[2].reshape(shape))
        self.register_buffer("stage1_dbz_std", vectors[3].reshape(shape))

        # C1 updates only the Stage-1 shared U-Net and rain head.  The T3D
        # classification head, when present, is retained in checkpoints but is
        # frozen and not evaluated because this controlled experiment has no
        # type loss.
        for parameter in self.stage2_model.parameters():
            parameter.requires_grad_(False)
        type_head = getattr(self.stage1_model, "type_head", None)
        if isinstance(type_head, nn.Module):
            for parameter in type_head.parameters():
                parameter.requires_grad_(False)
        self.stage2_model.eval()
        if isinstance(type_head, nn.Module):
            type_head.eval()

    @property
    def packed_in_channels(self) -> int:
        return self.stage2_in_channels + 2

    def train(self, mode: bool = True) -> "Stage3C1OracleCascade":
        """Keep frozen branches in evaluation mode after recursive ``train``."""

        super().train(mode)
        self.stage2_model.eval()
        type_head = getattr(self.stage1_model, "type_head", None)
        if isinstance(type_head, nn.Module):
            type_head.eval()
        return self

    def build_stage1_inputs(
        self, packed_inputs: torch.Tensor
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        """Run frozen Stage 2 and construct ``(B,3,D,H,Z)`` Stage-1 input."""

        if packed_inputs.ndim != 5:
            raise ValueError("packed Stage-3 inputs must have shape (B,C,D,H,Z)")
        if packed_inputs.shape[1] != self.packed_in_channels:
            raise ValueError(
                f"expected {self.packed_in_channels} packed channels, got "
                f"{packed_inputs.shape[1]}"
            )
        if packed_inputs.shape[-1] != self.stage2_dbz_mean.shape[-1]:
            raise ValueError("packed input height differs from normalization metadata")

        # (B,C2,D,H,Z) -> frozen Stage 2 -> (B,1,D,H,Z).  no_grad is
        # intentional in C1: rain loss must never update Stage 2.
        stage2_inputs = packed_inputs[:, : self.stage2_in_channels]
        with torch.no_grad():
            stage2_output = self.stage2_model(stage2_inputs)
            support_logits, predicted_dbz_s2 = stage2_predictions_from_output(
                stage2_output
            )

        # Oracle support and height each retain one channel and the full 3-D
        # grid: (B,1,D,H,Z).  Values >0.5 are exactly the Dataset's 0/1 mask.
        true_support = packed_inputs[
            :, self.stage2_in_channels : self.stage2_in_channels + 1
        ] > 0.5
        height_scaled = packed_inputs[:, self.stage2_in_channels + 1 :]

        # Standardized S2 dBZ -> physical dBZ -> standardized S1 dBZ. No hard
        # clipping is applied, so the bridge does not erase tail information.
        predicted_dbz_physical = (
            predicted_dbz_s2 * self.stage2_dbz_std + self.stage2_dbz_mean
        )
        predicted_dbz_s1 = (
            predicted_dbz_physical - self.stage1_dbz_mean
        ) / self.stage1_dbz_std
        oracle_dbz = torch.where(
            true_support, predicted_dbz_s1, torch.zeros_like(predicted_dbz_s1)
        )
        stage1_inputs = torch.cat(
            (oracle_dbz, true_support.to(oracle_dbz.dtype), height_scaled), dim=1
        )
        diagnostics = {
            "support_logits": support_logits,
            "predicted_dbz_standardized_stage2": predicted_dbz_s2,
            "predicted_dbz_physical": predicted_dbz_physical,
            "predicted_dbz_standardized_stage1": predicted_dbz_s1,
            "true_dpr_support": true_support,
        }
        return stage1_inputs, diagnostics

    def forward(self, packed_inputs: torch.Tensor) -> torch.Tensor:
        stage1_inputs, _ = self.build_stage1_inputs(packed_inputs)
        # Call the shared decoder and rain head explicitly. This avoids an
        # unused forward through the frozen T3D type head while still updating
        # every trainable Stage-1 trunk/rain parameter.
        features = self.stage1_model.forward_features(stage1_inputs)
        return self.stage1_model.output_head(features)


def assert_c1_freeze_contract(model: nn.Module) -> None:
    """Fail fast if a C1 optimizer could update a forbidden parameter."""

    candidate = model.module if hasattr(model, "module") else model
    if not isinstance(candidate, Stage3C1OracleCascade):
        raise TypeError("model must be Stage3C1OracleCascade (possibly DDP-wrapped)")
    if any(parameter.requires_grad for parameter in candidate.stage2_model.parameters()):
        raise RuntimeError("C1-O requires every Stage-2 parameter to be frozen")
    type_head = getattr(candidate.stage1_model, "type_head", None)
    if isinstance(type_head, nn.Module) and any(
        parameter.requires_grad for parameter in type_head.parameters()
    ):
        raise RuntimeError("C1-O requires the unused Stage-1 type head to be frozen")
    trainable = [
        parameter for parameter in candidate.stage1_model.parameters()
        if parameter.requires_grad
    ]
    if not trainable:
        raise RuntimeError("C1-O has no trainable Stage-1 rain parameters")


class Stage3C2OracleCascade(nn.Module):
    """Freeze Stage 1 and task-adapt a restricted part of Stage 2.

    Packed input and normalization semantics are identical to C1-O.  The
    important gradient difference is that Stage 2 is executed normally and
    the frozen Stage 1 is *not* wrapped in ``no_grad``.  Consequently the final
    rain loss follows this path without updating Stage-1 parameters::

        rain loss -> frozen Stage 1 operations -> normalization bridge
                  -> Stage-2 dBZ -> trainable Stage-2 decoder/head

    The returned mapping keeps all three physical tasks explicit.  Every
    tensor has shape ``(B,1,D,H,Z)``:

    - ``rain``: Stage-1 log-rain prediction;
    - ``support_logits``: Stage-2 DPR-support logits;
    - ``reflectivity``: Stage-2 standardized DPR dBZ.
    """

    def __init__(
        self,
        stage2_model: Stage2UNet3D,
        stage1_model: Stage1UNet3D,
        *,
        stage2_dbz_mean: Sequence[float],
        stage2_dbz_std: Sequence[float],
        stage1_dbz_mean: Sequence[float],
        stage1_dbz_std: Sequence[float],
        stage2_in_channels: int,
        trainable_scope: str = STAGE3_C2_TRAINABLE_SCOPE,
    ) -> None:
        super().__init__()
        if int(stage2_model.in_channels) != int(stage2_in_channels):
            raise ValueError("Stage-2 model and packed-input channel counts differ")
        if int(stage1_model.in_channels) != 3:
            raise ValueError("C2-O requires the three-channel Stage-1 interface")
        if trainable_scope != STAGE3_C2_TRAINABLE_SCOPE:
            raise ValueError(
                "C2-O trainable_scope must be "
                f"{STAGE3_C2_TRAINABLE_SCOPE!r}"
            )
        if not stage2_model.decoder:
            raise ValueError("Stage-2 model must contain at least one decoder block")

        vectors = [
            torch.as_tensor(value, dtype=torch.float32)
            for value in (
                stage2_dbz_mean,
                stage2_dbz_std,
                stage1_dbz_mean,
                stage1_dbz_std,
            )
        ]
        if any(value.ndim != 1 for value in vectors):
            raise ValueError("normalization mean/std values must be one-dimensional")
        level_count = int(vectors[0].numel())
        if level_count <= 0 or any(value.numel() != level_count for value in vectors):
            raise ValueError("all normalization vectors must have equal nonzero length")
        if any(not bool(torch.isfinite(value).all()) for value in vectors):
            raise ValueError("normalization vectors must be finite at every height")
        if bool((vectors[1] <= 0).any()) or bool((vectors[3] <= 0).any()):
            raise ValueError("normalization standard deviations must be positive")

        self.stage2_model = stage2_model
        self.stage1_model = stage1_model
        self.stage2_in_channels = int(stage2_in_channels)
        self.trainable_scope = trainable_scope
        shape = (1, 1, 1, 1, level_count)
        self.register_buffer("stage2_dbz_mean", vectors[0].reshape(shape))
        self.register_buffer("stage2_dbz_std", vectors[1].reshape(shape))
        self.register_buffer("stage1_dbz_mean", vectors[2].reshape(shape))
        self.register_buffer("stage1_dbz_std", vectors[3].reshape(shape))

        # Start from an entirely frozen two-network cascade and open only the
        # pre-registered Stage-2 interface modules. The support head remains
        # frozen, but its BCE gradient still reaches the shared last decoder
        # through the fixed head weights and therefore acts as an anchor.
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in self.stage2_model.decoder[-1].parameters():
            parameter.requires_grad_(True)
        for parameter in self.stage2_model.reflectivity_head.parameters():
            parameter.requires_grad_(True)
        self._set_branch_modes(training=False)

    @property
    def packed_in_channels(self) -> int:
        return self.stage2_in_channels + 2

    def _set_branch_modes(self, *, training: bool) -> None:
        # Keep every frozen dropout layer deterministic. Only trainable Stage-2
        # modules enter train mode; GroupNorm itself has no running statistics.
        self.stage1_model.eval()
        self.stage2_model.eval()
        if training:
            self.stage2_model.decoder[-1].train(True)
            self.stage2_model.reflectivity_head.train(True)

    def train(self, mode: bool = True) -> "Stage3C2OracleCascade":
        super().train(mode)
        self._set_branch_modes(training=bool(mode))
        return self

    def build_stage1_inputs(
        self, packed_inputs: torch.Tensor
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        """Build differentiable ``(B,3,D,H,Z)`` oracle-support Stage-1 input."""

        if packed_inputs.ndim != 5:
            raise ValueError("packed Stage-3 inputs must have shape (B,C,D,H,Z)")
        if packed_inputs.shape[1] != self.packed_in_channels:
            raise ValueError(
                f"expected {self.packed_in_channels} packed channels, got "
                f"{packed_inputs.shape[1]}"
            )
        if packed_inputs.shape[-1] != self.stage2_dbz_mean.shape[-1]:
            raise ValueError("packed input height differs from normalization metadata")

        # (B,C2,D,H,Z) -> trainable Stage 2 -> two (B,1,D,H,Z) outputs.
        stage2_inputs = packed_inputs[:, : self.stage2_in_channels]
        support_logits, predicted_dbz_s2 = stage2_predictions_from_output(
            self.stage2_model(stage2_inputs)
        )
        true_support = packed_inputs[
            :, self.stage2_in_channels : self.stage2_in_channels + 1
        ] > 0.5
        height_scaled = packed_inputs[:, self.stage2_in_channels + 1 :]

        # This affine bridge is intentionally differentiable. No detach,
        # threshold, or clipping is permitted on the reflectivity value path.
        predicted_dbz_physical = (
            predicted_dbz_s2 * self.stage2_dbz_std + self.stage2_dbz_mean
        )
        predicted_dbz_s1 = (
            predicted_dbz_physical - self.stage1_dbz_mean
        ) / self.stage1_dbz_std
        oracle_dbz = torch.where(
            true_support, predicted_dbz_s1, torch.zeros_like(predicted_dbz_s1)
        )
        stage1_inputs = torch.cat(
            (oracle_dbz, true_support.to(oracle_dbz.dtype), height_scaled), dim=1
        )
        return stage1_inputs, {
            "support_logits": support_logits,
            "reflectivity": predicted_dbz_s2,
            "predicted_dbz_physical": predicted_dbz_physical,
            "predicted_dbz_standardized_stage1": predicted_dbz_s1,
            "true_dpr_support": true_support,
        }

    def forward(self, packed_inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        stage1_inputs, diagnostics = self.build_stage1_inputs(packed_inputs)
        # Stage-1 parameters are frozen, but autograd must record these
        # operations because stage1_inputs depends on trainable Stage-2 dBZ.
        features = self.stage1_model.forward_features(stage1_inputs)
        rain = self.stage1_model.output_head(features)
        return {
            "rain": rain,
            "support_logits": diagnostics["support_logits"],
            "reflectivity": diagnostics["reflectivity"],
        }


def stage3_c2_predictions_from_output(
    output: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Validate and return ``rain, support_logits, reflectivity`` tensors."""

    if not isinstance(output, Mapping):
        raise TypeError("C2-O model output must be a mapping")
    rain = output.get("rain")
    support = output.get("support_logits")
    reflectivity = output.get("reflectivity")
    if not all(isinstance(value, torch.Tensor) for value in (rain, support, reflectivity)):
        raise TypeError(
            "C2-O output must contain Tensor values rain, support_logits, reflectivity"
        )
    assert isinstance(rain, torch.Tensor)
    assert isinstance(support, torch.Tensor)
    assert isinstance(reflectivity, torch.Tensor)
    if rain.ndim != 5 or rain.shape[1] != 1:
        raise ValueError("C2-O rain output must have shape (B,1,D,H,Z)")
    if support.shape != rain.shape or reflectivity.shape != rain.shape:
        raise ValueError("all C2-O outputs must share shape (B,1,D,H,Z)")
    return rain, support, reflectivity


def assert_c2_freeze_contract(model: nn.Module) -> None:
    """Require exactly the registered Stage-2 decoder/head scope to train."""

    candidate = model.module if hasattr(model, "module") else model
    if not isinstance(candidate, Stage3C2OracleCascade):
        raise TypeError("model must be Stage3C2OracleCascade (possibly DDP-wrapped)")
    if any(parameter.requires_grad for parameter in candidate.stage1_model.parameters()):
        raise RuntimeError("C2-O requires every Stage-1 parameter to be frozen")
    expected = {
        f"stage2_model.decoder.{len(candidate.stage2_model.decoder) - 1}.{name}"
        for name, _ in candidate.stage2_model.decoder[-1].named_parameters()
    }
    expected.update(
        f"stage2_model.reflectivity_head.{name}"
        for name, _ in candidate.stage2_model.reflectivity_head.named_parameters()
    )
    actual = {
        name for name, parameter in candidate.named_parameters()
        if parameter.requires_grad
    }
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise RuntimeError(
            f"C2-O trainable scope mismatch; missing={missing}, unexpected={unexpected}"
        )
