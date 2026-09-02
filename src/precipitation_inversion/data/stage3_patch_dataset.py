"""Aligned patch data for Stage-3 cascade adaptation experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset as TorchDataset
except (ImportError, OSError):  # pragma: no cover
    torch = None

    class TorchDataset:  # type: ignore[no-redef]
        pass

from .patch_dataset import Stage1PatchDataset
from .stage2_patch_dataset import Stage2PatchDataset


STAGE3_C1_ORACLE_EXTRA_CHANNELS = (
    "true_dpr_support_oracle",
    "stage1_height_scaled_copy",
)


class Stage3C1PatchDataset(TorchDataset):
    """Pair identical Stage-1/Stage-2 patches without DPR-dBZ leakage.

    The returned ``inputs`` tensor is *packed* for
    :class:`~precipitation_inversion.models.stage3_cascade.Stage3C1OracleCascade`:

    ``(C2,D,H,Z) + (1,D,H,Z) + (1,D,H,Z) -> (C2+2,D,H,Z)``.

    Its first ``C2`` channels are Stage-2 GR-only inputs. The last two are the
    true DPR reflectivity support (C1 oracle diagnostic) and physical-height
    coordinate copied from Stage 1.  Stage-1's true DPR reflectivity value
    channel is explicitly discarded and is never returned as a model feature.
    Rain targets, loss masks and physical diagnostics retain the exact Stage-1
    implementation and therefore remain independent of Stage-2 predictions.

    ``positive_only=True`` follows the Stage-1 logical subset. Its stored raw
    ``patch_index`` is used to address the matching, unfiltered Stage-2 index.
    """

    def __init__(
        self,
        *,
        stage1_index_metadata: str | Path,
        stage1_normalization_stats: str | Path,
        stage2_index_metadata: str | Path,
        stage2_normalization_stats: str | Path,
        stage2_input_channels: Sequence[str],
        positive_only: bool,
        cache_size: int = 1,
        verify_hashes: bool = True,
        stage1_options: Mapping[str, Any] | None = None,
        stage2_options: Mapping[str, Any] | None = None,
    ) -> None:
        if torch is None:
            raise ModuleNotFoundError("PyTorch is required for Stage3C1PatchDataset")
        options = dict(stage1_options or {})
        stage2_values = dict(stage2_options or {})
        self.stage1_dataset = Stage1PatchDataset(
            stage1_index_metadata,
            stage1_normalization_stats,
            positive_only=positive_only,
            cache_size=cache_size,
            verify_hashes=verify_hashes,
            **options,
        )
        self.stage2_dataset = Stage2PatchDataset(
            stage2_index_metadata,
            stage2_normalization_stats,
            input_channels=stage2_input_channels,
            cache_size=cache_size,
            verify_hashes=verify_hashes,
            **stage2_values,
        )
        self.stage2_feature_names = tuple(self.stage2_dataset.feature_names)
        self.feature_names = list(
            self.stage2_feature_names + STAGE3_C1_ORACLE_EXTRA_CHANNELS
        )
        self.files = self.stage1_dataset.files
        self.source_files = self.stage1_dataset.source_files
        self.records = self.stage1_dataset.records
        self.record_indices = self.stage1_dataset.record_indices
        self.z = self.stage1_dataset.z
        self.core_size = self.stage1_dataset.core_size
        self.halo_size = self.stage1_dataset.halo_size
        self.input_size = self.stage1_dataset.input_size
        self.nray = self.stage1_dataset.nray
        self.padded_shape = self.stage1_dataset.padded_shape
        self._validate_alignment()

    def _validate_alignment(self) -> None:
        stage1 = self.stage1_dataset
        stage2 = self.stage2_dataset
        geometry_names = ("core_size", "halo_size", "nray", "padded_shape")
        for name in geometry_names:
            if getattr(stage1, name) != getattr(stage2, name):
                raise ValueError(f"Stage-1/Stage-2 patch geometry differs at {name}")
        if stage1.z.shape != stage2.z.shape or not np.allclose(
            stage1.z, stage2.z, rtol=0.0, atol=1e-6
        ):
            raise ValueError("Stage-1/Stage-2 height coordinates differ")
        if len(stage1.records) != len(stage2.records):
            raise ValueError("Stage-1/Stage-2 raw patch counts differ")
        for field in ("file_id", "core_start", "core_length"):
            if not np.array_equal(stage1.records[field], stage2.records[field]):
                raise ValueError(f"Stage-1/Stage-2 patch indices differ at {field}")
        if len(stage1.source_files) != len(stage2.files):
            raise ValueError("Stage-1/Stage-2 source-file counts differ")
        for file_id, (left, right) in enumerate(zip(stage1.source_files, stage2.files)):
            if Path(left["file_path"]).resolve() != Path(right["file_path"]).resolve():
                raise ValueError(f"Stage-1/Stage-2 source path differs at file {file_id}")
            if int(left["nscan"]) != int(right["nscan"]):
                raise ValueError(f"Stage-1/Stage-2 nscan differs at file {file_id}")

    def __len__(self) -> int:
        return len(self.stage1_dataset)

    def file_index_range(self, file_id: int) -> range:
        return self.stage1_dataset.file_index_range(file_id)

    def clear_cache(self) -> None:
        self.stage1_dataset.clear_cache()
        self.stage2_dataset.clear_cache()

    def _aligned_items(self, index: int) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return matching Stage-1/Stage-2 items addressed by one raw index."""

        stage1_item = self.stage1_dataset[index]
        raw_index = int(stage1_item["patch_index"])
        stage2_item = self.stage2_dataset[raw_index]
        for name in ("file_id", "core_start", "core_length"):
            if int(stage1_item[name]) != int(stage2_item[name]):
                raise RuntimeError(f"Stage-1/Stage-2 item alignment failed at {name}")
        return stage1_item, stage2_item

    @staticmethod
    def _pack_items(
        stage1_item: dict[str, Any], stage2_item: dict[str, Any]
    ) -> dict[str, Any]:
        """Pack GR inputs plus oracle support/height without true-dBZ leakage."""

        stage1_inputs = stage1_item.pop("inputs")
        stage2_inputs = stage2_item["inputs"]
        if stage1_inputs.shape[0] != 3:
            raise ValueError("C1-O supports only baseline three-channel Stage-1 data")
        if stage1_inputs.shape[1:] != stage2_inputs.shape[1:]:
            raise ValueError("Stage-1/Stage-2 item tensor geometry differs")

        # Before: Stage2=(C2,D,H,Z), Stage1=(3,D,H,Z).  Drop Stage1 channel 0
        # (true DPR dBZ), retain only channel 1 support and channel 2 height.
        # After: packed=(C2+2,D,H,Z), e.g. (6,64,64,60) for W1.25.
        packed = torch.cat((stage2_inputs, stage1_inputs[1:3]), dim=0)
        result = dict(stage1_item)
        result["inputs"] = packed.contiguous()
        result["true_dpr_support"] = stage1_inputs[1:2].to(dtype=torch.bool)
        result["stage2_patch_index"] = stage2_item["patch_index"]
        return result

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._pack_items(*self._aligned_items(index))

    def __getstate__(self) -> dict[str, Any]:
        return dict(self.__dict__)


class Stage3C2PatchDataset(Stage3C1PatchDataset):
    """Return aligned final-rain and Stage-2 physical supervision for C2-O.

    C2 must iterate the complete Stage-2 patch population, including
    background patches, because support reconstruction is one of its physical
    anchors. Therefore ``positive_only`` is fixed to ``False`` and
    ``records`` exposes the Stage-2 structured index required by the original
    four-stratum sampler.

    Model input remains packed ``(C2+2,D,H,Z)``. Additional label-side tensors
    all have shape ``(1,D,H,Z)`` and never enter either network:

    - ``stage2_target_support`` and ``stage2_support_loss_mask`` supervise
      occurrence over the trustworthy occupancy domain;
    - ``stage2_target_dbz`` and ``stage2_regression_mask`` supervise dBZ only
      where a physical DPR reflectivity exists;
    - ``stage2_regression_weights`` carries W1.25 target-intensity weights.
    """

    def __init__(self, **kwargs: Any) -> None:
        if kwargs.pop("positive_only", False) is not False:
            raise ValueError("C2-O requires the complete, unfiltered patch index")
        super().__init__(positive_only=False, **kwargs)
        # Stage2StratifiedBatchSampler reads these structured count fields.
        self.records = self.stage2_dataset.records
        self.files = self.stage2_dataset.files
        self.record_indices = None

    def __getitem__(self, index: int) -> dict[str, Any]:
        stage1_item, stage2_item = self._aligned_items(index)
        result = self._pack_items(stage1_item, stage2_item)
        self._attach_stage2_supervision(result, stage2_item)
        return result

    @staticmethod
    def _attach_stage2_supervision(
        result: dict[str, Any], stage2_item: Mapping[str, Any]
    ) -> None:
        """Attach label-side support/dBZ fields shared by C2 and D0."""

        field_map = {
            "target_support": "stage2_target_support",
            "target_dbz": "stage2_target_dbz",
            "support_loss_mask": "stage2_support_loss_mask",
            "regression_mask": "stage2_regression_mask",
        }
        for source, destination in field_map.items():
            result[destination] = stage2_item[source]
        weights = stage2_item.get("regression_weights")
        if weights is not None:
            result["stage2_regression_weights"] = weights


class Stage3D0PatchDataset(Stage3C2PatchDataset):
    """Return deployable GR inputs with aligned rain/support/dBZ supervision.

    Unlike C1/C2 oracle datasets, model ``inputs`` contain exactly the Stage-2
    GR feature subset and have shape ``(C2,D,H,Z)``—normally
    ``(4,64,64,60)``. The aligned Stage-1 item supplies log-rain targets and
    reliable masks, while the Stage-2 item supplies physical auxiliary labels.
    None of those satellite-derived tensors are stacked into the input.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.feature_names = list(self.stage2_feature_names)

    def __getitem__(self, index: int) -> dict[str, Any]:
        stage1_item, stage2_item = self._aligned_items(index)
        stage1_inputs = stage1_item.pop("inputs")
        stage2_inputs = stage2_item["inputs"]
        if stage1_inputs.shape[1:] != stage2_inputs.shape[1:]:
            raise ValueError("D0 Stage-1/Stage-2 tensor geometry differs")
        # Before: Stage1=(3,D,H,Z), Stage2=(C2,D,H,Z). After: D0 input remains
        # only Stage2=(C2,D,H,Z); true DPR dBZ/support channels are discarded.
        result = dict(stage1_item)
        result["inputs"] = stage2_inputs.contiguous()
        result["stage2_patch_index"] = stage2_item["patch_index"]
        self._attach_stage2_supervision(result, stage2_item)
        return result
