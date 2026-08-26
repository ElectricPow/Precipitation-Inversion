"""Tests for core-with-halo patch indexing and tensor construction."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from precipitation_inversion.data.patch_dataset import (  # noqa: E402
    CFB_DISTANCE_INPUT_CHANNEL,
    CFB_QUALITY_CLUTTER,
    CFB_QUALITY_RELIABLE,
    CFB_QUALITY_UNKNOWN,
    CFB_QUALITY_WEAK,
    PATCH_INDEX_FORMAT_VERSION,
    PATCH_INPUT_CHANNELS,
    PATCH_INPUT_VARIABLE,
    PATCH_LABEL_VARIABLE,
    Stage1PatchDataset,
    build_stage1_patch_index_records,
    ceil_to_multiple,
    core_starts,
    save_patch_index,
    stage1_patch_dataset_kwargs,
)


HEIGHTS = np.array([0.125, 0.375, 0.625, 0.875, 1.125, 1.375], dtype=np.float32)
RAIN_CELLS = (
    (0, 0, 2, 1.0, 20.0),
    (31, 1, 2, 2.0, 22.0),
    (32, 2, 3, 3.0, 24.0),
    (63, 3, 4, 4.0, 35.0),
    (69, 4, 5, 5.0, 50.0),
)


def create_patch_nc(
    path: Path,
    *,
    nscan: int = 70,
    with_rain: bool = True,
    include_precipitation_type: bool = True,
) -> None:
    """Create source arrays shaped ``(nscan,5,6)`` with known edge rain cells."""

    shape = (nscan, 5, 6)
    with Dataset(path, "w") as dataset:
        dataset.createDimension("nscan", shape[0])
        dataset.createDimension("nray", shape[1])
        dataset.createDimension("z", shape[2])
        z = dataset.createVariable("z", "f4", ("z",), fill_value=np.nan)
        z[:] = HEIGHTS

        dbz_values = np.full(shape, np.nan, dtype=np.float32)
        rain_values = np.zeros(shape, dtype=np.float32)
        if with_rain:
            for scan, ray, level, rain, dbz in RAIN_CELLS:
                if scan < nscan:
                    rain_values[scan, ray, level] = rain
                    dbz_values[scan, ray, level] = dbz
            # Native positive/DPR-valid but below cfb, therefore excluded by QC.
            rain_values[10, 0, 0] = 9.0
            dbz_values[10, 0, 0] = 30.0

        dbz = dataset.createVariable(
            "dbz_dpr", "f4", ("nscan", "nray", "z"), fill_value=np.nan
        )
        dbz[:] = dbz_values
        pre = dataset.createVariable(
            "pre_dpr", "f4", ("nscan", "nray", "z"), fill_value=np.nan
        )
        pre[:] = rain_values
        cfb = dataset.createVariable("cfb", "i2", ("nscan", "nray"))
        cfb[:] = 1
        if include_precipitation_type:
            precipitation_type = dataset.createVariable(
                "typePrecip", "i2", ("nscan", "nray")
            )
            precipitation_type[:] = -1111
            precipitation_type[0, 0] = 1
            precipitation_type[min(31, nscan - 1), 1] = 2
            precipitation_type[min(32, nscan - 1), 2] = 3


def write_normalization(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "scope": "training_split_only",
                "selection_mask": "pre_positive_qc",
                "variables": {
                    "dbz_dpr": {
                        "heights_km": HEIGHTS.tolist(),
                        # Legacy label-selected statistics have no fitted
                        # values in the two lowest absolute height levels.
                        "mean": [None, None, 10.0, 20.0, 30.0, 40.0],
                        "std": [None, None, 2.0, 4.0, 5.0, 10.0],
                        "count": [0, 0, 16, 64, 16, 4],
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def write_variable_valid_normalization(
    path: Path, *, include_height_loss_reference: bool = True
) -> None:
    """Write input-selected stats plus independent reliable-label counts."""

    reliable_counts = [0, 0, 16, 64, 16, 4]
    value = {
        "scope": "training_split_only",
        "statistics_role": "model_input_normalization",
        "selection_mask": "variable_valid",
        "selection_semantics": "Each variable uses its own finite values.",
        "label_qc_applied_to_input_statistics": False,
        "height_loss_weight_reference": (
            {
                "selection_mask": "pre_positive_qc",
                "heights_km": HEIGHTS.tolist(),
                "count": reliable_counts,
                "total_count": sum(reliable_counts),
            }
            if include_height_loss_reference
            else None
        ),
        "variables": {
            "dbz_dpr": {
                "heights_km": HEIGHTS.tolist(),
                "mean": [20.0, 15.0, 10.0, 20.0, 30.0, 40.0],
                "std": [5.0, 2.0, 2.0, 4.0, 5.0, 10.0],
                # Deliberately unlike the reliable-label counts: these are
                # input frequencies and must never drive height loss weights.
                "count": [1000, 900, 800, 700, 600, 500],
            }
        },
    }
    path.write_text(json.dumps(value), encoding="utf-8")


def make_patch_dataset_fixture(root: Path) -> tuple[Stage1PatchDataset, Path, Path]:
    """Create two files, five cores, and return the all-core Dataset."""

    rainy = root / "rainy.nc"
    dry = root / "dry.nc"
    create_patch_nc(rainy, nscan=70, with_rain=True)
    create_patch_nc(
        dry,
        nscan=33,
        with_rain=False,
        include_precipitation_type=False,
    )
    records, files, heights, nray, _ = build_stage1_patch_index_records(
        [rainy, dry], core_size=32
    )
    index_path = root / "train.npy"
    metadata_path = root / "train.json"
    save_patch_index(
        index_path,
        metadata_path,
        records,
        {
            "format_version": PATCH_INDEX_FORMAT_VERSION,
            "input_variable": PATCH_INPUT_VARIABLE,
            "label_variable": PATCH_LABEL_VARIABLE,
            "label_transform": "log1p",
            "input_channels": list(PATCH_INPUT_CHANNELS),
            "core_size": 32,
            "halo_size": 16,
            "horizontal_multiple": 16,
            "height_padding": 0,
            "nray": nray,
            "z_size": len(heights),
            "heights_km": heights.tolist(),
            "padded_patch_shape": [64, 16, len(heights)],
            "patch_count": len(records),
            "files": files,
        },
    )
    normalization_path = root / "normalization.json"
    write_normalization(normalization_path)
    return Stage1PatchDataset(metadata_path, normalization_path), metadata_path, normalization_path


class PatchIndexTests(unittest.TestCase):
    def test_core_starts_and_padding_multiple(self) -> None:
        self.assertEqual(core_starts(70, 32), (0, 32, 64))
        self.assertEqual(core_starts(32, 32), (0,))
        self.assertEqual(ceil_to_multiple(49, 16), 64)
        self.assertEqual(ceil_to_multiple(64, 16), 64)

    def test_index_counts_only_nonoverlapping_core_not_halo(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rainy.nc"
            create_patch_nc(path)
            records, files, heights, nray, count = build_stage1_patch_index_records(
                [path], core_size=32
            )
        self.assertEqual(count, 1)
        self.assertEqual(nray, 5)
        np.testing.assert_allclose(heights, HEIGHTS)
        np.testing.assert_array_equal(records["core_start"], [0, 32, 64])
        np.testing.assert_array_equal(records["core_length"], [32, 32, 6])
        np.testing.assert_array_equal(records["positive_count"], [2, 2, 1])
        np.testing.assert_array_equal(records["input_count"], [2, 2, 1])
        self.assertEqual(files[0]["positive_count"], 5)


class Stage1PatchDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.dataset, self.metadata_path, self.normalization_path = (
            make_patch_dataset_fixture(self.root)
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_shapes_channels_padding_and_first_core_values(self) -> None:
        item = self.dataset[0]
        self.assertEqual(tuple(item["inputs"].shape), (3, 64, 16, 6))
        for name in (
            "target",
            "loss_mask",
            "loss_weights",
            "output_mask",
            "core_mask",
            "reliable_loss_mask",
            "weak_loss_mask",
            "diagnostic_target",
            "native_positive_mask",
            "cfb_distance_km",
            "relative_cfb_level",
            "cfb_quality_region",
        ):
            self.assertEqual(tuple(item[name].shape), (1, 64, 16, 6))
        self.assertEqual(tuple(item["height_km"].shape), (1, 1, 1, 6))
        self.assertEqual(tuple(item["height_index"].shape), (1, 1, 1, 6))
        self.assertEqual(tuple(item["cfb_profile_valid"].shape), (1, 64, 16, 1))
        self.assertEqual(tuple(item["precipitation_type"].shape), (1, 64, 16, 1))
        self.assertEqual(item["inputs"].dtype, item["target"].dtype)
        self.assertEqual(item["loss_mask"].dtype, __import__("torch").bool)
        self.assertEqual(item["unpadded_shape"].tolist(), [64, 5, 6])
        self.assertEqual(item["padded_shape"].tolist(), [64, 16, 6])

        # Source scan 0 maps to destination scan halo=16. At z=2,
        # standardized dbz=(20-10)/2=5 and validity channel equals one.
        self.assertAlmostEqual(float(item["inputs"][0, 16, 0, 2]), 5.0)
        self.assertEqual(float(item["inputs"][1, 16, 0, 2]), 1.0)
        # Height is the third channel: six physical levels map exactly to
        # [-1,1], while orbit-boundary and ray-padding positions remain zero.
        self.assertAlmostEqual(float(item["inputs"][2, 16, 0, 0]), -1.0)
        self.assertAlmostEqual(float(item["inputs"][2, 16, 0, 5]), 1.0)
        self.assertEqual(float(item["inputs"][2, 0, 0, 0]), 0.0)
        self.assertEqual(float(item["inputs"][2, 16, 5, 0]), 0.0)
        self.assertAlmostEqual(float(item["target"][0, 16, 0, 2]), np.log(2.0))
        self.assertEqual(int(item["loss_mask"].sum()), 2)
        self.assertEqual(int(item["reliable_loss_mask"].sum()), 2)
        self.assertEqual(int(item["weak_loss_mask"].sum()), 0)
        self.assertTrue(
            np.allclose(
                item["loss_weights"][item["loss_mask"]].numpy(), 1.0
            )
        )
        self.assertEqual(int(item["output_mask"].sum()), 2)
        self.assertEqual(int(item["core_mask"].sum()), 32 * 5 * 6)
        # Legacy pre_positive_qc normalization had no level-0 statistics, so
        # its finite below-CFB echo is unfortunately removed from model input.
        self.assertEqual(float(item["inputs"][1, 26, 0, 0]), 0.0)
        self.assertFalse(bool(item["loss_mask"][0, 26, 0, 0]))
        self.assertFalse(bool(item["core_mask"][0, 16, 5, 0]))
        self.assertEqual(
            int(item["cfb_quality_region"][0, 26, 0, 0]),
            CFB_QUALITY_CLUTTER,
        )
        self.assertEqual(
            int(item["cfb_quality_region"][0, 16, 0, 1]),
            CFB_QUALITY_RELIABLE,
        )
        self.assertAlmostEqual(float(item["cfb_distance_km"][0, 26, 0, 0]), -0.25)
        self.assertEqual(int(item["precipitation_type"][0, 16, 0, 0]), 1)

    def test_halo_is_input_context_but_not_second_cores_loss(self) -> None:
        item = self.dataset[1]  # core [32,64), context window [16,80)
        # Global scan31 maps to local15 and is visible to the model.
        self.assertEqual(float(item["inputs"][1, 15, 1, 2]), 1.0)
        # It is halo context, hence excluded from the core loss.
        self.assertFalse(bool(item["loss_mask"][0, 15, 1, 2]))
        # Global scan32 maps to local16, the first scan of this core.
        self.assertTrue(bool(item["loss_mask"][0, 16, 2, 3]))
        self.assertEqual(int(item["loss_mask"].sum()), 2)

    def test_e1_masks_below_cfb_input_without_calling_it_missing_label(self) -> None:
        native_stats = self.root / "native_normalization.json"
        write_variable_valid_normalization(native_stats)
        baseline = Stage1PatchDataset(
            self.metadata_path,
            native_stats,
            cfb_input_mode="baseline",
        )
        masked = Stage1PatchDataset(
            self.metadata_path,
            native_stats,
            cfb_input_mode="mask_below_cfb",
        )
        baseline_item = baseline[0]
        item = masked[0]
        # Source (scan10,ray0,z0) -> local scan26. E0 sees this finite echo;
        # E1 neutral-fills it and turns off only the model input-valid channel.
        self.assertAlmostEqual(float(baseline_item["inputs"][0, 26, 0, 0]), 2.0)
        self.assertEqual(float(baseline_item["inputs"][1, 26, 0, 0]), 1.0)
        self.assertEqual(float(item["inputs"][0, 26, 0, 0]), 0.0)
        self.assertEqual(float(item["inputs"][1, 26, 0, 0]), 0.0)
        self.assertTrue(bool(item["native_positive_mask"][0, 26, 0, 0]))
        self.assertFalse(bool(item["reliable_loss_mask"][0, 26, 0, 0]))
        self.assertEqual(tuple(item["inputs"].shape), (3, 64, 16, 6))

    def test_e2_adds_scaled_signed_cfb_distance_channel(self) -> None:
        signed = Stage1PatchDataset(
            self.metadata_path,
            self.normalization_path,
            cfb_input_mode="signed_distance",
            cfb_distance_scale_km=0.5,
        )
        item = signed[0]
        self.assertEqual(tuple(item["inputs"].shape), (4, 64, 16, 6))
        self.assertEqual(signed.feature_names[-1], CFB_DISTANCE_INPUT_CHANNEL)
        # CFB=1: z0 is -0.25 km -> -0.5 after scaling; z1 is zero.
        self.assertAlmostEqual(float(item["inputs"][3, 26, 0, 0]), -0.5)
        self.assertEqual(float(item["inputs"][3, 26, 0, 1]), 0.0)
        # Orbit-start halo and ray padding use neutral zero, never fake distance.
        self.assertEqual(float(item["inputs"][3, 0, 0, 0]), 0.0)
        self.assertEqual(float(item["inputs"][3, 16, 5, 0]), 0.0)

    def test_variable_valid_input_stats_do_not_change_label_qc(self) -> None:
        native_stats = self.root / "native_normalization.json"
        write_variable_valid_normalization(native_stats)
        dataset = Stage1PatchDataset(
            self.metadata_path,
            native_stats,
            height_loss_weighting="inverse_sqrt_frequency",
        )
        item = dataset[0]
        low_location = (0, 26, 0, 0)

        self.assertEqual(dataset.input_normalization_selection, "variable_valid")
        self.assertEqual(
            dataset.height_loss_weight_source,
            "height_loss_weight_reference:pre_positive_qc",
        )
        # Native input: (30 dBZ - mean 20) / std 5 = 2, and is marked valid.
        self.assertAlmostEqual(float(item["inputs"][0, 26, 0, 0]), 2.0)
        self.assertEqual(float(item["inputs"][1, 26, 0, 0]), 1.0)
        # Label semantics remain independent: the same below-CFB voxel is only
        # an auditable native target, never a reliable loss label.
        self.assertTrue(bool(item["native_positive_mask"][low_location]))
        self.assertFalse(bool(item["reliable_loss_mask"][low_location]))
        self.assertFalse(bool(item["loss_mask"][low_location]))
        self.assertEqual(float(item["target"][low_location]), 0.0)

        reliable_counts = np.asarray([0, 0, 16, 64, 16, 4], dtype=np.float64)
        input_counts = np.asarray([1000, 900, 800, 700, 600, 500], dtype=np.float64)
        self.assertAlmostEqual(
            float(
                np.sum(dataset.height_loss_weights * reliable_counts)
                / reliable_counts.sum()
            ),
            1.0,
            places=6,
        )
        self.assertNotAlmostEqual(
            float(
                np.sum(dataset.height_loss_weights * input_counts)
                / input_counts.sum()
            ),
            1.0,
            places=3,
        )

    def test_variable_valid_weighting_requires_independent_label_counts(self) -> None:
        native_stats = self.root / "native_without_loss_reference.json"
        write_variable_valid_normalization(
            native_stats, include_height_loss_reference=False
        )
        # Unweighted input normalization remains valid without loss statistics.
        Stage1PatchDataset(self.metadata_path, native_stats)
        with self.assertRaisesRegex(KeyError, "independent.*pre_positive_qc"):
            Stage1PatchDataset(
                self.metadata_path,
                native_stats,
                height_loss_weighting="inverse_sqrt_frequency",
            )

    def test_partial_debug_normalization_is_rejected_for_training(self) -> None:
        partial_stats = self.root / "partial_debug_normalization.json"
        write_variable_valid_normalization(partial_stats)
        value = json.loads(partial_stats.read_text(encoding="utf-8"))
        value["validated_file_count"] = 175
        value["processed_file_count"] = 1
        partial_stats.write_text(json.dumps(value), encoding="utf-8")

        with self.assertRaisesRegex(
            ValueError, "processed_file_count differs from validated_file_count"
        ):
            Stage1PatchDataset(self.metadata_path, partial_stats)

    def test_weak_output_support_does_not_depend_on_rain_label(self) -> None:
        weighted = Stage1PatchDataset(
            self.metadata_path,
            self.normalization_path,
            weak_cfb_layer_weights=(0.1,),
        )
        # Source (scan10,ray0,z0) maps to local (scan26,ray0,z0).  Its native
        # DPR echo is valid and it lies exactly one level below the valid CFB.
        weak_location = (0, 26, 0, 0)
        positive_item = weighted[0]
        self.assertTrue(bool(positive_item["weak_loss_mask"][weak_location]))
        self.assertTrue(bool(positive_item["output_mask"][weak_location]))

        # Changing only the precipitation label from 9 mm/h to zero removes
        # weak supervision, but must not alter where full-orbit inference may
        # retain the model output.
        with Dataset(self.root / "rainy.nc", "a") as source:
            source.variables["pre_dpr"][10, 0, 0] = 0.0
        weighted.clear_cache()
        zero_rain_item = weighted[0]
        self.assertFalse(bool(zero_rain_item["native_positive_mask"][weak_location]))
        self.assertFalse(bool(zero_rain_item["weak_loss_mask"][weak_location]))
        self.assertTrue(bool(zero_rain_item["output_mask"][weak_location]))
        self.assertEqual(float(zero_rain_item["target"][weak_location]), 0.0)

        # A configured zero-weight layer is disabled for both loss and output
        # support, preserving baseline/E1/E2 behavior when no positive weak
        # layer weight is active.
        disabled_item = Stage1PatchDataset(
            self.metadata_path,
            self.normalization_path,
            weak_cfb_layer_weights=(0.0,),
        )[0]
        self.assertFalse(bool(disabled_item["weak_loss_mask"][weak_location]))
        self.assertFalse(bool(disabled_item["output_mask"][weak_location]))

    def test_weak_cfb_height_and_intensity_weights_are_composed(self) -> None:
        weighted = Stage1PatchDataset(
            self.metadata_path,
            self.normalization_path,
            weak_cfb_layer_weights=(0.1,),
            height_loss_weighting="inverse_sqrt_frequency",
            height_loss_weight_min=0.5,
            height_loss_weight_max=3.0,
            intensity_loss_bin_edges=(1.0, 5.0),
            intensity_loss_bin_weights=(1.0, 2.0, 4.0),
        )
        item = weighted[0]
        weak_location = (0, 26, 0, 0)
        reliable_location = (0, 16, 0, 2)
        self.assertEqual(int(item["reliable_loss_mask"].sum()), 2)
        self.assertEqual(int(item["weak_loss_mask"].sum()), 1)
        self.assertEqual(int(item["loss_mask"].sum()), 3)
        self.assertTrue(bool(item["weak_loss_mask"][weak_location]))
        self.assertEqual(
            int(item["cfb_quality_region"][weak_location]), CFB_QUALITY_WEAK
        )
        self.assertAlmostEqual(
            float(item["target"][weak_location]), np.log1p(9.0), places=6
        )
        # 9 mm/h -> last intensity bin (4); below-CFB quality=0.1.
        expected_weak = float(weighted.height_loss_weights[0]) * 4.0 * 0.1
        self.assertAlmostEqual(
            float(item["loss_weights"][weak_location]), expected_weak, places=6
        )
        # 1 mm/h uses side='right', hence the [1,5) weight 2.
        expected_reliable = float(weighted.height_loss_weights[2]) * 2.0
        self.assertAlmostEqual(
            float(item["loss_weights"][reliable_location]),
            expected_reliable,
            places=6,
        )
        counts = np.asarray([0, 0, 16, 64, 16, 4], dtype=np.float64)
        self.assertAlmostEqual(
            float(np.sum(weighted.height_loss_weights * counts) / counts.sum()),
            1.0,
            places=6,
        )

    def test_missing_precipitation_type_uses_unknown_not_no_rain(self) -> None:
        dry_item = self.dataset[3]
        self.assertTrue(
            bool((dry_item["precipitation_type"] == -9999).all())
        )

    def test_invalid_cfb_is_explicit_unknown_and_never_fake_distance(self) -> None:
        with Dataset(self.root / "rainy.nc", "a") as source:
            source.variables["cfb"][0, 0] = -9999
        item = self.dataset[0]
        self.assertFalse(bool(item["cfb_profile_valid"][0, 16, 0, 0]))
        self.assertTrue(
            bool(__import__("torch").isnan(item["cfb_distance_km"][0, 16, 0]).all())
        )
        self.assertTrue(
            bool(
                (
                    item["cfb_quality_region"][0, 16, 0]
                    == CFB_QUALITY_UNKNOWN
                ).all()
            )
        )
        # Preserve baseline compatibility: invalid CFB does not invent clutter
        # and therefore does not silently discard an otherwise valid label.
        self.assertTrue(bool(item["reliable_loss_mask"][0, 16, 0, 2]))
        signed = Stage1PatchDataset(
            self.metadata_path,
            self.normalization_path,
            cfb_input_mode="signed_distance",
        )[0]
        self.assertTrue(bool((signed["inputs"][3, 16, 0] == 0.0).all()))

    def test_config_to_dataset_kwargs_has_one_canonical_mapping(self) -> None:
        options = stage1_patch_dataset_kwargs(
            {
                "cfb_input_mode": "signed_distance",
                "cfb_distance_scale_km": 1.5,
                "weak_cfb_layer_weights": [0.1, 0.05],
            },
            {
                "height_loss_weighting": "inverse_sqrt_frequency",
                "height_loss_weight_min": 0.6,
                "height_loss_weight_max": 2.5,
                "intensity_loss_bin_edges": [1, 5],
                "intensity_loss_bin_weights": [1, 2, 3],
            },
        )
        self.assertEqual(options["cfb_input_mode"], "signed_distance")
        self.assertEqual(options["weak_cfb_layer_weights"], (0.1, 0.05))
        self.assertEqual(options["intensity_loss_bin_weights"], (1.0, 2.0, 3.0))

    def test_ablation_configuration_rejects_ambiguous_weights(self) -> None:
        common = (self.metadata_path, self.normalization_path)
        with self.assertRaisesRegex(ValueError, "cfb_input_mode"):
            Stage1PatchDataset(*common, cfb_input_mode="unknown")
        with self.assertRaisesRegex(ValueError, "inside.*0,1"):
            Stage1PatchDataset(*common, weak_cfb_layer_weights=(1.1,))
        with self.assertRaisesRegex(ValueError, "one more value"):
            Stage1PatchDataset(
                *common,
                intensity_loss_bin_edges=(1.0, 5.0),
                intensity_loss_bin_weights=(1.0, 2.0),
            )
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            Stage1PatchDataset(
                *common,
                intensity_loss_bin_edges=(5.0, 1.0),
                intensity_loss_bin_weights=(1.0, 2.0, 3.0),
            )

    def test_last_short_core_and_positive_only_filter(self) -> None:
        last_rainy = self.dataset[2]
        self.assertEqual(int(last_rainy["core_start"]), 64)
        self.assertEqual(int(last_rainy["core_length"]), 6)
        self.assertEqual(int(last_rainy["core_mask"].sum()), 6 * 5 * 6)
        self.assertEqual(len(self.dataset), 5)  # three rainy-file + two dry-file cores

        filtered = Stage1PatchDataset(
            self.metadata_path,
            self.normalization_path,
            positive_only=True,
        )
        self.assertEqual(len(filtered), 3)
        self.assertEqual(list(filtered.file_index_range(0)), [0, 1, 2])
        self.assertEqual(list(filtered.file_index_range(1)), [])
        self.assertTrue(all(int(filtered[i]["file_id"]) == 0 for i in range(3)))

    def test_dataloader_batch_has_channel_first_5d_tensors(self) -> None:
        import torch

        loader = torch.utils.data.DataLoader(self.dataset, batch_size=2, num_workers=0)
        batch = next(iter(loader))
        self.assertEqual(tuple(batch["inputs"].shape), (2, 3, 64, 16, 6))
        self.assertEqual(tuple(batch["target"].shape), (2, 1, 64, 16, 6))
        self.assertEqual(tuple(batch["loss_mask"].shape), (2, 1, 64, 16, 6))


if __name__ == "__main__":
    unittest.main()
