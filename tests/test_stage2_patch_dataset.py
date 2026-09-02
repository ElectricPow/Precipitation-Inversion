"""Tests for Stage-2 normalization, indexing, and fixed tensor construction."""

from __future__ import annotations

import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.fit_stage2_normalization_stats import (  # noqa: E402
    fit_stage2_statistics,
)
from precipitation_inversion.data.stage2_patch_dataset import (  # noqa: E402
    STAGE2_FIVE_CHANNEL_INTERP_INPUT_CHANNELS,
    STAGE2_FOUR_CHANNEL_DENSITY_INPUT_CHANNELS,
    STAGE2_FOUR_CHANNEL_DISTANCE_INPUT_CHANNELS,
    STAGE2_INPUT_CHANNELS,
    STAGE2_INPUT_VARIABLE,
    STAGE2_PATCH_INDEX_FORMAT_VERSION,
    STAGE2_R1_ORACLE_SPARSE_VALUE_INPUT_CHANNELS,
    STAGE2_TARGET_VARIABLE,
    STAGE2_THREE_CHANNEL_NATIVE_INPUT_CHANNELS,
    Stage2PatchDataset,
    build_stage2_patch_index_records,
    physical_dbz_regression_weights,
    save_stage2_patch_index,
    stage2_patch_dataset_kwargs,
    validate_stage2_input_channels,
    validate_stage2_reflectivity_weighting,
)


HEIGHTS = np.array([0.125, 0.375, 0.625], dtype=np.float32)


def create_stage2_nc(path: Path, *, dpr_offset: float = 0.0) -> None:
    """Create shape ``(6,3,3)`` with native, sentinel, and physical states."""

    shape = (6, 3, 3)
    gr = np.full(shape, np.nan, dtype=np.float32)
    gr[0, 0, :] = [10.0, 20.0, 30.0]
    gr[1, 0, :] = [14.0, 24.0, 34.0]
    gr[0, 1, 0] = -9999.9

    interp = np.full(shape, np.nan, dtype=np.float32)
    interp[0, 0, :] = gr[0, 0, :]
    interp[1, 0, :] = gr[1, 0, :]
    interp[1, 1, :] = [16.0, 26.0, 36.0]

    dpr = np.full(shape, -9999.9, dtype=np.float32)
    dpr[0, 0, :] = np.array([12.0, 22.0, 32.0]) + dpr_offset
    dpr[1, 1, :] = np.array([16.0, 26.0, 36.0]) + dpr_offset
    dpr[2, 2, :] = np.array([18.0, 28.0, 38.0]) + dpr_offset
    precipitation = np.zeros(shape, dtype=np.float32)
    precipitation[dpr > -9990.0] = 1.0

    with Dataset(path, "w") as dataset:
        dataset.createDimension("nscan", shape[0])
        dataset.createDimension("nray", shape[1])
        dataset.createDimension("z", shape[2])
        z = dataset.createVariable("z", "f4", ("z",), fill_value=np.nan)
        z[:] = HEIGHTS
        for name, values in (
            ("dbz_gr_sparse", gr),
            ("dbz_gr_interp", interp),
            ("dbz_dpr", dpr),
            ("pre_dpr", precipitation),
        ):
            variable = dataset.createVariable(
                name, "f4", ("nscan", "nray", "z"), fill_value=np.nan
            )
            variable.units = "dBZ" if name.startswith("dbz") else "mm/h"
            variable[:] = values
        cfb = dataset.createVariable("cfb", "i2", ("nscan", "nray"))
        cfb[:] = 0


def write_normalization(path: Path) -> None:
    def values(mean: list[float]) -> dict[str, object]:
        return {
            "heights_km": HEIGHTS.tolist(),
            "mean": mean,
            "std": [2.0, 2.0, 2.0],
            "count": [2, 2, 2],
        }

    path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "stage": 2,
                "scope": "training_split_only",
                "selection_mask": "reflectivity_storage_value",
                "validated_file_count": 1,
                "processed_file_count": 1,
                "split_manifest_sha256": "same-split",
                "variables": {
                    "dbz_gr_sparse": values([12.0, 22.0, 32.0]),
                    "dbz_gr_interp": values([14.0, 24.0, 34.0]),
                    "dbz_dpr": values([14.0, 24.0, 34.0]),
                },
            }
        ),
        encoding="utf-8",
    )


def make_dataset(root: Path, *, source: Path | None = None) -> Stage2PatchDataset:
    source = source or root / "sample.nc"
    if not source.exists():
        create_stage2_nc(source)
    records, files, z, nray, _ = build_stage2_patch_index_records(
        [source], core_size=4, strong_dbz_threshold=30.0
    )
    index_path = root / "train.npy"
    metadata_path = root / "train.json"
    save_stage2_patch_index(
        index_path,
        metadata_path,
        records,
        {
            "format_version": STAGE2_PATCH_INDEX_FORMAT_VERSION,
            "stage": 2,
            "input_variable": STAGE2_INPUT_VARIABLE,
            "input_channels": list(STAGE2_INPUT_CHANNELS),
            "target_variable": STAGE2_TARGET_VARIABLE,
            "core_size": 4,
            "halo_size": 2,
            "horizontal_multiple": 4,
            "height_padding": 0,
            "nray": nray,
            "z_size": len(z),
            "heights_km": z.tolist(),
            "padded_patch_shape": [8, 4, len(z)],
            "patch_count": len(records),
            "split_manifest_sha256": "same-split",
            "files": files,
        },
    )
    normalization = root / "normalization.json"
    write_normalization(normalization)
    return Stage2PatchDataset(metadata_path, normalization)


class Stage2PatchIndexTests(unittest.TestCase):
    def test_index_counts_core_regions_and_strong_targets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.nc"
            create_stage2_nc(path)
            records, files, z, nray, count = build_stage2_patch_index_records(
                [path], core_size=4, strong_dbz_threshold=30.0
            )
        self.assertEqual((count, nray), (1, 3))
        np.testing.assert_allclose(z, HEIGHTS)
        np.testing.assert_array_equal(records["core_start"], [0, 4])
        np.testing.assert_array_equal(records["core_length"], [4, 2])
        first = records[0]
        self.assertEqual(int(first["gr_count"]), 6)
        self.assertEqual(int(first["dpr_count"]), 9)
        self.assertEqual(int(first["q11_count"]), 3)
        self.assertEqual(int(first["q01_count"]), 6)
        self.assertEqual(int(first["q10_count"]), 3)
        self.assertEqual(int(first["gap_target_count"]), 3)
        self.assertEqual(int(first["outside_target_count"]), 3)
        self.assertEqual(int(first["strong_dpr_count"]), 3)
        self.assertEqual(files[0]["target_patch_count"], 1)

    def test_statistics_fit_only_physical_states(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.nc"
            create_stage2_nc(path)
            accumulators, states, z, _, chunks = fit_stage2_statistics(
                [path],
                variables=("dbz_gr_sparse", "dbz_dpr"),
                scan_chunk_size=2,
            )
        self.assertEqual(chunks, 3)
        np.testing.assert_allclose(z, HEIGHTS)
        np.testing.assert_array_equal(
            accumulators["dbz_gr_sparse"].count, [2, 2, 2]
        )
        np.testing.assert_allclose(
            accumulators["dbz_gr_sparse"].mean, [12.0, 22.0, 32.0]
        )
        self.assertEqual(states["dbz_gr_sparse"]["sentinel"], 1)
        self.assertEqual(states["dbz_gr_sparse"]["value"], 6)
        self.assertEqual(states["dbz_dpr"]["value"], 9)


class Stage2PatchDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.dataset = make_dataset(self.root)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_shapes_channels_padding_and_mask_partition(self) -> None:
        item = self.dataset[0]
        self.assertEqual(tuple(item["inputs"].shape), (4, 8, 4, 3))
        mask_names = (
            "target_dbz",
            "target_support",
            "target_valid",
            "support_loss_mask",
            "occupancy_domain_mask",
            "regression_mask",
            "overlap_mask",
            "dpr_only_mask",
            "gr_only_mask",
            "neither_mask",
            "gap_proxy_mask",
            "outside_proxy_mask",
            "below_cfb_target_mask",
            "core_mask",
            "geometry_mask",
            "gr_value_mask",
            "dpr_sparse_anchor_mask",
            "dpr_sparse_anchor_distance_scaled",
            "gr_interp_value_mask",
            "gr_native_available",
            "gr_native_missing",
            "gr_sentinel",
        )
        for name in mask_names:
            self.assertEqual(tuple(item[name].shape), (1, 8, 4, 3), name)
        self.assertEqual(tuple(item["height_km"].shape), (1, 1, 1, 3))
        self.assertEqual(item["unpadded_shape"].tolist(), [8, 3, 3])
        self.assertEqual(item["padded_shape"].tolist(), [8, 4, 3])

        # Source scan 0 is destination scan 2 because halo=2.  Physical GR is
        # standardized per height; finite sentinel remains separately visible.
        np.testing.assert_allclose(
            item["inputs"][0, 2, 0].numpy(), [-1.0, -1.0, -1.0]
        )
        np.testing.assert_array_equal(
            item["inputs"][1, 2, 0].numpy(), [1.0, 1.0, 1.0]
        )
        self.assertEqual(float(item["inputs"][1, 2, 1, 0]), 0.0)
        self.assertEqual(float(item["inputs"][2, 2, 1, 0]), 1.0)
        self.assertTrue(bool(item["gr_sentinel"][0, 2, 1, 0]))
        self.assertEqual(float(item["inputs"][2, 2, 2, 0]), 0.0)

        # First two scan positions are orbit-boundary halo; ray index 3 is
        # horizontal padding. Every input channel uses neutral zero there.
        self.assertTrue(np.all(item["inputs"][:, :2].numpy() == 0.0))
        self.assertTrue(np.all(item["inputs"][:, :, 3].numpy() == 0.0))
        self.assertEqual(int(item["core_mask"].sum()), 4 * 3 * 3)
        self.assertEqual(int(item["support_loss_mask"].sum()), 4 * 3 * 3)
        self.assertEqual(int(item["regression_mask"].sum()), 9)
        self.assertEqual(int(item["overlap_mask"].sum()), 3)
        self.assertEqual(int(item["dpr_only_mask"].sum()), 6)
        self.assertEqual(int(item["gr_only_mask"].sum()), 3)
        self.assertEqual(int(item["gap_proxy_mask"].sum()), 3)
        self.assertEqual(int(item["outside_proxy_mask"].sum()), 3)
        partition = sum(
            item[name].to(dtype=__import__("torch").uint8)
            for name in (
                "overlap_mask",
                "dpr_only_mask",
                "gr_only_mask",
                "neither_mask",
            )
        )
        self.assertTrue(bool((partition <= 1).all()))
        self.assertEqual(
            int(partition.sum()), int(item["occupancy_domain_mask"].sum())
        )

    def test_r1_oracle_sparse_dpr_channels_expose_only_colocated_anchors(self) -> None:
        oracle = Stage2PatchDataset(
            self.root / "train.json",
            self.root / "normalization.json",
            input_channels=STAGE2_R1_ORACLE_SPARSE_VALUE_INPUT_CHANNELS,
        )
        item = oracle[0]
        self.assertEqual(tuple(item["inputs"].shape), (4, 8, 4, 3))
        self.assertEqual(oracle.feature_names, list(STAGE2_R1_ORACLE_SPARSE_VALUE_INPUT_CHANNELS))
        # Source scan 0/ray 0 is the sole direct GR-DPR overlap profile and is
        # placed at Patch scan 2 by the two-scan halo. DPR standardization is
        # ([12,22,32]-[14,24,34])/2 = [-1,-1,-1].
        np.testing.assert_allclose(item["inputs"][0, 2, 0], [-1.0, -1.0, -1.0])
        np.testing.assert_allclose(item["inputs"][1, 2, 0], [1.0, 1.0, 1.0])
        self.assertEqual(int(item["dpr_sparse_anchor_mask"].sum()), 3)
        self.assertTrue(bool((item["inputs"][0][~item["dpr_sparse_anchor_mask"][0]] == 0).all()))
        # Complete-orbit Chebyshev distance is 0 at the anchor, 1/8 at the
        # next diagonal target, and maximal 1 outside geometry/padded ray.
        np.testing.assert_allclose(item["inputs"][2, 2, 0], 0.0)
        np.testing.assert_allclose(item["inputs"][2, 3, 1], 1.0 / 8.0)
        self.assertTrue(bool((item["inputs"][2, :2] == 1.0).all()))
        self.assertTrue(bool((item["inputs"][2, :, 3] == 1.0).all()))

        with self.assertRaisesRegex(ValueError, "cannot be mixed"):
            validate_stage2_input_channels(
                (
                    "dbz_gr_sparse_standardized",
                    *STAGE2_R1_ORACLE_SPARSE_VALUE_INPUT_CHANNELS,
                )
            )

    def test_physical_dbz_weight_boundaries_and_strict_validation(self) -> None:
        physical = np.array(
            [24.999, 25.0, 34.999, 35.0, np.nan], dtype=np.float32
        ).reshape(1, 1, 5)
        selected = np.array([True, True, True, True, False]).reshape(1, 1, 5)
        weights = physical_dbz_regression_weights(
            physical,
            selected,
            bin_edges_dbz=(25.0, 35.0),
            bin_weights=(1.0, 1.25, 1.5),
        )
        self.assertEqual(weights.dtype, np.float32)
        np.testing.assert_allclose(
            weights.reshape(-1), [1.0, 1.25, 1.25, 1.5, 0.0]
        )

        invalid = (
            ((25.0, 25.0), (1.0, 1.25, 1.5), ValueError),
            ((35.0, 25.0), (1.0, 1.25, 1.5), ValueError),
            ((25.0, 35.0), (1.0, 1.25), ValueError),
            ((25.0,), (1.0, 0.0), ValueError),
            ((25.0,), (1.0, float("inf")), ValueError),
            ((25.0, True), (1.0, 1.25, 1.5), TypeError),
        )
        for edges, values, error in invalid:
            with self.subTest(edges=edges, weights=values), self.assertRaises(error):
                validate_stage2_reflectivity_weighting(edges, values)
        with self.assertRaises(TypeError):
            validate_stage2_reflectivity_weighting("25,35", (1.0, 1.25, 1.5))
        with self.assertRaisesRegex(ValueError, "one more"):
            stage2_patch_dataset_kwargs(
                {
                    "reflectivity": {
                        "intensity_bin_edges_dbz": [25.0, 35.0],
                        "intensity_bin_weights": [],
                    }
                }
            )

    def test_e3_weights_are_physical_supervision_only_and_zero_padded(self) -> None:
        distance_parent = Stage2PatchDataset(
            self.root / "train.json",
            self.root / "normalization.json",
            input_channels=STAGE2_FOUR_CHANNEL_DISTANCE_INPUT_CHANNELS,
        )
        weighted = Stage2PatchDataset(
            self.root / "train.json",
            self.root / "normalization.json",
            input_channels=STAGE2_FOUR_CHANNEL_DISTANCE_INPUT_CHANNELS,
            reflectivity_intensity_bin_edges_dbz=(25.0, 35.0),
            reflectivity_intensity_bin_weights=(1.0, 1.25, 1.5),
        )
        parent_item = distance_parent[0]
        item = weighted[0]

        self.assertNotIn("regression_weights", parent_item)
        self.assertEqual(tuple(item["regression_weights"].shape), (1, 8, 4, 3))
        self.assertEqual(item["regression_weights"].dtype, __import__("torch").float32)
        # E3 changes no model input, target, or mask relative to distance D.
        for name in (
            "inputs",
            "target_dbz",
            "target_support",
            "support_loss_mask",
            "regression_mask",
            "overlap_mask",
            "dpr_only_mask",
            "core_mask",
            "geometry_mask",
        ):
            np.testing.assert_array_equal(item[name].numpy(), parent_item[name].numpy())

        selected = item["regression_mask"]
        values = item["regression_weights"]
        self.assertTrue(bool((values[~selected] == 0.0).all()))
        self.assertTrue(bool((values[selected] > 0.0).all()))
        # Raw physical DPR values form 4/3/2 low/middle/high voxels. Their
        # standardized targets are all far below 25, so 10.75 also proves the
        # Dataset did not accidentally bin the normalized target tensor.
        self.assertAlmostEqual(float(values.sum()), 10.75, places=6)
        unique, counts = __import__("torch").unique(
            values[selected], return_counts=True
        )
        self.assertEqual(unique.tolist(), [1.0, 1.25, 1.5])
        self.assertEqual(counts.tolist(), [4, 3, 2])

    def test_target_changes_cannot_change_gr_only_inputs(self) -> None:
        other_root = self.root / "other"
        other_root.mkdir()
        source = other_root / "changed_target.nc"
        create_stage2_nc(source, dpr_offset=100.0)
        other = make_dataset(other_root, source=source)
        np.testing.assert_array_equal(
            self.dataset[0]["inputs"].numpy(), other[0]["inputs"].numpy()
        )
        self.assertFalse(
            np.array_equal(
                self.dataset[0]["target_dbz"].numpy(),
                other[0]["target_dbz"].numpy(),
            )
        )

    def test_three_channel_native_subset_preserves_selected_channel_values(self) -> None:
        three_channel = Stage2PatchDataset(
            self.root / "train.json",
            self.root / "normalization.json",
            input_channels=STAGE2_THREE_CHANNEL_NATIVE_INPUT_CHANNELS,
        )
        baseline_item = self.dataset[0]
        selected_item = three_channel[0]

        self.assertEqual(
            three_channel.feature_names,
            list(STAGE2_THREE_CHANNEL_NATIVE_INPUT_CHANNELS),
        )
        self.assertEqual(tuple(selected_item["inputs"].shape), (3, 8, 4, 3))
        # Three-channel output [0,1,2] is exactly canonical [0,2,3]. No value
        # transformation is changed by removing the GR physical-value mask.
        np.testing.assert_array_equal(
            selected_item["inputs"].numpy(),
            baseline_item["inputs"][[0, 2, 3]].numpy(),
        )
        np.testing.assert_array_equal(
            selected_item["target_dbz"].numpy(),
            baseline_item["target_dbz"].numpy(),
        )

    def test_five_channel_interp_standardizes_values_and_preserves_mask(self) -> None:
        five_channel = Stage2PatchDataset(
            self.root / "train.json",
            self.root / "normalization.json",
            input_channels=STAGE2_FIVE_CHANNEL_INTERP_INPUT_CHANNELS,
        )
        baseline_item = self.dataset[0]
        item = five_channel[0]

        self.assertEqual(
            five_channel.feature_names,
            list(STAGE2_FIVE_CHANNEL_INTERP_INPUT_CHANNELS),
        )
        self.assertEqual(tuple(item["inputs"].shape), (5, 8, 4, 3))
        # Five-channel [0,1,4] exactly preserves 3V's sparse value, value mask,
        # and height channels. Channels [2,3] are the sole intervention.
        np.testing.assert_array_equal(item["inputs"][0], baseline_item["inputs"][0])
        np.testing.assert_array_equal(item["inputs"][1], baseline_item["inputs"][1])
        np.testing.assert_array_equal(item["inputs"][4], baseline_item["inputs"][3])

        # Source scan 0/1 map to window scan 2/3 because halo=2. Per-height
        # interpolation means are [14,24,34] with std=2.
        np.testing.assert_allclose(item["inputs"][2, 2, 0], [-2.0, -2.0, -2.0])
        np.testing.assert_allclose(item["inputs"][2, 3, 0], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(item["inputs"][2, 3, 1], [1.0, 1.0, 1.0])
        np.testing.assert_array_equal(item["inputs"][3, 3, 1], [1.0, 1.0, 1.0])
        self.assertEqual(float(item["inputs"][3, 2, 1, 0]), 0.0)
        self.assertTrue(bool(item["gr_interp_value_mask"][0, 3, 1, 0]))
        self.assertTrue(np.all(item["inputs"][:, :2].numpy() == 0.0))
        self.assertTrue(np.all(item["inputs"][:, :, 3].numpy() == 0.0))
        np.testing.assert_array_equal(
            item["target_dbz"].numpy(), baseline_item["target_dbz"].numpy()
        )

    def test_four_channel_distance_is_full_orbit_scaled_and_padded_far(self) -> None:
        distance_dataset = Stage2PatchDataset(
            self.root / "train.json",
            self.root / "normalization.json",
            input_channels=STAGE2_FOUR_CHANNEL_DISTANCE_INPUT_CHANNELS,
        )
        baseline_item = self.dataset[0]
        item = distance_dataset[0]

        self.assertEqual(
            distance_dataset.feature_names,
            list(STAGE2_FOUR_CHANNEL_DISTANCE_INPUT_CHANNELS),
        )
        self.assertEqual(tuple(item["inputs"].shape), (4, 8, 4, 3))
        # S2-3V-D preserves 3V's sparse dBZ, value mask and height exactly;
        # channel 2 is the only new information.
        np.testing.assert_array_equal(item["inputs"][0], baseline_item["inputs"][0])
        np.testing.assert_array_equal(item["inputs"][1], baseline_item["inputs"][1])
        np.testing.assert_array_equal(item["inputs"][3], baseline_item["inputs"][3])

        distance = item["inputs"][2]
        # Source scan 0 maps to destination scan 2. Direct GR at ray 0 has
        # distance 0; same-height ray offsets 1/2 have distances 1/2, scaled
        # by the fixed eight-cell clipping radius.
        np.testing.assert_allclose(distance[2, 0], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(distance[2, 1], [0.125, 0.125, 0.125])
        np.testing.assert_allclose(distance[2, 2], [0.25, 0.25, 0.25])
        # Orbit-external halo and nray padding are not direct observations and
        # therefore use the maximally-far value 1.0.
        self.assertTrue(bool((distance[:2] == 1.0).all()))
        self.assertTrue(bool((distance[:, 3] == 1.0).all()))
        np.testing.assert_array_equal(
            item["target_dbz"].numpy(), baseline_item["target_dbz"].numpy()
        )

    def test_four_channel_density_is_full_orbit_same_height_and_zero_padded(self) -> None:
        density_dataset = Stage2PatchDataset(
            self.root / "train.json",
            self.root / "normalization.json",
            input_channels=STAGE2_FOUR_CHANNEL_DENSITY_INPUT_CHANNELS,
        )
        baseline_item = self.dataset[0]
        item = density_dataset[0]

        self.assertEqual(
            density_dataset.feature_names,
            list(STAGE2_FOUR_CHANNEL_DENSITY_INPUT_CHANNELS),
        )
        self.assertEqual(tuple(item["inputs"].shape), (4, 8, 4, 3))
        # S2-3V-rho preserves all three parent channels exactly. Channel 2 is
        # the sole intervention and is direct-GR density, never a DPR feature.
        np.testing.assert_array_equal(item["inputs"][0], baseline_item["inputs"][0])
        np.testing.assert_array_equal(item["inputs"][1], baseline_item["inputs"][1])
        np.testing.assert_array_equal(item["inputs"][3], baseline_item["inputs"][3])

        density = item["inputs"][2]
        # Physical GR exists at source scans 0 and 1, ray 0, at all 3 heights.
        # Source scan 0 maps to destination scan 2. Its fixed 5x5 window has
        # two observations per height, so rho=2/25. Sentinel is not counted.
        np.testing.assert_allclose(density[2, 0], [2.0 / 25.0] * 3)
        np.testing.assert_allclose(density[2, 1], [2.0 / 25.0] * 3)
        # Orbit-external halo and high-ray padding represent no observations.
        self.assertTrue(bool((density[:2] == 0.0).all()))
        self.assertTrue(bool((density[:, 3] == 0.0).all()))
        self.assertTrue(bool(((density >= 0.0) & (density <= 1.0)).all()))
        np.testing.assert_array_equal(
            item["target_dbz"].numpy(), baseline_item["target_dbz"].numpy()
        )

        # The second Patch begins at source scan 2. Density at that boundary
        # still sees scans 0/1 because it was computed on the complete orbit
        # before extraction; a Patch-local calculation would incorrectly be 0.
        second = density_dataset[1]["inputs"][2]
        np.testing.assert_allclose(second[0, 0], [2.0 / 25.0] * 3)

    def test_input_channel_contract_rejects_unknown_duplicate_and_reordering(self) -> None:
        self.assertEqual(
            validate_stage2_input_channels(None), STAGE2_INPUT_CHANNELS
        )
        with self.assertRaisesRegex(ValueError, "unknown"):
            validate_stage2_input_channels(
                ("dbz_gr_sparse_standardized", "unknown", "height_scaled")
            )
        with self.assertRaisesRegex(ValueError, "duplicates"):
            validate_stage2_input_channels(
                (
                    "dbz_gr_sparse_standardized",
                    "gr_value_mask",
                    "gr_value_mask",
                    "height_scaled",
                )
            )
        with self.assertRaisesRegex(ValueError, "canonical"):
            validate_stage2_input_channels(
                (
                    "height_scaled",
                    "gr_native_available",
                    "dbz_gr_sparse_standardized",
                )
            )
        with self.assertRaisesRegex(ValueError, "required"):
            validate_stage2_input_channels(("gr_native_available",))
        with self.assertRaisesRegex(ValueError, "selected together"):
            validate_stage2_input_channels(
                (
                    "dbz_gr_sparse_standardized",
                    "dbz_gr_interp_standardized",
                    "height_scaled",
                )
            )
        with self.assertRaisesRegex(ValueError, "selected together"):
            validate_stage2_input_channels(
                (
                    "dbz_gr_sparse_standardized",
                    "gr_interp_value_mask",
                    "height_scaled",
                )
            )
        self.assertEqual(
            validate_stage2_input_channels(
                STAGE2_FIVE_CHANNEL_INTERP_INPUT_CHANNELS
            ),
            STAGE2_FIVE_CHANNEL_INTERP_INPUT_CHANNELS,
        )
        self.assertEqual(
            validate_stage2_input_channels(
                STAGE2_FOUR_CHANNEL_DISTANCE_INPUT_CHANNELS
            ),
            STAGE2_FOUR_CHANNEL_DISTANCE_INPUT_CHANNELS,
        )
        self.assertEqual(
            validate_stage2_input_channels(
                STAGE2_FOUR_CHANNEL_DENSITY_INPUT_CHANNELS
            ),
            STAGE2_FOUR_CHANNEL_DENSITY_INPUT_CHANNELS,
        )

    def test_hash_and_partial_normalization_validation(self) -> None:
        metadata_path = self.root / "train.json"
        normalization_path = self.root / "normalization.json"
        value = json.loads(normalization_path.read_text(encoding="utf-8"))
        value["processed_file_count"] = 0
        normalization_path.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "partial"):
            Stage2PatchDataset(metadata_path, normalization_path)

    def test_dataloader_batch_and_pickled_copy_reopen_index(self) -> None:
        torch = __import__("torch")
        _ = self.dataset[0]
        self.assertEqual(self.dataset.cached_file_ids, (0,))
        restored = pickle.loads(pickle.dumps(self.dataset))
        self.assertEqual(restored.cached_file_ids, ())
        self.assertEqual(len(restored), len(self.dataset))
        loader = torch.utils.data.DataLoader(restored, batch_size=2, num_workers=0)
        batch = next(iter(loader))
        self.assertEqual(tuple(batch["inputs"].shape), (2, 4, 8, 4, 3))
        self.assertEqual(tuple(batch["target_dbz"].shape), (2, 1, 8, 4, 3))


if __name__ == "__main__":
    unittest.main()
