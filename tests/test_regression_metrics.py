"""Tests for streaming masked precipitation regression metrics."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from precipitation_inversion.metrics.regression import (  # noqa: E402
    FilewisePrecipitationMetrics,
    GroupedRegressionAccumulator,
    PhysicalRainGradientMetrics,
    PrecipitationRegressionMetrics,
    RegressionAccumulator,
    StratifiedPrecipitationMetrics,
    physical_vertical_rain_gradient,
)


class RegressionAccumulatorTests(unittest.TestCase):
    def test_masked_metrics_match_direct_values(self) -> None:
        # Arrays represent (B=1,C=1,D=1,H=1,Z=3). The final height is padding.
        prediction = torch.tensor([2.0, 4.0, 999.0]).reshape(1, 1, 1, 1, 3)
        target = torch.tensor([1.0, 5.0, 0.0]).reshape_as(prediction)
        mask = torch.tensor([True, True, False]).reshape_as(prediction)
        accumulator = RegressionAccumulator()
        accumulator.update(prediction, target, mask)
        result = accumulator.compute()
        self.assertEqual(result["count"], 2)
        self.assertAlmostEqual(result["mae"], 1.0)
        self.assertAlmostEqual(result["rmse"], 1.0)
        self.assertAlmostEqual(result["bias"], 0.0)
        self.assertAlmostEqual(result["r2"], 0.75)
        self.assertAlmostEqual(result["pearson_r"], 1.0)

    def test_streaming_updates_and_merge_equal_one_shot(self) -> None:
        prediction = np.array([1.0, 4.0, 8.0, 3.0], dtype=np.float32)
        target = np.array([2.0, 2.0, 7.0, 5.0], dtype=np.float32)
        mask = np.array([True, False, True, True])
        expected = RegressionAccumulator()
        expected.update(prediction, target, mask)

        first = RegressionAccumulator()
        second = RegressionAccumulator()
        first.update(prediction[:2], target[:2], mask[:2])
        second.update(prediction[2:], target[2:], mask[2:])
        first.merge(second)
        for key, expected_value in expected.compute().items():
            actual = first.compute()[key]
            if isinstance(expected_value, float):
                self.assertAlmostEqual(actual, expected_value)
            else:
                self.assertEqual(actual, expected_value)

    def test_empty_mask_and_invalid_selected_values(self) -> None:
        accumulator = RegressionAccumulator()
        accumulator.update(torch.tensor([math.nan]), torch.tensor([1.0]), torch.tensor([False]))
        result = accumulator.compute()
        self.assertEqual(result["count"], 0)
        self.assertTrue(math.isnan(result["mae"]))
        with self.assertRaisesRegex(ValueError, "finite"):
            accumulator.update(
                torch.tensor([math.nan]), torch.tensor([1.0]), torch.tensor([True])
            )
        with self.assertRaises(TypeError):
            accumulator.update(torch.ones(1), torch.ones(1), torch.ones(1))


class PrecipitationRegressionMetricsTests(unittest.TestCase):
    def test_log_conversion_physical_metrics_and_target_bins(self) -> None:
        target_rain = torch.tensor([0.5, 2.0, 7.0, 20.0, 40.0])
        prediction_rain = target_rain + 1.0
        mask = torch.ones_like(target_rain, dtype=torch.bool)
        metrics = PrecipitationRegressionMetrics((1, 5, 10, 30))
        metrics.update_log(
            torch.log1p(prediction_rain), torch.log1p(target_rain), mask
        )
        result = metrics.compute()
        self.assertEqual(result["rain"]["all"]["count"], 5)
        self.assertAlmostEqual(result["rain"]["all"]["mae"], 1.0, places=5)
        bins = result["rain"]["target_bins_mm_h"]
        self.assertEqual(bins["lt_1"]["count"], 1)
        self.assertEqual(bins["1_to_5"]["count"], 1)
        self.assertEqual(bins["5_to_10"]["count"], 1)
        self.assertEqual(bins["10_to_30"]["count"], 1)
        self.assertEqual(bins["ge_30"]["count"], 1)

    def test_negative_log_prediction_is_clamped_only_in_physical_space(self) -> None:
        metrics = PrecipitationRegressionMetrics((1,))
        metrics.update_log(
            torch.tensor([-2.0]), torch.tensor([math.log(2.0)]), torch.tensor([True])
        )
        result = metrics.compute()
        self.assertAlmostEqual(result["rain"]["all"]["bias"], -1.0)
        self.assertAlmostEqual(result["log"]["bias"], -2.0 - math.log(2.0))

    def test_update_rain_and_reset(self) -> None:
        metrics = PrecipitationRegressionMetrics((1, 5))
        metrics.update_rain(
            torch.tensor([1.0, 3.0]),
            torch.tensor([1.0, 2.0]),
            torch.tensor([True, False]),
        )
        self.assertEqual(metrics.compute()["rain"]["all"]["count"], 1)
        metrics.reset()
        self.assertEqual(metrics.compute()["rain"]["all"]["count"], 0)

    def test_invalid_thresholds_and_shape_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            PrecipitationRegressionMetrics((5, 1))
        metrics = PrecipitationRegressionMetrics()
        with self.assertRaisesRegex(ValueError, "shapes differ"):
            metrics.update_log(torch.ones(2), torch.ones(3), torch.ones(2, dtype=torch.bool))


class GroupedRegressionAccumulatorTests(unittest.TestCase):
    def test_grouped_update_empty_group_and_merge(self) -> None:
        prediction = torch.tensor([1.0, 3.0, 5.0, 7.0])
        target = torch.tensor([0.0, 2.0, 4.0, 8.0])
        mask = torch.ones(4, dtype=torch.bool)
        # -1 explicitly excludes the final selected voxel from this grouping.
        groups = torch.tensor([0, 0, 1, -1])

        accumulator = GroupedRegressionAccumulator(("first", "second", "empty"))
        accumulator.update(prediction, target, mask, groups)
        result = accumulator.compute()
        self.assertEqual(result["first"]["count"], 2)
        self.assertAlmostEqual(result["first"]["mae"], 1.0)
        self.assertAlmostEqual(result["first"]["bias"], 1.0)
        self.assertEqual(result["second"]["count"], 1)
        self.assertEqual(result["empty"]["count"], 0)
        self.assertTrue(math.isnan(result["empty"]["pearson_r"]))

        left = GroupedRegressionAccumulator(("first", "second", "empty"))
        right = GroupedRegressionAccumulator(("first", "second", "empty"))
        left.update(prediction[:2], target[:2], mask[:2], groups[:2])
        right.update(prediction[2:], target[2:], mask[2:], groups[2:])
        left.merge(right)
        for label in accumulator.labels:
            for key, expected in result[label].items():
                actual = left.compute()[label][key]
                if isinstance(expected, float) and math.isnan(expected):
                    self.assertTrue(math.isnan(actual))
                elif isinstance(expected, float):
                    self.assertAlmostEqual(actual, expected)
                else:
                    self.assertEqual(actual, expected)

    def test_group_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            GroupedRegressionAccumulator(("same", "same"))
        first = GroupedRegressionAccumulator(("a",))
        second = GroupedRegressionAccumulator(("b",))
        with self.assertRaisesRegex(ValueError, "different labels"):
            first.merge(second)


class StratifiedPrecipitationMetricsTests(unittest.TestCase):
    @staticmethod
    def _example() -> tuple[torch.Tensor, ...]:
        # Rain tensors: (B=1,C=1,D=2,H=1,Z=3). The final voxel is masked.
        target = torch.tensor(
            [[[[[0.5, 2.0, 7.0]], [[20.0, 40.0, 2.0]]]]]
        )
        prediction = target + 1.0
        mask = torch.ones_like(target, dtype=torch.bool)
        mask[..., 1, 0, 2] = False
        height = torch.tensor([0.5, 1.5, 2.5])
        # Channel-free voxel metadata: (B,D,H,Z). It is expanded without copy.
        cfb_distance = torch.tensor(
            [[[[-1.5, -0.5, 0.25]], [[0.0, 1.0, 2.0]]]]
        )
        # Profile metadata: (B,D,H): first profile stratiform, second convective.
        precipitation_type = torch.tensor([[[1.0], [2.0]]])
        return prediction, target, mask, height, cfb_distance, precipitation_type

    def test_exact_height_cfb_intensity_and_type_groups(self) -> None:
        prediction, target, mask, height, distance, precipitation_type = self._example()
        metrics = StratifiedPrecipitationMetrics(height)
        metrics.update_log(
            torch.log1p(prediction),
            torch.log1p(target),
            mask,
            height_km=height,
            cfb_distance_km=distance,
            precipitation_type=precipitation_type,
        )
        result = metrics.compute()

        by_height = result["by_height_km"]
        self.assertEqual(by_height["z_0p5_km"]["count"], 2)
        self.assertEqual(by_height["z_1p5_km"]["count"], 2)
        self.assertEqual(by_height["z_2p5_km"]["count"], 1)
        for values in by_height.values():
            self.assertAlmostEqual(values["mae"], 1.0, places=5)
            self.assertAlmostEqual(values["bias"], 1.0, places=5)

        by_distance = result["by_cfb_distance_km"]
        self.assertEqual(by_distance["lt_m1_km"]["count"], 1)
        self.assertEqual(by_distance["m1_to_0_km"]["count"], 1)
        self.assertEqual(by_distance["0_to_0p5_km"]["count"], 2)
        self.assertEqual(by_distance["0p5_to_2_km"]["count"], 1)
        self.assertEqual(by_distance["ge_2_km"]["count"], 0)

        crossed = result["by_height_and_intensity_mm_h"]
        self.assertEqual(crossed["z_0p5_km"]["lt_1_mm_h"]["count"], 1)
        self.assertEqual(crossed["z_0p5_km"]["10_to_30_mm_h"]["count"], 1)
        self.assertEqual(crossed["z_1p5_km"]["1_to_5_mm_h"]["count"], 1)
        self.assertEqual(crossed["z_1p5_km"]["ge_30_mm_h"]["count"], 1)
        self.assertEqual(crossed["z_2p5_km"]["5_to_10_mm_h"]["count"], 1)

        by_type = result["by_precipitation_type"]
        self.assertEqual(by_type["stratiform"]["count"], 3)
        self.assertEqual(by_type["convective"]["count"], 2)
        self.assertEqual(by_type["other"]["count"], 0)
        self.assertTrue(math.isnan(by_type["other"]["rmse"]))

    def test_height_band_mode_optional_metadata_and_nan_handling(self) -> None:
        prediction, target, mask, height, _, _ = self._example()
        metrics = StratifiedPrecipitationMetrics(
            height_bin_edges_km=(0.0, 1.0, 2.0, 3.0)
        )
        # NaN auxiliary metadata is not treated as a numerical CFB distance or
        # precipitation class, while height statistics remain complete.
        nan_distance = torch.full_like(target, math.nan)
        nan_type = torch.full((1, 2, 1), math.nan)
        metrics.update_rain(
            prediction,
            target,
            mask,
            height_km=height,
            cfb_distance_km=nan_distance,
            precipitation_type=nan_type,
        )
        result = metrics.compute()
        self.assertEqual(result["by_height_km"]["0_to_1_km"]["count"], 2)
        self.assertEqual(result["by_height_km"]["1_to_2_km"]["count"], 2)
        self.assertEqual(result["by_height_km"]["2_to_3_km"]["count"], 1)
        self.assertTrue(
            all(
                values["count"] == 0
                for values in result["by_cfb_distance_km"].values()
            )
        )
        self.assertTrue(
            all(
                values["count"] == 0
                for values in result["by_precipitation_type"].values()
            )
        )

    def test_streaming_merge_matches_one_shot_and_reset(self) -> None:
        prediction, target, mask, height, distance, precipitation_type = self._example()
        expected = StratifiedPrecipitationMetrics(height)
        expected.update_rain(
            prediction,
            target,
            mask,
            height_km=height,
            cfb_distance_km=distance,
            precipitation_type=precipitation_type,
        )

        merged = StratifiedPrecipitationMetrics(height)
        second = StratifiedPrecipitationMetrics(height)
        merged.update_rain(
            prediction[:, :, :1],
            target[:, :, :1],
            mask[:, :, :1],
            height_km=height,
            cfb_distance_km=distance[:, :1],
            precipitation_type=precipitation_type[:, :1],
        )
        second.update_rain(
            prediction[:, :, 1:],
            target[:, :, 1:],
            mask[:, :, 1:],
            height_km=height,
            cfb_distance_km=distance[:, 1:],
            precipitation_type=precipitation_type[:, 1:],
        )
        merged.merge(second)
        expected_result = expected.compute()
        merged_result = merged.compute()
        for label in expected.height_labels:
            self.assertEqual(
                merged_result["by_height_km"][label]["count"],
                expected_result["by_height_km"][label]["count"],
            )
            self.assertAlmostEqual(
                merged_result["by_height_km"][label]["rmse"],
                expected_result["by_height_km"][label]["rmse"],
            )
        merged.reset()
        self.assertTrue(
            all(
                values["count"] == 0
                for values in merged.compute()["by_height_km"].values()
            )
        )

    def test_invalid_configuration_metadata_shape_and_selected_nan(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one"):
            StratifiedPrecipitationMetrics()
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            StratifiedPrecipitationMetrics((1.0, 0.0))
        with self.assertRaisesRegex(ValueError, "positive"):
            StratifiedPrecipitationMetrics((1.0,), intensity_thresholds_mm_h=(0, 1))

        metrics = StratifiedPrecipitationMetrics((1.0,))
        with self.assertRaisesRegex(ValueError, "height_km shape"):
            metrics.update_rain(
                torch.ones(1, 1, 1, 1, 1),
                torch.ones(1, 1, 1, 1, 1),
                torch.ones(1, 1, 1, 1, 1, dtype=torch.bool),
                height_km=torch.ones(2),
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            metrics.update_rain(
                torch.tensor([math.nan]),
                torch.tensor([1.0]),
                torch.tensor([True]),
                height_km=torch.tensor([1.0]),
            )


class FilewisePrecipitationMetricsTests(unittest.TestCase):
    def test_macro_average_per_file_and_reproducible_bootstrap(self) -> None:
        # Tensor meaning: (B=2,C=1,D=1,H=1,Z=2), one batch item per orbit.
        prediction = torch.tensor([2.0, 4.0, 3.0, 5.0]).reshape(2, 1, 1, 1, 2)
        target = torch.tensor([1.0, 5.0, 1.0, 1.0]).reshape_as(prediction)
        mask = torch.ones_like(prediction, dtype=torch.bool)
        first = FilewisePrecipitationMetrics(
            ("orbit_a", "orbit_b", "empty_orbit"),
            bootstrap_seed=17,
            bootstrap_replicates=250,
            confidence_level=0.9,
        )
        first.update_rain(
            prediction, target, mask, file_id=torch.tensor([0, 1])
        )
        result = first.compute()
        self.assertEqual(result["file_count_total"], 3)
        self.assertEqual(result["file_count_nonempty"], 2)
        self.assertEqual(result["valid_voxel_count"], 4)
        self.assertEqual(result["per_file"]["orbit_a"]["count"], 2)
        self.assertEqual(result["per_file"]["empty_orbit"]["count"], 0)
        self.assertAlmostEqual(result["per_file"]["orbit_a"]["rmse"], 1.0)
        self.assertAlmostEqual(
            result["per_file"]["orbit_b"]["rmse"], math.sqrt(10.0)
        )
        self.assertAlmostEqual(result["macro_average"]["mae"], 2.0)
        self.assertAlmostEqual(
            result["macro_average"]["rmse"], (1.0 + math.sqrt(10.0)) / 2.0
        )
        self.assertAlmostEqual(result["macro_average"]["bias"], 1.5)
        # Orbit B has a constant target and undefined r; macro r uses orbit A.
        self.assertAlmostEqual(result["macro_average"]["pearson_r"], 1.0)

        second = FilewisePrecipitationMetrics(
            ("orbit_a", "orbit_b", "empty_orbit"),
            bootstrap_seed=17,
            bootstrap_replicates=250,
            confidence_level=0.9,
        )
        second.update_rain(
            prediction, target, mask, file_id=torch.tensor([0, 1])
        )
        self.assertEqual(result["bootstrap"], second.compute()["bootstrap"])

    def test_log_update_reset_empty_files_and_invalid_ids(self) -> None:
        metrics = FilewisePrecipitationMetrics(
            ("only",), bootstrap_replicates=10
        )
        empty = metrics.compute()
        self.assertEqual(empty["file_count_nonempty"], 0)
        self.assertTrue(math.isnan(empty["macro_average"]["rmse"]))
        self.assertEqual(
            empty["bootstrap"]["confidence_interval"]["rmse"][
                "valid_replicates"
            ],
            0,
        )
        metrics.update_log(
            torch.log1p(torch.tensor([2.0])),
            torch.log1p(torch.tensor([1.0])),
            torch.tensor([True]),
            file_id=torch.tensor([0]),
        )
        self.assertAlmostEqual(metrics.compute()["macro_average"]["mae"], 1.0)
        metrics.reset()
        self.assertEqual(metrics.compute()["valid_voxel_count"], 0)
        with self.assertRaisesRegex(ValueError, "outside"):
            metrics.update_rain(
                torch.ones(1),
                torch.ones(1),
                torch.tensor([True]),
                file_id=torch.tensor([1]),
            )
        with self.assertRaisesRegex(ValueError, "finite integers"):
            metrics.update_rain(
                torch.ones(1),
                torch.ones(1),
                torch.tensor([True]),
                file_id=torch.tensor([math.nan]),
            )


class PhysicalRainGradientMetricsTests(unittest.TestCase):
    def test_nonuniform_height_gradient_values_and_strict_pair_mask(self) -> None:
        # Physical profiles: (...,Z=3). Nonuniform z proves dz is not hard-coded.
        prediction = torch.tensor([2.0, 5.0, 8.0]).reshape(1, 1, 1, 1, 3)
        target = torch.tensor([1.0, 3.0, 9.0]).reshape_as(prediction)
        mask = torch.ones_like(prediction, dtype=torch.bool)
        predicted, observed, pairs, midpoint = physical_vertical_rain_gradient(
            prediction,
            target,
            mask,
            height_km=torch.tensor([0.0, 0.5, 1.5]),
        )
        # prediction dR/dz=[6,3], target dR/dz=[4,6] mm h^-1 km^-1.
        torch.testing.assert_close(
            predicted.flatten(), torch.tensor([6.0, 3.0])
        )
        torch.testing.assert_close(observed.flatten(), torch.tensor([4.0, 6.0]))
        torch.testing.assert_close(midpoint.flatten(), torch.tensor([0.25, 1.0]))
        self.assertTrue(bool(pairs.all()))

        # A missing middle endpoint invalidates both adjacent pairs. The
        # implementation must never bridge directly from level 0 to level 2.
        gap_mask = torch.tensor([True, False, True]).reshape_as(mask)
        _, _, gap_pairs, _ = physical_vertical_rain_gradient(
            prediction,
            target,
            gap_mask,
            height_km=torch.tensor([0.0, 0.5, 1.5]),
        )
        self.assertEqual(int(gap_pairs.sum()), 0)

    def test_overall_grouped_filewise_and_gradient_amplitude_diagnostics(self) -> None:
        prediction = torch.tensor([2.0, 5.0, 8.0]).reshape(1, 1, 1, 1, 3)
        target = torch.tensor([1.0, 3.0, 9.0]).reshape_as(prediction)
        mask = torch.ones_like(prediction, dtype=torch.bool)
        metrics = PhysicalRainGradientMetrics(
            (0.0, 0.5, 1.5),
            cfb_distance_edges_km=(0.0, 1.0),
            intensity_thresholds_mm_h=(1.0, 5.0),
            sign_epsilon_mm_h_km=0.1,
            file_labels=("orbit",),
            bootstrap_replicates=25,
        )
        metrics.update_rain(
            prediction,
            target,
            mask,
            height_km=torch.tensor([0.0, 0.5, 1.5]),
            cfb_distance_km=torch.tensor([-0.5, 0.0, 1.0]),
            precipitation_type=torch.tensor([[[1.0]]]),
            file_id=torch.tensor([0]),
        )
        result = metrics.compute()
        overall = result["all"]
        self.assertEqual(overall["count"], 2)
        self.assertAlmostEqual(overall["mae"], 2.5)
        self.assertAlmostEqual(overall["rmse"], math.sqrt(6.5))
        self.assertAlmostEqual(overall["bias"], -0.5)
        self.assertAlmostEqual(overall["pearson_r"], -1.0)
        self.assertAlmostEqual(overall["mean_abs_gradient_ratio"], 0.9)
        self.assertAlmostEqual(overall["sign_agreement_fraction"], 1.0)
        self.assertEqual(
            result["by_midpoint_height_km"]["z_0p25_km"]["count"], 1
        )
        self.assertEqual(
            result["by_midpoint_height_km"]["z_1_km"]["count"], 1
        )
        self.assertEqual(
            result["by_pair_mean_target_intensity_mm_h"]["1_to_5_mm_h"][
                "count"
            ],
            1,
        )
        self.assertEqual(
            result["by_pair_mean_target_intensity_mm_h"]["ge_5_mm_h"][
                "count"
            ],
            1,
        )
        self.assertEqual(result["by_precipitation_type"]["stratiform"]["count"], 2)
        self.assertEqual(result["filewise"]["valid_voxel_count"], 2)

    def test_log_conversion_empty_pairs_and_invalid_height_contract(self) -> None:
        metrics = PhysicalRainGradientMetrics((0.0, 1.0, 2.0))
        target_rain = torch.tensor([1.0, 2.0, 4.0]).reshape(1, 1, 1, 1, 3)
        # Negative log prediction becomes zero physical rain before dR/dz.
        prediction_log = torch.full_like(target_rain, -2.0)
        mask = torch.tensor([True, True, False]).reshape_as(target_rain)
        metrics.update_log(
            prediction_log,
            torch.log1p(target_rain),
            mask,
            height_km=torch.tensor([0.0, 1.0, 2.0]),
        )
        result = metrics.compute()["all"]
        self.assertEqual(result["count"], 1)
        self.assertAlmostEqual(result["bias"], -1.0)

        metrics.reset()
        metrics.update_rain(
            target_rain,
            target_rain,
            torch.tensor([True, False, True]).reshape_as(target_rain),
            height_km=torch.tensor([0.0, 1.0, 2.0]),
        )
        self.assertEqual(metrics.compute()["all"]["count"], 0)
        with self.assertRaisesRegex(ValueError, "fixed physical"):
            metrics.update_rain(
                target_rain,
                target_rain,
                torch.ones_like(target_rain, dtype=torch.bool),
                height_km=torch.tensor([0.0, 1.0, 3.0]),
            )
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            PhysicalRainGradientMetrics((0.0, 1.0, 1.0))
        with self.assertRaisesRegex(TypeError, "boolean"):
            physical_vertical_rain_gradient(
                target_rain,
                target_rain,
                torch.ones_like(target_rain),
                height_km=torch.tensor([0.0, 1.0, 2.0]),
            )


if __name__ == "__main__":
    unittest.main()
