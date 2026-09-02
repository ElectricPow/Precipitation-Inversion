"""Tests for validation-only regional Stage-2 counterfactual inputs."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from precipitation_inversion.inference.stage2_oracles import (  # noqa: E402
    build_regional_oracle_input,
)


class Stage2RegionalOracleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.prediction = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
        self.predicted_support = np.array(
            [[[True, False], [True, True]], [[False, True], [True, False]]]
        )
        self.target = self.prediction + 100.0
        self.target_support = np.array(
            [[[True, True], [False, True]], [[False, True], [False, False]]]
        )

    def test_value_oracle_changes_only_selected_true_echo_values(self) -> None:
        region = np.zeros_like(self.target_support)
        region[0, 0, 0:2] = True
        original_prediction = self.prediction.copy()
        original_support = self.predicted_support.copy()
        result = build_regional_oracle_input(
            self.prediction,
            self.predicted_support,
            self.target,
            self.target_support,
            region,
            component="value",
        )
        np.testing.assert_allclose(result.reflectivity_dbz[region], self.target[region])
        np.testing.assert_allclose(
            result.reflectivity_dbz[~region], self.prediction[~region]
        )
        np.testing.assert_array_equal(
            result.reflectivity_support, self.predicted_support
        )
        np.testing.assert_array_equal(self.prediction, original_prediction)
        np.testing.assert_array_equal(self.predicted_support, original_support)

    def test_support_oracle_repairs_false_negative_and_false_positive(self) -> None:
        region = np.zeros_like(self.target_support)
        region[0, 0, 1] = True  # predicted 0, target 1
        region[0, 1, 0] = True  # predicted 1, target 0
        result = build_regional_oracle_input(
            self.prediction,
            self.predicted_support,
            self.target,
            self.target_support,
            region,
            component="support",
        )
        np.testing.assert_array_equal(
            result.reflectivity_support[region], self.target_support[region]
        )
        np.testing.assert_array_equal(
            result.reflectivity_support[~region], self.predicted_support[~region]
        )
        np.testing.assert_allclose(result.reflectivity_dbz, self.prediction)

    def test_joint_oracle_changes_support_and_only_defined_target_values(self) -> None:
        region = np.zeros_like(self.target_support)
        region[0, 0, 1] = True  # true echo: expose and replace its dBZ
        region[0, 1, 0] = True  # true no-echo: remove support, keep unused dBZ
        result = build_regional_oracle_input(
            self.prediction,
            self.predicted_support,
            self.target,
            self.target_support,
            region,
            component="joint",
        )
        self.assertTrue(result.reflectivity_support[0, 0, 1])
        self.assertFalse(result.reflectivity_support[0, 1, 0])
        self.assertEqual(result.reflectivity_dbz[0, 0, 1], self.target[0, 0, 1])
        self.assertEqual(
            result.reflectivity_dbz[0, 1, 0], self.prediction[0, 1, 0]
        )

    def test_value_oracle_rejects_no_echo_and_support_rejects_nan_exposure(self) -> None:
        no_echo = np.zeros_like(self.target_support)
        no_echo[0, 1, 0] = True
        with self.assertRaisesRegex(ValueError, "subset"):
            build_regional_oracle_input(
                self.prediction,
                self.predicted_support,
                self.target,
                self.target_support,
                no_echo,
                component="value",
            )

        missing_prediction = self.prediction.copy()
        missing_prediction[0, 0, 1] = np.nan
        exposes_echo = np.zeros_like(self.target_support)
        exposes_echo[0, 0, 1] = True
        with self.assertRaisesRegex(ValueError, "non-finite"):
            build_regional_oracle_input(
                missing_prediction,
                self.predicted_support,
                self.target,
                self.target_support,
                exposes_echo,
                component="support",
            )


if __name__ == "__main__":
    unittest.main()
