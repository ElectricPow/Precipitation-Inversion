"""Tests for the preregistered C1 rain-only objective."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from precipitation_inversion.losses.stage3_losses import (  # noqa: E402
    build_stage3_c1_loss,
)


class Stage3LossTests(unittest.TestCase):
    def test_c1_requires_i_plus_exact_g002(self) -> None:
        criterion = build_stage3_c1_loss(
            {
                "name": "masked_smooth_l1",
                "beta": 0.2,
                "physical_gradient": {"enabled": True, "weight": 0.02, "beta": 1.0},
            }
        )
        self.assertAlmostEqual(criterion.physical_gradient_weight, 0.02)
        with self.assertRaisesRegex(ValueError, "exactly 0.02"):
            build_stage3_c1_loss(
                {
                    "name": "masked_smooth_l1",
                    "beta": 0.2,
                    "physical_gradient": {
                        "enabled": True,
                        "weight": 0.01,
                        "beta": 1.0,
                    },
                }
            )


if __name__ == "__main__":
    unittest.main()
