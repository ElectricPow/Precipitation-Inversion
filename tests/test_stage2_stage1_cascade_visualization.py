"""Synthetic report test for reusable two-stage orbit visualization."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.visualize_stage2_stage1_cascade import (  # noqa: E402
    EXPECTED_FORMAT,
    compute_shared_geographic_extent,
    visualize_cascade_orbit_bundle,
)


class CascadeVisualizationTests(unittest.TestCase):
    def test_saved_orbit_produces_consistent_png_pdf_and_manifest(self) -> None:
        nscan, nray, z_size = 5, 4, 6
        z = np.linspace(0.5, 5.5, z_size, dtype=np.float32)
        lat = np.linspace(20.0, 21.0, nscan)[:, None] + np.zeros((nscan, nray))
        lon = np.linspace(110.0, 111.0, nray)[None, :] + np.zeros((nscan, nray))
        target = np.zeros((nscan, nray, z_size), dtype=np.float32)
        target[1:4, 1:3, :4] = np.linspace(0.1, 8.0, 24).reshape(3, 2, 4)
        qc = np.ones_like(target, dtype=bool)
        positive = target > 0.0
        first = target * 0.9
        second = target * 0.7
        first_support = np.ones_like(target, dtype=bool)
        second_support = positive.copy()
        modes = [
            {
                "slug": "oracle",
                "display_name": "Oracle",
                "rain_field": "rain__oracle",
                "input_support_field": "input_support__oracle",
                "output_support_field": "output_support__oracle",
            },
            {
                "slug": "deploy",
                "display_name": "Deployable",
                "rain_field": "rain__deploy",
                "input_support_field": "input_support__deploy",
                "output_support_field": "output_support__deploy",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "bundle"
            output = root / "figures"
            bundle.mkdir()
            np.savez_compressed(
                bundle / "fields.npz",
                target_rain_mm_h=target,
                reliable_positive_mask=positive,
                qc_label_mask=qc,
                heights_km=z,
                lat=lat,
                lon=lon,
                cfb=np.zeros((nscan, nray), dtype=np.float32),
                precipitation_type=np.ones((nscan, nray), dtype=np.float32),
                true_dpr_support=positive,
                rain__oracle=first,
                input_support__oracle=first_support,
                output_support__oracle=first_support,
                rain__deploy=second,
                input_support__deploy=second_support,
                output_support__deploy=second_support,
            )
            metadata = {
                "format": EXPECTED_FORMAT,
                "sample_id": "synthetic",
                "source_file": "/synthetic.nc",
                "fields_file": "fields.npz",
                "modes": modes,
            }
            metadata_path = bundle / "metadata.json"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            result = visualize_cascade_orbit_bundle(
                metadata_path,
                output_dir=output,
                height_km=2.0,
                max_points=1000,
                dpi=25,
                seed=7,
            )
            self.assertEqual(result["sample_id"], "synthetic")
            self.assertEqual(set(result["modes"]), {"oracle", "deploy"})
            expected_extent = compute_shared_geographic_extent(
                lon, lat, np.any(qc, axis=-1)
            )
            np.testing.assert_allclose(
                result["shared_geographic_extent_lon_lat"], expected_extent
            )
            for mode in result["modes"].values():
                np.testing.assert_allclose(
                    mode["geographic_extent"], expected_extent
                )
            self.assertTrue((output / "diagnostics.pdf").is_file())
            self.assertTrue((output / "figure_manifest.md").is_file())
            self.assertTrue((output / ".complete").is_file())
            pngs = sorted((output / "comparisons").glob("*.png"))
            self.assertEqual(len(pngs), 4)
            self.assertTrue(all(path.stat().st_size > 0 for path in pngs))

    def test_extent_uses_qc_footprint_not_method_support(self) -> None:
        lon = np.array([[100.0, 101.0, 102.0], [100.0, 101.0, 102.0]])
        lat = np.array([[20.0, 20.0, 20.0], [21.0, 21.0, 21.0]])
        qc = np.ones_like(lon, dtype=bool)
        extent = compute_shared_geographic_extent(
            lon, lat, qc, padding_fraction=0.0
        )
        self.assertEqual(extent, (100.0, 102.0, 20.0, 21.0))
        # A method occupying only the center cell must still use this full QC
        # extent; method support is deliberately absent from the helper API.


if __name__ == "__main__":
    unittest.main()
