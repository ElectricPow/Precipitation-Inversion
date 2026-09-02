"""Tests for S2-R0 static and frozen-cascade decomposition helpers."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.analyze_stage2_task_decomposition import (  # noqa: E402
    R0_DIAGNOSTICS_FORMAT,
    R0_LOCAL_SHIFT_FORMAT,
    R0_ORACLE_AUDIT_FORMAT,
    build_alignment_oracle_summary,
    build_stage2_region_error_budget,
    plot_r0_summary,
    validate_complete_validation_evaluation,
    validate_r0_local_shift_audit,
    validate_r0_oracle_audit,
)
from scripts.evaluate_stage2_stage1_cascade import (  # noqa: E402
    R0_DIAGNOSTICS_FORMAT as CASCADE_R0_DIAGNOSTICS_FORMAT,
    R0_ORACLE_AUDIT_FORMAT as CASCADE_R0_ORACLE_AUDIT_FORMAT,
    R0_SAMPLE_ID_HASH_CONTRACT,
    build_mode_definitions,
    build_r0_subtask_masks,
    build_r0_decomposition_oracle_audit,
    ordered_sample_ids_sha256,
    parse_stage2_run_specs,
    r0_decomposition_oracle_rows,
    replace_stage2_dbz_with_oracle_region,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _region_row(
    region: str,
    *,
    target_count: int = 0,
    rmse: float = math.nan,
    fn: int = 0,
    fp: int = 0,
) -> dict[str, object]:
    return {
        "region": region,
        "reflectivity_on_target_support_count": target_count,
        "reflectivity_on_target_support_mae_dbz": rmse / 2 if math.isfinite(rmse) else "",
        "reflectivity_on_target_support_rmse_dbz": rmse if math.isfinite(rmse) else "",
        "reflectivity_on_target_support_bias_dbz": 0.0 if target_count else "",
        "reflectivity_on_target_support_pearson_r": 0.5 if target_count else "",
        "support_false_negative": fn,
        "support_false_positive": fp,
        "support_recall": 0.5 if target_count else "",
        "support_false_alarm_ratio": 0.2 if fp else "",
    }


def _cascade_result(value: float) -> dict[str, object]:
    rain_names = ("mae", "rmse", "bias", "r2", "pearson_r", "ccc")
    drdz_names = rain_names + (
        "mean_abs_gradient_ratio",
        "sign_agreement_fraction",
    )
    rain = {name: value for name in rain_names}
    drdz = {name: value for name in drdz_names}
    return {
        "reliable_positive": {"rain": {"all": dict(rain)}},
        "qc_label_domain_including_zero": {"rain": {"all": dict(rain)}},
        "physical_drdz_reliable_positive": {"all": drdz},
    }


def _oracle_validation_fixture(
    root: Path,
    *,
    file_count: int = 2,
    declared_formal: bool | None = None,
) -> tuple[dict[str, object], dict[str, object], dict[str, object], Path]:
    """Create a complete on-disk provenance graph for strict validator tests."""

    root.mkdir(parents=True, exist_ok=True)
    sample_ids = ["orbit-a", "orbit-b"]
    selected_ids = sample_ids[:file_count]
    patch_index = {
        "stage": 2,
        "split": "val",
        "file_count": len(sample_ids),
        "files": [{"sample_id": value} for value in sample_ids],
    }
    patch_index_path = root / "stage2_val.json"
    patch_index_path.write_text(json.dumps(patch_index), encoding="utf-8")
    stage1_index = root / "stage1_val.json"
    stage1_index.write_text(json.dumps({"files": sample_ids}), encoding="utf-8")
    stage1_checkpoint = root / "stage1.pt"
    stage1_checkpoint.write_bytes(b"sealed-stage1")
    stage2_checkpoint = root / "stage2.pt"
    stage2_checkpoint.write_bytes(b"w1p25-stage2")
    threshold_file = root / "support_threshold.json"
    threshold_file.write_text(
        json.dumps(
            {"threshold": 0.8, "selected_on_split": "val"},
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    formal = file_count == len(sample_ids) if declared_formal is None else declared_formal
    provenance = {
        "split": "val",
        "file_count": file_count,
        "expected_file_count": len(sample_ids),
        "formal_complete_validation": formal,
        "sample_ids": selected_ids,
        "sample_ids_sha256": ordered_sample_ids_sha256(selected_ids),
        "sample_id_hash_contract": R0_SAMPLE_ID_HASH_CONTRACT,
        "stage1": {
            "checkpoint_path": str(stage1_checkpoint),
            "checkpoint_sha256": _sha256(stage1_checkpoint),
            "checkpoint_epoch": 22,
            "index_path": str(stage1_index),
            "index_sha256": _sha256(stage1_index),
        },
        "stage2_runs": {
            "w1_25": {
                "label": "W1.25",
                "checkpoint_path": str(stage2_checkpoint),
                "checkpoint_sha256": _sha256(stage2_checkpoint),
                "checkpoint_epoch": 27,
                "support_threshold": 0.8,
                "threshold_file_path": str(threshold_file),
                "threshold_file_sha256": _sha256(threshold_file),
                "index_path": str(patch_index_path),
                "index_sha256": _sha256(patch_index_path),
            }
        },
    }
    oracle = {
        "format": R0_ORACLE_AUDIT_FORMAT,
        "provenance": provenance,
        "runs": {"w1_25": {"label": "W1.25", "metrics": {}}},
    }
    metrics = {
        "stage": 2,
        "split": "val",
        "file_count": len(sample_ids),
        "checkpoint": str(stage2_checkpoint),
        "checkpoint_epoch": 27,
        "support_threshold": 0.8,
    }
    return oracle, metrics, patch_index, patch_index_path


class Stage2StaticDecompositionTests(unittest.TestCase):
    def test_complete_validation_contract_rejects_test_or_partial(self) -> None:
        index = {"stage": 2, "split": "val", "file_count": 3}
        metrics = {
            "stage": 2,
            "split": "val",
            "file_count": 3,
            "checkpoint": "best.pt",
        }
        result = validate_complete_validation_evaluation(metrics, index)
        self.assertTrue(result["complete_validation"])
        with self.assertRaisesRegex(ValueError, "split=val"):
            validate_complete_validation_evaluation(
                {**metrics, "split": "test"}, index
            )
        with self.assertRaisesRegex(ValueError, "2/3"):
            validate_complete_validation_evaluation(
                {**metrics, "file_count": 2}, index
            )
        partial = validate_complete_validation_evaluation(
            {**metrics, "file_count": 2}, index, allow_incomplete=True
        )
        self.assertFalse(partial["complete_validation"])

    def test_oracle_schema_accepts_only_matching_complete_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            oracle, metrics, index, index_path = _oracle_validation_fixture(
                Path(temporary)
            )
            result = validate_r0_oracle_audit(
                oracle,
                metrics,
                index,
                patch_index_path=index_path,
            )
            self.assertTrue(result["complete_validation"])
            self.assertTrue(result["accepted_as_formal"])
            self.assertEqual(result["matched_stage2_run"], "w1_25")
            self.assertEqual(result["evaluated_file_count"], 2)

            diagnostics = dict(oracle)
            diagnostics["format"] = R0_DIAGNOSTICS_FORMAT
            with self.assertRaisesRegex(ValueError, "diagnostics schema"):
                validate_r0_oracle_audit(
                    diagnostics,
                    metrics,
                    index,
                    patch_index_path=index_path,
                )

            bad_hash = json.loads(json.dumps(oracle))
            bad_hash["provenance"]["sample_ids_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "sample_ids_sha256"):
                validate_r0_oracle_audit(
                    bad_hash,
                    metrics,
                    index,
                    patch_index_path=index_path,
                )

    def test_oracle_smoke_requires_opt_in_and_is_never_formal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            oracle, metrics, index, index_path = _oracle_validation_fixture(
                Path(temporary), file_count=1
            )
            with self.assertRaisesRegex(ValueError, "1/2"):
                validate_r0_oracle_audit(
                    oracle,
                    metrics,
                    index,
                    patch_index_path=index_path,
                )
            result = validate_r0_oracle_audit(
                oracle,
                metrics,
                index,
                patch_index_path=index_path,
                allow_incomplete=True,
            )
            self.assertFalse(result["complete_validation"])
            self.assertFalse(result["accepted_as_formal"])

            falsely_formal, _, _, _ = _oracle_validation_fixture(
                Path(temporary) / "false-formal",
                file_count=1,
                declared_formal=True,
            )
            with self.assertRaisesRegex(ValueError, "inconsistent with orbit coverage"):
                validate_r0_oracle_audit(
                    falsely_formal,
                    metrics,
                    index,
                    patch_index_path=index_path,
                    allow_incomplete=True,
                )

    def test_oracle_checkpoint_and_threshold_must_match_static_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            oracle, metrics, index, index_path = _oracle_validation_fixture(root)
            foreign_checkpoint = root / "foreign_stage2.pt"
            foreign_checkpoint.write_bytes(b"different-stage2")
            mismatched_metrics = {**metrics, "checkpoint": str(foreign_checkpoint)}
            with self.assertRaisesRegex(ValueError, "exactly one Stage-2 run"):
                validate_r0_oracle_audit(
                    oracle,
                    mismatched_metrics,
                    index,
                    patch_index_path=index_path,
                )

            mismatched_threshold = {**metrics, "support_threshold": 0.7}
            with self.assertRaisesRegex(ValueError, "threshold differs"):
                validate_r0_oracle_audit(
                    oracle,
                    mismatched_threshold,
                    index,
                    patch_index_path=index_path,
                )

    def test_local_shift_attachment_requires_same_ordered_validation_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "split_manifest.csv"
            manifest.write_text("sample_id,file_path,split\na,x,val\nb,y,val\n", encoding="utf-8")
            sample_ids = ["a", "b"]
            index = {
                "stage": 2,
                "split": "val",
                "file_count": 2,
                "files": [{"sample_id": value} for value in sample_ids],
            }
            payload = {
                "format": R0_LOCAL_SHIFT_FORMAT,
                "split": "val",
                "test_set_accessed": False,
                "formal_result": True,
                "selected_file_count": 2,
                "expected_validation_file_count": 2,
                "split_manifest": str(manifest),
                "split_manifest_sha256": _sha256(manifest),
                "sample_ids": sample_ids,
                "sample_ids_sha256": ordered_sample_ids_sha256(sample_ids),
                "sample_id_hash_contract": R0_SAMPLE_ID_HASH_CONTRACT,
                "metrics": {
                    "exact_support_csi": 0.4,
                    "one_shift_all_validation_oracle_support_csi": 0.41,
                    "per_orbit_height_oracle_support_csi": 0.43,
                    "local_window_height_oracle_support_csi": 0.46,
                    "orbit_count": 2,
                    "opposing_local_shift_orbit_count": 2,
                    "cancellation_evidence_orbit_count": 1,
                },
            }
            result = validate_r0_local_shift_audit(payload, index)
            self.assertTrue(result["accepted_as_formal"])

            wrong_order = json.loads(json.dumps(payload))
            wrong_order["sample_ids"] = ["b", "a"]
            wrong_order["sample_ids_sha256"] = ordered_sample_ids_sha256(["b", "a"])
            with self.assertRaisesRegex(ValueError, "ordered sample IDs"):
                validate_r0_local_shift_audit(wrong_order, index)

    def test_region_budget_checks_partitions_and_sse_contributions(self) -> None:
        q01_rmse = math.sqrt(44.0 / 6.0)
        all_rmse = math.sqrt(48.0 / 10.0)
        rows = [
            _region_row("all_domain", target_count=10, rmse=all_rmse, fn=4, fp=5),
            _region_row("q11_direct_overlap", target_count=4, rmse=1.0, fn=1),
            _region_row("q01_direct_missing", target_count=6, rmse=q01_rmse, fn=3),
            _region_row("q10_gr_only", fp=2),
            _region_row("q00_neither", fp=3),
            _region_row("dpr_gap_proxy", target_count=2, rmse=2.0, fn=1),
            _region_row("dpr_outside_proxy", target_count=4, rmse=3.0, fn=2),
            _region_row("dpr_dbz_ge35", target_count=2, rmse=3.0, fn=1),
        ]
        budget, summary = build_stage2_region_error_budget(rows)
        by_name = {row["region"]: row for row in budget}

        self.assertAlmostEqual(
            by_name["q11_direct_overlap"]["target_fraction"], 0.4
        )
        self.assertAlmostEqual(
            by_name["dpr_outside_proxy"]["squared_error_fraction"], 36.0 / 48.0
        )
        self.assertAlmostEqual(
            by_name["dpr_outside_proxy"]["false_negative_fraction"], 0.5
        )
        self.assertTrue(summary["partition_checks"]["gap_plus_outside_equals_q01"])

        broken = [dict(row) for row in rows]
        next(row for row in broken if row["region"] == "dpr_gap_proxy")[
            "reflectivity_on_target_support_count"
        ] = 3
        with self.assertRaisesRegex(ValueError, r"gap\+outside"):
            build_stage2_region_error_budget(broken)

    def test_alignment_oracle_compares_best_with_zero_at_each_height(self) -> None:
        best = [
            {
                "group": "val",
                "height_index": "0",
                "height_km": "0.25",
                "best_scan_shift": "0",
                "best_ray_shift": "0",
                "best_support_csi": "0.4",
                "pearson_at_best_support_shift": "0.7",
                "overlap_count": "10",
            },
            {
                "group": "val",
                "height_index": "1",
                "height_km": "0.75",
                "best_scan_shift": "1",
                "best_ray_shift": "0",
                "best_support_csi": "0.5",
                "pearson_at_best_support_shift": "0.8",
                "overlap_count": "12",
            },
        ]
        shifts = [
            {
                "group": "val",
                "height_index": str(index),
                "scan_shift": "0",
                "ray_shift": "0",
                "support_csi": value,
                "pearson_r": "0.6",
            }
            for index, value in ((0, "0.4"), (1, "0.45"))
        ]
        rows, summary = build_alignment_oracle_summary(best, shifts)
        self.assertEqual(summary["nonzero_best_height_count"], 1)
        self.assertAlmostEqual(summary["max_support_csi_gain"], 0.05)
        self.assertTrue(rows[0]["best_is_zero_shift"])

    def test_summary_plotter_supports_static_local_and_two_oracle_components(self) -> None:
        region_names = (
            "q11_direct_overlap",
            "dpr_gap_proxy",
            "dpr_outside_proxy",
            "dpr_dbz_ge35",
        )
        regions = [
            {
                "region": name,
                "target_fraction": 0.25,
                "squared_error_fraction": 0.25,
                "false_negative_fraction": 0.25,
            }
            for name in region_names
        ]

        def component(region: str) -> dict[str, object]:
            return {
                "regions": {region: "synthetic"},
                "metrics": {
                    "reliable_positive_rain": {
                        "rmse": {
                            "regions": {
                                region: {
                                    "fraction_of_baseline_to_reference_gap_closed": 0.5
                                }
                            }
                        }
                    }
                },
            }

        summary = {
            "regions": regions,
            "local_shift_oracle": {
                "metrics": {
                    "exact_support_csi": 0.29,
                    "one_shift_all_validation_oracle_support_csi": 0.29,
                    "per_orbit_single_shift_oracle_support_csi": 0.291,
                    "per_orbit_height_oracle_support_csi": 0.295,
                    "local_window_height_oracle_support_csi": 0.301,
                }
            },
            "oracle_replacement": {
                "runs": {
                    "w1_25": {
                        "value_oracle": component("q11"),
                        "support_oracle": component("q00"),
                    }
                }
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            paths = plot_r0_summary(output, summary)
            self.assertEqual(len(paths), 3)
            self.assertTrue(all((output / path).is_file() for path in paths))


class Stage2CascadeDecompositionTests(unittest.TestCase):
    def _spec(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        checkpoint = root / "best.pt"
        checkpoint.write_bytes(b"placeholder")
        threshold = root / "threshold.json"
        threshold.write_text(
            json.dumps({"threshold": 0.8, "selected_on_split": "val"}),
            encoding="utf-8",
        )
        return parse_stage2_run_specs(
            [["W1.25", str(checkpoint), str(threshold)]]
        )[0]

    def test_oracle_replacement_changes_only_selected_true_support(self) -> None:
        prediction = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
        target = prediction + 100.0
        support = np.ones_like(prediction, dtype=bool)
        selected = np.zeros_like(support)
        selected[0, 1, 1] = True
        result = replace_stage2_dbz_with_oracle_region(
            prediction, target, support, selected
        )
        np.testing.assert_allclose(result[~selected], prediction[~selected])
        np.testing.assert_allclose(result[selected], target[selected])
        invalid = selected.copy()
        support[0, 1, 1] = False
        with self.assertRaisesRegex(ValueError, "subset"):
            replace_stage2_dbz_with_oracle_region(
                prediction, target, support, invalid
            )

    def test_r0_routing_is_full_gr_only_and_supervision_uses_native_domain(self) -> None:
        shape = (1, 2, 4)
        gr_sparse = np.array(
            [1, 0, 0, 1, 0, 0, 1, 0], dtype=bool
        ).reshape(shape)
        gr_interp = np.array(
            [1, 1, 0, 1, 1, 0, 1, 1], dtype=bool
        ).reshape(shape)
        native_a = np.array(
            [1, 1, 1, 1, 0, 0, 0, 0], dtype=bool
        ).reshape(shape)
        dpr_a = np.array(
            [1, 1, 0, 0, 1, 1, 0, 0], dtype=bool
        ).reshape(shape)
        dbz_a = np.array(
            [10.0, 40.0, np.nan, np.nan, 20.0, 45.0, np.nan, np.nan],
            dtype=np.float32,
        ).reshape(shape)
        source_a = {
            "gr_sparse_valid": gr_sparse,
            "gr_interp_valid": gr_interp,
            "pre_valid_native_mask": native_a,
            # Deliberately different: this mask is for final-rain metrics and
            # must not crop either R0 routing or Stage-2 supervision.
            "qc_label_mask": np.zeros(shape, dtype=bool),
            "dpr_valid": dpr_a,
            "dbz_dpr": dbz_a,
        }
        masks_a = build_r0_subtask_masks(source_a)

        # Change every target-side quantity while preserving only the two GR
        # masks.  The returned deployment mapping must remain bit-identical.
        native_b = ~native_a
        dpr_b = np.array(
            [0, 0, 1, 1, 0, 0, 1, 1], dtype=bool
        ).reshape(shape)
        dbz_b = np.full(shape, 50.0, dtype=np.float32)
        source_b = {
            **source_a,
            "pre_valid_native_mask": native_b,
            "qc_label_mask": np.ones(shape, dtype=bool),
            "dpr_valid": dpr_b,
            "dbz_dpr": dbz_b,
        }
        masks_b = build_r0_subtask_masks(source_b)

        for name, values in masks_a.routing.as_dict().items():
            np.testing.assert_array_equal(values, masks_b.routing.as_dict()[name])
        self.assertTrue(np.all(masks_a.routing.domain))
        self.assertEqual(
            set(masks_a.routing.as_dict()),
            {
                "gr_routing_domain",
                "gr_route_observed",
                "gr_route_near",
                "gr_route_far",
            },
        )

        # All target-derived partitions stay strictly inside their own native
        # label domain even though GR routing covers the full orbit.
        for masks, native in ((masks_a, native_a), (masks_b, native_b)):
            np.testing.assert_array_equal(masks.label_domain, native)
            supervision = masks.as_dict()
            self.assertTrue(
                all(not np.any(values & ~native) for values in supervision.values())
            )
            q_union = (
                masks.q11_overlap
                | masks.q10_gr_only
                | masks.q01_dpr_only
                | masks.q00_neither
            )
            np.testing.assert_array_equal(q_union, native)

    def test_r0_modes_and_closed_gap_fraction_are_complete(self) -> None:
        spec = self._spec()
        self.assertEqual(R0_ORACLE_AUDIT_FORMAT, CASCADE_R0_ORACLE_AUDIT_FORMAT)
        self.assertEqual(R0_DIAGNOSTICS_FORMAT, CASCADE_R0_DIAGNOSTICS_FORMAT)
        self.assertNotEqual(R0_ORACLE_AUDIT_FORMAT, R0_DIAGNOSTICS_FORMAT)
        modes = build_mode_definitions(
            [spec],
            include_gr_interp=False,
            include_r0_decomposition_oracles=True,
        )
        for region in ("q11", "q01", "gap", "outside", "strong_ge35"):
            self.assertIn(f"w1_25_oracle_value_{region}", modes)
        for region in (
            "q11",
            "q01",
            "gap",
            "outside",
            "q10",
            "q00",
            "strong_ge35",
        ):
            self.assertIn(f"w1_25_oracle_support_{region}", modes)

        computed = {
            "dpr_oracle": _cascade_result(2.0),
            "w1_25_oracle_mask": _cascade_result(4.0),
            "w1_25_predicted_mask": _cascade_result(6.0),
        }
        for region in ("q11", "q01", "gap", "outside", "strong_ge35"):
            computed[f"w1_25_oracle_value_{region}"] = _cascade_result(3.0)
        for region in (
            "q11",
            "q01",
            "gap",
            "outside",
            "q10",
            "q00",
            "strong_ge35",
        ):
            computed[f"w1_25_oracle_support_{region}"] = _cascade_result(5.0)
        sample_ids = ["orbit-a"]
        provenance = {
            "split": "val",
            "file_count": 1,
            "expected_file_count": 1,
            "formal_complete_validation": True,
            "sample_ids": sample_ids,
            "sample_ids_sha256": ordered_sample_ids_sha256(sample_ids),
            "sample_id_hash_contract": R0_SAMPLE_ID_HASH_CONTRACT,
            "stage1": {},
            "stage2_runs": {spec.slug: {"label": spec.label}},
        }
        audit = build_r0_decomposition_oracle_audit(
            [spec], computed, provenance=provenance
        )
        self.assertEqual(audit["format"], R0_ORACLE_AUDIT_FORMAT)
        self.assertEqual(audit["provenance"], provenance)
        rmse = audit["runs"]["w1_25"]["metrics"][
            "reliable_positive_rain"
        ]["rmse"]
        self.assertAlmostEqual(
            rmse["regions"]["outside"][
                "fraction_of_baseline_to_reference_gap_closed"
            ],
            0.5,
        )
        support_rmse = audit["runs"]["w1_25"]["support_oracle"]["metrics"][
            "reliable_positive_rain"
        ]["rmse"]
        self.assertAlmostEqual(
            support_rmse["regions"]["q00"][
                "fraction_of_baseline_to_reference_gap_closed"
            ],
            0.5,
        )
        rows = r0_decomposition_oracle_rows(audit)
        self.assertEqual(len(rows), 240)
        self.assertEqual({row["oracle_component"] for row in rows}, {"value", "support"})


if __name__ == "__main__":
    unittest.main()
