"""Tests for deterministic and leakage-safe manifest splitting."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from precipitation_inversion.data.splits import (  # noqa: E402
    DEFAULT_BALANCE_WEIGHTS,
    allocate_group_counts,
    assert_valid_split,
    balanced_group_split,
    chronological_group_split,
    normalized_ratios,
)


def synthetic_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    # Two files per date. Rain statistics vary strongly to exercise balancing.
    for date_index in range(12):
        for file_index in range(2):
            scale = date_index + 1
            records.append(
                {
                    "sample_id": f"sample-{date_index:02d}-{file_index}",
                    "date": f"2017-05-{date_index + 1:02d}",
                    "file_count": 1,
                    "total_voxel_count": 1_000 + 10 * scale,
                    "gr_sparse_valid_count": 100 + 3 * scale,
                    "pre_positive_count": 50 * scale,
                    "pre_gt_10_count": 10 * scale,
                    "pre_gt_20_count": 3 * scale,
                    "pre_gt_50_count": scale if date_index % 3 == 0 else 0,
                    "stratiform_profile_count": 20 * scale,
                    "convective_profile_count": 5 * scale,
                }
            )
    return records


class SplitTests(unittest.TestCase):
    def test_ratios_are_normalized_without_changing_order(self) -> None:
        names, ratios = normalized_ratios({"train": 7, "val": 2, "test": 1})
        self.assertEqual(names, ("train", "val", "test"))
        np.testing.assert_allclose(ratios, [0.7, 0.2, 0.1])

    def test_group_counts_sum_and_keep_every_split_nonempty(self) -> None:
        counts = allocate_group_counts(7, [0.98, 0.01, 0.01])
        self.assertEqual(int(counts.sum()), 7)
        self.assertTrue(np.all(counts >= 1))

    def test_balanced_split_is_deterministic_and_complete(self) -> None:
        records = synthetic_records()
        first = balanced_group_split(records, seed=42, trials=500)
        second = balanced_group_split(records, seed=42, trials=500)
        self.assertEqual(first.group_assignments, second.group_assignments)
        self.assertEqual(first.record_assignments, second.record_assignments)
        self.assertEqual(first.score, second.score)
        assert_valid_split(records, first)
        self.assertEqual(len(first.record_assignments), len(records))

    def test_same_date_never_crosses_splits(self) -> None:
        records = synthetic_records()
        result = balanced_group_split(records, seed=7, trials=300)
        for date in {str(record["date"]) for record in records}:
            assigned = {
                result.record_assignments[str(record["sample_id"])]
                for record in records
                if record["date"] == date
            }
            self.assertEqual(len(assigned), 1)

    def test_balanced_search_is_no_worse_than_first_random_candidate(self) -> None:
        records = synthetic_records()
        one_trial = balanced_group_split(records, seed=11, trials=1)
        many_trials = balanced_group_split(records, seed=11, trials=500)
        self.assertLessEqual(many_trials.score, one_trial.score)

    def test_chronological_split_uses_consecutive_dates(self) -> None:
        records = synthetic_records()
        result = chronological_group_split(records)
        assert_valid_split(records, result)
        dates_by_split = {
            split: sorted(
                date
                for date, assigned_split in result.group_assignments.items()
                if assigned_split == split
            )
            for split in result.split_names
        }
        self.assertLess(max(dates_by_split["train"]), min(dates_by_split["val"]))
        self.assertLess(max(dates_by_split["val"]), min(dates_by_split["test"]))

    def test_invalid_or_duplicate_records_are_rejected(self) -> None:
        records = synthetic_records()
        records.append(dict(records[0]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            balanced_group_split(records, trials=1)

    def test_default_balance_fields_are_present_in_fixture(self) -> None:
        record = synthetic_records()[0]
        self.assertTrue(set(DEFAULT_BALANCE_WEIGHTS).difference({"file_count"}) <= set(record))


if __name__ == "__main__":
    unittest.main()

