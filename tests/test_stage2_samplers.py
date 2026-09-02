"""Tests for stratified and distributed Stage-2 patch sampling."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from precipitation_inversion.data.stage2_samplers import (  # noqa: E402
    STAGE2_STRATA,
    Stage2StratifiedBatchSampler,
    allocate_stage2_epoch_quotas,
    classify_stage2_patch_strata,
)


RECORD_DTYPE = np.dtype(
    [
        ("file_id", "<u2"),
        ("dpr_count", "<u4"),
        ("q11_count", "<u4"),
        ("q01_count", "<u4"),
        ("strong_dpr_count", "<u4"),
    ]
)


class PatchRecordDataset:
    def __init__(self) -> None:
        self.records = np.zeros(12, dtype=RECORD_DTYPE)
        self.records["file_id"] = np.repeat([0, 1, 2], 4)
        # 0:3 background
        self.records[3:6]["dpr_count"] = 10
        self.records[3:6]["q11_count"] = 10
        # 6:9 fill targets
        self.records[6:9]["dpr_count"] = 10
        self.records[6:9]["q11_count"] = 2
        self.records[6:9]["q01_count"] = 8
        # 9:12 strong targets; Q01 presence does not override strong priority.
        self.records[9:12]["dpr_count"] = 10
        self.records[9:12]["q11_count"] = 5
        self.records[9:12]["q01_count"] = 5
        self.records[9:12]["strong_dpr_count"] = 2

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> int:
        return index


def batches(sampler: Stage2StratifiedBatchSampler) -> list[list[int]]:
    return list(iter(sampler))


class Stage2StratumTests(unittest.TestCase):
    def test_priority_classification_is_mutually_exclusive(self) -> None:
        dataset = PatchRecordDataset()
        strata = classify_stage2_patch_strata(dataset.records)
        np.testing.assert_array_equal(strata, np.repeat(np.arange(4), 3))

    def test_largest_remainder_quotas_sum_exactly(self) -> None:
        quotas = allocate_stage2_epoch_quotas(
            11, np.array([0.15, 0.15, 0.35, 0.35])
        )
        # Tied remainders are resolved by stable stratum order.
        np.testing.assert_array_equal(quotas, [2, 1, 4, 4])
        self.assertEqual(int(quotas.sum()), 11)

    def test_inconsistent_counts_and_missing_fields_are_rejected(self) -> None:
        records = PatchRecordDataset().records.copy()
        records[0]["dpr_count"] = 1
        with self.assertRaisesRegex(ValueError, "q11_count"):
            classify_stage2_patch_strata(records)
        with self.assertRaisesRegex(ValueError, "missing fields"):
            classify_stage2_patch_strata(
                np.zeros(2, dtype=[("file_id", "u1")])
            )


class Stage2StratifiedBatchSamplerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = PatchRecordDataset()
        self.equal_weights = {name: 1.0 for name in STAGE2_STRATA}

    def test_equal_quota_epoch_visits_every_patch_once(self) -> None:
        sampler = Stage2StratifiedBatchSampler(
            self.dataset,
            batch_size=2,
            stratum_weights=self.equal_weights,
            group_by_file=False,
            seed=7,
        )
        result = batches(sampler)
        flattened = [index for batch in result for index in batch]
        self.assertEqual(sampler.stratum_counts, {name: 3 for name in STAGE2_STRATA})
        self.assertEqual(sampler.epoch_quotas, {name: 3 for name in STAGE2_STRATA})
        self.assertEqual(sorted(flattened), list(range(12)))
        self.assertEqual(len(sampler), 6)

    def test_oversampling_uses_full_cycles_and_exact_quotas(self) -> None:
        sampler = Stage2StratifiedBatchSampler(
            self.dataset,
            batch_size=5,
            epoch_size=20,
            stratum_weights=self.equal_weights,
            group_by_file=False,
            seed=2,
        )
        flattened = np.array([i for batch in sampler for i in batch])
        labels = sampler.strata[flattened]
        np.testing.assert_array_equal(np.bincount(labels, minlength=4), [5, 5, 5, 5])
        # Five draws from three candidates must contain all three before repeats.
        for stratum_id in range(4):
            self.assertEqual(np.unique(flattened[labels == stratum_id]).size, 3)

    def test_file_grouping_is_cache_local_and_epoch_reproducible(self) -> None:
        sampler = Stage2StratifiedBatchSampler(
            self.dataset,
            batch_size=1,
            stratum_weights=self.equal_weights,
            group_by_file=True,
            seed=19,
        )
        epoch_zero = batches(sampler)
        self.assertEqual(epoch_zero, batches(sampler))
        file_sequence = [int(self.dataset.records[batch[0]]["file_id"]) for batch in epoch_zero]
        compressed = [
            file_id
            for position, file_id in enumerate(file_sequence)
            if position == 0 or file_sequence[position - 1] != file_id
        ]
        self.assertEqual(len(compressed), 3)
        self.assertEqual(set(compressed), {0, 1, 2})
        sampler.set_epoch(1)
        self.assertNotEqual(epoch_zero, batches(sampler))

    def test_ddp_shards_one_global_batch_sequence(self) -> None:
        global_batches = batches(
            Stage2StratifiedBatchSampler(
                self.dataset,
                batch_size=2,
                stratum_weights=self.equal_weights,
                group_by_file=False,
                seed=11,
                even_batches=False,
            )
        )
        shards = [
            batches(
                Stage2StratifiedBatchSampler(
                    self.dataset,
                    batch_size=2,
                    stratum_weights=self.equal_weights,
                    group_by_file=False,
                    seed=11,
                    num_replicas=3,
                    rank=rank,
                    even_batches=True,
                )
            )
            for rank in range(3)
        ]
        self.assertEqual([len(shard) for shard in shards], [2, 2, 2])
        reconstructed = [None] * 6
        for rank, shard in enumerate(shards):
            for local_index, batch in enumerate(shard):
                reconstructed[rank + local_index * 3] = batch
        self.assertEqual(reconstructed, global_batches)

    def test_dataloader_and_invalid_configuration(self) -> None:
        torch = __import__("torch")
        sampler = Stage2StratifiedBatchSampler(
            self.dataset,
            batch_size=3,
            stratum_weights=self.equal_weights,
            group_by_file=False,
            shuffle=False,
        )
        loader = torch.utils.data.DataLoader(
            self.dataset, batch_sampler=sampler, num_workers=0
        )
        self.assertEqual([batch.tolist() for batch in loader], batches(sampler))
        with self.assertRaisesRegex(ValueError, "exactly"):
            Stage2StratifiedBatchSampler(
                self.dataset,
                batch_size=1,
                stratum_weights={"background": 1.0},
            )
        empty_stratum = PatchRecordDataset()
        empty_stratum.records = empty_stratum.records[:3]
        with self.assertRaisesRegex(ValueError, "no patches"):
            Stage2StratifiedBatchSampler(
                empty_stratum,
                batch_size=1,
                stratum_weights=self.equal_weights,
            )


if __name__ == "__main__":
    unittest.main()
