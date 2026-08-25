"""Tests for cache-local and distributed-aware batch sampling."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from precipitation_inversion.data.samplers import (  # noqa: E402
    FileBlockBatchSampler,
    FileIndexRange,
    file_ranges_from_dataset,
)


class IntegerDataset:
    """Tiny Dataset with the same file-range metadata contract as stage one."""

    def __init__(self, file_sizes: list[int]) -> None:
        self.files = []
        start = 0
        for file_id, size in enumerate(file_sizes):
            stop = start + size
            self.files.append(
                {
                    "file_id": file_id,
                    "index_start": start,
                    "index_stop": stop,
                    "sample_count": size,
                }
            )
            start = stop
        self.size = start

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> int:
        return index


def batches(sampler: FileBlockBatchSampler) -> list[list[int]]:
    return list(iter(sampler))


class FileRangeTests(unittest.TestCase):
    def test_ranges_include_empty_files_and_cover_dataset(self) -> None:
        dataset = IntegerDataset([5, 0, 3])
        self.assertEqual(
            file_ranges_from_dataset(dataset),
            (
                FileIndexRange(file_id=0, start=0, stop=5),
                FileIndexRange(file_id=1, start=5, stop=5),
                FileIndexRange(file_id=2, start=5, stop=8),
            ),
        )

    def test_gaps_overlaps_and_inconsistent_counts_are_rejected(self) -> None:
        dataset = IntegerDataset([3, 2])
        dataset.files[1]["index_start"] = 2
        with self.assertRaisesRegex(ValueError, "contiguous"):
            file_ranges_from_dataset(dataset)

        dataset = IntegerDataset([3, 2])
        dataset.files[0]["sample_count"] = 4
        with self.assertRaisesRegex(ValueError, "sample_count"):
            file_ranges_from_dataset(dataset)

        dataset = IntegerDataset([3, 2])
        dataset.size = 6
        with self.assertRaisesRegex(ValueError, "dataset has"):
            file_ranges_from_dataset(dataset)


class FileBlockBatchSamplerTests(unittest.TestCase):
    def setUp(self) -> None:
        # Includes one empty file and several incomplete final batches.
        self.dataset = IntegerDataset([10, 0, 7, 3])

    def test_sequential_order_never_mixes_files_and_covers_every_sample(self) -> None:
        sampler = FileBlockBatchSampler(
            self.dataset,
            batch_size=4,
            block_size=8,
            shuffle=False,
        )
        result = batches(sampler)
        self.assertEqual(
            result,
            [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [8, 9],
                [10, 11, 12, 13],
                [14, 15, 16],
                [17, 18, 19],
            ],
        )
        self.assertEqual(len(sampler), len(result))
        self.assertEqual(sorted(index for batch in result for index in batch), list(range(20)))
        for batch in result:
            file_ids = {self._file_id(index) for index in batch}
            self.assertEqual(len(file_ids), 1)

    def test_drop_last_drops_only_each_files_incomplete_batch(self) -> None:
        sampler = FileBlockBatchSampler(
            self.dataset,
            batch_size=4,
            block_size=8,
            shuffle=False,
            drop_last=True,
        )
        result = batches(sampler)
        self.assertEqual(result, [[0, 1, 2, 3], [4, 5, 6, 7], [10, 11, 12, 13]])
        self.assertTrue(all(len(batch) == 4 for batch in result))
        self.assertEqual(len(sampler), 3)

    def test_seed_and_epoch_control_reproducible_shuffling(self) -> None:
        sampler = FileBlockBatchSampler(
            IntegerDataset([24, 24, 24]),
            batch_size=4,
            block_size=8,
            seed=123,
        )
        epoch_zero = batches(sampler)
        self.assertEqual(epoch_zero, batches(sampler))
        sampler.set_epoch(1)
        epoch_one = batches(sampler)
        self.assertNotEqual(epoch_zero, epoch_one)
        sampler.set_epoch(0)
        self.assertEqual(epoch_zero, batches(sampler))

        # A file appears in one contiguous run of batches, preserving cache locality.
        file_sequence = [self._file_id_for_sizes(batch[0], [24, 24, 24]) for batch in epoch_one]
        compressed = [
            file_id
            for position, file_id in enumerate(file_sequence)
            if position == 0 or file_sequence[position - 1] != file_id
        ]
        self.assertEqual(len(compressed), 3)
        self.assertEqual(set(compressed), {0, 1, 2})

    def test_distributed_even_shards_are_disjoint_and_equal_length(self) -> None:
        dataset = IntegerDataset([11, 8, 5])  # seven global batches
        global_sampler = FileBlockBatchSampler(
            dataset, batch_size=4, block_size=8, seed=9, even_batches=False
        )
        global_batches = batches(global_sampler)
        rank_batches = []
        for rank in range(3):
            sampler = FileBlockBatchSampler(
                dataset,
                batch_size=4,
                block_size=8,
                seed=9,
                num_replicas=3,
                rank=rank,
                even_batches=True,
            )
            current = batches(sampler)
            self.assertEqual(len(current), 2)
            rank_batches.append(current)

        expected = [global_batches[index] for index in range(6)]
        actual = [batch for shard in rank_batches for batch in shard]
        self.assertEqual(
            {tuple(batch) for batch in actual}, {tuple(batch) for batch in expected}
        )
        flattened = [set(index for batch in shard for index in batch) for shard in rank_batches]
        self.assertTrue(flattened[0].isdisjoint(flattened[1]))
        self.assertTrue(flattened[0].isdisjoint(flattened[2]))
        self.assertTrue(flattened[1].isdisjoint(flattened[2]))

    def test_uneven_distributed_shards_preserve_all_global_batches(self) -> None:
        dataset = IntegerDataset([11, 8, 5])
        global_batches = batches(
            FileBlockBatchSampler(
                dataset, batch_size=4, block_size=8, seed=5, even_batches=False
            )
        )
        shards = [
            batches(
                FileBlockBatchSampler(
                    dataset,
                    batch_size=4,
                    block_size=8,
                    seed=5,
                    num_replicas=3,
                    rank=rank,
                    even_batches=False,
                )
            )
            for rank in range(3)
        ]
        self.assertEqual([len(shard) for shard in shards], [3, 2, 2])
        self.assertEqual(
            {tuple(batch) for shard in shards for batch in shard},
            {tuple(batch) for batch in global_batches},
        )

    def test_pytorch_dataloader_accepts_sampler_as_batch_sampler(self) -> None:
        try:
            import torch
        except (ImportError, OSError):
            self.skipTest("a usable PyTorch is not installed")
        sampler = FileBlockBatchSampler(
            self.dataset, batch_size=4, block_size=8, shuffle=False
        )
        loader = torch.utils.data.DataLoader(
            self.dataset, batch_sampler=sampler, num_workers=0
        )
        loaded = [batch.tolist() for batch in loader]
        self.assertEqual(loaded, batches(sampler))

    def test_invalid_configuration_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "multiple"):
            FileBlockBatchSampler(self.dataset, batch_size=4, block_size=6)
        with self.assertRaisesRegex(ValueError, "positive"):
            FileBlockBatchSampler(self.dataset, batch_size=0)
        with self.assertRaisesRegex(ValueError, "smaller"):
            FileBlockBatchSampler(
                self.dataset, batch_size=4, num_replicas=2, rank=2
            )
        sampler = FileBlockBatchSampler(self.dataset, batch_size=4)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            sampler.set_epoch(-1)

    def _file_id(self, index: int) -> int:
        return self._file_id_for_sizes(index, [10, 0, 7, 3])

    @staticmethod
    def _file_id_for_sizes(index: int, sizes: list[int]) -> int:
        stop = 0
        for file_id, size in enumerate(sizes):
            stop += size
            if index < stop:
                return file_id
        raise IndexError(index)


if __name__ == "__main__":
    unittest.main()
