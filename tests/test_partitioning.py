import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


import numpy as np

from fl_baselines.datasets.partitioning import create_iid_partitions, create_dirichlet_partitions


class PartitioningTest(unittest.TestCase):
    def test_iid_partitions_are_deterministic_and_cover_all_samples(self) -> None:
        first = create_iid_partitions(num_samples=10, num_partitions=3, seed=123)
        second = create_iid_partitions(num_samples=10, num_partitions=3, seed=123)

        self.assertEqual([p.tolist() for p in first], [p.tolist() for p in second])
        combined = np.concatenate(first)
        self.assertEqual(sorted(combined.tolist()), list(range(10)))

    def test_dirichlet_partitions_are_deterministic_and_cover_all_samples(self) -> None:
        targets = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])

        first = create_dirichlet_partitions(targets, num_partitions=3, alpha=0.5, seed=123)
        second = create_dirichlet_partitions(targets, num_partitions=3, alpha=0.5, seed=123)

        self.assertEqual([p.tolist() for p in first], [p.tolist() for p in second])
        combined = np.concatenate(first)
        self.assertEqual(sorted(combined.tolist()), list(range(len(targets))))

    def test_dirichlet_partitions_rebalance_empty_clients_when_possible(self) -> None:
        targets = np.repeat(np.arange(10), 10)

        partitions = create_dirichlet_partitions(
            targets,
            num_partitions=10,
            alpha=0.001,
            seed=1,
        )

        self.assertTrue(all(len(partition) > 0 for partition in partitions))
        combined = np.concatenate(partitions)
        self.assertEqual(sorted(combined.tolist()), list(range(len(targets))))

    def test_invalid_partition_arguments_fail_fast(self) -> None:
        with self.assertRaisesRegex(ValueError, "num_partitions must be positive"):
            create_iid_partitions(num_samples=10, num_partitions=0, seed=1)

        with self.assertRaisesRegex(ValueError, "alpha must be positive"):
            create_dirichlet_partitions(np.array([0, 1]), num_partitions=2, alpha=0.0, seed=1)


if __name__ == "__main__":
    unittest.main()
