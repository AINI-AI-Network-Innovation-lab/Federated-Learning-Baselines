import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


import torch
from torch.utils.data import Dataset

from fl_baselines.core.config import ExperimentConfig
from fl_baselines.datasets.vision import TorchVisionDatasetBuilder


class FakeVisionDataset(Dataset):
    def __init__(self, targets: list[int]) -> None:
        self.targets = targets

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        return torch.zeros(1, 28, 28), self.targets[index]


class FakeVisionBuilder(TorchVisionDatasetBuilder):
    name = "fake"

    def _load(self, train: bool, config: ExperimentConfig):
        if train:
            return FakeVisionDataset([0, 0, 1, 1, 2, 2])
        return FakeVisionDataset([0, 1, 2])


class DatasetBuilderTest(unittest.TestCase):
    def test_torchvision_builder_partitions_client_loaders(self) -> None:
        config = ExperimentConfig.from_run_config({"batch-size": 2})

        loaders = FakeVisionBuilder().build_client_loaders(
            config,
            partition_id=0,
            num_partitions=2,
        )

        self.assertGreater(len(loaders.train.dataset), 0)
        self.assertGreater(len(loaders.test.dataset), 0)

    def test_torchvision_builder_rejects_invalid_partition_id(self) -> None:
        config = ExperimentConfig.from_run_config({})

        with self.assertRaisesRegex(ValueError, "partition_id must be in"):
            FakeVisionBuilder().build_client_loaders(
                config,
                partition_id=2,
                num_partitions=2,
            )

    def test_torchvision_builder_builds_server_loader(self) -> None:
        config = ExperimentConfig.from_run_config({"batch-size": 2})

        loader = FakeVisionBuilder().build_server_loader(config)

        self.assertEqual(len(loader.dataset), 3)

    def test_torchvision_builder_splits_client_partition_into_train_and_test(self) -> None:
        config = ExperimentConfig.from_run_config(
            {"batch-size": 2, "client-test-fraction": 0.5, "seed": 7}
        )

        loaders = FakeVisionBuilder().build_client_loaders(
            config,
            partition_id=0,
            num_partitions=2,
        )

        self.assertGreater(len(loaders.train.dataset), 0)
        self.assertGreater(len(loaders.test.dataset), 0)
        self.assertEqual(
            len(loaders.train.dataset) + len(loaders.test.dataset),
            3,
        )

    def test_torchvision_builder_client_split_is_deterministic(self) -> None:
        config = ExperimentConfig.from_run_config(
            {"batch-size": 2, "client-test-fraction": 0.5, "seed": 11}
        )

        first = FakeVisionBuilder().build_client_loaders(
            config,
            partition_id=0,
            num_partitions=2,
        )
        second = FakeVisionBuilder().build_client_loaders(
            config,
            partition_id=0,
            num_partitions=2,
        )

        self.assertEqual(len(first.train.dataset), len(second.train.dataset))
        self.assertEqual(len(first.test.dataset), len(second.test.dataset))


if __name__ == "__main__":
    unittest.main()
