"""Shared builders for vision datasets."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from fl_baselines.core.config import ExperimentConfig
from fl_baselines.core.types import ClientDataLoaders
from fl_baselines.datasets.partitioning import create_partitions


class TorchVisionDatasetBuilder(ABC):
    name: str

    def build_client_loaders(
        self,
        config: ExperimentConfig,
        partition_id: int,
        num_partitions: int,
    ) -> ClientDataLoaders:
        train_dataset = self._load(train=True, config=config)

        partition_indices = self._partition_indices(
            self._targets(train_dataset),
            config,
            partition_id,
            num_partitions,
            seed_offset=0,
        )
        client_train_indices, client_test_indices = self._split_client_train_test(
            partition_indices,
            config,
            partition_id,
        )

        return ClientDataLoaders(
            train=DataLoader(
                Subset(train_dataset, client_train_indices.tolist()),
                batch_size=config.batch_size,
                shuffle=True,
            ),
            test=DataLoader(
                Subset(train_dataset, client_test_indices.tolist()),
                batch_size=config.batch_size,
                shuffle=False,
            ),
        )

    def build_server_loader(self, config: ExperimentConfig) -> DataLoader:
        dataset = self._load(train=False, config=config)
        return DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    @abstractmethod
    def _load(self, train: bool, config: ExperimentConfig):
        """Load train or test split."""

    def _partition_indices(
        self,
        targets: Sequence[int] | np.ndarray,
        config: ExperimentConfig,
        partition_id: int,
        num_partitions: int,
        seed_offset: int,
    ) -> np.ndarray:
        if partition_id < 0 or partition_id >= num_partitions:
            raise ValueError("partition_id must be in [0, num_partitions)")

        partitions = create_partitions(
            np.asarray(targets),
            num_partitions=num_partitions,
            partitioner=config.partitioner,
            dirichlet_alpha=config.dirichlet_alpha,
            seed=config.seed + seed_offset,
        )
        return partitions[partition_id]

    def _targets(self, dataset) -> Sequence[int] | np.ndarray:
        if hasattr(dataset, "targets"):
            return dataset.targets
        if hasattr(dataset, "labels"):
            return dataset.labels
        raise ValueError(f"{type(dataset).__name__} does not expose targets or labels")

    def _split_client_train_test(
        self,
        indices: np.ndarray,
        config: ExperimentConfig,
        partition_id: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(indices) < 2:
            return indices, indices[:0]

        rng = np.random.default_rng(config.seed + 10_000 + partition_id)
        shuffled = indices.copy()
        rng.shuffle(shuffled)

        test_size = int(round(len(shuffled) * config.client_test_fraction))
        test_size = max(1, min(len(shuffled) - 1, test_size))
        test_indices = shuffled[:test_size]
        train_indices = shuffled[test_size:]
        return train_indices, test_indices


def build_image_transform(config: ExperimentConfig, source_channels: int):
    steps = []
    if (config.input_height, config.input_width) != (0, 0):
        steps.append(transforms.Resize((config.input_height, config.input_width)))
    if config.input_channels != source_channels:
        steps.append(transforms.Grayscale(num_output_channels=config.input_channels))
    steps.append(transforms.ToTensor())
    return transforms.Compose(steps)
