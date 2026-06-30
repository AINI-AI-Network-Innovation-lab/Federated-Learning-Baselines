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
        validation_dataset = self._load(train=False, config=config)

        train_indices = self._partition_indices(
            self._targets(train_dataset),
            config,
            partition_id,
            num_partitions,
            seed_offset=0,
        )
        validation_indices = self._partition_indices(
            self._targets(validation_dataset),
            config,
            partition_id,
            num_partitions,
            seed_offset=1,
        )

        return ClientDataLoaders(
            train=DataLoader(
                Subset(train_dataset, train_indices.tolist()),
                batch_size=config.batch_size,
                shuffle=True,
            ),
            validation=DataLoader(
                Subset(validation_dataset, validation_indices.tolist()),
                batch_size=config.batch_size,
                shuffle=False,
            ),
        )

    def build_server_loader(self, config: ExperimentConfig) -> DataLoader:
        dataset = self._load(train=False, config=config)
        return DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    @abstractmethod
    def _load(self, train: bool, config: ExperimentConfig):
        """Load train or validation/test split."""

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


def build_image_transform(config: ExperimentConfig, source_channels: int):
    steps = []
    if (config.input_height, config.input_width) != (0, 0):
        steps.append(transforms.Resize((config.input_height, config.input_width)))
    if config.input_channels != source_channels:
        steps.append(transforms.Grayscale(num_output_channels=config.input_channels))
    steps.append(transforms.ToTensor())
    return transforms.Compose(steps)
