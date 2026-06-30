"""Dataset extension interface."""

from __future__ import annotations

from typing import Protocol

from torch.utils.data import DataLoader

from fl_baselines.core.config import ExperimentConfig
from fl_baselines.core.types import ClientDataLoaders


class DatasetBuilder(Protocol):
    name: str

    def build_client_loaders(
        self,
        config: ExperimentConfig,
        partition_id: int,
        num_partitions: int,
    ) -> ClientDataLoaders:
        """Build train and validation loaders for one client partition."""

    def build_server_loader(self, config: ExperimentConfig) -> DataLoader:
        """Build a server-side evaluation loader."""
