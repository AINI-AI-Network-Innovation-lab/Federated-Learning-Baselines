"""MNIST dataset builder."""

from __future__ import annotations

from torchvision import datasets

from fl_baselines.core.config import ExperimentConfig
from fl_baselines.datasets.vision import TorchVisionDatasetBuilder, build_image_transform


class MnistDatasetBuilder(TorchVisionDatasetBuilder):
    name = "mnist"

    def _load(self, train: bool, config: ExperimentConfig) -> datasets.MNIST:
        return datasets.MNIST(
            root=config.data_dir,
            train=train,
            download=True,
            transform=build_image_transform(config, source_channels=1),
        )
