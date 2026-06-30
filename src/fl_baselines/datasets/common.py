"""Common torchvision dataset builders."""

from __future__ import annotations

from torchvision import datasets

from fl_baselines.core.config import ExperimentConfig
from fl_baselines.datasets.vision import TorchVisionDatasetBuilder, build_image_transform


class FashionMnistDatasetBuilder(TorchVisionDatasetBuilder):
    name = "fmnist"

    def _load(self, train: bool, config: ExperimentConfig):
        return datasets.FashionMNIST(
            root=config.data_dir,
            train=train,
            download=True,
            transform=build_image_transform(config, source_channels=1),
        )


class EmnistDatasetBuilder(TorchVisionDatasetBuilder):
    name = "emnist"

    def _load(self, train: bool, config: ExperimentConfig):
        return datasets.EMNIST(
            root=config.data_dir,
            split=config.emnist_split,
            train=train,
            download=True,
            transform=build_image_transform(config, source_channels=1),
        )


class Cifar10DatasetBuilder(TorchVisionDatasetBuilder):
    name = "cifar10"

    def _load(self, train: bool, config: ExperimentConfig):
        return datasets.CIFAR10(
            root=config.data_dir,
            train=train,
            download=True,
            transform=build_image_transform(config, source_channels=3),
        )


class Cifar100DatasetBuilder(TorchVisionDatasetBuilder):
    name = "cifar100"

    def _load(self, train: bool, config: ExperimentConfig):
        return datasets.CIFAR100(
            root=config.data_dir,
            train=train,
            download=True,
            transform=build_image_transform(config, source_channels=3),
        )


class ImageNetDatasetBuilder(TorchVisionDatasetBuilder):
    name = "imagenet"

    def _load(self, train: bool, config: ExperimentConfig):
        return datasets.ImageNet(
            root=config.data_dir,
            split="train" if train else "val",
            transform=build_image_transform(config, source_channels=3),
        )
