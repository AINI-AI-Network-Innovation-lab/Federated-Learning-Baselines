"""Default dataset, model, and algorithm registrations."""

from fl_baselines.core.registry import ALGORITHMS, DATASETS, MODELS


def register_default_components() -> None:
    """Register built-in components.

    The function is idempotent so apps and tests can call it freely.
    """

    from fl_baselines.algorithms.fedavg import FedAvgBuilder
    from fl_baselines.algorithms.fedavgm import FedAvgMBuilder
    from fl_baselines.algorithms.fedprox import FedProxBuilder
    from fl_baselines.algorithms.moon import MoonBuilder
    from fl_baselines.algorithms.scaffold import ScaffoldBuilder
    from fl_baselines.datasets.common import (
        Cifar10DatasetBuilder,
        Cifar100DatasetBuilder,
        EmnistDatasetBuilder,
        FashionMnistDatasetBuilder,
        ImageNetDatasetBuilder,
    )
    from fl_baselines.datasets.mnist import MnistDatasetBuilder
    from fl_baselines.models.inception import InceptionBuilder
    from fl_baselines.models.lenet import LeNetBuilder
    from fl_baselines.models.mnist_cnn import MnistCnnBuilder
    from fl_baselines.models.resnet import ResNet9Builder, ResNet18Builder, ResNet34Builder

    if "mnist" not in DATASETS:
        DATASETS.register("mnist", MnistDatasetBuilder())
    if "fmnist" not in DATASETS:
        DATASETS.register("fmnist", FashionMnistDatasetBuilder())
    if "emnist" not in DATASETS:
        DATASETS.register("emnist", EmnistDatasetBuilder())
    if "cifar10" not in DATASETS:
        DATASETS.register("cifar10", Cifar10DatasetBuilder())
    if "cifar100" not in DATASETS:
        DATASETS.register("cifar100", Cifar100DatasetBuilder())
    if "imagenet" not in DATASETS:
        DATASETS.register("imagenet", ImageNetDatasetBuilder())
    if "mnist_cnn" not in MODELS:
        MODELS.register("mnist_cnn", MnistCnnBuilder())
    if "lenet" not in MODELS:
        MODELS.register("lenet", LeNetBuilder())
    if "resnet9" not in MODELS:
        MODELS.register("resnet9", ResNet9Builder())
    if "resnet18" not in MODELS:
        MODELS.register("resnet18", ResNet18Builder())
    if "resnet34" not in MODELS:
        MODELS.register("resnet34", ResNet34Builder())
    if "inception" not in MODELS:
        MODELS.register("inception", InceptionBuilder())
    if "fedavg" not in ALGORITHMS:
        ALGORITHMS.register("fedavg", FedAvgBuilder())
    if "fedavgm" not in ALGORITHMS:
        ALGORITHMS.register("fedavgm", FedAvgMBuilder())
    if "fedprox" not in ALGORITHMS:
        ALGORITHMS.register("fedprox", FedProxBuilder())
    if "scaffold" not in ALGORITHMS:
        ALGORITHMS.register("scaffold", ScaffoldBuilder())
    if "moon" not in ALGORITHMS:
        ALGORITHMS.register("moon", MoonBuilder())
