"""Default dataset, model, and algorithm registrations."""

from fl_baselines.core.registry import ALGORITHMS, DATASETS, MODELS


def register_default_components() -> None:
    """Register built-in components.

    The function is idempotent so apps and tests can call it freely.
    """

    from fl_baselines.algorithms.fedavg import FedAvgBuilder
    from fl_baselines.algorithms.fedavgm import FedAvgMBuilder
    from fl_baselines.algorithms.fedadp import FedAdpBuilder
    from fl_baselines.algorithms.fedaaw import FedAAWBuilder
    from fl_baselines.algorithms.ditto import DittoBuilder
    from fl_baselines.algorithms.feddc import FedDCBuilder
    from fl_baselines.algorithms.feddecorr import FedDecorrBuilder
    from fl_baselines.algorithms.fedent import FedEntBuilder
    from fl_baselines.algorithms.fedvck import FedVCKBuilder
    from fl_baselines.algorithms.feddyn import FedDynBuilder
    from fl_baselines.algorithms.fedexp import FedExPBuilder
    from fl_baselines.algorithms.fedsam import FedSAMBuilder
    from fl_baselines.algorithms.fedspeed import FedSpeedBuilder
    from fl_baselines.algorithms.fedntd import FedNTDBuilder
    from fl_baselines.algorithms.fedproto import FedProtoBuilder
    from fl_baselines.algorithms.pfedme import PFedMeBuilder
    from fl_baselines.algorithms.fedper import FedPerBuilder
    from fl_baselines.algorithms.fedrep import FedRepBuilder
    from fl_baselines.algorithms.fednova import FedNovaBuilder
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
    if "fedadp" not in ALGORITHMS:
        ALGORITHMS.register("fedadp", FedAdpBuilder())
    if "ditto" not in ALGORITHMS:
        ALGORITHMS.register("ditto", DittoBuilder())
    if "feddyn" not in ALGORITHMS:
        ALGORITHMS.register("feddyn", FedDynBuilder())
    if "feddc" not in ALGORITHMS:
        ALGORITHMS.register("feddc", FedDCBuilder())
    if "feddecorr" not in ALGORITHMS:
        ALGORITHMS.register("feddecorr", FedDecorrBuilder())
    if "fedexp" not in ALGORITHMS:
        ALGORITHMS.register("fedexp", FedExPBuilder())
    if "fedspeed" not in ALGORITHMS:
        ALGORITHMS.register("fedspeed", FedSpeedBuilder())
    if "fedsam" not in ALGORITHMS:
        ALGORITHMS.register("fedsam", FedSAMBuilder())
    if "fedent" not in ALGORITHMS:
        ALGORITHMS.register("fedent", FedEntBuilder())
    if "fedaaw" not in ALGORITHMS:
        ALGORITHMS.register("fedaaw", FedAAWBuilder())
    if "fedvck" not in ALGORITHMS:
        ALGORITHMS.register("fedvck", FedVCKBuilder())
    if "fedntd" not in ALGORITHMS:
        ALGORITHMS.register("fedntd", FedNTDBuilder())
    if "fedproto" not in ALGORITHMS:
        ALGORITHMS.register("fedproto", FedProtoBuilder())
    if "pfedme" not in ALGORITHMS:
        ALGORITHMS.register("pfedme", PFedMeBuilder())
    if "fednova" not in ALGORITHMS:
        ALGORITHMS.register("fednova", FedNovaBuilder())
    if "fedper" not in ALGORITHMS:
        ALGORITHMS.register("fedper", FedPerBuilder())
    if "fedrep" not in ALGORITHMS:
        ALGORITHMS.register("fedrep", FedRepBuilder())
    if "fedprox" not in ALGORITHMS:
        ALGORITHMS.register("fedprox", FedProxBuilder())
    if "scaffold" not in ALGORITHMS:
        ALGORITHMS.register("scaffold", ScaffoldBuilder())
    if "moon" not in ALGORITHMS:
        ALGORITHMS.register("moon", MoonBuilder())
