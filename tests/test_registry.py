import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


from fl_baselines.core.registry import Registry
from fl_baselines.defaults import register_default_components
from fl_baselines.core.registry import ALGORITHMS, DATASETS


class RegistryTest(unittest.TestCase):
    def test_register_and_lookup_component(self) -> None:
        registry = Registry("algorithm")
        component = object()

        registry.register("fedavg", component)

        self.assertIs(registry.get("fedavg"), component)
        self.assertEqual(registry.names(), ["fedavg"])

    def test_duplicate_key_fails_with_clear_message(self) -> None:
        registry = Registry("dataset")
        registry.register("mnist", object())

        with self.assertRaisesRegex(ValueError, "Dataset 'mnist' is already registered"):
            registry.register("mnist", object())

    def test_unknown_key_lists_available_components(self) -> None:
        registry = Registry("algorithm")
        registry.register("fedavg", object())

        with self.assertRaisesRegex(KeyError, "Unknown algorithm 'fedprox'. Available: fedavg"):
            registry.get("fedprox")

    def test_default_components_include_algorithms(self) -> None:
        register_default_components()

        for algorithm_name in [
            "fedavg",
            "fedavgm",
            "fedadagrad",
            "fedadam",
            "fedadp",
            "fedyogi",
            "gamf",
            "fedma",
            "fedcda",
            "feddrl",
            "feddyn",
            "feddc",
            "feddecorr",
            "fedexp",
            "fedspeed",
            "fedsam",
            "fedgen",
            "fedent",
            "fedlaw",
            "fedaaw",
            "feddisco",
            "fedvck",
            "fedntd",
            "fedlc",
            "fedrs",
            "fedlama",
            "fedproto",
            "fedmeta",
            "fednp",
            "fedcurv",
            "fedmmd",
            "apfl",
            "ditto",
            "pfedme",
            "fednova",
            "fedper",
            "fedrep",
            "fedala",
            "fedamp",
            "fedlaa",
            "fedprox",
            "scaffold",
            "moon",
        ]:
            self.assertIn(algorithm_name, ALGORITHMS.names())

    def test_default_components_include_common_datasets(self) -> None:
        register_default_components()

        for dataset_name in ["mnist", "fmnist", "emnist", "cifar10", "cifar100", "imagenet"]:
            self.assertIn(dataset_name, DATASETS.names())


if __name__ == "__main__":
    unittest.main()
