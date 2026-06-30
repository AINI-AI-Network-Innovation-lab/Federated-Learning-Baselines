import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg, FedAvgM, FedProx

from fl_baselines.clients.torch_client import TorchFlowerClient, get_model_parameters
from fl_baselines.algorithms.fedavg import FedAvgBuilder
from fl_baselines.algorithms.fedavgm import FedAvgMBuilder
from fl_baselines.algorithms.fedper import FedPerBuilder, FedPerStrategy
from fl_baselines.algorithms.fedrep import FedRepBuilder, FedRepStrategy
from fl_baselines.algorithms.fednova import FedNovaBuilder, FedNovaStrategy
from fl_baselines.algorithms.fedprox import FedProxBuilder
from fl_baselines.algorithms.moon import MoonBuilder, MoonStrategy
from fl_baselines.algorithms.scaffold import ScaffoldBuilder, ScaffoldStrategy
from fl_baselines.core.config import ExperimentConfig
from fl_baselines.defaults import register_default_components
from fl_baselines.core.registry import MODELS
from fl_baselines.models.lenet import LeNetBuilder
from fl_baselines.models.mnist_cnn import MnistCnnBuilder
from fl_baselines.models.resnet import ResNet9Builder, ResNet18Builder, ResNet34Builder
from fl_baselines.models.inception import InceptionBuilder
from fl_baselines.training.evaluate import evaluate_model
from fl_baselines.training.moon import train_moon_client
from fl_baselines.training.scaffold import train_scaffold_client
from fl_baselines.training.train import train_one_client


class ModelAndAlgorithmTest(unittest.TestCase):
    def test_evaluate_model_accepts_tuple_output_models(self) -> None:
        class TupleOutputModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, inputs):
                logits = self.linear(inputs)
                return inputs, logits, logits

        loader = DataLoader(
            TensorDataset(torch.ones(4, 2), torch.zeros(4, dtype=torch.long)),
            batch_size=2,
        )

        loss, metrics = evaluate_model(TupleOutputModel(), loader, device="cpu")

        self.assertGreaterEqual(loss, 0.0)
        self.assertIn("accuracy", metrics)

    def test_mnist_cnn_forward_shape(self) -> None:
        model = MnistCnnBuilder().build_model(ExperimentConfig.from_run_config({}))

        output = model(torch.zeros(2, 1, 28, 28))

        self.assertEqual(tuple(output.shape), (2, 10))

    def test_configurable_models_forward_to_requested_num_classes(self) -> None:
        cases = [
            (LeNetBuilder(), (2, 1, 28, 28)),
            (ResNet9Builder(), (2, 3, 32, 32)),
            (ResNet18Builder(), (2, 3, 32, 32)),
            (ResNet34Builder(), (2, 3, 32, 32)),
            (InceptionBuilder(), (2, 3, 75, 75)),
        ]

        for builder, input_shape in cases:
            with self.subTest(model=builder.name):
                _, channels, height, width = input_shape
                config = ExperimentConfig.from_run_config(
                    {
                        "input-channels": channels,
                        "input-height": height,
                        "input-width": width,
                        "num-classes": 7,
                    }
                )
                model = builder.build_model(config)
                model.eval()

                with torch.no_grad():
                    output = model(torch.zeros(*input_shape))

                self.assertEqual(tuple(output.shape), (2, 7))

    def test_default_components_include_common_models(self) -> None:
        register_default_components()

        for model_name in ["lenet", "resnet9", "resnet18", "resnet34", "inception"]:
            self.assertIn(model_name, MODELS.names())

    def test_inception_rejects_too_small_input_size(self) -> None:
        config = ExperimentConfig.from_run_config(
            {
                "model": "inception",
                "input-channels": 3,
                "input-height": 32,
                "input-width": 32,
                "num-classes": 10,
            }
        )

        with self.assertRaisesRegex(ValueError, "Inception requires input height and width"):
            InceptionBuilder().build_model(config)

    def test_fedavg_builder_creates_flower_strategy(self) -> None:
        config = ExperimentConfig.from_run_config({"num-supernodes": 4})
        model = MnistCnnBuilder().build_model(config)

        strategy = FedAvgBuilder().build_strategy(config, model, evaluate_fn=None)

        self.assertIsInstance(strategy, FedAvg)
        self.assertEqual(strategy.min_fit_clients, 4)

    def test_fedprox_builder_creates_flower_strategy(self) -> None:
        config = ExperimentConfig.from_run_config(
            {"num-supernodes": 4, "proximal-mu": 0.25}
        )
        model = MnistCnnBuilder().build_model(config)

        strategy = FedProxBuilder().build_strategy(config, model, evaluate_fn=None)

        self.assertIsInstance(strategy, FedProx)
        self.assertEqual(strategy.min_fit_clients, 4)
        self.assertEqual(strategy.proximal_mu, 0.25)

    def test_fedavgm_builder_creates_flower_strategy(self) -> None:
        config = ExperimentConfig.from_run_config(
            {
                "num-supernodes": 4,
                "server-learning-rate": 0.8,
                "server-momentum": 0.95,
            }
        )
        model = MnistCnnBuilder().build_model(config)

        strategy = FedAvgMBuilder().build_strategy(config, model, evaluate_fn=None)

        self.assertIsInstance(strategy, FedAvgM)
        self.assertEqual(strategy.min_fit_clients, 4)
        self.assertEqual(strategy.server_learning_rate, 0.8)
        self.assertEqual(strategy.server_momentum, 0.95)

    def test_fednova_builder_creates_strategy(self) -> None:
        config = ExperimentConfig.from_run_config(
            {
                "num-supernodes": 4,
                "fednova-server-momentum": 0.3,
            }
        )
        model = MnistCnnBuilder().build_model(config)

        strategy = FedNovaBuilder().build_strategy(config, model, evaluate_fn=None)
        fit_config = strategy.on_fit_config_fn(1)

        self.assertIsInstance(strategy, FedNovaStrategy)
        self.assertEqual(strategy.min_fit_clients, 4)
        self.assertEqual(strategy.server_momentum, 0.3)
        self.assertEqual(fit_config["algorithm"], "fednova")

    def test_fedper_builder_creates_shared_only_strategy(self) -> None:
        config = ExperimentConfig.from_run_config(
            {"num-supernodes": 4, "fedper-personal-layers": 1}
        )
        model = MnistCnnBuilder().build_model(config)
        full_parameter_count = len(get_model_parameters(model))

        strategy = FedPerBuilder().build_strategy(config, model, evaluate_fn=None)
        fit_config = strategy.on_fit_config_fn(1)

        self.assertIsInstance(strategy, FedPerStrategy)
        self.assertEqual(strategy.min_fit_clients, 4)
        self.assertEqual(fit_config["algorithm"], "fedper")
        self.assertLess(len(strategy.shared_parameter_indices), full_parameter_count)

    def test_fedrep_builder_creates_representation_strategy(self) -> None:
        config = ExperimentConfig.from_run_config(
            {
                "num-supernodes": 4,
                "fedrep-personal-layers": 1,
                "fedrep-representation-epochs": 2,
            }
        )
        model = MnistCnnBuilder().build_model(config)
        full_parameter_count = len(get_model_parameters(model))

        strategy = FedRepBuilder().build_strategy(config, model, evaluate_fn=None)
        fit_config = strategy.on_fit_config_fn(1)

        self.assertIsInstance(strategy, FedRepStrategy)
        self.assertEqual(strategy.min_fit_clients, 4)
        self.assertEqual(fit_config["algorithm"], "fedrep")
        self.assertEqual(fit_config["fedrep_representation_epochs"], 2)
        self.assertLess(len(strategy.shared_parameter_indices), full_parameter_count)

    def test_scaffold_builder_creates_strategy(self) -> None:
        config = ExperimentConfig.from_run_config({"num-supernodes": 4})
        model = MnistCnnBuilder().build_model(config)

        strategy = ScaffoldBuilder().build_strategy(config, model, evaluate_fn=None)

        self.assertIsInstance(strategy, ScaffoldStrategy)
        self.assertEqual(strategy.min_fit_clients, 4)

    def test_moon_builder_creates_strategy(self) -> None:
        config = ExperimentConfig.from_run_config(
            {"num-supernodes": 4, "moon-mu": 1.25, "moon-temperature": 0.6}
        )
        model = MnistCnnBuilder().build_model(config)

        strategy = MoonBuilder().build_strategy(config, model, evaluate_fn=None)
        fit_config = strategy.on_fit_config_fn(1)

        self.assertIsInstance(strategy, MoonStrategy)
        self.assertEqual(strategy.min_fit_clients, 4)
        self.assertEqual(fit_config["algorithm"], "moon")
        self.assertEqual(fit_config["moon_mu"], 1.25)
        self.assertEqual(fit_config["moon_temperature"], 0.6)

    def test_scaffold_strategy_aggregates_model_and_server_control(self) -> None:
        model = torch.nn.Linear(2, 2)
        model_parameters = get_model_parameters(model)
        control_parameters = [
            np.zeros_like(parameter.detach().numpy()) for parameter in model.parameters()
        ]

        with tempfile.TemporaryDirectory() as output_dir:
            strategy = ScaffoldStrategy(
                fraction_fit=1.0,
                fraction_evaluate=1.0,
                min_fit_clients=2,
                min_evaluate_clients=2,
                min_available_clients=2,
                initial_parameters=ndarrays_to_parameters(model_parameters),
                num_total_clients=4,
                server_control=control_parameters,
                checkpoint_model=model,
                output_dir=output_dir,
            )

            first_model = [parameter + 1.0 for parameter in model_parameters]
            second_model = [parameter + 3.0 for parameter in model_parameters]
            first_delta = [np.ones_like(control) for control in control_parameters]
            second_delta = [2.0 * np.ones_like(control) for control in control_parameters]
            results = [
                (
                    None,
                    FitRes(
                        Status(Code.OK, ""),
                        ndarrays_to_parameters(first_model + first_delta),
                        1,
                        {},
                    ),
                ),
                (
                    None,
                    FitRes(
                        Status(Code.OK, ""),
                        ndarrays_to_parameters(second_model + second_delta),
                        3,
                        {},
                    ),
                ),
            ]

            aggregated_parameters, _ = strategy.aggregate_fit(1, results, [])

        aggregated_model = parameters_to_ndarrays(aggregated_parameters)
        for aggregated, first, second in zip(
            aggregated_model, first_model, second_model
        ):
            np.testing.assert_allclose(aggregated, (first + 3.0 * second) / 4.0)
        for server_control in strategy.server_control:
            np.testing.assert_allclose(server_control, 0.75 * np.ones_like(server_control))

    def test_fednova_strategy_applies_normalized_client_updates(self) -> None:
        model = torch.nn.Linear(2, 1, bias=False)
        initial_parameters = [np.array([[10.0, 20.0]], dtype=np.float32)]

        with tempfile.TemporaryDirectory() as output_dir:
            strategy = FedNovaStrategy(
                fraction_fit=1.0,
                fraction_evaluate=1.0,
                min_fit_clients=2,
                min_evaluate_clients=2,
                min_available_clients=2,
                initial_parameters=ndarrays_to_parameters(initial_parameters),
                server_momentum=0.0,
                checkpoint_model=model,
                output_dir=output_dir,
            )

            first_update = [np.array([[2.0, 4.0]], dtype=np.float32)]
            second_update = [np.array([[1.0, 3.0]], dtype=np.float32)]
            results = [
                (
                    None,
                    FitRes(
                        Status(Code.OK, ""),
                        ndarrays_to_parameters(first_update),
                        2,
                        {"local_norm": 2.0},
                    ),
                ),
                (
                    None,
                    FitRes(
                        Status(Code.OK, ""),
                        ndarrays_to_parameters(second_update),
                        6,
                        {"local_norm": 4.0},
                    ),
                ),
            ]

            aggregated_parameters, _ = strategy.aggregate_fit(1, results, [])

        expected_update = np.array([[1.4, 3.4]], dtype=np.float32)
        expected_parameters = initial_parameters[0] - expected_update
        np.testing.assert_allclose(
            parameters_to_ndarrays(aggregated_parameters)[0],
            expected_parameters,
        )

    def test_fedper_strategy_aggregates_only_shared_parameters(self) -> None:
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
        full_parameters = get_model_parameters(model)

        with tempfile.TemporaryDirectory() as output_dir:
            strategy = FedPerStrategy(
                fraction_fit=1.0,
                fraction_evaluate=1.0,
                min_fit_clients=2,
                min_evaluate_clients=2,
                min_available_clients=2,
                initial_parameters=ndarrays_to_parameters(full_parameters[:-2]),
                shared_parameter_indices=[0, 1],
                personal_parameter_indices=[2, 3],
                checkpoint_model=model,
                output_dir=output_dir,
            )
            first_shared = [parameter + 1.0 for parameter in full_parameters[:-2]]
            second_shared = [parameter + 3.0 for parameter in full_parameters[:-2]]
            results = [
                (
                    None,
                    FitRes(
                        Status(Code.OK, ""),
                        ndarrays_to_parameters(first_shared),
                        1,
                        {},
                    ),
                ),
                (
                    None,
                    FitRes(
                        Status(Code.OK, ""),
                        ndarrays_to_parameters(second_shared),
                        3,
                        {},
                    ),
                ),
            ]

            aggregated_parameters, _ = strategy.aggregate_fit(1, results, [])

        aggregated_shared = parameters_to_ndarrays(aggregated_parameters)
        self.assertEqual(len(aggregated_shared), 2)
        for aggregated, first, second in zip(
            aggregated_shared,
            first_shared,
            second_shared,
        ):
            np.testing.assert_allclose(aggregated, (first + 3.0 * second) / 4.0)

    def test_fednova_client_fit_returns_updates_for_any_model_shape(self) -> None:
        config = ExperimentConfig.from_run_config({"algorithm": "fednova"})
        model = torch.nn.Linear(3, 2)
        loader = DataLoader(
            TensorDataset(torch.ones(4, 3), torch.zeros(4, dtype=torch.long)),
            batch_size=2,
        )
        client = TorchFlowerClient(
            model,
            loaders=type(
                "Loaders",
                (),
                {"train": loader, "validation": loader},
            )(),
            config=config,
        )
        initial_parameters = get_model_parameters(model)

        updates, num_examples, metrics = client.fit(
            initial_parameters,
            {"algorithm": "fednova", "local_epochs": 1, "learning_rate": 0.1},
        )

        self.assertEqual(num_examples, 4)
        self.assertEqual(len(updates), len(initial_parameters))
        self.assertEqual(updates[0].shape, initial_parameters[0].shape)
        self.assertGreater(metrics["local_norm"], 0.0)
        self.assertGreater(metrics["tau"], 0.0)

    def test_fedper_client_fit_returns_shared_parameters_and_saves_personal_state(self) -> None:
        with tempfile.TemporaryDirectory() as output_dir:
            config = ExperimentConfig.from_run_config(
                {
                    "algorithm": "fedper",
                    "output-dir": output_dir,
                    "fedper-personal-layers": 1,
                }
            )
            model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Linear(2, 2))
            loader = DataLoader(
                TensorDataset(torch.ones(4, 3), torch.zeros(4, dtype=torch.long)),
                batch_size=2,
            )
            client = TorchFlowerClient(
                model,
                loaders=type(
                    "Loaders",
                    (),
                    {"train": loader, "validation": loader},
                )(),
                config=config,
                client_id="client-1",
            )
            initial_parameters = get_model_parameters(model)

            shared_parameters, num_examples, metrics = client.fit(
                initial_parameters[:2],
                {"algorithm": "fedper", "local_epochs": 1, "learning_rate": 0.1},
            )

            self.assertEqual(num_examples, 4)
            self.assertEqual(len(shared_parameters), 2)
            self.assertIn("train_loss", metrics)
            self.assertTrue(
                (Path(output_dir) / "fedper_clients" / "client-1" / "personal.pt").exists()
            )

    def test_fedrep_client_fit_returns_shared_parameters_and_saves_personal_state(self) -> None:
        with tempfile.TemporaryDirectory() as output_dir:
            config = ExperimentConfig.from_run_config(
                {
                    "algorithm": "fedrep",
                    "output-dir": output_dir,
                    "fedrep-personal-layers": 1,
                    "fedrep-representation-epochs": 1,
                }
            )
            model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Linear(2, 2))
            loader = DataLoader(
                TensorDataset(torch.ones(4, 3), torch.zeros(4, dtype=torch.long)),
                batch_size=2,
            )
            client = TorchFlowerClient(
                model,
                loaders=type(
                    "Loaders",
                    (),
                    {"train": loader, "validation": loader},
                )(),
                config=config,
                client_id="client-1",
            )
            initial_parameters = get_model_parameters(model)

            shared_parameters, num_examples, metrics = client.fit(
                initial_parameters[:2],
                {
                    "algorithm": "fedrep",
                    "local_epochs": 1,
                    "learning_rate": 0.1,
                    "fedrep_representation_epochs": 1,
                },
            )

            self.assertEqual(num_examples, 4)
            self.assertEqual(len(shared_parameters), 2)
            self.assertIn("fedrep_head_train_loss", metrics)
            self.assertIn("fedrep_representation_train_loss", metrics)
            self.assertTrue(
                (Path(output_dir) / "fedrep_clients" / "client-1" / "personal.pt").exists()
            )

    def test_scaffold_training_returns_control_delta(self) -> None:
        model = torch.nn.Linear(2, 2)
        loader = DataLoader(
            TensorDataset(torch.ones(4, 2), torch.zeros(4, dtype=torch.long)),
            batch_size=2,
        )
        parameters = [parameter.detach().numpy() for parameter in model.parameters()]
        zero_controls = [torch.zeros_like(parameter).numpy() for parameter in model.parameters()]

        metrics, new_control, control_delta = train_scaffold_client(
            model,
            loader,
            epochs=1,
            learning_rate=0.1,
            device="cpu",
            server_control=zero_controls,
            client_control=zero_controls,
        )

        self.assertIn("train_loss", metrics)
        self.assertEqual(len(new_control), len(parameters))
        self.assertEqual(len(control_delta), len(parameters))
        self.assertEqual(new_control[0].shape, parameters[0].shape)

    def test_moon_training_returns_contrastive_metrics(self) -> None:
        model = torch.nn.Linear(2, 2)
        global_model = torch.nn.Linear(2, 2)
        previous_model = torch.nn.Linear(2, 2)
        global_model.load_state_dict(model.state_dict())
        previous_model.load_state_dict(model.state_dict())
        loader = DataLoader(
            TensorDataset(torch.ones(4, 2), torch.zeros(4, dtype=torch.long)),
            batch_size=2,
        )

        metrics = train_moon_client(
            model,
            global_model,
            previous_model,
            loader,
            epochs=1,
            learning_rate=0.1,
            device="cpu",
            moon_mu=1.0,
            temperature=0.5,
        )

        self.assertIn("train_loss", metrics)
        self.assertIn("moon_contrastive_loss", metrics)
        self.assertIn("moon_ce_loss", metrics)

    def test_training_loop_accepts_fedprox_proximal_term(self) -> None:
        model = torch.nn.Linear(2, 2)
        loader = DataLoader(
            TensorDataset(torch.zeros(4, 2), torch.zeros(4, dtype=torch.long)),
            batch_size=2,
        )

        metrics = train_one_client(
            model,
            loader,
            epochs=1,
            learning_rate=0.1,
            device="cpu",
            proximal_mu=0.1,
        )

        self.assertIn("train_loss", metrics)
        self.assertIn("train_accuracy", metrics)


if __name__ == "__main__":
    unittest.main()
