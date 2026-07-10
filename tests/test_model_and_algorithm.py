import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAdagrad, FedAdam, FedAvg, FedAvgM, FedProx, FedYogi

import fl_baselines.clients.torch_client as torch_client_module
from fl_baselines.clients.torch_client import TorchFlowerClient, get_model_parameters
from fl_baselines.algorithms.fedavg import FedAvgBuilder
from fl_baselines.algorithms.fedavgm import FedAvgMBuilder
from fl_baselines.algorithms.fedadagrad import FedAdagradBuilder
from fl_baselines.algorithms.fedadam import FedAdamBuilder
from fl_baselines.algorithms.fedadp import FedAdpBuilder, FedAdpStrategy
from fl_baselines.algorithms.fedyogi import FedYogiBuilder
from fl_baselines.algorithms.gamf import GAMFBuilder, GAMFStrategy
from fl_baselines.algorithms.fedma import FedMABuilder, FedMAStrategy
from fl_baselines.algorithms.fedcda import FedCDABuilder, FedCDAStrategy
from fl_baselines.algorithms.fedgen import FedGENBuilder, FedGENStrategy
from fl_baselines.algorithms.feddrl import FedDRLBuilder, FedDRLStrategy
from fl_baselines.algorithms.fedlaw import FedLAWBuilder, FedLAWStrategy
from fl_baselines.algorithms.ditto import DittoBuilder
from fl_baselines.algorithms.feddc import FedDCBuilder, FedDCStrategy
from fl_baselines.algorithms.feddecorr import FedDecorrBuilder
from fl_baselines.algorithms.fedaaw import FedAAWBuilder, FedAAWStrategy
from fl_baselines.algorithms.feddisco import FedDiscoBuilder, FedDiscoStrategy
from fl_baselines.algorithms.fedent import FedEntBuilder
from fl_baselines.algorithms.fedvck import FedVCKBuilder, FedVCKStrategy
from fl_baselines.algorithms.feddyn import FedDynBuilder, FedDynStrategy
from fl_baselines.algorithms.fedexp import FedExPBuilder, FedExPStrategy
from fl_baselines.algorithms.fedsam import FedSAMBuilder
from fl_baselines.algorithms.fedspeed import FedSpeedBuilder
from fl_baselines.algorithms.fedntd import FedNTDBuilder
from fl_baselines.algorithms.fedlc import FedLCBuilder
from fl_baselines.algorithms.fedrs import FedRSBuilder
from fl_baselines.algorithms.fedproto import FedProtoBuilder, FedProtoStrategy
from fl_baselines.algorithms.pfedme import PFedMeBuilder, PFedMeStrategy
from fl_baselines.algorithms.fedper import FedPerBuilder, FedPerStrategy
from fl_baselines.algorithms.fedrep import FedRepBuilder, FedRepStrategy
from fl_baselines.algorithms.fedala import FedALABuilder
from fl_baselines.algorithms.fedlama import FedLAMABuilder, FedLAMAStrategy
from fl_baselines.algorithms.fedamp import FedAMPBuilder, FedAMPStrategy
from fl_baselines.algorithms.fedlaa import FedLAABuilder, FedLAAStrategy
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
from fl_baselines.training.features import extract_features
from fl_baselines.training.feddecorr import feddecorr_loss, train_feddecorr_client
from fl_baselines.training.fedaaw import train_fedaaw_client
from fl_baselines.training.fedma import train_fedma_client
from fl_baselines.training.fedgen import evaluate_fedgen_model, train_fedgen_client
from fl_baselines.training.feddrl import compute_feddrl_reward, train_feddrl_client
from fl_baselines.training.feddisco import (
    compute_label_distribution,
    compute_label_distribution_discrepancy,
    train_feddisco_client,
)
from fl_baselines.training.fedent import (
    apply_fedent_eta_decay,
    compute_fedent_learning_rate,
    train_fedent_client,
)
from fl_baselines.training.fedvck import train_fedvck_client
from fl_baselines.training.fedsam import train_fedsam_client
from fl_baselines.training.fedspeed import train_fedspeed_client
from fl_baselines.training.fedntd import train_fedntd_client
from fl_baselines.training.fedlc import (
    fedlc_loss,
    local_class_counts,
    train_fedlc_client,
)
from fl_baselines.training.fedrs import (
    fedrs_loss,
    observed_class_mask,
    train_fedrs_client,
)
from fl_baselines.training.fedlama import (
    compute_fedlama_layer_discrepancies,
    fedlama_sync_mask_for_round,
    parse_fedlama_sync_mask,
    train_fedlama_client,
)
from fl_baselines.training.moon import train_moon_client
from fl_baselines.training.fedala import adaptive_local_aggregation
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
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1", metrics)

    def test_evaluate_model_returns_macro_classification_metrics(self) -> None:
        class FixedLogitModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer(
                    "logits",
                    torch.tensor(
                        [
                            [5.0, 1.0, 0.0],
                            [1.0, 5.0, 0.0],
                            [0.0, 5.0, 1.0],
                            [1.0, 0.0, 5.0],
                        ],
                        dtype=torch.float32,
                    ),
                )

            def forward(self, inputs):
                indices = inputs.squeeze(-1).long()
                return self.logits[indices]

        loader = DataLoader(
            TensorDataset(
                torch.arange(4, dtype=torch.float32).unsqueeze(-1),
                torch.tensor([0, 1, 2, 2], dtype=torch.long),
            ),
            batch_size=2,
        )

        _, metrics = evaluate_model(FixedLogitModel(), loader, device="cpu")

        self.assertAlmostEqual(metrics["accuracy"], 0.75, places=6)
        self.assertAlmostEqual(metrics["precision"], 5.0 / 6.0, places=6)
        self.assertAlmostEqual(metrics["recall"], 5.0 / 6.0, places=6)
        self.assertAlmostEqual(metrics["f1"], 7.0 / 9.0, places=6)

    def test_evaluate_model_handles_missing_class_predictions_in_macro_metrics(self) -> None:
        class MissingPredictionModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer(
                    "logits",
                    torch.tensor(
                        [
                            [5.0, 1.0, 0.0],
                            [5.0, 1.0, 0.0],
                            [0.0, 5.0, 1.0],
                        ],
                        dtype=torch.float32,
                    ),
                )

            def forward(self, inputs):
                indices = inputs.squeeze(-1).long()
                return self.logits[indices]

        loader = DataLoader(
            TensorDataset(
                torch.arange(3, dtype=torch.float32).unsqueeze(-1),
                torch.tensor([0, 2, 2], dtype=torch.long),
            ),
            batch_size=3,
        )

        _, metrics = evaluate_model(MissingPredictionModel(), loader, device="cpu")

        self.assertAlmostEqual(metrics["accuracy"], 1.0 / 3.0, places=6)
        self.assertAlmostEqual(metrics["precision"], 1.0 / 6.0, places=6)
        self.assertAlmostEqual(metrics["recall"], 1.0 / 3.0, places=6)
        self.assertAlmostEqual(metrics["f1"], 2.0 / 9.0, places=6)

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

    def test_fedadam_builder_creates_flower_strategy(self) -> None:
        config = ExperimentConfig.from_run_config(
            {
                "num-supernodes": 4,
                "fedadam-eta": 0.2,
                "fedadam-eta-l": 0.03,
                "fedadam-beta-1": 0.8,
                "fedadam-beta-2": 0.97,
                "fedadam-tau": 1e-6,
            }
        )
        model = MnistCnnBuilder().build_model(config)

        strategy = FedAdamBuilder().build_strategy(config, model, evaluate_fn=None)
        fit_config = strategy.on_fit_config_fn(1)

        self.assertIsInstance(strategy, FedAdam)
        self.assertEqual(strategy.min_fit_clients, 4)
        self.assertEqual(strategy.eta, 0.2)
        self.assertEqual(strategy.eta_l, 0.03)
        self.assertEqual(strategy.beta_1, 0.8)
        self.assertEqual(strategy.beta_2, 0.97)
        self.assertEqual(strategy.tau, 1e-6)
        self.assertEqual(fit_config["algorithm"], "fedadam")

    def test_fedadagrad_builder_creates_flower_strategy(self) -> None:
        config = ExperimentConfig.from_run_config(
            {
                "num-supernodes": 4,
                "fedadagrad-eta": 0.15,
                "fedadagrad-eta-l": 0.04,
                "fedadagrad-tau": 1e-7,
            }
        )
        model = MnistCnnBuilder().build_model(config)

        strategy = FedAdagradBuilder().build_strategy(config, model, evaluate_fn=None)
        fit_config = strategy.on_fit_config_fn(1)

        self.assertIsInstance(strategy, FedAdagrad)
        self.assertEqual(strategy.min_fit_clients, 4)
        self.assertEqual(strategy.eta, 0.15)
        self.assertEqual(strategy.eta_l, 0.04)
        self.assertEqual(strategy.tau, 1e-7)
        self.assertEqual(fit_config["algorithm"], "fedadagrad")

    def test_fedyogi_builder_creates_flower_strategy(self) -> None:
        config = ExperimentConfig.from_run_config(
            {
                "num-supernodes": 4,
                "fedyogi-eta": 0.02,
                "fedyogi-eta-l": 0.025,
                "fedyogi-beta-1": 0.85,
                "fedyogi-beta-2": 0.96,
                "fedyogi-tau": 1e-4,
            }
        )
        model = MnistCnnBuilder().build_model(config)

        strategy = FedYogiBuilder().build_strategy(config, model, evaluate_fn=None)
        fit_config = strategy.on_fit_config_fn(1)

        self.assertIsInstance(strategy, FedYogi)
        self.assertEqual(strategy.min_fit_clients, 4)
        self.assertEqual(strategy.eta, 0.02)
        self.assertEqual(strategy.eta_l, 0.025)
        self.assertEqual(strategy.beta_1, 0.85)
        self.assertEqual(strategy.beta_2, 0.96)
        self.assertEqual(strategy.tau, 1e-4)
        self.assertEqual(fit_config["algorithm"], "fedyogi")

    def test_fedadp_builder_creates_strategy(self) -> None:
        config = ExperimentConfig.from_run_config(
            {
                "num-supernodes": 4,
                "fedadp-alpha": 4.0,
            }
        )
        model = MnistCnnBuilder().build_model(config)

        strategy = FedAdpBuilder().build_strategy(config, model, evaluate_fn=None)
        fit_config = strategy.on_fit_config_fn(1)

        self.assertIsInstance(strategy, FedAdpStrategy)
        self.assertEqual(strategy.min_fit_clients, 4)
        self.assertEqual(strategy.alpha, 4.0)
        self.assertEqual(fit_config["algorithm"], "fedadp")

    def test_gamf_builder_creates_strategy(self) -> None:
        config = ExperimentConfig.from_run_config(
            {
                "algorithm": "gamf",
                "num-supernodes": 4,
                "gamf-sigma": 2.0,
                "gamf-max-iters": 20,
            }
        )
        model = MnistCnnBuilder().build_model(config)

        strategy = GAMFBuilder().build_strategy(config, model, evaluate_fn=None)
        fit_config = strategy.on_fit_config_fn(1)

        self.assertIsInstance(strategy, GAMFStrategy)
        self.assertEqual(strategy.min_fit_clients, 4)
        self.assertEqual(strategy.sigma, 2.0)
        self.assertEqual(strategy.max_iters, 20)
        self.assertEqual(fit_config["algorithm"], "gamf")

    def test_gamf_builder_supports_lenet_and_mnist_cnn(self) -> None:
        cases = [
            (MnistCnnBuilder(), {}),
            (LeNetBuilder(), {}),
        ]

        for model_builder, overrides in cases:
            with self.subTest(model=model_builder.name):
                config = ExperimentConfig.from_run_config(
                    {"algorithm": "gamf", "num-supernodes": 2, **overrides}
                )
                model = model_builder.build_model(config)

                strategy = GAMFBuilder().build_strategy(config, model, evaluate_fn=None)

                self.assertIsInstance(strategy, GAMFStrategy)

    def test_gamf_builder_rejects_unsupported_models(self) -> None:
        cases = [
            (
                ResNet9Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                ResNet18Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                ResNet34Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                InceptionBuilder(),
                {"input-channels": 3, "input-height": 75, "input-width": 75},
            ),
        ]

        for model_builder, overrides in cases:
            with self.subTest(model=model_builder.name):
                config = ExperimentConfig.from_run_config(
                    {"algorithm": "gamf", "num-supernodes": 2, **overrides}
                )
                model = model_builder.build_model(config)

                with self.assertRaisesRegex(ValueError, "GAMF currently supports"):
                    GAMFBuilder().build_strategy(config, model, evaluate_fn=None)

    def test_fedma_builder_creates_strategy(self) -> None:
        config = ExperimentConfig.from_run_config(
            {
                "algorithm": "fedma",
                "num-supernodes": 4,
                "fedma-matching-epsilon": 0.0,
            }
        )
        model = MnistCnnBuilder().build_model(config)

        strategy = FedMABuilder().build_strategy(config, model, evaluate_fn=None)
        fit_config = strategy.on_fit_config_fn(1)

        self.assertIsInstance(strategy, FedMAStrategy)
        self.assertEqual(strategy.min_fit_clients, 4)
        self.assertEqual(fit_config["algorithm"], "fedma")
        self.assertEqual(fit_config["fedma_stage"], 0)
        self.assertEqual(fit_config["fedma_matching_epsilon"], 0.0)

    def test_fedma_builder_supports_lenet_and_mnist_cnn(self) -> None:
        cases = [
            (MnistCnnBuilder(), {}),
            (LeNetBuilder(), {}),
        ]

        for model_builder, overrides in cases:
            with self.subTest(model=model_builder.name):
                config = ExperimentConfig.from_run_config(
                    {"algorithm": "fedma", "num-supernodes": 2, **overrides}
                )
                model = model_builder.build_model(config)
                strategy = FedMABuilder().build_strategy(config, model, evaluate_fn=None)

                self.assertIsInstance(strategy, FedMAStrategy)

    def test_fedma_builder_rejects_unsupported_models(self) -> None:
        cases = [
            (
                ResNet9Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                ResNet18Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                ResNet34Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                InceptionBuilder(),
                {"input-channels": 3, "input-height": 75, "input-width": 75},
            ),
        ]

        for model_builder, overrides in cases:
            with self.subTest(model=model_builder.name):
                config = ExperimentConfig.from_run_config(
                    {"algorithm": "fedma", "num-supernodes": 2, **overrides}
                )
                model = model_builder.build_model(config)

                with self.assertRaisesRegex(ValueError, "FedMA currently supports"):
                    FedMABuilder().build_strategy(config, model, evaluate_fn=None)

    def test_fedadp_builder_supports_current_models(self) -> None:
        cases = [
            (MnistCnnBuilder(), {}),
            (LeNetBuilder(), {}),
            (
                ResNet9Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                ResNet18Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                ResNet34Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                InceptionBuilder(),
                {"input-channels": 3, "input-height": 75, "input-width": 75},
            ),
        ]

        for model_builder, overrides in cases:
            with self.subTest(model=model_builder.name):
                config = ExperimentConfig.from_run_config(
                    {"algorithm": "fedadp", "num-supernodes": 2, **overrides}
                )
                model = model_builder.build_model(config)

                strategy = FedAdpBuilder().build_strategy(
                    config,
                    model,
                    evaluate_fn=None,
                )

                self.assertIsInstance(strategy, FedAdpStrategy)

    def test_fedcda_builder_creates_strategy(self) -> None:
        config = ExperimentConfig.from_run_config(
            {
                "algorithm": "fedcda",
                "num-supernodes": 4,
                "fedcda-memory-size": 4,
                "fedcda-num-batches": 2,
                "fedcda-warmup-rounds": 3,
            }
        )
        model = MnistCnnBuilder().build_model(config)

        strategy = FedCDABuilder().build_strategy(config, model, evaluate_fn=None)
        fit_config = strategy.on_fit_config_fn(1)

        self.assertIsInstance(strategy, FedCDAStrategy)
        self.assertEqual(strategy.min_fit_clients, 4)
        self.assertEqual(fit_config["algorithm"], "fedcda")
        self.assertEqual(fit_config["local_epochs"], 1)
        self.assertEqual(fit_config["learning_rate"], 0.01)

    def test_fedcda_builder_supports_current_models(self) -> None:
        cases = [
            (MnistCnnBuilder(), {}),
            (LeNetBuilder(), {}),
            (
                ResNet9Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                ResNet18Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                ResNet34Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                InceptionBuilder(),
                {"input-channels": 3, "input-height": 75, "input-width": 75},
            ),
        ]

        for model_builder, overrides in cases:
            with self.subTest(model=model_builder.name):
                config = ExperimentConfig.from_run_config(
                    {"algorithm": "fedcda", "num-supernodes": 2, **overrides}
                )
                model = model_builder.build_model(config)
                strategy = FedCDABuilder().build_strategy(config, model, evaluate_fn=None)

                self.assertIsInstance(strategy, FedCDAStrategy)
                self.assertEqual(strategy.min_fit_clients, 2)

    def test_fedgen_builder_creates_strategy(self) -> None:
        config = ExperimentConfig.from_run_config(
            {
                "algorithm": "fedgen",
                "num-supernodes": 4,
                "fedgen-alpha": 1.5,
                "fedgen-warmup-epochs": 2,
            }
        )
        model = MnistCnnBuilder().build_model(config)

        strategy = FedGENBuilder().build_strategy(config, model, evaluate_fn=None)
        fit_config = strategy.on_fit_config_fn(1)

        self.assertIsInstance(strategy, FedGENStrategy)
        self.assertEqual(strategy.min_fit_clients, 4)
        self.assertEqual(fit_config["algorithm"], "fedgen")
        self.assertEqual(fit_config["fedgen_alpha"], 1.5)
        self.assertEqual(fit_config["fedgen_warmup_epochs"], 2)

    def test_fedgen_builder_supports_current_models(self) -> None:
        cases = [
            (MnistCnnBuilder(), {}),
            (LeNetBuilder(), {}),
            (
                ResNet9Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                ResNet18Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                ResNet34Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                InceptionBuilder(),
                {"input-channels": 3, "input-height": 75, "input-width": 75},
            ),
        ]

        for model_builder, overrides in cases:
            with self.subTest(model=model_builder.name):
                config = ExperimentConfig.from_run_config(
                    {"algorithm": "fedgen", "num-supernodes": 2, **overrides}
                )
                model = model_builder.build_model(config)
                strategy = FedGENBuilder().build_strategy(config, model, evaluate_fn=None)

                self.assertIsInstance(strategy, FedGENStrategy)
                self.assertEqual(strategy.min_fit_clients, 2)

    def test_fedexp_builder_creates_strategy(self) -> None:
        config = ExperimentConfig.from_run_config(
            {"num-supernodes": 4, "fedexp-epsilon": 0.01}
        )
        model = MnistCnnBuilder().build_model(config)

        strategy = FedExPBuilder().build_strategy(config, model, evaluate_fn=None)
        fit_config = strategy.on_fit_config_fn(1)

        self.assertIsInstance(strategy, FedExPStrategy)
        self.assertEqual(strategy.min_fit_clients, 4)
        self.assertEqual(strategy.epsilon, 0.01)
        self.assertEqual(fit_config["algorithm"], "fedexp")

    def test_fedexp_builder_supports_current_models(self) -> None:
        cases = [
            (MnistCnnBuilder(), {}),
            (LeNetBuilder(), {}),
            (
                ResNet9Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                ResNet18Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                ResNet34Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                InceptionBuilder(),
                {"input-channels": 3, "input-height": 75, "input-width": 75},
            ),
        ]

        for model_builder, overrides in cases:
            with self.subTest(model=model_builder.name):
                config = ExperimentConfig.from_run_config(
                    {"algorithm": "fedexp", "num-supernodes": 2, **overrides}
                )
                model = model_builder.build_model(config)
                strategy = FedExPBuilder().build_strategy(config, model, evaluate_fn=None)
                self.assertIsInstance(strategy, FedExPStrategy)

    def test_fedsam_builder_creates_strategy(self) -> None:
        config = ExperimentConfig.from_run_config(
            {"num-supernodes": 4, "fedsam-rho": 0.75}
        )
        model = MnistCnnBuilder().build_model(config)

        strategy = FedSAMBuilder().build_strategy(config, model, evaluate_fn=None)
        fit_config = strategy.on_fit_config_fn(1)

        self.assertIsInstance(strategy, FedAvg)
        self.assertEqual(strategy.min_fit_clients, 4)
        self.assertEqual(fit_config["algorithm"], "fedsam")
        self.assertEqual(fit_config["fedsam_rho"], 0.75)

    def test_fedsam_builder_supports_current_models(self) -> None:
        cases = [
            (MnistCnnBuilder(), {}),
            (LeNetBuilder(), {}),
            (
                ResNet9Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                ResNet18Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                ResNet34Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                InceptionBuilder(),
                {"input-channels": 3, "input-height": 75, "input-width": 75},
            ),
        ]

        for model_builder, overrides in cases:
            with self.subTest(model=model_builder.name):
                config = ExperimentConfig.from_run_config(
                    {"algorithm": "fedsam", "num-supernodes": 2, **overrides}
                )
                model = model_builder.build_model(config)
                strategy = FedSAMBuilder().build_strategy(config, model, evaluate_fn=None)
                self.assertIsInstance(strategy, FedAvg)

    def test_fedent_builder_creates_strategy(self) -> None:
        config = ExperimentConfig.from_run_config(
            {
                "algorithm": "fedent",
                "num-supernodes": 4,
                "fedent-beta": 0.99,
                "fedent-gamma": 0.95,
            }
        )
        model = MnistCnnBuilder().build_model(config)

        strategy = FedEntBuilder().build_strategy(config, model, evaluate_fn=None)
        fit_config = strategy.on_fit_config_fn(1)

        self.assertIsInstance(strategy, FedAvg)
        self.assertEqual(strategy.min_fit_clients, 4)
        self.assertEqual(fit_config["algorithm"], "fedent")
        self.assertIn("fedent_phi1", fit_config)
        self.assertIn("fedent_phi2", fit_config)
        self.assertEqual(fit_config["fedent_beta"], 0.99)
        self.assertEqual(fit_config["fedent_gamma"], 0.95)

    def test_feddrl_builder_creates_strategy(self) -> None:
        config = ExperimentConfig.from_run_config(
            {
                "algorithm": "feddrl",
                "num-supernodes": 4,
                "feddrl-hidden-size": 128,
                "feddrl-updates-per-round": 2,
            }
        )
        model = MnistCnnBuilder().build_model(config)

        strategy = FedDRLBuilder().build_strategy(config, model, evaluate_fn=None)
        fit_config = strategy.on_fit_config_fn(1)

        self.assertIsInstance(strategy, FedDRLStrategy)
        self.assertEqual(strategy.min_fit_clients, 4)
        self.assertEqual(fit_config["algorithm"], "feddrl")
        self.assertEqual(fit_config["local_epochs"], 1)
        self.assertEqual(fit_config["learning_rate"], 0.01)

    def test_feddrl_builder_supports_current_models(self) -> None:
        cases = [
            (MnistCnnBuilder(), {}),
            (LeNetBuilder(), {}),
            (
                ResNet9Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                ResNet18Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                ResNet34Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                InceptionBuilder(),
                {"input-channels": 3, "input-height": 75, "input-width": 75},
            ),
        ]

        for model_builder, overrides in cases:
            with self.subTest(model=model_builder.name):
                config = ExperimentConfig.from_run_config(
                    {"algorithm": "feddrl", "num-supernodes": 2, **overrides}
                )
                model = model_builder.build_model(config)
                strategy = FedDRLBuilder().build_strategy(config, model, evaluate_fn=None)

                self.assertIsInstance(strategy, FedDRLStrategy)
                self.assertEqual(strategy.min_fit_clients, 2)

    def test_fedent_builder_supports_current_models(self) -> None:
        cases = [
            (MnistCnnBuilder(), {}),
            (LeNetBuilder(), {}),
            (
                ResNet9Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                ResNet18Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                ResNet34Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                InceptionBuilder(),
                {"input-channels": 3, "input-height": 75, "input-width": 75},
            ),
        ]

        for model_builder, overrides in cases:
            with self.subTest(model=model_builder.name):
                config = ExperimentConfig.from_run_config(
                    {"algorithm": "fedent", "num-supernodes": 2, **overrides}
                )
                model = model_builder.build_model(config)
                strategy = FedEntBuilder().build_strategy(config, model, evaluate_fn=None)

                self.assertIsInstance(strategy, FedAvg)
                self.assertEqual(strategy.min_fit_clients, 2)

    def test_fedaaw_builder_creates_strategy(self) -> None:
        config = ExperimentConfig.from_run_config(
            {
                "algorithm": "fedaaw",
                "num-supernodes": 4,
                "fedaaw-beta": 0.02,
                "fedaaw-gamma": 1.5,
            }
        )
        model = MnistCnnBuilder().build_model(config)

        strategy = FedAAWBuilder().build_strategy(config, model, evaluate_fn=None)
        fit_config = strategy.on_fit_config_fn(1)

        self.assertIsInstance(strategy, FedAAWStrategy)
        self.assertEqual(strategy.min_fit_clients, 4)
        self.assertEqual(fit_config["algorithm"], "fedaaw")
        self.assertEqual(fit_config["fedaaw_beta"], 0.02)
        self.assertEqual(fit_config["fedaaw_gamma"], 1.5)
        self.assertEqual(fit_config["fedaaw_epsilon"], 1e-8)

    def test_fedlaw_builder_creates_strategy(self) -> None:
        config = ExperimentConfig.from_run_config(
            {
                "algorithm": "fedlaw",
                "num-supernodes": 4,
                "fedlaw-server-epochs": 3,
                "fedlaw-server-learning-rate": 0.02,
                "fedlaw-gamma-init": 0.9,
            }
        )
        model = MnistCnnBuilder().build_model(config)

        strategy = FedLAWBuilder().build_strategy(config, model, evaluate_fn=None)
        fit_config = strategy.on_fit_config_fn(1)

        self.assertIsInstance(strategy, FedLAWStrategy)
        self.assertEqual(strategy.min_fit_clients, 4)
        self.assertEqual(fit_config["algorithm"], "fedlaw")
        self.assertEqual(fit_config["fedlaw_server_epochs"], 3)
        self.assertEqual(fit_config["fedlaw_server_learning_rate"], 0.02)
        self.assertEqual(fit_config["fedlaw_gamma_init"], 0.9)

    def test_fedlaw_builder_supports_current_models(self) -> None:
        cases = [
            (MnistCnnBuilder(), {}),
            (LeNetBuilder(), {}),
            (
                ResNet9Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                ResNet18Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                ResNet34Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                InceptionBuilder(),
                {"input-channels": 3, "input-height": 75, "input-width": 75},
            ),
        ]

        for model_builder, overrides in cases:
            with self.subTest(model=model_builder.name):
                config = ExperimentConfig.from_run_config(
                    {"algorithm": "fedlaw", "num-supernodes": 2, **overrides}
                )
                model = model_builder.build_model(config)
                strategy = FedLAWBuilder().build_strategy(config, model, evaluate_fn=None)

                self.assertIsInstance(strategy, FedLAWStrategy)
                self.assertEqual(strategy.on_fit_config_fn(1)["algorithm"], "fedlaw")

    def test_fedaaw_builder_supports_current_models(self) -> None:
        cases = [
            (MnistCnnBuilder(), {}),
            (LeNetBuilder(), {}),
            (
                ResNet9Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                ResNet18Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                ResNet34Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                InceptionBuilder(),
                {"input-channels": 3, "input-height": 75, "input-width": 75},
            ),
        ]

        for model_builder, overrides in cases:
            with self.subTest(model=model_builder.name):
                config = ExperimentConfig.from_run_config(
                    {"algorithm": "fedaaw", "num-supernodes": 2, **overrides}
                )
                model = model_builder.build_model(config)
                strategy = FedAAWBuilder().build_strategy(config, model, evaluate_fn=None)

                self.assertIsInstance(strategy, FedAAWStrategy)
                self.assertEqual(strategy.min_fit_clients, 2)

    def test_feddisco_builder_creates_strategy(self) -> None:
        config = ExperimentConfig.from_run_config(
            {
                "algorithm": "feddisco",
                "num-supernodes": 4,
                "feddisco-discrepancy-weight": 0.4,
                "feddisco-bias": 0.2,
                "feddisco-metric": "l2",
            }
        )
        model = MnistCnnBuilder().build_model(config)

        strategy = FedDiscoBuilder().build_strategy(config, model, evaluate_fn=None)
        fit_config = strategy.on_fit_config_fn(1)

        self.assertIsInstance(strategy, FedDiscoStrategy)
        self.assertEqual(strategy.min_fit_clients, 4)
        self.assertEqual(strategy.discrepancy_weight, 0.4)
        self.assertEqual(strategy.bias, 0.2)
        self.assertEqual(fit_config["algorithm"], "feddisco")
        self.assertEqual(fit_config["feddisco_metric"], "l2")

    def test_feddisco_builder_supports_current_models(self) -> None:
        cases = [
            (MnistCnnBuilder(), {}),
            (LeNetBuilder(), {}),
            (
                ResNet9Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                ResNet18Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                ResNet34Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                InceptionBuilder(),
                {"input-channels": 3, "input-height": 75, "input-width": 75},
            ),
        ]

        for model_builder, overrides in cases:
            with self.subTest(model=model_builder.name):
                config = ExperimentConfig.from_run_config(
                    {"algorithm": "feddisco", "num-supernodes": 2, **overrides}
                )
                model = model_builder.build_model(config)
                strategy = FedDiscoBuilder().build_strategy(
                    config,
                    model,
                    evaluate_fn=None,
                )

                self.assertIsInstance(strategy, FedDiscoStrategy)
                self.assertEqual(strategy.on_fit_config_fn(1)["algorithm"], "feddisco")

    def test_fedvck_builder_creates_strategy(self) -> None:
        config = ExperimentConfig.from_run_config(
            {
                "algorithm": "fedvck",
                "num-supernodes": 4,
                "fedvck-condensed-ratio": 0.02,
                "fedvck-condensed-steps": 3,
                "fedvck-server-replay-epochs": 2,
            }
        )
        model = MnistCnnBuilder().build_model(config)

        strategy = FedVCKBuilder().build_strategy(config, model, evaluate_fn=None)
        fit_config = strategy.on_fit_config_fn(1)

        self.assertIsInstance(strategy, FedVCKStrategy)
        self.assertEqual(strategy.min_fit_clients, 4)
        self.assertEqual(fit_config["algorithm"], "fedvck")
        self.assertEqual(fit_config["fedvck_condensed_ratio"], 0.02)
        self.assertEqual(fit_config["fedvck_condensed_steps"], 3)
        self.assertEqual(fit_config["fedvck_server_replay_epochs"], 2)

    def test_fedvck_builder_supports_current_models(self) -> None:
        cases = [
            (MnistCnnBuilder(), {}),
            (LeNetBuilder(), {}),
            (
                ResNet9Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                ResNet18Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                ResNet34Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                InceptionBuilder(),
                {"input-channels": 3, "input-height": 75, "input-width": 75},
            ),
        ]

        for model_builder, overrides in cases:
            with self.subTest(model=model_builder.name):
                config = ExperimentConfig.from_run_config(
                    {"algorithm": "fedvck", "num-supernodes": 2, **overrides}
                )
                model = model_builder.build_model(config)
                strategy = FedVCKBuilder().build_strategy(config, model, evaluate_fn=None)

                self.assertIsInstance(strategy, FedVCKStrategy)
                self.assertEqual(strategy.min_fit_clients, 2)

    def test_feddyn_builder_creates_strategy(self) -> None:
        config = ExperimentConfig.from_run_config(
            {"num-supernodes": 4, "feddyn-alpha": 0.2}
        )
        model = MnistCnnBuilder().build_model(config)

        strategy = FedDynBuilder().build_strategy(config, model, evaluate_fn=None)
        fit_config = strategy.on_fit_config_fn(1)

        self.assertIsInstance(strategy, FedDynStrategy)
        self.assertEqual(strategy.min_fit_clients, 4)
        self.assertEqual(strategy.alpha, 0.2)
        self.assertEqual(fit_config["algorithm"], "feddyn")
        self.assertEqual(fit_config["feddyn_alpha"], 0.2)

    def test_feddyn_builder_supports_current_models(self) -> None:
        cases = [
            (MnistCnnBuilder(), {}),
            (LeNetBuilder(), {}),
            (
                ResNet9Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                ResNet18Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                ResNet34Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                InceptionBuilder(),
                {"input-channels": 3, "input-height": 75, "input-width": 75},
            ),
        ]

        for model_builder, overrides in cases:
            with self.subTest(model=model_builder.name):
                config = ExperimentConfig.from_run_config(
                    {"algorithm": "feddyn", "num-supernodes": 2, **overrides}
                )
                model = model_builder.build_model(config)

                strategy = FedDynBuilder().build_strategy(
                    config,
                    model,
                    evaluate_fn=None,
                )

                self.assertIsInstance(strategy, FedDynStrategy)

    def test_feddc_builder_creates_strategy(self) -> None:
        config = ExperimentConfig.from_run_config(
            {"num-supernodes": 4, "feddc-alpha": 0.05}
        )
        model = MnistCnnBuilder().build_model(config)

        strategy = FedDCBuilder().build_strategy(config, model, evaluate_fn=None)
        fit_config = strategy.on_fit_config_fn(1)

        self.assertIsInstance(strategy, FedDCStrategy)
        self.assertEqual(strategy.min_fit_clients, 4)
        self.assertEqual(strategy.alpha, 0.05)
        self.assertEqual(fit_config["algorithm"], "feddc")
        self.assertEqual(fit_config["feddc_alpha"], 0.05)

    def test_feddc_builder_supports_current_models(self) -> None:
        cases = [
            (MnistCnnBuilder(), {}),
            (LeNetBuilder(), {}),
            (
                ResNet9Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                ResNet18Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                ResNet34Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                InceptionBuilder(),
                {"input-channels": 3, "input-height": 75, "input-width": 75},
            ),
        ]

        for model_builder, overrides in cases:
            with self.subTest(model=model_builder.name):
                config = ExperimentConfig.from_run_config(
                    {"algorithm": "feddc", "num-supernodes": 2, **overrides}
                )
                model = model_builder.build_model(config)
                strategy = FedDCBuilder().build_strategy(config, model, evaluate_fn=None)
                self.assertEqual(strategy.on_fit_config_fn(1)["algorithm"], "feddc")

    def test_feddecorr_builder_creates_strategy(self) -> None:
        config = ExperimentConfig.from_run_config(
            {"num-supernodes": 4, "feddecorr-beta": 0.3}
        )
        model = MnistCnnBuilder().build_model(config)

        strategy = FedDecorrBuilder().build_strategy(config, model, evaluate_fn=None)
        fit_config = strategy.on_fit_config_fn(1)

        self.assertIsInstance(strategy, FedAvg)
        self.assertEqual(strategy.min_fit_clients, 4)
        self.assertEqual(fit_config["algorithm"], "feddecorr")
        self.assertEqual(fit_config["feddecorr_beta"], 0.3)

    def test_feddecorr_builder_supports_current_models(self) -> None:
        cases = [
            (MnistCnnBuilder(), {"input-channels": 1, "input-height": 28, "input-width": 28}),
            (LeNetBuilder(), {"input-channels": 1, "input-height": 28, "input-width": 28}),
            (
                ResNet9Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                ResNet18Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                ResNet34Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                InceptionBuilder(),
                {"input-channels": 3, "input-height": 75, "input-width": 75},
            ),
        ]

        for model_builder, overrides in cases:
            with self.subTest(model=model_builder.name):
                config = ExperimentConfig.from_run_config(
                    {"algorithm": "feddecorr", "num-supernodes": 2, **overrides}
                )
                model = model_builder.build_model(config)
                strategy = FedDecorrBuilder().build_strategy(config, model, evaluate_fn=None)
                self.assertEqual(strategy.on_fit_config_fn(1)["algorithm"], "feddecorr")

    def test_fedspeed_builder_creates_strategy(self) -> None:
        config = ExperimentConfig.from_run_config(
            {
                "num-supernodes": 4,
                "fedspeed-lambda": 0.2,
                "fedspeed-alpha": 0.8,
                "fedspeed-rho": 0.15,
            }
        )
        model = MnistCnnBuilder().build_model(config)

        strategy = FedSpeedBuilder().build_strategy(config, model, evaluate_fn=None)
        fit_config = strategy.on_fit_config_fn(1)

        self.assertIsInstance(strategy, FedAvg)
        self.assertEqual(strategy.min_fit_clients, 4)
        self.assertEqual(fit_config["algorithm"], "fedspeed")
        self.assertEqual(fit_config["fedspeed_lambda"], 0.2)
        self.assertEqual(fit_config["fedspeed_alpha"], 0.8)
        self.assertEqual(fit_config["fedspeed_rho"], 0.15)

    def test_fedspeed_builder_supports_current_models(self) -> None:
        cases = [
            (MnistCnnBuilder(), {}),
            (LeNetBuilder(), {}),
            (
                ResNet9Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                ResNet18Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                ResNet34Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                InceptionBuilder(),
                {"input-channels": 3, "input-height": 75, "input-width": 75},
            ),
        ]

        for model_builder, overrides in cases:
            with self.subTest(model=model_builder.name):
                config = ExperimentConfig.from_run_config(
                    {"algorithm": "fedspeed", "num-supernodes": 2, **overrides}
                )
                model = model_builder.build_model(config)
                strategy = FedSpeedBuilder().build_strategy(config, model, evaluate_fn=None)
                self.assertEqual(strategy.on_fit_config_fn(1)["algorithm"], "fedspeed")

    def test_fedproto_builder_creates_strategy(self) -> None:
        config = ExperimentConfig.from_run_config(
            {"num-supernodes": 4, "fedproto-lambda": 0.2}
        )
        model = MnistCnnBuilder().build_model(config)

        strategy = FedProtoBuilder().build_strategy(config, model, evaluate_fn=None)
        fit_config = strategy.on_fit_config_fn(1)

        self.assertIsInstance(strategy, FedProtoStrategy)
        self.assertEqual(strategy.min_fit_clients, 4)
        self.assertEqual(fit_config["algorithm"], "fedproto")
        self.assertEqual(fit_config["fedproto_lambda"], 0.2)

    def test_fedproto_builder_supports_current_models(self) -> None:
        cases = [
            (MnistCnnBuilder(), {"input-channels": 1, "input-height": 28, "input-width": 28}),
            (LeNetBuilder(), {"input-channels": 1, "input-height": 28, "input-width": 28}),
            (
                ResNet9Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                ResNet18Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                ResNet34Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                InceptionBuilder(),
                {"input-channels": 3, "input-height": 75, "input-width": 75},
            ),
        ]

        for model_builder, overrides in cases:
            with self.subTest(model=model_builder.name):
                config = ExperimentConfig.from_run_config(
                    {"algorithm": "fedproto", "num-supernodes": 2, "num-classes": 7, **overrides}
                )
                model = model_builder.build_model(config)
                strategy = FedProtoBuilder().build_strategy(config, model, evaluate_fn=None)
                self.assertEqual(strategy.on_fit_config_fn(1)["algorithm"], "fedproto")

    def test_fedproto_feature_extraction_supports_current_models(self) -> None:
        cases = [
            (MnistCnnBuilder(), {"input-channels": 1, "input-height": 28, "input-width": 28}, (2, 1, 28, 28)),
            (LeNetBuilder(), {"input-channels": 1, "input-height": 28, "input-width": 28}, (2, 1, 28, 28)),
            (ResNet9Builder(), {"input-channels": 3, "input-height": 32, "input-width": 32}, (2, 3, 32, 32)),
            (ResNet18Builder(), {"input-channels": 3, "input-height": 32, "input-width": 32}, (2, 3, 32, 32)),
            (ResNet34Builder(), {"input-channels": 3, "input-height": 32, "input-width": 32}, (2, 3, 32, 32)),
            (InceptionBuilder(), {"input-channels": 3, "input-height": 75, "input-width": 75}, (2, 3, 75, 75)),
        ]

        for model_builder, overrides, input_shape in cases:
            with self.subTest(model=model_builder.name):
                config = ExperimentConfig.from_run_config({"num-classes": 7, **overrides})
                model = model_builder.build_model(config)
                inputs = torch.zeros(*input_shape)
                features = extract_features(model, inputs)
                self.assertEqual(features.ndim, 2)
                self.assertEqual(features.shape[0], 2)

    def test_fedmeta_builder_creates_strategy(self) -> None:
        from fl_baselines.algorithms.fedmeta import FedMetaBuilder, FedMetaStrategy

        config = ExperimentConfig.from_run_config(
            {
                "algorithm": "fedmeta",
                "num-supernodes": 4,
                "fedmeta-method": "meta-sgd",
                "fedmeta-inner-learning-rate": 0.05,
                "fedmeta-outer-learning-rate": 0.01,
                "fedmeta-support-fraction": 0.4,
                "fedmeta-inner-steps": 2,
                "fedmeta-first-order": False,
                "fedmeta-alpha-init": 0.03,
            }
        )
        model = torch.nn.Linear(3, 2)

        strategy = FedMetaBuilder().build_strategy(config, model, evaluate_fn=None)
        fit_config = strategy.on_fit_config_fn(3)

        self.assertIsInstance(strategy, FedMetaStrategy)
        self.assertEqual(strategy.min_fit_clients, 4)
        self.assertEqual(fit_config["algorithm"], "fedmeta")
        self.assertEqual(fit_config["fedmeta_method"], "meta-sgd")
        self.assertEqual(fit_config["fedmeta_inner_learning_rate"], 0.05)
        self.assertEqual(fit_config["fedmeta_support_fraction"], 0.4)
        self.assertEqual(fit_config["fedmeta_inner_steps"], 2)
        self.assertFalse(fit_config["fedmeta_first_order"])

    def test_fedmeta_strategy_applies_weighted_meta_gradients(self) -> None:
        from fl_baselines.algorithms.fedmeta import FedMetaStrategy

        model = torch.nn.Linear(2, 1, bias=False)
        initial_parameters = [np.array([[2.0, 4.0]], dtype=np.float32)]
        strategy = FedMetaStrategy(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=2,
            min_evaluate_clients=2,
            min_available_clients=2,
            initial_parameters=ndarrays_to_parameters(initial_parameters),
            meta_sgd=False,
            outer_learning_rate=0.5,
            checkpoint_model=model,
            output_dir=tempfile.mkdtemp(),
        )
        first_gradient = [np.array([[2.0, 0.0]], dtype=np.float32)]
        second_gradient = [np.array([[0.0, 4.0]], dtype=np.float32)]
        results = [
            (
                None,
                FitRes(
                    Status(Code.OK, ""),
                    ndarrays_to_parameters(first_gradient),
                    1,
                    {},
                ),
            ),
            (
                None,
                FitRes(
                    Status(Code.OK, ""),
                    ndarrays_to_parameters(second_gradient),
                    3,
                    {},
                ),
            ),
        ]

        updated_parameters, _ = strategy.aggregate_fit(1, results, [])

        expected_gradient = np.array([[0.5, 3.0]], dtype=np.float32)
        expected_parameters = initial_parameters[0] - 0.5 * expected_gradient
        np.testing.assert_allclose(parameters_to_ndarrays(updated_parameters)[0], expected_parameters)

    def test_fednp_builder_creates_strategy(self) -> None:
        from fl_baselines.algorithms.fednp import FedNPBuilder, FedNPStrategy

        config = ExperimentConfig.from_run_config(
            {
                "algorithm": "fednp",
                "num-supernodes": 4,
                "num-classes": 3,
                "fednp-lambda": 0.6,
                "fednp-prior-variance": 1.5,
                "fednp-stability-eps": 1e-5,
            }
        )
        model = MnistCnnBuilder().build_model(config)

        strategy = FedNPBuilder().build_strategy(config, model, evaluate_fn=None)
        fit_config = strategy.on_fit_config_fn(1)

        self.assertIsInstance(strategy, FedNPStrategy)
        self.assertEqual(strategy.min_fit_clients, 4)
        self.assertEqual(fit_config["algorithm"], "fednp")
        self.assertEqual(fit_config["fednp_lambda"], 0.6)
        self.assertEqual(fit_config["fednp_prior_variance"], 1.5)
        self.assertEqual(fit_config["fednp_stability_eps"], 1e-5)

    def test_fedcurv_builder_creates_strategy(self) -> None:
        from fl_baselines.algorithms.fedcurv import FedCurvBuilder, FedCurvStrategy

        config = ExperimentConfig.from_run_config(
            {
                "algorithm": "fedcurv",
                "num-supernodes": 4,
                "fedcurv-lambda": 0.4,
                "fedcurv-fisher-batches": 3,
                "fedcurv-stability-eps": 1e-6,
            }
        )
        model = torch.nn.Linear(3, 2)

        strategy = FedCurvBuilder().build_strategy(config, model, evaluate_fn=None)
        fit_config = strategy.on_fit_config_fn(2)

        self.assertIsInstance(strategy, FedCurvStrategy)
        self.assertEqual(strategy.min_fit_clients, 4)
        self.assertEqual(fit_config["algorithm"], "fedcurv")
        self.assertEqual(fit_config["fedcurv_lambda"], 0.4)
        self.assertEqual(fit_config["fedcurv_fisher_batches"], 3)
        self.assertEqual(fit_config["fedcurv_stability_eps"], 1e-6)

    def test_fedmmd_builder_creates_strategy(self) -> None:
        from fl_baselines.algorithms.fedmmd import FedMMDBuilder, FedMMDStrategy

        config = ExperimentConfig.from_run_config(
            {
                "algorithm": "fedmmd",
                "num-supernodes": 4,
                "fedmmd-sigma": 0.5,
                "fedmmd-sknq-threshold": 0.4,
                "fedmmd-min-clients": 2,
                "fedmmd-entropy-eps": 1e-8,
            }
        )
        model = torch.nn.Linear(3, 2)

        strategy = FedMMDBuilder().build_strategy(config, model, evaluate_fn=None)
        fit_config = strategy.on_fit_config_fn(2)

        self.assertIsInstance(strategy, FedMMDStrategy)
        self.assertEqual(strategy.min_fit_clients, 4)
        self.assertEqual(fit_config["algorithm"], "fedmmd")
        self.assertEqual(fit_config["fedmmd_sigma"], 0.5)
        self.assertEqual(fit_config["fedmmd_sknq_threshold"], 0.4)
        self.assertEqual(fit_config["fedmmd_min_clients"], 2)
        self.assertEqual(fit_config["fedmmd_entropy_eps"], 1e-8)

    def test_fedntd_builder_creates_strategy(self) -> None:
        config = ExperimentConfig.from_run_config(
            {"num-supernodes": 4, "fedntd-beta": 1.2, "fedntd-temperature": 2.0}
        )
        model = MnistCnnBuilder().build_model(config)

        strategy = FedNTDBuilder().build_strategy(config, model, evaluate_fn=None)
        fit_config = strategy.on_fit_config_fn(1)

        self.assertEqual(strategy.min_fit_clients, 4)
        self.assertEqual(fit_config["algorithm"], "fedntd")
        self.assertEqual(fit_config["fedntd_beta"], 1.2)
        self.assertEqual(fit_config["fedntd_temperature"], 2.0)

    def test_fedntd_builder_supports_current_models(self) -> None:
        cases = [
            (MnistCnnBuilder(), {}),
            (LeNetBuilder(), {}),
            (
                ResNet9Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                ResNet18Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                ResNet34Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                InceptionBuilder(),
                {"input-channels": 3, "input-height": 75, "input-width": 75},
            ),
        ]

        for model_builder, overrides in cases:
            with self.subTest(model=model_builder.name):
                config = ExperimentConfig.from_run_config(
                    {"algorithm": "fedntd", "num-supernodes": 2, **overrides}
                )
                model = model_builder.build_model(config)
                strategy = FedNTDBuilder().build_strategy(config, model, evaluate_fn=None)
                self.assertEqual(strategy.on_fit_config_fn(1)["algorithm"], "fedntd")

    def test_fedlc_builder_creates_strategy(self) -> None:
        config = ExperimentConfig.from_run_config(
            {"num-supernodes": 4, "fedlc-tau": 0.5, "fedlc-epsilon": 1e-4}
        )
        model = MnistCnnBuilder().build_model(config)

        strategy = FedLCBuilder().build_strategy(config, model, evaluate_fn=None)
        fit_config = strategy.on_fit_config_fn(1)

        self.assertEqual(strategy.min_fit_clients, 4)
        self.assertEqual(fit_config["algorithm"], "fedlc")
        self.assertEqual(fit_config["fedlc_tau"], 0.5)
        self.assertEqual(fit_config["fedlc_epsilon"], 1e-4)
        self.assertEqual(fit_config["num_classes"], 10)

    def test_fedrs_builder_creates_strategy(self) -> None:
        config = ExperimentConfig.from_run_config(
            {"num-supernodes": 4, "fedrs-alpha": 0.5}
        )
        model = MnistCnnBuilder().build_model(config)

        strategy = FedRSBuilder().build_strategy(config, model, evaluate_fn=None)
        fit_config = strategy.on_fit_config_fn(1)

        self.assertIsInstance(strategy, FedAvg)
        self.assertEqual(strategy.min_fit_clients, 4)
        self.assertEqual(fit_config["algorithm"], "fedrs")
        self.assertEqual(fit_config["fedrs_alpha"], 0.5)
        self.assertEqual(fit_config["num_classes"], 10)

    def test_fedlc_builder_supports_current_models(self) -> None:
        cases = [
            (MnistCnnBuilder(), {}),
            (LeNetBuilder(), {}),
            (
                ResNet9Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                ResNet18Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                ResNet34Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                InceptionBuilder(),
                {"input-channels": 3, "input-height": 75, "input-width": 75},
            ),
        ]

        for model_builder, overrides in cases:
            with self.subTest(model=model_builder.name):
                config = ExperimentConfig.from_run_config(
                    {"algorithm": "fedlc", "num-supernodes": 2, **overrides}
                )
                model = model_builder.build_model(config)
                strategy = FedLCBuilder().build_strategy(config, model, evaluate_fn=None)
                self.assertEqual(strategy.on_fit_config_fn(1)["algorithm"], "fedlc")

    def test_fedrs_builder_supports_current_models(self) -> None:
        cases = [
            (MnistCnnBuilder(), {}),
            (LeNetBuilder(), {}),
            (
                ResNet9Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                ResNet18Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                ResNet34Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                InceptionBuilder(),
                {"input-channels": 3, "input-height": 75, "input-width": 75},
            ),
        ]

        for model_builder, overrides in cases:
            with self.subTest(model=model_builder.name):
                config = ExperimentConfig.from_run_config(
                    {"algorithm": "fedrs", "num-supernodes": 2, **overrides}
                )
                model = model_builder.build_model(config)
                strategy = FedRSBuilder().build_strategy(config, model, evaluate_fn=None)
                self.assertEqual(strategy.on_fit_config_fn(1)["algorithm"], "fedrs")

    def test_ditto_builder_creates_strategy(self) -> None:
        config = ExperimentConfig.from_run_config(
            {"num-supernodes": 4, "ditto-lambda": 0.3}
        )
        model = MnistCnnBuilder().build_model(config)

        strategy = DittoBuilder().build_strategy(config, model, evaluate_fn=None)
        fit_config = strategy.on_fit_config_fn(1)

        self.assertEqual(strategy.min_fit_clients, 4)
        self.assertEqual(fit_config["algorithm"], "ditto")
        self.assertEqual(fit_config["ditto_lambda"], 0.3)

    def test_ditto_builder_supports_current_models(self) -> None:
        cases = [
            (MnistCnnBuilder(), {}),
            (LeNetBuilder(), {}),
            (
                ResNet9Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                ResNet18Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                ResNet34Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                InceptionBuilder(),
                {"input-channels": 3, "input-height": 75, "input-width": 75},
            ),
        ]

        for model_builder, overrides in cases:
            with self.subTest(model=model_builder.name):
                config = ExperimentConfig.from_run_config(
                    {"algorithm": "ditto", "num-supernodes": 2, **overrides}
                )
                model = model_builder.build_model(config)
                strategy = DittoBuilder().build_strategy(config, model, evaluate_fn=None)
                self.assertEqual(strategy.on_fit_config_fn(1)["algorithm"], "ditto")

    def test_pfedme_builder_creates_strategy(self) -> None:
        config = ExperimentConfig.from_run_config(
            {
                "num-supernodes": 4,
                "pfedme-lambda": 15.0,
                "pfedme-beta": 0.7,
                "pfedme-personal-learning-rate": 0.02,
                "pfedme-personal-steps": 4,
            }
        )
        model = MnistCnnBuilder().build_model(config)

        strategy = PFedMeBuilder().build_strategy(config, model, evaluate_fn=None)
        fit_config = strategy.on_fit_config_fn(1)

        self.assertIsInstance(strategy, PFedMeStrategy)
        self.assertEqual(strategy.min_fit_clients, 4)
        self.assertEqual(strategy.beta, 0.7)
        self.assertEqual(fit_config["algorithm"], "pfedme")
        self.assertEqual(fit_config["pfedme_lambda"], 15.0)
        self.assertEqual(fit_config["pfedme_beta"], 0.7)
        self.assertEqual(fit_config["pfedme_personal_learning_rate"], 0.02)
        self.assertEqual(fit_config["pfedme_personal_steps"], 4)

    def test_pfedme_builder_supports_current_models(self) -> None:
        cases = [
            (MnistCnnBuilder(), {}),
            (LeNetBuilder(), {}),
            (
                ResNet9Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                ResNet18Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                ResNet34Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                InceptionBuilder(),
                {"input-channels": 3, "input-height": 75, "input-width": 75},
            ),
        ]

        for model_builder, overrides in cases:
            with self.subTest(model=model_builder.name):
                config = ExperimentConfig.from_run_config(
                    {"algorithm": "pfedme", "num-supernodes": 2, **overrides}
                )
                model = model_builder.build_model(config)
                strategy = PFedMeBuilder().build_strategy(config, model, evaluate_fn=None)
                self.assertEqual(strategy.on_fit_config_fn(1)["algorithm"], "pfedme")

    def test_apfl_builder_creates_strategy(self) -> None:
        from fl_baselines.algorithms.apfl import APFLBuilder

        config = ExperimentConfig.from_run_config(
            {
                "algorithm": "apfl",
                "num-supernodes": 4,
                "apfl-alpha": 0.6,
                "apfl-personal-learning-rate": 0.02,
                "apfl-adaptive-alpha": False,
                "apfl-alpha-learning-rate": 0.01,
            }
        )
        model = MnistCnnBuilder().build_model(config)

        strategy = APFLBuilder().build_strategy(config, model, evaluate_fn=None)
        fit_config = strategy.on_fit_config_fn(1)

        self.assertEqual(strategy.min_fit_clients, 4)
        self.assertEqual(fit_config["algorithm"], "apfl")
        self.assertEqual(fit_config["apfl_alpha"], 0.6)
        self.assertEqual(fit_config["apfl_personal_learning_rate"], 0.02)
        self.assertFalse(fit_config["apfl_adaptive_alpha"])
        self.assertEqual(fit_config["apfl_alpha_learning_rate"], 0.01)

    def test_apfl_builder_supports_current_models(self) -> None:
        from fl_baselines.algorithms.apfl import APFLBuilder

        cases = [
            (MnistCnnBuilder(), {}),
            (LeNetBuilder(), {}),
            (
                ResNet9Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                ResNet18Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                ResNet34Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                InceptionBuilder(),
                {"input-channels": 3, "input-height": 75, "input-width": 75},
            ),
        ]

        for model_builder, overrides in cases:
            with self.subTest(model=model_builder.name):
                config = ExperimentConfig.from_run_config(
                    {"algorithm": "apfl", "num-supernodes": 2, **overrides}
                )
                model = model_builder.build_model(config)
                strategy = APFLBuilder().build_strategy(config, model, evaluate_fn=None)
                self.assertEqual(strategy.on_fit_config_fn(1)["algorithm"], "apfl")

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

    def test_fedala_builder_creates_fedavg_strategy_with_ala_config(self) -> None:
        config = ExperimentConfig.from_run_config(
            {
                "num-supernodes": 4,
                "fedala-eta": 0.5,
                "fedala-rand-percent": 40,
                "fedala-layer-count": 2,
                "fedala-threshold": 0.02,
                "fedala-num-pre-loss": 4,
                "fedala-start-max-steps": 8,
            }
        )
        model = MnistCnnBuilder().build_model(config)

        strategy = FedALABuilder().build_strategy(config, model, evaluate_fn=None)
        fit_config = strategy.on_fit_config_fn(2)

        self.assertIsInstance(strategy, FedAvg)
        self.assertEqual(strategy.min_fit_clients, 4)
        self.assertEqual(fit_config["algorithm"], "fedala")
        self.assertEqual(fit_config["server_round"], 2)
        self.assertEqual(fit_config["fedala_eta"], 0.5)
        self.assertEqual(fit_config["fedala_rand_percent"], 40)
        self.assertEqual(fit_config["fedala_layer_count"], 2)
        self.assertEqual(fit_config["fedala_threshold"], 0.02)
        self.assertEqual(fit_config["fedala_num_pre_loss"], 4)
        self.assertEqual(fit_config["fedala_start_max_steps"], 8)

    def test_fedala_builder_supports_current_models(self) -> None:
        cases = [
            (MnistCnnBuilder(), {}),
            (LeNetBuilder(), {}),
            (
                ResNet9Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                ResNet18Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                ResNet34Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                InceptionBuilder(),
                {"input-channels": 3, "input-height": 75, "input-width": 75},
            ),
        ]

        for model_builder, overrides in cases:
            with self.subTest(model=model_builder.name):
                config = ExperimentConfig.from_run_config(
                    {"algorithm": "fedala", "num-supernodes": 2, **overrides}
                )
                model = model_builder.build_model(config)
                strategy = FedALABuilder().build_strategy(config, model, evaluate_fn=None)
                self.assertEqual(strategy.on_fit_config_fn(1)["algorithm"], "fedala")

    def test_adaptive_local_aggregation_blends_selected_higher_layers(self) -> None:
        local_model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
        global_model = copy.deepcopy(local_model)
        with torch.no_grad():
            for parameter in local_model.parameters():
                parameter.fill_(0.0)
            for parameter in global_model.parameters():
                parameter.fill_(1.0)

        loader = DataLoader(
            TensorDataset(torch.ones(4, 2), torch.zeros(4, dtype=torch.long)),
            batch_size=2,
        )

        result = adaptive_local_aggregation(
            local_model,
            global_model,
            loader,
            weights=None,
            layer_count=1,
            eta=0.1,
            rand_percent=100,
            threshold=0.01,
            num_pre_loss=2,
            start_max_steps=1,
            device="cpu",
            start_phase=True,
        )

        parameters = list(local_model.parameters())
        self.assertTrue(torch.allclose(parameters[0], torch.ones_like(parameters[0])))
        self.assertTrue(torch.allclose(parameters[1], torch.ones_like(parameters[1])))
        self.assertEqual(len(result.weights), 2)
        for weight in result.weights:
            self.assertGreaterEqual(float(weight.min()), 0.0)
            self.assertLessEqual(float(weight.max()), 1.0)
        self.assertFalse(result.start_phase)

    def test_fedala_client_persists_local_model_and_ala_weights(self) -> None:
        model = torch.nn.Linear(2, 2)
        loaders = torch_client_module.ClientDataLoaders(
            train=DataLoader(
                TensorDataset(torch.ones(4, 2), torch.zeros(4, dtype=torch.long)),
                batch_size=2,
            ),
            test=DataLoader(
                TensorDataset(torch.ones(2, 2), torch.zeros(2, dtype=torch.long)),
                batch_size=1,
            ),
        )

        with tempfile.TemporaryDirectory() as output_dir:
            config = ExperimentConfig.from_run_config(
                {
                    "algorithm": "fedala",
                    "output-dir": output_dir,
                    "fedala-layer-count": 1,
                    "fedala-rand-percent": 100,
                    "fedala-start-max-steps": 1,
                    "fedala-num-pre-loss": 2,
                    "local-epochs": 1,
                    "learning-rate": 0.01,
                }
            )
            client = TorchFlowerClient(model, loaders, config, client_id="client-1")
            global_parameters = [
                np.ones_like(parameter) for parameter in get_model_parameters(model)
            ]

            returned_parameters, num_examples, metrics = client.fit(
                global_parameters,
                {
                    "algorithm": "fedala",
                    "server_round": 2,
                    "local_epochs": 1,
                    "learning_rate": 0.01,
                    "fedala_eta": 1.0,
                    "fedala_rand_percent": 100,
                    "fedala_layer_count": 1,
                    "fedala_threshold": 0.01,
                    "fedala_num_pre_loss": 2,
                    "fedala_start_max_steps": 1,
                },
            )

            self.assertEqual(num_examples, 4)
            self.assertEqual(len(returned_parameters), len(global_parameters))
            self.assertIn("train_loss", metrics)
            state_path = Path(output_dir) / "fedala_clients" / "client-1" / "state.pt"
            self.assertTrue(state_path.exists())

    def test_fedamp_builder_creates_personalized_strategy(self) -> None:
        config = ExperimentConfig.from_run_config(
            {
                "num-supernodes": 4,
                "fedamp-lambda": 0.2,
                "fedamp-alpha": 0.05,
                "fedamp-sigma": 2.0,
            }
        )
        model = MnistCnnBuilder().build_model(config)

        strategy = FedAMPBuilder().build_strategy(config, model, evaluate_fn=None)
        fit_config = strategy.on_fit_config_fn(1)

        self.assertIsInstance(strategy, FedAMPStrategy)
        self.assertEqual(strategy.min_fit_clients, 4)
        self.assertEqual(strategy.fedamp_lambda, 0.2)
        self.assertEqual(strategy.alpha, 0.05)
        self.assertEqual(strategy.sigma, 2.0)
        self.assertEqual(fit_config["algorithm"], "fedamp")
        self.assertEqual(fit_config["fedamp_proximal_mu"], 4.0)

    def test_fedamp_builder_supports_current_models(self) -> None:
        cases = [
            (MnistCnnBuilder(), {}),
            (LeNetBuilder(), {}),
            (
                ResNet9Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                ResNet18Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                ResNet34Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                InceptionBuilder(),
                {"input-channels": 3, "input-height": 75, "input-width": 75},
            ),
        ]

        for model_builder, overrides in cases:
            with self.subTest(model=model_builder.name):
                config = ExperimentConfig.from_run_config(
                    {"algorithm": "fedamp", "num-supernodes": 2, **overrides}
                )
                model = model_builder.build_model(config)
                strategy = FedAMPBuilder().build_strategy(config, model, evaluate_fn=None)
                self.assertEqual(strategy.on_fit_config_fn(1)["algorithm"], "fedamp")

    def test_fedamp_strategy_builds_personalized_cloud_models(self) -> None:
        model = torch.nn.Linear(1, 1, bias=False)
        initial_parameters = ndarrays_to_parameters(get_model_parameters(model))

        with tempfile.TemporaryDirectory() as output_dir:
            strategy = FedAMPStrategy(
                fraction_fit=1.0,
                fraction_evaluate=1.0,
                min_fit_clients=2,
                min_evaluate_clients=2,
                min_available_clients=2,
                initial_parameters=initial_parameters,
                fedamp_lambda=0.1,
                alpha=0.1,
                sigma=10.0,
                checkpoint_model=model,
                output_dir=output_dir,
            )
            results = []
            for cid, value in [("0", 0.0), ("1", 1.0)]:
                client_proxy = type("ClientProxy", (), {"cid": cid})()
                parameters = ndarrays_to_parameters(
                    [np.asarray([[value]], dtype=np.float32)]
                )
                fit_res = FitRes(
                    status=Status(Code.OK, ""),
                    parameters=parameters,
                    num_examples=1,
                    metrics={"train_loss": 0.1},
                )
                results.append((client_proxy, fit_res))

            aggregated_parameters, metrics = strategy.aggregate_fit(1, results, [])

            self.assertIsNotNone(aggregated_parameters)
            self.assertEqual(metrics["fedamp_client_count"], 2)
            self.assertEqual(sorted(strategy.personalized_cloud_models.keys()), ["0", "1"])
            self.assertEqual(len(strategy.last_message_weights), 2)
            for weights in strategy.last_message_weights.values():
                self.assertAlmostEqual(sum(weights.values()), 1.0, places=6)
                self.assertGreater(weights["0"], 0.0)
                self.assertGreater(weights["1"], 0.0)

    def test_fedamp_configure_fit_sends_personalized_cloud_parameters(self) -> None:
        model = torch.nn.Linear(1, 1, bias=False)
        initial_parameters = ndarrays_to_parameters(get_model_parameters(model))
        client_proxy = type("ClientProxy", (), {"cid": "0"})()

        class FakeClientManager:
            def num_available(self) -> int:
                return 1

            def sample(self, num_clients: int, min_num_clients: int):
                return [client_proxy]

        with tempfile.TemporaryDirectory() as output_dir:
            strategy = FedAMPStrategy(
                fraction_fit=1.0,
                fraction_evaluate=1.0,
                min_fit_clients=1,
                min_evaluate_clients=1,
                min_available_clients=1,
                initial_parameters=initial_parameters,
                on_fit_config_fn=lambda _: {"algorithm": "fedamp"},
                fedamp_lambda=0.1,
                alpha=0.1,
                sigma=10.0,
                checkpoint_model=model,
                output_dir=output_dir,
            )
            strategy.personalized_cloud_models["0"] = [
                np.asarray([[3.0]], dtype=np.float32)
            ]

            fit_instructions = strategy.configure_fit(
                2,
                initial_parameters,
                FakeClientManager(),
            )

            sent_parameters = parameters_to_ndarrays(fit_instructions[0][1].parameters)
            self.assertEqual(float(sent_parameters[0][0, 0]), 3.0)
            self.assertEqual(fit_instructions[0][1].config["algorithm"], "fedamp")

    def test_fedamp_client_routes_to_proximal_training(self) -> None:
        model = torch.nn.Linear(2, 2)
        loaders = torch_client_module.ClientDataLoaders(
            train=DataLoader(
                TensorDataset(torch.ones(4, 2), torch.zeros(4, dtype=torch.long)),
                batch_size=2,
            ),
            test=DataLoader(
                TensorDataset(torch.ones(2, 2), torch.zeros(2, dtype=torch.long)),
                batch_size=1,
            ),
        )
        config = ExperimentConfig.from_run_config(
            {
                "algorithm": "fedamp",
                "fedamp-lambda": 0.2,
                "fedamp-alpha": 0.1,
                "local-epochs": 1,
                "learning-rate": 0.01,
            }
        )
        client = TorchFlowerClient(model, loaders, config, client_id="client-1")
        cloud_parameters = [
            np.ones_like(parameter) for parameter in get_model_parameters(model)
        ]

        returned_parameters, num_examples, metrics = client.fit(
            cloud_parameters,
            {
                "algorithm": "fedamp",
                "local_epochs": 1,
                "learning_rate": 0.01,
                "fedamp_proximal_mu": 2.0,
            },
        )

        self.assertEqual(num_examples, 4)
        self.assertEqual(len(returned_parameters), len(cloud_parameters))
        self.assertIn("train_loss", metrics)

    def test_fedlaa_builder_creates_strategy(self) -> None:
        config = ExperimentConfig.from_run_config(
            {
                "num-supernodes": 4,
                "fedlaa-beta": 3.0,
            }
        )
        model = MnistCnnBuilder().build_model(config)

        strategy = FedLAABuilder().build_strategy(config, model, evaluate_fn=None)
        fit_config = strategy.on_fit_config_fn(3)

        self.assertIsInstance(strategy, FedLAAStrategy)
        self.assertEqual(strategy.min_fit_clients, 4)
        self.assertEqual(strategy.beta, 3.0)
        self.assertEqual(fit_config["algorithm"], "fedlaa")
        self.assertEqual(fit_config["server_round"], 3)

    def test_fedlaa_builder_supports_current_models(self) -> None:
        cases = [
            (MnistCnnBuilder(), {}),
            (LeNetBuilder(), {}),
            (
                ResNet9Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                ResNet18Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                ResNet34Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                InceptionBuilder(),
                {"input-channels": 3, "input-height": 75, "input-width": 75},
            ),
        ]

        for model_builder, overrides in cases:
            with self.subTest(model=model_builder.name):
                config = ExperimentConfig.from_run_config(
                    {"algorithm": "fedlaa", "num-supernodes": 2, **overrides}
                )
                model = model_builder.build_model(config)
                strategy = FedLAABuilder().build_strategy(config, model, evaluate_fn=None)
                self.assertEqual(strategy.on_fit_config_fn(1)["algorithm"], "fedlaa")

    def test_fedlaa_strategy_weights_layers_independently(self) -> None:
        model = torch.nn.Sequential(
            torch.nn.Linear(1, 1, bias=False),
            torch.nn.Linear(1, 1, bias=False),
        )
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        initial_parameters = ndarrays_to_parameters(get_model_parameters(model))

        with tempfile.TemporaryDirectory() as output_dir:
            strategy = FedLAAStrategy(
                fraction_fit=1.0,
                fraction_evaluate=1.0,
                min_fit_clients=2,
                min_evaluate_clients=2,
                min_available_clients=2,
                initial_parameters=initial_parameters,
                on_fit_config_fn=lambda _: {
                    "algorithm": "fedlaa",
                    "learning_rate": 1.0,
                    "local_epochs": 1,
                },
                beta=5.0,
                checkpoint_model=model,
                output_dir=output_dir,
            )
            results = []
            for cid, values in [
                ("0", [np.asarray([[2.0]], dtype=np.float32), np.asarray([[-0.1]], dtype=np.float32)]),
                ("1", [np.asarray([[-0.5]], dtype=np.float32), np.asarray([[2.0]], dtype=np.float32)]),
            ]:
                client_proxy = type("ClientProxy", (), {"cid": cid})()
                fit_res = FitRes(
                    status=Status(Code.OK, ""),
                    parameters=ndarrays_to_parameters(values),
                    num_examples=1,
                    metrics={"train_loss": 0.1},
                )
                results.append((client_proxy, fit_res))

            aggregated_parameters, metrics = strategy.aggregate_fit(1, results, [])

            self.assertIsNotNone(aggregated_parameters)
            self.assertEqual(metrics["fedlaa_layer_count"], 2)
            self.assertGreater(strategy.last_layer_weights[0][0], strategy.last_layer_weights[0][1])
            self.assertGreater(strategy.last_layer_weights[1][1], strategy.last_layer_weights[1][0])
            self.assertEqual(len(strategy.smoothed_angles["0"]), 2)
            self.assertEqual(len(strategy.smoothed_angles["1"]), 2)

    def test_fedlama_builder_creates_strategy(self) -> None:
        config = ExperimentConfig.from_run_config(
            {
                "num-supernodes": 4,
                "fedlama-base-interval": 1,
                "fedlama-interval-factor": 2.0,
            }
        )
        model = MnistCnnBuilder().build_model(config)

        strategy = FedLAMABuilder().build_strategy(config, model, evaluate_fn=None)
        fit_config = strategy.on_fit_config_fn(3)

        self.assertIsInstance(strategy, FedLAMAStrategy)
        self.assertEqual(strategy.min_fit_clients, 4)
        self.assertEqual(strategy.base_interval, 1)
        self.assertEqual(strategy.interval_factor, 2.0)
        self.assertEqual(fit_config["algorithm"], "fedlama")
        self.assertEqual(fit_config["fedlama_base_interval"], 1)
        self.assertEqual(fit_config["fedlama_interval_factor"], 2.0)

    def test_fedlama_builder_supports_current_models(self) -> None:
        cases = [
            (MnistCnnBuilder(), {}),
            (LeNetBuilder(), {}),
            (
                ResNet9Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                ResNet18Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                ResNet34Builder(),
                {"input-channels": 3, "input-height": 32, "input-width": 32},
            ),
            (
                InceptionBuilder(),
                {"input-channels": 3, "input-height": 75, "input-width": 75},
            ),
        ]

        for model_builder, overrides in cases:
            with self.subTest(model=model_builder.name):
                config = ExperimentConfig.from_run_config(
                    {
                        "algorithm": "fedlama",
                        "num-supernodes": 2,
                        "fedlama-base-interval": 1,
                        "fedlama-interval-factor": 2.0,
                        **overrides,
                    }
                )
                model = model_builder.build_model(config)
                strategy = FedLAMABuilder().build_strategy(config, model, evaluate_fn=None)
                self.assertEqual(strategy.on_fit_config_fn(1)["algorithm"], "fedlama")

    def test_fedlama_strategy_reweights_layer_sync_intervals(self) -> None:
        model = torch.nn.Sequential(
            torch.nn.Linear(1, 1, bias=False),
            torch.nn.Linear(1, 1, bias=False),
        )
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        initial_parameters = ndarrays_to_parameters(get_model_parameters(model))

        with tempfile.TemporaryDirectory() as output_dir:
            strategy = FedLAMAStrategy(
                fraction_fit=1.0,
                fraction_evaluate=1.0,
                min_fit_clients=2,
                min_evaluate_clients=2,
                min_available_clients=2,
                initial_parameters=initial_parameters,
                on_fit_config_fn=lambda _: {
                    "algorithm": "fedlama",
                    "learning_rate": 1.0,
                    "local_epochs": 1,
                    "fedlama_base_interval": 1,
                    "fedlama_interval_factor": 2.0,
                },
                base_interval=1,
                interval_factor=2.0,
                checkpoint_model=model,
                output_dir=output_dir,
            )
            results = []
            for cid, values in [
                ("0", [np.asarray([[0.1]], dtype=np.float32), np.asarray([[3.0]], dtype=np.float32)]),
                ("1", [np.asarray([[0.1]], dtype=np.float32), np.asarray([[2.5]], dtype=np.float32)]),
            ]:
                client_proxy = type("ClientProxy", (), {"cid": cid})()
                fit_res = FitRes(
                    status=Status(Code.OK, ""),
                    parameters=ndarrays_to_parameters(values),
                    num_examples=1,
                    metrics={
                        "train_loss": 0.1,
                        "fedlama_layer_discrepancies": json.dumps([0.0, 9.0]),
                    },
                )
                results.append((client_proxy, fit_res))

            aggregated_parameters, metrics = strategy.aggregate_fit(1, results, [])

            self.assertIsNotNone(aggregated_parameters)
            self.assertEqual(strategy.layer_intervals, [2, 1])
            self.assertEqual(metrics["fedlama_sync_layer_count"], 2)
            self.assertEqual(fedlama_sync_mask_for_round(2, strategy.layer_intervals), [False, True])

    def test_fedlama_client_routes_to_stateful_trainer(self) -> None:
        with tempfile.TemporaryDirectory() as output_dir:
            config = ExperimentConfig.from_run_config(
                {
                    "algorithm": "fedlama",
                    "output-dir": output_dir,
                }
            )
            model = torch.nn.Linear(2, 2)
            loader = DataLoader(
                TensorDataset(torch.ones(4, 2), torch.zeros(4, dtype=torch.long)),
                batch_size=2,
            )
            client = TorchFlowerClient(
                model,
                loaders=type("Loaders", (), {"train": loader, "test": loader})(),
                config=config,
                client_id="fedlama-1",
            )
            initial_parameters = get_model_parameters(model)

            returned_parameters, num_examples, metrics = client.fit(
                initial_parameters,
                {
                    "algorithm": "fedlama",
                    "local_epochs": 1,
                    "learning_rate": 0.1,
                    "fedlama_sync_mask": json.dumps([True, False]),
                    "fedlama_layer_intervals": json.dumps([1, 2]),
                },
            )

            self.assertEqual(num_examples, 4)
            self.assertEqual(len(returned_parameters), 1)
            self.assertIn("train_loss", metrics)
            self.assertIn("fedlama_layer_discrepancies", metrics)
            self.assertTrue(
                (Path(output_dir) / "fedlama_clients" / "fedlama-1" / "state.pt").exists()
            )
            self.assertEqual(parse_fedlama_sync_mask(json.dumps([True, False]), 2), [True, False])

    def test_fedlama_discrepancy_helper_normalizes_by_interval(self) -> None:
        server_parameters = [
            np.asarray([0.0], dtype=np.float32),
            np.asarray([0.0], dtype=np.float32),
        ]
        local_parameters = [
            np.asarray([1.0], dtype=np.float32),
            np.asarray([2.0], dtype=np.float32),
        ]

        discrepancies = compute_fedlama_layer_discrepancies(
            server_parameters,
            local_parameters,
            [1, 2],
        )

        self.assertAlmostEqual(discrepancies[0], 1.0)
        self.assertAlmostEqual(discrepancies[1], 2.0)

    def test_fedlama_training_helper_updates_model(self) -> None:
        model = torch.nn.Linear(2, 2)
        loader = DataLoader(
            TensorDataset(torch.ones(4, 2), torch.zeros(4, dtype=torch.long)),
            batch_size=2,
        )

        metrics = train_fedlama_client(
            model,
            loader,
            epochs=1,
            learning_rate=0.1,
            device="cpu",
        )

        self.assertIn("train_loss", metrics)
        self.assertIn("train_accuracy", metrics)

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

    def test_fedadp_strategy_reweights_clients_by_update_alignment(self) -> None:
        model = torch.nn.Linear(2, 1, bias=False)
        initial_parameters = [np.array([[0.0, 0.0]], dtype=np.float32)]

        with tempfile.TemporaryDirectory() as output_dir:
            strategy = FedAdpStrategy(
                fraction_fit=1.0,
                fraction_evaluate=1.0,
                min_fit_clients=2,
                min_evaluate_clients=2,
                min_available_clients=2,
                initial_parameters=ndarrays_to_parameters(initial_parameters),
                alpha=5.0,
                checkpoint_model=model,
                output_dir=output_dir,
            )
            aligned_model = [np.array([[2.0, 0.0]], dtype=np.float32)]
            orthogonal_model = [np.array([[0.0, 1.0]], dtype=np.float32)]
            results = [
                (
                    None,
                    FitRes(
                        Status(Code.OK, ""),
                        ndarrays_to_parameters(aligned_model),
                        1,
                        {},
                    ),
                ),
                (
                    None,
                    FitRes(
                        Status(Code.OK, ""),
                        ndarrays_to_parameters(orthogonal_model),
                        1,
                        {},
                    ),
                ),
            ]

            aggregated_parameters, _ = strategy.aggregate_fit(1, results, [])

        aggregated = parameters_to_ndarrays(aggregated_parameters)[0]
        fedavg = (aligned_model[0] + orthogonal_model[0]) / 2.0
        self.assertGreater(aggregated[0, 0], fedavg[0, 0])
        self.assertLess(aggregated[0, 1], fedavg[0, 1])
        self.assertAlmostEqual(sum(strategy.last_aggregation_weights), 1.0)

    def test_feddyn_strategy_updates_auxiliary_state_and_global_model(self) -> None:
        model = torch.nn.Linear(2, 1, bias=False)
        initial_parameters = [np.array([[10.0, 20.0]], dtype=np.float32)]

        with tempfile.TemporaryDirectory() as output_dir:
            strategy = FedDynStrategy(
                fraction_fit=1.0,
                fraction_evaluate=1.0,
                min_fit_clients=2,
                min_evaluate_clients=2,
                min_available_clients=2,
                initial_parameters=ndarrays_to_parameters(initial_parameters),
                alpha=0.5,
                num_total_clients=4,
                checkpoint_model=model,
                output_dir=output_dir,
            )
            first_model = [np.array([[8.0, 18.0]], dtype=np.float32)]
            second_model = [np.array([[6.0, 14.0]], dtype=np.float32)]
            results = [
                (
                    None,
                    FitRes(
                        Status(Code.OK, ""),
                        ndarrays_to_parameters(first_model),
                        1,
                        {},
                    ),
                ),
                (
                    None,
                    FitRes(
                        Status(Code.OK, ""),
                        ndarrays_to_parameters(second_model),
                        3,
                        {},
                    ),
                ),
            ]

            aggregated_parameters, _ = strategy.aggregate_fit(1, results, [])

        aggregated = parameters_to_ndarrays(aggregated_parameters)[0]
        expected_avg = (first_model[0] + 3.0 * second_model[0]) / 4.0
        expected_h = -0.5 * (
            (
                (first_model[0] - initial_parameters[0])
                + (second_model[0] - initial_parameters[0])
            )
            / 4.0
        )
        expected_model = expected_avg - expected_h / 0.5
        np.testing.assert_allclose(aggregated, expected_model)
        np.testing.assert_allclose(strategy.h[0], expected_h)

    def test_feddc_strategy_updates_average_update_state(self) -> None:
        model = torch.nn.Linear(2, 1, bias=False)
        initial_parameters = [np.array([[10.0, 20.0]], dtype=np.float32)]
        zero_state = [np.zeros_like(initial_parameters[0])]

        with tempfile.TemporaryDirectory() as output_dir:
            strategy = FedDCStrategy(
                fraction_fit=1.0,
                fraction_evaluate=1.0,
                min_fit_clients=2,
                min_evaluate_clients=2,
                min_available_clients=2,
                initial_parameters=ndarrays_to_parameters(initial_parameters),
                alpha=0.5,
                num_total_clients=4,
                average_update_state=zero_state,
                checkpoint_model=model,
                output_dir=output_dir,
            )
            first_corrected = [np.array([[9.0, 19.0]], dtype=np.float32)]
            second_corrected = [np.array([[7.0, 17.0]], dtype=np.float32)]
            first_update = [np.array([[-1.0, -1.0]], dtype=np.float32)]
            second_update = [np.array([[-3.0, -3.0]], dtype=np.float32)]
            results = [
                (
                    None,
                    FitRes(
                        Status(Code.OK, ""),
                        ndarrays_to_parameters(first_corrected + first_update),
                        1,
                        {},
                    ),
                ),
                (
                    None,
                    FitRes(
                        Status(Code.OK, ""),
                        ndarrays_to_parameters(second_corrected + second_update),
                        3,
                        {},
                    ),
                ),
            ]

            aggregated_parameters, _ = strategy.aggregate_fit(1, results, [])

        expected_model = (first_corrected[0] + 3.0 * second_corrected[0]) / 4.0
        expected_update = (first_update[0] + 3.0 * second_update[0]) / 4.0
        np.testing.assert_allclose(parameters_to_ndarrays(aggregated_parameters)[0], expected_model)
        np.testing.assert_allclose(strategy.average_update_state[0], expected_update)

    def test_fedent_strategy_updates_phi_state_after_aggregate_fit(self) -> None:
        config = ExperimentConfig.from_run_config({"algorithm": "fedent", "num-supernodes": 2})
        model = MnistCnnBuilder().build_model(config)
        strategy = FedEntBuilder().build_strategy(config, model, evaluate_fn=None)

        initial_phi2 = strategy._phi2
        initial_parameters = get_model_parameters(model)
        first_model = [parameter + 1.0 for parameter in initial_parameters]
        second_model = [parameter + 3.0 for parameter in initial_parameters]
        results = [
            (
                None,
                FitRes(
                    Status(Code.OK, ""),
                    ndarrays_to_parameters(first_model),
                    1,
                    {"fedent_weight_sq_norm": 10.0},
                ),
            ),
            (
                None,
                FitRes(
                    Status(Code.OK, ""),
                    ndarrays_to_parameters(second_model),
                    3,
                    {"fedent_weight_sq_norm": 30.0},
                ),
            ),
        ]

        aggregated_parameters, _ = strategy.aggregate_fit(1, results, [])

        self.assertIsNotNone(aggregated_parameters)
        self.assertNotEqual(strategy._phi2, initial_phi2)
        self.assertEqual(len(strategy._phi1), len(initial_parameters))

    def test_fedaaw_strategy_updates_trackers_after_aggregate_fit(self) -> None:
        config = ExperimentConfig.from_run_config({"algorithm": "fedaaw", "num-supernodes": 2})
        model = MnistCnnBuilder().build_model(config)
        strategy = FedAAWBuilder().build_strategy(config, model, evaluate_fn=None)
        initial_parameters = get_model_parameters(model)
        results = [
            (
                type("Proxy", (), {"cid": "client-1"})(),
                FitRes(
                    Status(Code.OK, ""),
                    ndarrays_to_parameters(initial_parameters),
                    5,
                    {"fedaaw_grad_norm_sq": 4.0},
                ),
            ),
            (
                type("Proxy", (), {"cid": "client-2"})(),
                FitRes(
                    Status(Code.OK, ""),
                    ndarrays_to_parameters(initial_parameters),
                    5,
                    {"fedaaw_grad_norm_sq": 9.0},
                ),
            ),
        ]

        aggregated_parameters, _ = strategy.aggregate_fit(1, results, [])

        self.assertIsNotNone(aggregated_parameters)
        self.assertEqual(strategy.gradient_trackers["client-1"], 4.0)
        self.assertEqual(strategy.gradient_trackers["client-2"], 9.0)

    def test_fedlaw_strategy_learns_proxy_aware_weights_and_gamma(self) -> None:
        config = ExperimentConfig.from_run_config(
            {
                "algorithm": "fedlaw",
                "num-supernodes": 2,
                "fedlaw-server-epochs": 25,
                "fedlaw-server-learning-rate": 0.1,
                "fedlaw-gamma-init": 1.0,
                "input-channels": 1,
                "input-height": 1,
                "input-width": 2,
                "num-classes": 2,
            }
        )
        model = torch.nn.Linear(2, 2)
        strategy = FedLAWBuilder().build_strategy(config, model, evaluate_fn=None)
        proxy_loader = DataLoader(
            TensorDataset(
                torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32),
                torch.tensor([0, 0], dtype=torch.long),
            ),
            batch_size=2,
        )
        strategy.set_proxy_loader(proxy_loader)

        first_weight = np.array([[6.0, 0.0], [-6.0, 0.0]], dtype=np.float32)
        second_weight = np.array([[-6.0, 0.0], [6.0, 0.0]], dtype=np.float32)
        first_bias = np.zeros(2, dtype=np.float32)
        second_bias = np.zeros(2, dtype=np.float32)
        results = [
            (
                type("Proxy", (), {"cid": "client-a"})(),
                FitRes(
                    Status(Code.OK, ""),
                    ndarrays_to_parameters([first_weight, first_bias]),
                    1,
                    {},
                ),
            ),
            (
                type("Proxy", (), {"cid": "client-b"})(),
                FitRes(
                    Status(Code.OK, ""),
                    ndarrays_to_parameters([second_weight, second_bias]),
                    1,
                    {},
                ),
            ),
        ]

        aggregated_parameters, metrics = strategy.aggregate_fit(1, results, [])

        self.assertIsNotNone(aggregated_parameters)
        self.assertAlmostEqual(sum(strategy.last_lambda), 1.0, places=5)
        self.assertGreater(strategy.last_lambda[0], strategy.last_lambda[1])
        self.assertGreater(strategy.last_gamma, 0.0)
        self.assertIn("fedlaw_gamma", metrics)
        self.assertIn("fedlaw_proxy_loss", metrics)

    def test_fedaaw_strategy_gives_higher_weight_to_smaller_tracker(self) -> None:
        config = ExperimentConfig.from_run_config(
            {"algorithm": "fedaaw", "num-supernodes": 2, "fedaaw-beta": 0.5}
        )
        model = MnistCnnBuilder().build_model(config)
        strategy = FedAAWBuilder().build_strategy(config, model, evaluate_fn=None)
        initial_parameters = get_model_parameters(model)
        shifted_parameters = [parameter + 1.0 for parameter in initial_parameters]
        results = [
            (
                type("Proxy", (), {"cid": "client-a"})(),
                FitRes(
                    Status(Code.OK, ""),
                    ndarrays_to_parameters(initial_parameters),
                    5,
                    {"fedaaw_grad_norm_sq": 1.0},
                ),
            ),
            (
                type("Proxy", (), {"cid": "client-b"})(),
                FitRes(
                    Status(Code.OK, ""),
                    ndarrays_to_parameters(shifted_parameters),
                    5,
                    {"fedaaw_grad_norm_sq": 10.0},
                ),
            ),
        ]

        aggregated_parameters, _ = strategy.aggregate_fit(1, results, [])

        self.assertIsNotNone(aggregated_parameters)
        self.assertGreater(strategy.last_aggregation_weights[0], strategy.last_aggregation_weights[1])

    def test_fedaaw_strategy_updates_tracker_running_average(self) -> None:
        config = ExperimentConfig.from_run_config({"algorithm": "fedaaw", "num-supernodes": 1})
        model = MnistCnnBuilder().build_model(config)
        strategy = FedAAWBuilder().build_strategy(config, model, evaluate_fn=None)
        initial_parameters = get_model_parameters(model)

        first_results = [
            (
                type("Proxy", (), {"cid": "client-1"})(),
                FitRes(
                    Status(Code.OK, ""),
                    ndarrays_to_parameters(initial_parameters),
                    5,
                    {"fedaaw_grad_norm_sq": 4.0},
                ),
            )
        ]
        second_results = [
            (
                type("Proxy", (), {"cid": "client-1"})(),
                FitRes(
                    Status(Code.OK, ""),
                    ndarrays_to_parameters(initial_parameters),
                    5,
                    {"fedaaw_grad_norm_sq": 10.0},
                ),
            )
        ]

        strategy.aggregate_fit(1, first_results, [])
        strategy.aggregate_fit(2, second_results, [])

        self.assertAlmostEqual(strategy.gradient_trackers["client-1"], 7.0)

    def test_fedaaw_strategy_falls_back_when_weights_are_invalid(self) -> None:
        config = ExperimentConfig.from_run_config(
            {
                "algorithm": "fedaaw",
                "num-supernodes": 2,
                "fedaaw-beta": 0.01,
                "fedaaw-epsilon": 1e-8,
            }
        )
        model = MnistCnnBuilder().build_model(config)
        strategy = FedAAWBuilder().build_strategy(config, model, evaluate_fn=None)
        initial_parameters = get_model_parameters(model)
        results = [
            (
                type("Proxy", (), {"cid": "client-a"})(),
                FitRes(
                    Status(Code.OK, ""),
                    ndarrays_to_parameters(initial_parameters),
                    3,
                    {"fedaaw_grad_norm_sq": float("nan")},
                ),
            ),
            (
                type("Proxy", (), {"cid": "client-b"})(),
                FitRes(
                    Status(Code.OK, ""),
                    ndarrays_to_parameters(initial_parameters),
                    1,
                    {"fedaaw_grad_norm_sq": float("nan")},
                ),
            ),
        ]

        strategy.aggregate_fit(1, results, [])

        self.assertAlmostEqual(strategy.last_aggregation_weights[0], 0.75)
        self.assertAlmostEqual(strategy.last_aggregation_weights[1], 0.25)

    def test_feddisco_strategy_gives_higher_weight_to_lower_discrepancy(self) -> None:
        config = ExperimentConfig.from_run_config(
            {"algorithm": "feddisco", "num-supernodes": 2}
        )
        model = MnistCnnBuilder().build_model(config)
        strategy = FedDiscoBuilder().build_strategy(config, model, evaluate_fn=None)
        first_parameters = get_model_parameters(model)
        second_parameters = [parameter + 1.0 for parameter in first_parameters]
        results = [
            (
                type("Proxy", (), {"cid": "low"})(),
                FitRes(
                    Status(Code.OK, ""),
                    ndarrays_to_parameters(first_parameters),
                    10,
                    {"feddisco_discrepancy": 0.0},
                ),
            ),
            (
                type("Proxy", (), {"cid": "high"})(),
                FitRes(
                    Status(Code.OK, ""),
                    ndarrays_to_parameters(second_parameters),
                    10,
                    {"feddisco_discrepancy": 1.0},
                ),
            ),
        ]

        aggregated_parameters, _ = strategy.aggregate_fit(1, results, [])

        self.assertIsNotNone(aggregated_parameters)
        self.assertGreater(
            strategy.last_aggregation_weights[0],
            strategy.last_aggregation_weights[1],
        )

    def test_feddisco_strategy_falls_back_when_relu_scores_are_zero(self) -> None:
        config = ExperimentConfig.from_run_config(
            {
                "algorithm": "feddisco",
                "num-supernodes": 2,
                "feddisco-discrepancy-weight": 10.0,
                "feddisco-bias": 0.0,
            }
        )
        model = MnistCnnBuilder().build_model(config)
        strategy = FedDiscoBuilder().build_strategy(config, model, evaluate_fn=None)
        first_parameters = get_model_parameters(model)
        second_parameters = [parameter + 1.0 for parameter in first_parameters]
        results = [
            (
                type("Proxy", (), {"cid": "a"})(),
                FitRes(
                    Status(Code.OK, ""),
                    ndarrays_to_parameters(first_parameters),
                    1,
                    {"feddisco_discrepancy": 1.0},
                ),
            ),
            (
                type("Proxy", (), {"cid": "b"})(),
                FitRes(
                    Status(Code.OK, ""),
                    ndarrays_to_parameters(second_parameters),
                    3,
                    {"feddisco_discrepancy": 1.0},
                ),
            ),
        ]

        strategy.aggregate_fit(1, results, [])

        self.assertEqual(strategy.last_aggregation_weights, [0.25, 0.75])

    def test_fedcda_strategy_uses_current_round_models_during_warmup(self) -> None:
        config = ExperimentConfig.from_run_config(
            {
                "algorithm": "fedcda",
                "num-supernodes": 2,
                "fedcda-memory-size": 2,
                "fedcda-num-batches": 1,
                "fedcda-warmup-rounds": 2,
            }
        )
        model = MnistCnnBuilder().build_model(config)
        strategy = FedCDABuilder().build_strategy(config, model, evaluate_fn=None)
        first_parameters = get_model_parameters(model)
        second_parameters = [parameter + 2.0 for parameter in first_parameters]
        results = [
            (
                type("Proxy", (), {"cid": "a"})(),
                FitRes(
                    Status(Code.OK, ""),
                    ndarrays_to_parameters(first_parameters),
                    10,
                    {"train_loss": 0.4},
                ),
            ),
            (
                type("Proxy", (), {"cid": "b"})(),
                FitRes(
                    Status(Code.OK, ""),
                    ndarrays_to_parameters(second_parameters),
                    10,
                    {"train_loss": 0.6},
                ),
            ),
        ]

        aggregated_parameters, metrics = strategy.aggregate_fit(1, results, [])

        self.assertIsNotNone(aggregated_parameters)
        self.assertEqual(metrics["fedcda_cache_size"], 2)
        self.assertEqual(metrics["fedcda_selected_client_count"], 2)
        self.assertEqual(metrics["fedcda_used_cross_round"], 0.0)
        self.assertEqual(strategy.last_selected_rounds, {"a": 1, "b": 1})

    def test_fedma_strategy_matches_permuted_hidden_units_before_averaging(self) -> None:
        config = ExperimentConfig.from_run_config({"algorithm": "fedma", "num-supernodes": 2})
        model = MnistCnnBuilder().build_model(config)
        strategy = FedMABuilder().build_strategy(config, model, evaluate_fn=None)

        base = get_model_parameters(model)
        client_one = [parameter.copy() for parameter in base]
        client_two = [parameter.copy() for parameter in base]
        permutation = np.arange(client_two[4].shape[0])[::-1]
        client_two[4] = client_two[4][permutation]
        client_two[5] = client_two[5][permutation]
        client_two[6] = client_two[6][:, permutation]

        results = [
            (
                object(),
                FitRes(
                    Status(Code.OK, ""),
                    ndarrays_to_parameters(client_one),
                    5,
                    {"fedma_label_counts": json.dumps([1] * 10)},
                ),
            ),
            (
                object(),
                FitRes(
                    Status(Code.OK, ""),
                    ndarrays_to_parameters(client_two),
                    5,
                    {"fedma_label_counts": json.dumps([1] * 10)},
                ),
            ),
        ]

        aggregated, metrics = strategy.aggregate_fit(3, results, [])

        self.assertIsNotNone(aggregated)
        aggregated_ndarrays = parameters_to_ndarrays(aggregated)
        np.testing.assert_allclose(aggregated_ndarrays[4], client_one[4], atol=1e-6)
        np.testing.assert_allclose(aggregated_ndarrays[5], client_one[5], atol=1e-6)
        np.testing.assert_allclose(aggregated_ndarrays[6], client_one[6], atol=1e-6)
        self.assertEqual(metrics["fedma_stage"], 2)

    def test_gamf_strategy_matches_permuted_hidden_units_before_averaging(self) -> None:
        config = ExperimentConfig.from_run_config({"algorithm": "gamf", "num-supernodes": 2})
        model = MnistCnnBuilder().build_model(config)
        strategy = GAMFBuilder().build_strategy(config, model, evaluate_fn=None)

        base = get_model_parameters(model)
        client_one = [parameter.copy() for parameter in base]
        client_two = [parameter.copy() for parameter in base]
        permutation = np.arange(client_two[4].shape[0])[::-1]
        client_two[4] = client_two[4][permutation]
        client_two[5] = client_two[5][permutation]
        client_two[6] = client_two[6][:, permutation]

        results = [
            (
                object(),
                FitRes(
                    Status(Code.OK, ""),
                    ndarrays_to_parameters(client_one),
                    5,
                    {},
                ),
            ),
            (
                object(),
                FitRes(
                    Status(Code.OK, ""),
                    ndarrays_to_parameters(client_two),
                    5,
                    {},
                ),
            ),
        ]

        aggregated, metrics = strategy.aggregate_fit(1, results, [])

        self.assertIsNotNone(aggregated)
        aggregated_ndarrays = parameters_to_ndarrays(aggregated)
        np.testing.assert_allclose(aggregated_ndarrays[4], client_one[4], atol=1e-6)
        np.testing.assert_allclose(aggregated_ndarrays[5], client_one[5], atol=1e-6)
        np.testing.assert_allclose(aggregated_ndarrays[6], client_one[6], atol=1e-6)
        self.assertGreaterEqual(metrics["gamf_num_matched_layers"], 1)

    def test_fedcda_strategy_selects_cross_round_models_with_lower_divergence(self) -> None:
        config = ExperimentConfig.from_run_config(
            {
                "algorithm": "fedcda",
                "num-supernodes": 2,
                "fedcda-memory-size": 2,
                "fedcda-num-batches": 1,
                "fedcda-warmup-rounds": 0,
                "fedcda-loss-weight": 1.0,
            }
        )
        model = MnistCnnBuilder().build_model(config)
        strategy = FedCDABuilder().build_strategy(config, model, evaluate_fn=None)
        base_parameters = get_model_parameters(model)
        far_parameters = [parameter + 10.0 for parameter in base_parameters]
        near_parameters = [parameter + 1.0 for parameter in base_parameters]

        first_round_results = [
            (
                type("Proxy", (), {"cid": "a"})(),
                FitRes(
                    Status(Code.OK, ""),
                    ndarrays_to_parameters(base_parameters),
                    5,
                    {"train_loss": 0.1},
                ),
            ),
            (
                type("Proxy", (), {"cid": "b"})(),
                FitRes(
                    Status(Code.OK, ""),
                    ndarrays_to_parameters(far_parameters),
                    5,
                    {"train_loss": 0.1},
                ),
            ),
        ]
        second_round_results = [
            (
                type("Proxy", (), {"cid": "a"})(),
                FitRes(
                    Status(Code.OK, ""),
                    ndarrays_to_parameters(near_parameters),
                    5,
                    {"train_loss": 0.1},
                ),
            ),
        ]

        strategy.aggregate_fit(1, first_round_results, [])
        aggregated_parameters, metrics = strategy.aggregate_fit(2, second_round_results, [])

        self.assertIsNotNone(aggregated_parameters)
        self.assertEqual(metrics["fedcda_used_cross_round"], 1.0)
        self.assertEqual(strategy.last_selected_rounds, {"a": 2, "b": 1})
        self.assertEqual(strategy.last_selected_client_ids, ["a", "b"])
        aggregated_ndarrays = parameters_to_ndarrays(aggregated_parameters)
        expected = [(left + right) / 2.0 for left, right in zip(near_parameters, far_parameters)]
        for actual, target in zip(aggregated_ndarrays, expected):
            np.testing.assert_allclose(actual, target)

    def test_fedgen_strategy_aggregates_global_mask(self) -> None:
        config = ExperimentConfig.from_run_config(
            {
                "algorithm": "fedgen",
                "num-supernodes": 2,
                "num-classes": 2,
            }
        )
        model = torch.nn.Linear(3, 2)
        strategy = FedGENBuilder().build_strategy(config, model, evaluate_fn=None)
        initial_parameters = get_model_parameters(model)
        first_mask = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        second_mask = np.array([3.0, 4.0, 5.0], dtype=np.float32)

        results = [
            (
                type("Proxy", (), {"cid": "a"})(),
                FitRes(
                    Status(Code.OK, ""),
                    ndarrays_to_parameters(initial_parameters + [first_mask]),
                    1,
                    {"train_loss": 0.5},
                ),
            ),
            (
                type("Proxy", (), {"cid": "b"})(),
                FitRes(
                    Status(Code.OK, ""),
                    ndarrays_to_parameters(initial_parameters + [second_mask]),
                    3,
                    {"train_loss": 0.4},
                ),
            ),
        ]

        aggregated_parameters, metrics = strategy.aggregate_fit(1, results, [])

        self.assertIsNotNone(aggregated_parameters)
        self.assertIn("fedgen_mask_mean", metrics)
        np.testing.assert_allclose(
            strategy.global_mask,
            np.array([2.5, 3.5, 4.5], dtype=np.float32),
        )

    def test_feddrl_strategy_uses_actor_weights_for_aggregation(self) -> None:
        config = ExperimentConfig.from_run_config(
            {
                "algorithm": "feddrl",
                "num-supernodes": 2,
                "feddrl-updates-per-round": 0,
            }
        )
        model = MnistCnnBuilder().build_model(config)
        strategy = FedDRLBuilder().build_strategy(config, model, evaluate_fn=None)
        first_parameters = get_model_parameters(model)
        second_parameters = [parameter + 1.0 for parameter in first_parameters]
        results = [
            (
                type("Proxy", (), {"cid": "a"})(),
                FitRes(
                    Status(Code.OK, ""),
                    ndarrays_to_parameters(first_parameters),
                    10,
                    {
                        "feddrl_pre_train_loss": 0.4,
                        "feddrl_post_train_loss": 0.2,
                    },
                ),
            ),
            (
                type("Proxy", (), {"cid": "b"})(),
                FitRes(
                    Status(Code.OK, ""),
                    ndarrays_to_parameters(second_parameters),
                    10,
                    {
                        "feddrl_pre_train_loss": 0.9,
                        "feddrl_post_train_loss": 0.6,
                    },
                ),
            ),
        ]

        with patch.object(
            strategy,
            "_select_action",
            return_value=(np.array([1.0, -1.0], dtype=np.float32), [0.75, 0.25]),
        ):
            aggregated_parameters, _ = strategy.aggregate_fit(1, results, [])

        self.assertIsNotNone(aggregated_parameters)
        self.assertEqual(strategy.last_aggregation_weights, [0.75, 0.25])

    def test_feddrl_reward_penalizes_average_loss_and_bias_gap(self) -> None:
        reward = compute_feddrl_reward([1.0, 3.0, 2.0])

        self.assertEqual(reward, -4.0)

    def test_fedvck_strategy_updates_memory_after_aggregate_fit(self) -> None:
        config = ExperimentConfig.from_run_config({"algorithm": "fedvck", "num-supernodes": 2})
        model = MnistCnnBuilder().build_model(config)
        strategy = FedVCKBuilder().build_strategy(config, model, evaluate_fn=None)

        initial_parameters = get_model_parameters(model)
        first_model = [parameter + 1.0 for parameter in initial_parameters]
        second_model = [parameter + 3.0 for parameter in initial_parameters]
        condensed_inputs = np.zeros((2, 1, 28, 28), dtype=np.float32)
        condensed_labels = np.array([0, 1], dtype=np.int64)
        first_prototype_sums = np.ones((10, 10), dtype=np.float32)
        second_prototype_sums = np.full((10, 10), 3.0, dtype=np.float32)
        first_counts = np.ones(10, dtype=np.float32)
        second_counts = np.full(10, 3.0, dtype=np.float32)
        results = [
            (
                None,
                FitRes(
                    Status(Code.OK, ""),
                    ndarrays_to_parameters(
                        first_model
                        + [condensed_inputs, condensed_labels, first_prototype_sums, first_counts]
                    ),
                    1,
                    {},
                ),
            ),
            (
                None,
                FitRes(
                    Status(Code.OK, ""),
                    ndarrays_to_parameters(
                        second_model
                        + [condensed_inputs, condensed_labels, second_prototype_sums, second_counts]
                    ),
                    3,
                    {},
                ),
            ),
        ]

        aggregated_parameters, metrics = strategy.aggregate_fit(1, results, [])

        self.assertIsNotNone(aggregated_parameters)
        self.assertEqual(metrics["fedvck_memory_size"], 2)
        self.assertEqual(len(strategy.condensed_memory), 2)

    def test_fedvck_strategy_caps_memory_rounds(self) -> None:
        config = ExperimentConfig.from_run_config(
            {
                "algorithm": "fedvck",
                "num-supernodes": 1,
                "fedvck-max-memory-rounds": 2,
            }
        )
        model = MnistCnnBuilder().build_model(config)
        strategy = FedVCKBuilder().build_strategy(config, model, evaluate_fn=None)
        initial_parameters = get_model_parameters(model)

        for round_index in range(3):
            payload = ndarrays_to_parameters(
                initial_parameters
                + [
                    np.full((1, 1, 28, 28), float(round_index), dtype=np.float32),
                    np.array([round_index % 10], dtype=np.int64),
                    np.ones((10, 10), dtype=np.float32),
                    np.ones(10, dtype=np.float32),
                ]
            )
            strategy.aggregate_fit(
                round_index + 1,
                [(None, FitRes(Status(Code.OK, ""), payload, 1, {}))],
                [],
            )

        self.assertLessEqual(len(strategy.condensed_memory), 2)

    def test_fedvck_server_replay_updates_global_parameters(self) -> None:
        config = ExperimentConfig.from_run_config(
            {
                "algorithm": "fedvck",
                "num-supernodes": 1,
                "fedvck-server-replay-epochs": 1,
                "fedvck-server-replay-learning-rate": 0.05,
            }
        )
        model = torch.nn.Linear(2, 2)
        strategy = FedVCKBuilder().build_strategy(config, model, evaluate_fn=None)
        initial_parameters = [parameter.copy() for parameter in get_model_parameters(model)]
        local_model = [parameter.copy() for parameter in initial_parameters]
        payload = ndarrays_to_parameters(
            local_model
            + [
                np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
                np.array([0, 1], dtype=np.int64),
                np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32),
                np.array([1.0, 1.0], dtype=np.float32),
            ]
        )

        aggregated_parameters, _ = strategy.aggregate_fit(
            1,
            [(None, FitRes(Status(Code.OK, ""), payload, 1, {}))],
            [],
        )

        updated_arrays = parameters_to_ndarrays(aggregated_parameters)
        self.assertTrue(
            any(
                not np.array_equal(before, after)
                for before, after in zip(initial_parameters, updated_arrays)
            )
        )

    def test_fedproto_strategy_aggregates_prototypes_by_class_without_model_averaging(self) -> None:
        model = torch.nn.Linear(2, 1, bias=False)
        initial_parameters = [np.array([[1.0, 2.0]], dtype=np.float32)]
        initial_prototypes = np.zeros((3, 2), dtype=np.float32)

        with tempfile.TemporaryDirectory() as output_dir:
            strategy = FedProtoStrategy(
                fraction_fit=1.0,
                fraction_evaluate=1.0,
                min_fit_clients=2,
                min_evaluate_clients=2,
                min_available_clients=2,
                initial_parameters=ndarrays_to_parameters(initial_parameters),
                num_classes=3,
                global_prototypes=initial_prototypes,
                checkpoint_model=model,
                output_dir=output_dir,
            )
            first_model = [np.array([[3.0, 5.0]], dtype=np.float32)]
            second_model = [np.array([[5.0, 9.0]], dtype=np.float32)]
            first_proto_sums = np.array([[2.0, 2.0], [0.0, 0.0], [4.0, 4.0]], dtype=np.float32)
            second_proto_sums = np.array([[6.0, 6.0], [3.0, 3.0], [0.0, 0.0]], dtype=np.float32)
            first_counts = np.array([1.0, 0.0, 2.0], dtype=np.float32)
            second_counts = np.array([3.0, 1.0, 0.0], dtype=np.float32)
            results = [
                (
                    None,
                    FitRes(
                        Status(Code.OK, ""),
                        ndarrays_to_parameters(first_model + [first_proto_sums, first_counts]),
                        1,
                        {},
                    ),
                ),
                (
                    None,
                    FitRes(
                        Status(Code.OK, ""),
                        ndarrays_to_parameters(second_model + [second_proto_sums, second_counts]),
                        3,
                        {},
                    ),
                ),
            ]

            aggregated_parameters, _ = strategy.aggregate_fit(1, results, [])

        np.testing.assert_allclose(
            parameters_to_ndarrays(aggregated_parameters)[0],
            initial_parameters[0],
        )
        np.testing.assert_allclose(strategy.global_prototypes[0], np.array([2.0, 2.0], dtype=np.float32))
        np.testing.assert_allclose(strategy.global_prototypes[1], np.array([3.0, 3.0], dtype=np.float32))
        np.testing.assert_allclose(strategy.global_prototypes[2], np.array([2.0, 2.0], dtype=np.float32))

    def test_fednp_strategy_aggregates_latent_statistics(self) -> None:
        from fl_baselines.algorithms.fednp import FedNPStrategy

        model = torch.nn.Linear(2, 1, bias=False)
        initial_parameters = [np.array([[1.0, 2.0]], dtype=np.float32)]
        initial_mean = np.zeros(2, dtype=np.float32)
        initial_var = np.ones(2, dtype=np.float32)

        with tempfile.TemporaryDirectory() as output_dir:
            strategy = FedNPStrategy(
                fraction_fit=1.0,
                fraction_evaluate=1.0,
                min_fit_clients=2,
                min_evaluate_clients=2,
                min_available_clients=2,
                initial_parameters=ndarrays_to_parameters(initial_parameters),
                latent_mean=initial_mean,
                latent_var=initial_var,
                prior_variance=1.0,
                model_parameter_count=1,
                checkpoint_model=model,
                output_dir=output_dir,
            )
            first_payload = ndarrays_to_parameters(
                [
                    np.array([[3.0, 5.0]], dtype=np.float32),
                    np.array([4.0, 2.0], dtype=np.float32),
                    np.array([10.0, 8.0], dtype=np.float32),
                    np.array([2.0], dtype=np.float32),
                ]
            )
            second_payload = ndarrays_to_parameters(
                [
                    np.array([[5.0, 9.0]], dtype=np.float32),
                    np.array([6.0, 10.0], dtype=np.float32),
                    np.array([18.0, 34.0], dtype=np.float32),
                    np.array([2.0], dtype=np.float32),
                ]
            )
            aggregated_parameters, _ = strategy.aggregate_fit(
                1,
                [
                    (None, FitRes(Status(Code.OK, ""), first_payload, 1, {})),
                    (None, FitRes(Status(Code.OK, ""), second_payload, 3, {})),
                ],
                [],
            )

        expected_model = (
            np.array([[3.0, 5.0]], dtype=np.float32)
            + 3.0 * np.array([[5.0, 9.0]], dtype=np.float32)
        ) / 4.0
        np.testing.assert_allclose(parameters_to_ndarrays(aggregated_parameters)[0], expected_model)
        np.testing.assert_allclose(strategy.latent_mean, np.array([2.5, 3.0], dtype=np.float32))
        np.testing.assert_allclose(strategy.latent_var, np.array([0.75, 1.5], dtype=np.float32))

    def test_fedcurv_strategy_aggregates_curvature_statistics(self) -> None:
        from fl_baselines.algorithms.fedcurv import FedCurvStrategy

        model = torch.nn.Linear(2, 1, bias=False)
        initial_parameters = [np.array([[1.0, 2.0]], dtype=np.float32)]

        with tempfile.TemporaryDirectory() as output_dir:
            strategy = FedCurvStrategy(
                fraction_fit=1.0,
                fraction_evaluate=1.0,
                min_fit_clients=2,
                min_evaluate_clients=2,
                min_available_clients=2,
                initial_parameters=ndarrays_to_parameters(initial_parameters),
                curvature_aggregate=[np.zeros((1, 2), dtype=np.float32)],
                weighted_parameter_aggregate=[np.zeros((1, 2), dtype=np.float32)],
                model_parameter_count=1,
                checkpoint_model=model,
                output_dir=output_dir,
            )
            first_payload = ndarrays_to_parameters(
                [
                    np.array([[3.0, 5.0]], dtype=np.float32),
                    np.array([[2.0, 1.0]], dtype=np.float32),
                    np.array([[6.0, 5.0]], dtype=np.float32),
                ]
            )
            second_payload = ndarrays_to_parameters(
                [
                    np.array([[5.0, 9.0]], dtype=np.float32),
                    np.array([[4.0, 3.0]], dtype=np.float32),
                    np.array([[20.0, 27.0]], dtype=np.float32),
                ]
            )
            aggregated_parameters, _ = strategy.aggregate_fit(
                1,
                [
                    (None, FitRes(Status(Code.OK, ""), first_payload, 1, {})),
                    (None, FitRes(Status(Code.OK, ""), second_payload, 3, {})),
                ],
                [],
            )

        expected_model = (
            np.array([[3.0, 5.0]], dtype=np.float32)
            + np.array([[5.0, 9.0]], dtype=np.float32)
        ) / 2.0
        np.testing.assert_allclose(parameters_to_ndarrays(aggregated_parameters)[0], expected_model)
        np.testing.assert_allclose(
            strategy.curvature_aggregate[0],
            np.array([[6.0, 4.0]], dtype=np.float32),
        )
        np.testing.assert_allclose(
            strategy.weighted_parameter_aggregate[0],
            np.array([[26.0, 32.0]], dtype=np.float32),
        )

    def test_fedmmd_strategy_downweights_outlier_models(self) -> None:
        from fl_baselines.algorithms.fedmmd import FedMMDStrategy

        model = torch.nn.Linear(2, 1, bias=False)
        initial_parameters = [np.array([[0.0, 0.0]], dtype=np.float32)]

        with tempfile.TemporaryDirectory() as output_dir:
            strategy = FedMMDStrategy(
                fraction_fit=1.0,
                fraction_evaluate=1.0,
                min_fit_clients=3,
                min_evaluate_clients=3,
                min_available_clients=3,
                initial_parameters=ndarrays_to_parameters(initial_parameters),
                sigma=0.5,
                sknq_threshold=0.4,
                min_clients=2,
                entropy_eps=1e-8,
                checkpoint_model=model,
                output_dir=output_dir,
            )
            first_payload = ndarrays_to_parameters([np.array([[0.0, 0.0]], dtype=np.float32)])
            second_payload = ndarrays_to_parameters([np.array([[0.1, 0.1]], dtype=np.float32)])
            outlier_payload = ndarrays_to_parameters([np.array([[8.0, 8.0]], dtype=np.float32)])

            aggregated_parameters, _ = strategy.aggregate_fit(
                1,
                [
                    (None, FitRes(Status(Code.OK, ""), first_payload, 5, {})),
                    (None, FitRes(Status(Code.OK, ""), second_payload, 5, {})),
                    (None, FitRes(Status(Code.OK, ""), outlier_payload, 5, {})),
                ],
                [],
            )

        aggregated = parameters_to_ndarrays(aggregated_parameters)[0]
        self.assertLess(float(aggregated.max()), 1.0)
        self.assertGreaterEqual(len(strategy.last_client_weights), 2)
        self.assertNotIn(2, strategy.last_selected_indices)

    def test_fedexp_strategy_applies_extrapolation_step(self) -> None:
        initial = [np.array([[0.0]], dtype=np.float32)]
        first_local = [np.array([[1.0]], dtype=np.float32)]
        second_local = [np.array([[-3.0]], dtype=np.float32)]

        with tempfile.TemporaryDirectory() as output_dir:
            strategy = FedExPStrategy(
                fraction_fit=1.0,
                fraction_evaluate=1.0,
                min_fit_clients=2,
                min_evaluate_clients=2,
                min_available_clients=2,
                evaluate_fn=None,
                on_fit_config_fn=lambda _: {"algorithm": "fedexp"},
                fit_metrics_aggregation_fn=None,
                evaluate_metrics_aggregation_fn=None,
                initial_parameters=ndarrays_to_parameters(initial),
                epsilon=0.0,
                checkpoint_model=torch.nn.Linear(1, 1, bias=False),
                output_dir=output_dir,
            )

            results = [
                (
                    object(),
                    FitRes(
                        status=Status(code=Code.OK, message=""),
                        parameters=ndarrays_to_parameters(first_local),
                        num_examples=1,
                        metrics={},
                    ),
                ),
                (
                    object(),
                    FitRes(
                        status=Status(code=Code.OK, message=""),
                        parameters=ndarrays_to_parameters(second_local),
                        num_examples=1,
                        metrics={},
                    ),
                ),
            ]

            aggregated_parameters, _ = strategy.aggregate_fit(1, results, [])

        aggregated_arrays = parameters_to_ndarrays(aggregated_parameters)
        self.assertAlmostEqual(float(aggregated_arrays[0][0][0]), -2.5, places=6)
        self.assertAlmostEqual(strategy.last_server_step_size, 2.5, places=6)

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

    def test_pfedme_strategy_mixes_previous_global_with_client_average(self) -> None:
        model = torch.nn.Linear(2, 1, bias=False)
        initial_parameters = [np.array([[10.0, 20.0]], dtype=np.float32)]

        with tempfile.TemporaryDirectory() as output_dir:
            strategy = PFedMeStrategy(
                fraction_fit=1.0,
                fraction_evaluate=1.0,
                min_fit_clients=2,
                min_evaluate_clients=2,
                min_available_clients=2,
                initial_parameters=ndarrays_to_parameters(initial_parameters),
                beta=0.25,
                checkpoint_model=model,
                output_dir=output_dir,
            )
            first_model = [np.array([[6.0, 10.0]], dtype=np.float32)]
            second_model = [np.array([[2.0, 6.0]], dtype=np.float32)]
            results = [
                (
                    None,
                    FitRes(
                        Status(Code.OK, ""),
                        ndarrays_to_parameters(first_model),
                        1,
                        {},
                    ),
                ),
                (
                    None,
                    FitRes(
                        Status(Code.OK, ""),
                        ndarrays_to_parameters(second_model),
                        3,
                        {},
                    ),
                ),
            ]

            aggregated_parameters, _ = strategy.aggregate_fit(1, results, [])

        client_average = (first_model[0] + 3.0 * second_model[0]) / 4.0
        expected = (1.0 - 0.25) * initial_parameters[0] + 0.25 * client_average
        np.testing.assert_allclose(parameters_to_ndarrays(aggregated_parameters)[0], expected)

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
                {"train": loader, "test": loader},
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
                    {"train": loader, "test": loader},
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
                    {"train": loader, "test": loader},
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

    def test_feddyn_client_fit_persists_dynamic_state(self) -> None:
        with tempfile.TemporaryDirectory() as output_dir:
            config = ExperimentConfig.from_run_config(
                {
                    "algorithm": "feddyn",
                    "output-dir": output_dir,
                    "feddyn-alpha": 0.2,
                }
            )
            model = torch.nn.Linear(3, 2)
            loader = DataLoader(
                TensorDataset(torch.ones(4, 3), torch.zeros(4, dtype=torch.long)),
                batch_size=2,
            )
            client = TorchFlowerClient(
                model,
                loaders=type("Loaders", (), {"train": loader, "test": loader})(),
                config=config,
                client_id="7",
            )
            initial_parameters = get_model_parameters(model)

            updated_parameters, num_examples, metrics = client.fit(
                initial_parameters,
                {
                    "algorithm": "feddyn",
                    "local_epochs": 1,
                    "learning_rate": 0.1,
                    "feddyn_alpha": 0.2,
                },
            )

            self.assertEqual(num_examples, 4)
            self.assertEqual(len(updated_parameters), len(initial_parameters))
            self.assertIn("train_loss", metrics)
            state_path = Path(output_dir) / "feddyn_clients" / "7" / "state.pt"
            self.assertTrue(state_path.exists())

    def test_feddyn_client_fit_reuses_saved_dynamic_state(self) -> None:
        with tempfile.TemporaryDirectory() as output_dir:
            config = ExperimentConfig.from_run_config(
                {
                    "algorithm": "feddyn",
                    "output-dir": output_dir,
                    "feddyn-alpha": 0.2,
                }
            )
            model = torch.nn.Linear(3, 2)
            loader = DataLoader(
                TensorDataset(torch.ones(4, 3), torch.zeros(4, dtype=torch.long)),
                batch_size=2,
            )
            client = TorchFlowerClient(
                model,
                loaders=type("Loaders", (), {"train": loader, "test": loader})(),
                config=config,
                client_id="3",
            )
            initial_parameters = get_model_parameters(model)

            client.fit(
                initial_parameters,
                {
                    "algorithm": "feddyn",
                    "local_epochs": 1,
                    "learning_rate": 0.1,
                    "feddyn_alpha": 0.2,
                },
            )
            state_path = Path(output_dir) / "feddyn_clients" / "3" / "state.pt"
            first_state = torch.load(state_path, map_location="cpu", weights_only=True)

            client.fit(
                initial_parameters,
                {
                    "algorithm": "feddyn",
                    "local_epochs": 1,
                    "learning_rate": 0.1,
                    "feddyn_alpha": 0.2,
                },
            )
            second_state = torch.load(state_path, map_location="cpu", weights_only=True)

            self.assertEqual(len(first_state), len(second_state))
            self.assertTrue(
                any(
                    not torch.equal(first, second)
                    for first, second in zip(first_state, second_state)
                )
            )

    def test_feddc_client_fit_persists_state(self) -> None:
        with tempfile.TemporaryDirectory() as output_dir:
            config = ExperimentConfig.from_run_config(
                {
                    "algorithm": "feddc",
                    "output-dir": output_dir,
                    "feddc-alpha": 0.05,
                }
            )
            model = torch.nn.Linear(3, 2)
            loader = DataLoader(
                TensorDataset(torch.ones(4, 3), torch.zeros(4, dtype=torch.long)),
                batch_size=2,
            )
            client = TorchFlowerClient(
                model,
                loaders=type("Loaders", (), {"train": loader, "test": loader})(),
                config=config,
                client_id="feddc-1",
            )
            initial_parameters = get_model_parameters(model)
            server_state = [np.zeros_like(parameter) for parameter in initial_parameters]

            updated_parameters, num_examples, metrics = client.fit(
                initial_parameters + server_state,
                {
                    "algorithm": "feddc",
                    "local_epochs": 1,
                    "learning_rate": 0.1,
                    "feddc_alpha": 0.05,
                },
            )

            self.assertEqual(num_examples, 4)
            self.assertEqual(len(updated_parameters), 2 * len(initial_parameters))
            self.assertIn("train_loss", metrics)
            state_path = Path(output_dir) / "feddc_clients" / "feddc-1" / "state.pt"
            self.assertTrue(state_path.exists())

    def test_feddc_client_fit_reuses_saved_state(self) -> None:
        with tempfile.TemporaryDirectory() as output_dir:
            config = ExperimentConfig.from_run_config(
                {
                    "algorithm": "feddc",
                    "output-dir": output_dir,
                    "feddc-alpha": 0.05,
                }
            )
            model = torch.nn.Linear(3, 2)
            loader = DataLoader(
                TensorDataset(torch.ones(4, 3), torch.zeros(4, dtype=torch.long)),
                batch_size=2,
            )
            client = TorchFlowerClient(
                model,
                loaders=type("Loaders", (), {"train": loader, "test": loader})(),
                config=config,
                client_id="feddc-2",
            )
            initial_parameters = get_model_parameters(model)
            server_state = [np.zeros_like(parameter) for parameter in initial_parameters]

            client.fit(
                initial_parameters + server_state,
                {
                    "algorithm": "feddc",
                    "local_epochs": 1,
                    "learning_rate": 0.1,
                    "feddc_alpha": 0.05,
                },
            )
            state_path = Path(output_dir) / "feddc_clients" / "feddc-2" / "state.pt"
            first_state = torch.load(state_path, map_location="cpu", weights_only=True)

            client.fit(
                initial_parameters + server_state,
                {
                    "algorithm": "feddc",
                    "local_epochs": 1,
                    "learning_rate": 0.1,
                    "feddc_alpha": 0.05,
                },
            )
            second_state = torch.load(state_path, map_location="cpu", weights_only=True)

            self.assertEqual(set(first_state.keys()), {"drift", "local_update"})
            self.assertEqual(set(second_state.keys()), {"drift", "local_update"})
            self.assertTrue(
                any(
                    not torch.equal(first_tensor, second_tensor)
                    for first_tensor, second_tensor in zip(
                        first_state["drift"],
                        second_state["drift"],
                    )
                )
            )

    def test_fedntd_client_fit_returns_metrics(self) -> None:
        config = ExperimentConfig.from_run_config(
            {
                "algorithm": "fedntd",
                "fedntd-beta": 1.2,
                "fedntd-temperature": 2.0,
            }
        )
        model = torch.nn.Linear(3, 2)
        loader = DataLoader(
            TensorDataset(torch.ones(4, 3), torch.zeros(4, dtype=torch.long)),
            batch_size=2,
        )
        client = TorchFlowerClient(
            model,
            loaders=type("Loaders", (), {"train": loader, "test": loader})(),
            config=config,
            client_id="fedntd-1",
        )
        initial_parameters = get_model_parameters(model)

        updated_parameters, num_examples, metrics = client.fit(
            initial_parameters,
            {
                "algorithm": "fedntd",
                "local_epochs": 1,
                "learning_rate": 0.1,
                "fedntd_beta": 1.2,
                "fedntd_temperature": 2.0,
            },
        )

        self.assertEqual(num_examples, 4)
        self.assertEqual(len(updated_parameters), len(initial_parameters))
        self.assertIn("train_loss", metrics)
        self.assertIn("train_accuracy", metrics)

    def test_fedproto_client_fit_returns_prototype_payload(self) -> None:
        config = ExperimentConfig.from_run_config(
            {
                "algorithm": "fedproto",
                "fedproto-lambda": 0.2,
                "num-classes": 3,
            }
        )
        model = torch.nn.Linear(3, 3)
        loader = DataLoader(
            TensorDataset(
                torch.tensor(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 1.0]],
                    dtype=torch.float32,
                ),
                torch.tensor([0, 1, 0, 2], dtype=torch.long),
            ),
            batch_size=2,
        )
        client = TorchFlowerClient(
            model,
            loaders=type("Loaders", (), {"train": loader, "test": loader})(),
            config=config,
            client_id="fedproto-1",
        )
        initial_parameters = get_model_parameters(model)
        global_prototypes = np.zeros((3, 3), dtype=np.float32)

        updated_payload, num_examples, metrics = client.fit(
            initial_parameters + [global_prototypes],
            {
                "algorithm": "fedproto",
                "local_epochs": 1,
                "learning_rate": 0.1,
                "fedproto_lambda": 0.2,
            },
        )

        self.assertEqual(num_examples, 4)
        self.assertEqual(len(updated_payload), len(initial_parameters) + 2)
        self.assertEqual(updated_payload[-2].shape[0], 3)
        self.assertEqual(updated_payload[-1].shape, (3,))
        self.assertIn("train_loss", metrics)
        self.assertIn("train_accuracy", metrics)

    def test_fedproto_client_fit_reuses_persisted_local_model(self) -> None:
        with tempfile.TemporaryDirectory() as output_dir:
            config = ExperimentConfig.from_run_config(
                {
                    "algorithm": "fedproto",
                    "num-classes": 3,
                    "output-dir": output_dir,
                }
            )
            model = torch.nn.Linear(3, 3)
            loader = DataLoader(
                TensorDataset(
                    torch.eye(3, dtype=torch.float32),
                    torch.tensor([0, 1, 2], dtype=torch.long),
                ),
                batch_size=3,
            )
            client = TorchFlowerClient(
                model,
                loaders=type("Loaders", (), {"train": loader, "test": loader})(),
                config=config,
                client_id="fedproto-2",
            )

            persisted_model = torch.nn.Linear(3, 3)
            with torch.no_grad():
                persisted_model.weight.fill_(5.0)
                persisted_model.bias.fill_(2.0)
            persisted_path = Path(output_dir) / "fedproto_clients" / "fedproto-2" / "local_model.pt"
            persisted_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(persisted_model.state_dict(), persisted_path)

            incoming_model = torch.nn.Linear(3, 3)
            with torch.no_grad():
                incoming_model.weight.zero_()
                incoming_model.bias.zero_()
            global_prototypes = np.zeros((3, 3), dtype=np.float32)

            updated_payload, _, _ = client.fit(
                get_model_parameters(incoming_model) + [global_prototypes],
                {
                    "algorithm": "fedproto",
                    "local_epochs": 0,
                    "learning_rate": 0.1,
                    "fedproto_lambda": 0.0,
                },
            )

        np.testing.assert_allclose(updated_payload[0], np.full((3, 3), 5.0, dtype=np.float32))
        np.testing.assert_allclose(updated_payload[1], np.full((3,), 2.0, dtype=np.float32))

    def test_fedmeta_client_fit_returns_meta_gradient_payload(self) -> None:
        config = ExperimentConfig.from_run_config(
            {
                "algorithm": "fedmeta",
                "num-classes": 2,
                "fedmeta-support-fraction": 0.5,
                "fedmeta-inner-learning-rate": 0.1,
                "fedmeta-inner-steps": 1,
            }
        )
        model = torch.nn.Linear(2, 2)
        loader = DataLoader(
            TensorDataset(
                torch.tensor(
                    [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]],
                    dtype=torch.float32,
                ),
                torch.tensor([0, 1, 0, 1], dtype=torch.long),
            ),
            batch_size=4,
        )
        client = TorchFlowerClient(
            model,
            loaders=type("Loaders", (), {"train": loader, "test": loader})(),
            config=config,
            client_id="fedmeta-1",
        )
        initial_parameters = get_model_parameters(model)

        gradient_payload, num_examples, metrics = client.fit(
            initial_parameters,
            {
                "algorithm": "fedmeta",
                "fedmeta_method": "maml",
                "fedmeta_inner_learning_rate": 0.1,
                "fedmeta_support_fraction": 0.5,
                "fedmeta_inner_steps": 1,
                "fedmeta_first_order": True,
            },
        )

        self.assertEqual(num_examples, 4)
        self.assertEqual(len(gradient_payload), len(initial_parameters))
        self.assertEqual(gradient_payload[0].shape, initial_parameters[0].shape)
        self.assertTrue(any(np.any(np.abs(gradient) > 0) for gradient in gradient_payload))
        self.assertIn("meta_query_loss", metrics)
        self.assertIn("meta_query_accuracy", metrics)

    def test_fednp_client_fit_returns_latent_statistics_payload(self) -> None:
        config = ExperimentConfig.from_run_config(
            {
                "algorithm": "fednp",
                "num-classes": 3,
                "fednp-lambda": 0.6,
                "fednp-prior-variance": 1.5,
            }
        )
        model = torch.nn.Linear(3, 3)
        loader = DataLoader(
            TensorDataset(
                torch.tensor(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 1.0]],
                    dtype=torch.float32,
                ),
                torch.tensor([0, 1, 0, 2], dtype=torch.long),
            ),
            batch_size=2,
        )
        client = TorchFlowerClient(
            model,
            loaders=type("Loaders", (), {"train": loader, "test": loader})(),
            config=config,
            client_id="fednp-1",
        )
        initial_parameters = get_model_parameters(model)
        latent_mean = np.zeros(3, dtype=np.float32)
        latent_var = np.ones(3, dtype=np.float32)

        updated_payload, num_examples, metrics = client.fit(
            initial_parameters + [latent_mean, latent_var],
            {
                "algorithm": "fednp",
                "local_epochs": 1,
                "learning_rate": 0.1,
                "fednp_lambda": 0.6,
                "fednp_prior_variance": 1.5,
                "fednp_stability_eps": 1e-5,
            },
        )

        self.assertEqual(num_examples, 4)
        self.assertEqual(len(updated_payload), len(initial_parameters) + 3)
        self.assertEqual(updated_payload[-3].shape, (3,))
        self.assertEqual(updated_payload[-2].shape, (3,))
        self.assertEqual(updated_payload[-1].shape, (1,))
        self.assertTrue(np.isfinite(updated_payload[-3]).all())
        self.assertTrue(np.isfinite(updated_payload[-2]).all())
        self.assertIn("train_loss", metrics)
        self.assertIn("train_accuracy", metrics)
        self.assertIn("fednp_reg_loss", metrics)

    def test_fedcurv_client_fit_returns_curvature_payload(self) -> None:
        config = ExperimentConfig.from_run_config(
            {
                "algorithm": "fedcurv",
                "fedcurv-lambda": 0.4,
                "fedcurv-fisher-batches": 2,
                "fedcurv-stability-eps": 1e-6,
            }
        )
        model = torch.nn.Linear(3, 3)
        loader = DataLoader(
            TensorDataset(
                torch.tensor(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 1.0]],
                    dtype=torch.float32,
                ),
                torch.tensor([0, 1, 0, 2], dtype=torch.long),
            ),
            batch_size=2,
        )
        client = TorchFlowerClient(
            model,
            loaders=type("Loaders", (), {"train": loader, "test": loader})(),
            config=config,
            client_id="fedcurv-1",
        )
        initial_parameters = get_model_parameters(model)
        global_u = [np.zeros_like(array) for array in initial_parameters]
        global_v = [np.zeros_like(array) for array in initial_parameters]

        updated_payload, num_examples, metrics = client.fit(
            initial_parameters + global_u + global_v,
            {
                "algorithm": "fedcurv",
                "local_epochs": 1,
                "learning_rate": 0.1,
                "fedcurv_lambda": 0.4,
                "fedcurv_fisher_batches": 2,
                "fedcurv_stability_eps": 1e-6,
            },
        )

        self.assertEqual(num_examples, 4)
        self.assertEqual(len(updated_payload), len(initial_parameters) * 3)
        curvature_payload = updated_payload[len(initial_parameters) : 2 * len(initial_parameters)]
        weighted_payload = updated_payload[2 * len(initial_parameters) :]
        self.assertTrue(all(array.shape == param.shape for array, param in zip(curvature_payload, initial_parameters)))
        self.assertTrue(all(array.shape == param.shape for array, param in zip(weighted_payload, initial_parameters)))
        self.assertTrue(all(np.isfinite(array).all() for array in curvature_payload))
        self.assertTrue(all(np.isfinite(array).all() for array in weighted_payload))
        self.assertIn("train_loss", metrics)
        self.assertIn("train_accuracy", metrics)
        self.assertIn("fedcurv_reg_loss", metrics)

    def test_train_fedgen_client_returns_local_mask_and_metrics(self) -> None:
        model = torch.nn.Linear(3, 2)
        loader = DataLoader(
            TensorDataset(torch.ones(4, 3), torch.zeros(4, dtype=torch.long)),
            batch_size=2,
        )
        global_mask = np.ones(3, dtype=np.float32)

        metrics, local_mask = train_fedgen_client(
            model,
            loader,
            epochs=1,
            learning_rate=0.1,
            device="cpu",
            global_mask=global_mask,
            fedgen_alpha=1.5,
            fedgen_lambda=0.1,
            fedgen_beta=0.9,
            fedgen_delta=0.9,
            fedgen_warmup_epochs=0,
            fedgen_l1_weight=1e-4,
        )

        self.assertEqual(local_mask.shape, (3,))
        self.assertTrue(np.isfinite(local_mask).all())
        self.assertIn("train_loss", metrics)
        self.assertIn("train_accuracy", metrics)
        self.assertIn("fedgen_mask_mean", metrics)

    def test_train_fedma_client_freezes_prefix_layers(self) -> None:
        model = MnistCnnBuilder().build_model(ExperimentConfig())
        loader = DataLoader(
            TensorDataset(torch.randn(8, 1, 28, 28), torch.randint(0, 10, (8,))),
            batch_size=4,
        )
        initial = {
            name: parameter.detach().clone() for name, parameter in model.named_parameters()
        }

        metrics = train_fedma_client(
            model,
            loader,
            epochs=1,
            learning_rate=0.01,
            device="cpu",
            frozen_layer_prefixes=["conv1", "conv2"],
        )

        self.assertIn("fedma_label_counts", metrics)
        self.assertTrue(torch.equal(initial["conv1.weight"], model.conv1.weight.detach()))
        self.assertTrue(torch.equal(initial["conv2.weight"], model.conv2.weight.detach()))

    def test_evaluate_fedgen_model_returns_metrics(self) -> None:
        model = torch.nn.Linear(3, 2)
        loader = DataLoader(
            TensorDataset(torch.ones(4, 3), torch.zeros(4, dtype=torch.long)),
            batch_size=2,
        )

        loss, metrics = evaluate_fedgen_model(
            model,
            loader,
            device="cpu",
            global_mask=np.ones(3, dtype=np.float32),
        )

        self.assertGreaterEqual(loss, 0.0)
        self.assertIn("accuracy", metrics)
        self.assertIn("precision", metrics)

    def test_feddecorr_loss_returns_finite_scalar(self) -> None:
        features = torch.tensor(
            [[1.0, 0.0, 2.0], [0.5, 1.0, 1.5], [2.0, 0.5, 0.0]],
            dtype=torch.float32,
        )

        loss = feddecorr_loss(features)

        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(float(loss.item()), 0.0)

    def test_feddecorr_training_updates_model_parameters(self) -> None:
        model = torch.nn.Linear(3, 2)
        loader = DataLoader(
            TensorDataset(torch.ones(4, 3), torch.zeros(4, dtype=torch.long)),
            batch_size=2,
        )
        initial_state = {
            key: value.detach().clone() for key, value in model.state_dict().items()
        }

        metrics = train_feddecorr_client(
            model,
            loader,
            epochs=1,
            learning_rate=0.05,
            device="cpu",
            feddecorr_beta=0.1,
        )

        self.assertIn("train_loss", metrics)
        self.assertIn("train_accuracy", metrics)
        self.assertIn("feddecorr_loss", metrics)
        self.assertTrue(
            any(
                not torch.equal(initial_state[key], model.state_dict()[key])
                for key in initial_state
            )
        )

    def test_feddecorr_training_with_zero_beta_still_trains(self) -> None:
        model = torch.nn.Linear(3, 2)
        loader = DataLoader(
            TensorDataset(torch.ones(4, 3), torch.zeros(4, dtype=torch.long)),
            batch_size=2,
        )

        metrics = train_feddecorr_client(
            model,
            loader,
            epochs=1,
            learning_rate=0.05,
            device="cpu",
            feddecorr_beta=0.0,
        )

        self.assertIn("train_loss", metrics)
        self.assertIn("train_accuracy", metrics)
        self.assertIn("feddecorr_loss", metrics)

    def test_train_fedaaw_client_returns_grad_norm_metric(self) -> None:
        model = torch.nn.Linear(3, 2)
        loader = DataLoader(
            TensorDataset(torch.ones(4, 3), torch.zeros(4, dtype=torch.long)),
            batch_size=2,
        )

        metrics = train_fedaaw_client(
            model,
            loader,
            epochs=1,
            learning_rate=0.05,
            device="cpu",
        )

        self.assertIn("train_loss", metrics)
        self.assertIn("train_accuracy", metrics)
        self.assertIn("fedaaw_grad_norm_sq", metrics)
        self.assertGreaterEqual(metrics["fedaaw_grad_norm_sq"], 0.0)

    def test_train_feddrl_client_returns_round_loss_metrics(self) -> None:
        model = torch.nn.Linear(2, 2)
        loader = DataLoader(
            TensorDataset(
                torch.tensor(
                    [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]],
                    dtype=torch.float32,
                ),
                torch.tensor([0, 1, 0, 1], dtype=torch.long),
            ),
            batch_size=2,
        )

        metrics = train_feddrl_client(
            model,
            loader,
            epochs=1,
            learning_rate=0.05,
            device="cpu",
        )

        self.assertIn("feddrl_pre_train_loss", metrics)
        self.assertIn("feddrl_post_train_loss", metrics)
        self.assertGreaterEqual(metrics["feddrl_pre_train_loss"], 0.0)
        self.assertGreaterEqual(metrics["feddrl_post_train_loss"], 0.0)

    def test_feddisco_discrepancy_is_lower_for_uniform_labels(self) -> None:
        uniform_loader = DataLoader(
            TensorDataset(torch.ones(4, 1), torch.tensor([0, 1, 0, 1])),
            batch_size=2,
        )
        skewed_loader = DataLoader(
            TensorDataset(torch.ones(4, 1), torch.tensor([0, 0, 0, 0])),
            batch_size=2,
        )

        uniform = compute_label_distribution(uniform_loader, num_classes=2)
        skewed = compute_label_distribution(skewed_loader, num_classes=2)

        self.assertLess(
            compute_label_distribution_discrepancy(
                uniform,
                metric="kl",
                epsilon=1e-8,
            ),
            compute_label_distribution_discrepancy(
                skewed,
                metric="kl",
                epsilon=1e-8,
            ),
        )

    def test_feddisco_discrepancy_supports_all_metrics(self) -> None:
        distribution = torch.tensor([1.0, 0.0])

        for metric in ["kl", "l1", "l2", "cosine"]:
            with self.subTest(metric=metric):
                discrepancy = compute_label_distribution_discrepancy(
                    distribution,
                    metric=metric,
                    epsilon=1e-8,
                )
                self.assertGreaterEqual(discrepancy, 0.0)

    def test_train_feddisco_client_returns_discrepancy_metric(self) -> None:
        model = torch.nn.Linear(1, 2)
        loader = DataLoader(
            TensorDataset(torch.ones(4, 1), torch.tensor([0, 1, 0, 1])),
            batch_size=2,
        )

        metrics = train_feddisco_client(
            model,
            loader,
            epochs=1,
            learning_rate=0.01,
            device="cpu",
            num_classes=2,
            metric="kl",
            epsilon=1e-8,
        )

        self.assertIn("feddisco_discrepancy", metrics)
        self.assertGreaterEqual(metrics["feddisco_discrepancy"], 0.0)

    def test_torch_flower_client_routes_fedaaw_fit(self) -> None:
        config = ExperimentConfig.from_run_config(
            {
                "algorithm": "fedaaw",
                "local-epochs": 1,
                "learning-rate": 0.05,
            }
        )
        model = torch.nn.Linear(2, 2)
        loader = DataLoader(
            TensorDataset(
                torch.tensor(
                    [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]],
                    dtype=torch.float32,
                ),
                torch.tensor([0, 1, 0, 1], dtype=torch.long),
            ),
            batch_size=2,
        )
        client = TorchFlowerClient(
            model,
            loaders=type("Loaders", (), {"train": loader, "test": loader})(),
            config=config,
            client_id="fedaaw-route",
        )
        initial_parameters = get_model_parameters(model)

        with patch.object(
            torch_client_module,
            "train_fedaaw_client",
            return_value={
                "train_loss": 1.0,
                "train_accuracy": 0.5,
                "fedaaw_grad_norm_sq": 2.5,
            },
        ) as mocked_trainer:
            updated_parameters, num_examples, metrics = client.fit(
                initial_parameters,
                {
                    "algorithm": "fedaaw",
                    "local_epochs": 1,
                    "learning_rate": 0.05,
                },
            )

        mocked_trainer.assert_called_once()
        self.assertEqual(num_examples, len(loader.dataset))
        self.assertEqual(len(updated_parameters), len(initial_parameters))
        self.assertIn("fedaaw_grad_norm_sq", metrics)

    def test_torch_flower_client_routes_feddrl_fit(self) -> None:
        config = ExperimentConfig.from_run_config(
            {
                "algorithm": "feddrl",
                "local-epochs": 1,
                "learning-rate": 0.05,
            }
        )
        model = torch.nn.Linear(2, 2)
        loader = DataLoader(
            TensorDataset(
                torch.tensor(
                    [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]],
                    dtype=torch.float32,
                ),
                torch.tensor([0, 1, 0, 1], dtype=torch.long),
            ),
            batch_size=2,
        )
        client = TorchFlowerClient(
            model,
            loaders=type("Loaders", (), {"train": loader, "test": loader})(),
            config=config,
            client_id="feddrl-route",
        )
        initial_parameters = get_model_parameters(model)

        with patch.object(
            torch_client_module,
            "train_feddrl_client",
            return_value={
                "train_loss": 0.7,
                "train_accuracy": 0.5,
                "feddrl_pre_train_loss": 1.2,
                "feddrl_post_train_loss": 0.8,
            },
        ) as mocked_trainer:
            updated_parameters, num_examples, metrics = client.fit(
                initial_parameters,
                {
                    "algorithm": "feddrl",
                    "local_epochs": 1,
                    "learning_rate": 0.05,
                },
            )

        mocked_trainer.assert_called_once()
        self.assertEqual(num_examples, len(loader.dataset))
        self.assertEqual(len(updated_parameters), len(initial_parameters))
        self.assertIn("feddrl_pre_train_loss", metrics)
        self.assertIn("feddrl_post_train_loss", metrics)

    def test_torch_flower_client_routes_fedgen_fit(self) -> None:
        config = ExperimentConfig.from_run_config(
            {
                "algorithm": "fedgen",
                "fedgen-alpha": 1.5,
                "fedgen-lambda": 0.2,
                "fedgen-beta": 0.9,
                "fedgen-delta": 0.8,
                "fedgen-warmup-epochs": 1,
                "fedgen-l1-weight": 1e-4,
            }
        )
        model = torch.nn.Linear(3, 2)
        loader = DataLoader(
            TensorDataset(torch.ones(4, 3), torch.zeros(4, dtype=torch.long)),
            batch_size=2,
        )
        client = TorchFlowerClient(
            model,
            loaders=type("Loaders", (), {"train": loader, "test": loader})(),
            config=config,
            client_id="fedgen-route",
        )
        initial_parameters = get_model_parameters(model)
        global_mask = np.ones(3, dtype=np.float32)

        with patch.object(
            torch_client_module,
            "train_fedgen_client",
            return_value=(
                {
                    "train_loss": 0.5,
                    "train_accuracy": 0.75,
                    "fedgen_mask_mean": 0.8,
                },
                np.full(3, 2.0, dtype=np.float32),
            ),
        ):
            updated_payload, num_examples, metrics = client.fit(
                initial_parameters + [global_mask],
                {
                    "algorithm": "fedgen",
                    "local_epochs": 1,
                    "learning_rate": 0.1,
                    "fedgen_alpha": 1.5,
                    "fedgen_lambda": 0.2,
                    "fedgen_beta": 0.9,
                    "fedgen_delta": 0.8,
                    "fedgen_warmup_epochs": 1,
                    "fedgen_l1_weight": 1e-4,
                },
            )

        self.assertEqual(num_examples, 4)
        self.assertEqual(len(updated_payload), len(initial_parameters) + 1)
        self.assertTrue(np.array_equal(updated_payload[-1], np.full(3, 2.0, dtype=np.float32)))
        self.assertIn("fedgen_mask_mean", metrics)

    def test_torch_flower_client_routes_fedma_fit(self) -> None:
        config = ExperimentConfig.from_run_config({"algorithm": "fedma"})
        model = MnistCnnBuilder().build_model(config)
        dataset = TensorDataset(torch.randn(8, 1, 28, 28), torch.randint(0, 10, (8,)))
        loader = DataLoader(dataset, batch_size=4)
        client = TorchFlowerClient(
            model,
            loaders=type("Loaders", (), {"train": loader, "test": loader})(),
            config=config,
            client_id="fedma-route",
        )
        initial_parameters = get_model_parameters(model)

        with patch.object(
            torch_client_module,
            "train_fedma_client",
            return_value={"fedma_label_counts": json.dumps([1] * 10)},
        ) as mocked_trainer:
            updated_parameters, num_examples, metrics = client.fit(
                initial_parameters,
                {
                    "algorithm": "fedma",
                    "local_epochs": 1,
                    "learning_rate": 0.01,
                    "fedma_stage": 1,
                },
            )

        self.assertEqual(num_examples, len(dataset))
        self.assertEqual(len(updated_parameters), len(initial_parameters))
        self.assertIn("fedma_label_counts", metrics)
        self.assertEqual(mocked_trainer.call_args.kwargs["frozen_layer_prefixes"], ["conv1"])

    def test_torch_flower_client_evaluates_fedgen_with_mask(self) -> None:
        config = ExperimentConfig.from_run_config({"algorithm": "fedgen"})
        model = torch.nn.Linear(3, 2)
        loader = DataLoader(
            TensorDataset(torch.ones(4, 3), torch.zeros(4, dtype=torch.long)),
            batch_size=2,
        )
        client = TorchFlowerClient(
            model,
            loaders=type("Loaders", (), {"train": loader, "test": loader})(),
            config=config,
            client_id="fedgen-eval",
        )
        initial_parameters = get_model_parameters(model)
        global_mask = np.ones(3, dtype=np.float32)

        with patch.object(
            torch_client_module,
            "evaluate_fedgen_model",
            return_value=(0.4, {"accuracy": 0.8, "precision": 0.8, "recall": 0.8, "f1": 0.8}),
        ):
            loss, num_examples, metrics = client.evaluate(
                initial_parameters + [global_mask],
                {"algorithm": "fedgen"},
            )

        self.assertEqual(loss, 0.4)
        self.assertEqual(num_examples, 4)
        self.assertEqual(metrics["accuracy"], 0.8)

    def test_torch_flower_client_routes_feddisco_fit(self) -> None:
        config = ExperimentConfig.from_run_config(
            {
                "algorithm": "feddisco",
                "local-epochs": 1,
                "learning-rate": 0.05,
                "num-classes": 2,
                "feddisco-metric": "kl",
            }
        )
        model = torch.nn.Linear(1, 2)
        loader = DataLoader(
            TensorDataset(torch.ones(4, 1), torch.tensor([0, 1, 0, 1])),
            batch_size=2,
        )
        client = TorchFlowerClient(
            model,
            loaders=type("Loaders", (), {"train": loader, "test": loader})(),
            config=config,
            client_id="feddisco-route",
        )
        initial_parameters = get_model_parameters(model)

        with patch.object(
            torch_client_module,
            "train_feddisco_client",
            return_value={
                "train_loss": 0.5,
                "train_accuracy": 0.5,
                "feddisco_discrepancy": 0.25,
            },
        ) as mocked_trainer:
            updated_parameters, num_examples, metrics = client.fit(
                initial_parameters,
                {
                    "algorithm": "feddisco",
                    "local_epochs": 1,
                    "learning_rate": 0.05,
                    "num_classes": 2,
                    "feddisco_metric": "kl",
                    "feddisco_epsilon": 1e-8,
                },
            )

        mocked_trainer.assert_called_once()
        self.assertEqual(num_examples, len(loader.dataset))
        self.assertEqual(len(updated_parameters), len(initial_parameters))
        self.assertIn("feddisco_discrepancy", metrics)

    def test_torch_flower_client_routes_feddecorr_to_dedicated_trainer(self) -> None:
        config = ExperimentConfig.from_run_config(
            {
                "algorithm": "feddecorr",
                "local-epochs": 1,
                "learning-rate": 0.05,
                "feddecorr-beta": 0.1,
            }
        )
        model = torch.nn.Linear(3, 2)
        loader = DataLoader(
            TensorDataset(torch.ones(4, 3), torch.zeros(4, dtype=torch.long)),
            batch_size=2,
        )
        client = TorchFlowerClient(
            model,
            loaders=type("Loaders", (), {"train": loader, "test": loader})(),
            config=config,
            client_id="feddecorr-route",
        )
        initial_parameters = get_model_parameters(model)

        with patch.object(
            torch_client_module,
            "train_feddecorr_client",
            return_value={
                "train_loss": 1.0,
                "train_accuracy": 0.5,
                "feddecorr_loss": 0.25,
            },
        ) as mocked_trainer:
            updated_parameters, num_examples, metrics = client.fit(
                initial_parameters,
                {
                    "algorithm": "feddecorr",
                    "local_epochs": 1,
                    "learning_rate": 0.05,
                    "feddecorr_beta": 0.1,
                },
            )

        mocked_trainer.assert_called_once()
        self.assertEqual(num_examples, len(loader.dataset))
        self.assertEqual(len(updated_parameters), len(initial_parameters))
        self.assertEqual(metrics["feddecorr_loss"], 0.25)

    def test_torch_flower_client_routes_fedlc_to_dedicated_trainer(self) -> None:
        config = ExperimentConfig.from_run_config(
            {
                "algorithm": "fedlc",
                "local-epochs": 1,
                "learning-rate": 0.05,
                "num-classes": 2,
                "fedlc-tau": 0.5,
                "fedlc-epsilon": 1e-4,
            }
        )
        model = torch.nn.Linear(3, 2)
        loader = DataLoader(
            TensorDataset(torch.ones(4, 3), torch.zeros(4, dtype=torch.long)),
            batch_size=2,
        )
        client = TorchFlowerClient(
            model,
            loaders=type("Loaders", (), {"train": loader, "test": loader})(),
            config=config,
            client_id="fedlc-route",
        )
        initial_parameters = get_model_parameters(model)

        with patch.object(
            torch_client_module,
            "train_fedlc_client",
            return_value={
                "train_loss": 1.0,
                "train_accuracy": 0.5,
                "fedlc_loss": 1.0,
            },
        ) as mocked_trainer:
            updated_parameters, num_examples, metrics = client.fit(
                initial_parameters,
                {
                    "algorithm": "fedlc",
                    "local_epochs": 1,
                    "learning_rate": 0.05,
                    "num_classes": 2,
                    "fedlc_tau": 0.5,
                    "fedlc_epsilon": 1e-4,
                },
            )

        mocked_trainer.assert_called_once()
        self.assertEqual(num_examples, len(loader.dataset))
        self.assertEqual(len(updated_parameters), len(initial_parameters))
        self.assertEqual(metrics["fedlc_loss"], 1.0)

    def test_torch_flower_client_routes_fedrs_to_dedicated_trainer(self) -> None:
        config = ExperimentConfig.from_run_config(
            {
                "algorithm": "fedrs",
                "local-epochs": 1,
                "learning-rate": 0.05,
                "num-classes": 2,
                "fedrs-alpha": 0.5,
            }
        )
        model = torch.nn.Linear(3, 2)
        loader = DataLoader(
            TensorDataset(torch.ones(4, 3), torch.zeros(4, dtype=torch.long)),
            batch_size=2,
        )
        client = TorchFlowerClient(
            model,
            loaders=type("Loaders", (), {"train": loader, "test": loader})(),
            config=config,
            client_id="fedrs-route",
        )
        initial_parameters = get_model_parameters(model)

        with patch.object(
            torch_client_module,
            "train_fedrs_client",
            return_value={
                "train_loss": 1.0,
                "train_accuracy": 0.5,
                "fedrs_loss": 1.0,
            },
        ) as mocked_trainer:
            updated_parameters, num_examples, metrics = client.fit(
                initial_parameters,
                {
                    "algorithm": "fedrs",
                    "local_epochs": 1,
                    "learning_rate": 0.05,
                    "num_classes": 2,
                    "fedrs_alpha": 0.5,
                },
            )

        mocked_trainer.assert_called_once()
        self.assertEqual(num_examples, len(loader.dataset))
        self.assertEqual(len(updated_parameters), len(initial_parameters))
        self.assertEqual(metrics["fedrs_loss"], 1.0)

    def test_fedspeed_training_updates_model_parameters_and_returns_state(self) -> None:
        model = torch.nn.Linear(3, 2)
        global_model = torch.nn.Linear(3, 2)
        global_model.load_state_dict(model.state_dict())
        loader = DataLoader(
            TensorDataset(torch.ones(4, 3), torch.zeros(4, dtype=torch.long)),
            batch_size=2,
        )
        state = [torch.zeros_like(parameter) for parameter in model.parameters()]
        initial_state = {
            key: value.detach().clone() for key, value in model.state_dict().items()
        }

        metrics, new_state, payload = train_fedspeed_client(
            model,
            global_model,
            loader,
            epochs=1,
            learning_rate=0.05,
            device="cpu",
            fedspeed_lambda=0.2,
            fedspeed_alpha=1.0,
            fedspeed_rho=0.1,
            state=state,
        )

        self.assertIn("train_loss", metrics)
        self.assertIn("train_accuracy", metrics)
        self.assertEqual(len(new_state), len(state))
        self.assertEqual(len(payload), len(state))
        self.assertTrue(
            any(
                not torch.equal(initial_state[key], model.state_dict()[key])
                for key in initial_state
            )
        )

    def test_fedspeed_client_fit_persists_state_and_returns_payload(self) -> None:
        with tempfile.TemporaryDirectory() as output_dir:
            config = ExperimentConfig.from_run_config(
                {
                    "algorithm": "fedspeed",
                    "output-dir": output_dir,
                    "fedspeed-lambda": 0.2,
                    "fedspeed-alpha": 1.0,
                    "fedspeed-rho": 0.1,
                }
            )
            model = torch.nn.Linear(3, 2)
            loader = DataLoader(
                TensorDataset(torch.ones(4, 3), torch.zeros(4, dtype=torch.long)),
                batch_size=2,
            )
            client = TorchFlowerClient(
                model,
                loaders=type("Loaders", (), {"train": loader, "test": loader})(),
                config=config,
                client_id="fedspeed-1",
            )
            initial_parameters = get_model_parameters(model)

            updated_parameters, num_examples, metrics = client.fit(
                initial_parameters,
                {
                    "algorithm": "fedspeed",
                    "local_epochs": 1,
                    "learning_rate": 0.05,
                    "fedspeed_lambda": 0.2,
                    "fedspeed_alpha": 1.0,
                    "fedspeed_rho": 0.1,
                },
            )

            self.assertEqual(num_examples, 4)
            self.assertEqual(len(updated_parameters), len(initial_parameters))
            self.assertIn("train_loss", metrics)
            state_path = Path(output_dir) / "fedspeed_clients" / "fedspeed-1" / "state.pt"
            self.assertTrue(state_path.exists())

    def test_fedspeed_client_fit_reuses_saved_state(self) -> None:
        with tempfile.TemporaryDirectory() as output_dir:
            config = ExperimentConfig.from_run_config(
                {
                    "algorithm": "fedspeed",
                    "output-dir": output_dir,
                    "fedspeed-lambda": 0.2,
                    "fedspeed-alpha": 1.0,
                    "fedspeed-rho": 0.1,
                }
            )
            model = torch.nn.Linear(3, 2)
            loader = DataLoader(
                TensorDataset(torch.ones(4, 3), torch.zeros(4, dtype=torch.long)),
                batch_size=2,
            )
            client = TorchFlowerClient(
                model,
                loaders=type("Loaders", (), {"train": loader, "test": loader})(),
                config=config,
                client_id="fedspeed-2",
            )
            initial_parameters = get_model_parameters(model)

            client.fit(
                initial_parameters,
                {
                    "algorithm": "fedspeed",
                    "local_epochs": 1,
                    "learning_rate": 0.05,
                    "fedspeed_lambda": 0.2,
                    "fedspeed_alpha": 1.0,
                    "fedspeed_rho": 0.1,
                },
            )
            state_path = Path(output_dir) / "fedspeed_clients" / "fedspeed-2" / "state.pt"
            first_state = torch.load(state_path, map_location="cpu", weights_only=True)

            client.fit(
                initial_parameters,
                {
                    "algorithm": "fedspeed",
                    "local_epochs": 1,
                    "learning_rate": 0.05,
                    "fedspeed_lambda": 0.2,
                    "fedspeed_alpha": 1.0,
                    "fedspeed_rho": 0.1,
                },
            )
            second_state = torch.load(state_path, map_location="cpu", weights_only=True)

            self.assertEqual(len(first_state), len(second_state))
            self.assertTrue(
                any(
                    not torch.equal(first, second)
                    for first, second in zip(first_state, second_state)
                )
            )

    def test_fedsam_training_updates_model_parameters(self) -> None:
        model = torch.nn.Linear(2, 2)
        loader = DataLoader(
            TensorDataset(
                torch.tensor(
                    [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]],
                    dtype=torch.float32,
                ),
                torch.tensor([0, 1, 0, 1], dtype=torch.long),
            ),
            batch_size=2,
        )
        initial_state = {
            key: value.detach().clone() for key, value in model.state_dict().items()
        }

        metrics = train_fedsam_client(
            model,
            loader,
            epochs=1,
            learning_rate=0.05,
            device="cpu",
            fedsam_rho=0.5,
        )

        self.assertIn("train_loss", metrics)
        self.assertIn("train_accuracy", metrics)
        self.assertTrue(
            any(
                not torch.equal(initial_state[key], model.state_dict()[key])
                for key in initial_state
            )
        )

    def test_fedsam_training_with_zero_rho_still_trains(self) -> None:
        model = torch.nn.Linear(2, 2)
        loader = DataLoader(
            TensorDataset(
                torch.tensor(
                    [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]],
                    dtype=torch.float32,
                ),
                torch.tensor([0, 1, 0, 1], dtype=torch.long),
            ),
            batch_size=2,
        )

        metrics = train_fedsam_client(
            model,
            loader,
            epochs=1,
            learning_rate=0.05,
            device="cpu",
            fedsam_rho=0.0,
        )

        self.assertIn("train_loss", metrics)
        self.assertIn("train_accuracy", metrics)

    def test_torch_flower_client_routes_fedsam_to_dedicated_trainer(self) -> None:
        config = ExperimentConfig.from_run_config(
            {
                "algorithm": "fedsam",
                "local-epochs": 1,
                "learning-rate": 0.05,
                "fedsam-rho": 0.5,
            }
        )
        model = torch.nn.Linear(2, 2)
        loader = DataLoader(
            TensorDataset(
                torch.tensor(
                    [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]],
                    dtype=torch.float32,
                ),
                torch.tensor([0, 1, 0, 1], dtype=torch.long),
            ),
            batch_size=2,
        )
        client = TorchFlowerClient(
            model,
            loaders=type("Loaders", (), {"train": loader, "test": loader})(),
            config=config,
            client_id="fedsam-route",
        )
        initial_parameters = get_model_parameters(model)

        with patch.object(
            torch_client_module,
            "train_fedsam_client",
            return_value={"train_loss": 1.0, "train_accuracy": 0.5},
        ) as mocked_trainer:
            updated_parameters, num_examples, metrics = client.fit(
                initial_parameters,
                {
                    "algorithm": "fedsam",
                    "local_epochs": 1,
                    "learning_rate": 0.05,
                    "fedsam_rho": 0.5,
                },
            )

        mocked_trainer.assert_called_once()
        self.assertEqual(num_examples, len(loader.dataset))
        self.assertEqual(len(updated_parameters), len(initial_parameters))
        self.assertEqual(metrics["train_loss"], 1.0)

    def test_fedntd_training_keeps_teacher_fixed(self) -> None:
        student_model = torch.nn.Linear(3, 2)
        teacher_model = torch.nn.Linear(3, 2)
        teacher_model.load_state_dict(student_model.state_dict())
        teacher_before = {
            key: value.detach().clone()
            for key, value in teacher_model.state_dict().items()
        }
        student_before = {
            key: value.detach().clone()
            for key, value in student_model.state_dict().items()
        }
        loader = DataLoader(
            TensorDataset(torch.ones(4, 3), torch.zeros(4, dtype=torch.long)),
            batch_size=2,
        )

        metrics = train_fedntd_client(
            student_model,
            teacher_model,
            loader,
            epochs=1,
            learning_rate=0.1,
            device="cpu",
            fedntd_beta=1.2,
            fedntd_temperature=2.0,
        )

        self.assertIn("train_loss", metrics)
        self.assertTrue(
            any(
                not torch.equal(student_before[key], student_model.state_dict()[key])
                for key in student_before
            )
        )
        self.assertTrue(
            all(
                torch.equal(teacher_before[key], teacher_model.state_dict()[key])
                for key in teacher_before
            )
        )

    def test_fedlc_loss_handles_missing_classes(self) -> None:
        logits = torch.tensor([[2.0, 0.0, 1.0], [0.0, 2.0, 1.0]])
        targets = torch.tensor([0, 1])
        class_counts = torch.tensor([3.0, 1.0, 0.0])

        loss = fedlc_loss(logits, targets, class_counts, tau=0.5, epsilon=1e-4)

        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(loss.item()), 0.0)

    def test_fedrs_loss_handles_missing_classes(self) -> None:
        logits = torch.tensor([[2.0, 0.0, 1.0], [0.0, 2.0, 1.0]])
        targets = torch.tensor([0, 1])
        class_mask = torch.tensor([1.0, 1.0, 0.0])

        loss = fedrs_loss(logits, targets, class_mask, alpha=0.5)

        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(loss.item()), 0.0)

    def test_local_class_counts_counts_valid_labels(self) -> None:
        loader = DataLoader(
            TensorDataset(
                torch.ones(5, 2),
                torch.tensor([0, 1, 1, 3, -1], dtype=torch.long),
            ),
            batch_size=2,
        )

        counts = local_class_counts(loader, num_classes=3)

        self.assertEqual(counts.tolist(), [1.0, 2.0, 0.0])

    def test_observed_class_mask_marks_seen_labels(self) -> None:
        loader = DataLoader(
            TensorDataset(
                torch.ones(5, 2),
                torch.tensor([0, 1, 1, 3, -1], dtype=torch.long),
            ),
            batch_size=2,
        )

        mask = observed_class_mask(loader, num_classes=3)

        self.assertEqual(mask.tolist(), [1.0, 1.0, 0.0])

    def test_fedlc_training_updates_model_parameters(self) -> None:
        model = torch.nn.Linear(3, 2)
        initial_state = {
            key: value.detach().clone() for key, value in model.state_dict().items()
        }
        loader = DataLoader(
            TensorDataset(
                torch.ones(4, 3),
                torch.tensor([0, 0, 0, 1], dtype=torch.long),
            ),
            batch_size=2,
        )

        metrics = train_fedlc_client(
            model,
            loader,
            epochs=1,
            learning_rate=0.1,
            device="cpu",
            num_classes=2,
            fedlc_tau=0.5,
            fedlc_epsilon=1e-4,
        )

        self.assertIn("fedlc_loss", metrics)
        self.assertTrue(
            any(
                not torch.equal(initial_state[key], model.state_dict()[key])
                for key in initial_state
            )
        )

    def test_fedrs_training_updates_model_parameters(self) -> None:
        model = torch.nn.Linear(3, 2)
        initial_state = {
            key: value.detach().clone() for key, value in model.state_dict().items()
        }
        loader = DataLoader(
            TensorDataset(
                torch.ones(4, 3),
                torch.tensor([0, 0, 0, 1], dtype=torch.long),
            ),
            batch_size=2,
        )

        metrics = train_fedrs_client(
            model,
            loader,
            epochs=1,
            learning_rate=0.1,
            device="cpu",
            num_classes=2,
            fedrs_alpha=0.5,
        )

        self.assertIn("fedrs_loss", metrics)
        self.assertTrue(
            any(
                not torch.equal(initial_state[key], model.state_dict()[key])
                for key in initial_state
            )
        )

    def test_ditto_client_fit_persists_personalized_state(self) -> None:
        with tempfile.TemporaryDirectory() as output_dir:
            config = ExperimentConfig.from_run_config(
                {
                    "algorithm": "ditto",
                    "output-dir": output_dir,
                    "ditto-lambda": 0.3,
                }
            )
            model = torch.nn.Linear(3, 2)
            loader = DataLoader(
                TensorDataset(torch.ones(4, 3), torch.zeros(4, dtype=torch.long)),
                batch_size=2,
            )
            client = TorchFlowerClient(
                model,
                loaders=type("Loaders", (), {"train": loader, "test": loader})(),
                config=config,
                client_id="ditto-1",
            )
            initial_parameters = get_model_parameters(model)

            updated_parameters, num_examples, metrics = client.fit(
                initial_parameters,
                {
                    "algorithm": "ditto",
                    "local_epochs": 1,
                    "learning_rate": 0.1,
                    "ditto_lambda": 0.3,
                },
            )

            self.assertEqual(num_examples, 4)
            self.assertEqual(len(updated_parameters), len(initial_parameters))
            self.assertIn("train_loss", metrics)
            self.assertTrue(
                (
                    Path(output_dir)
                    / "ditto_clients"
                    / "ditto-1"
                    / "personalized.pt"
                ).exists()
            )

    def test_ditto_client_fit_reuses_personalized_state(self) -> None:
        with tempfile.TemporaryDirectory() as output_dir:
            config = ExperimentConfig.from_run_config(
                {
                    "algorithm": "ditto",
                    "output-dir": output_dir,
                    "ditto-lambda": 0.3,
                }
            )
            model = torch.nn.Linear(3, 2)
            loader = DataLoader(
                TensorDataset(torch.ones(4, 3), torch.zeros(4, dtype=torch.long)),
                batch_size=2,
            )
            client = TorchFlowerClient(
                model,
                loaders=type("Loaders", (), {"train": loader, "test": loader})(),
                config=config,
                client_id="ditto-2",
            )
            initial_parameters = get_model_parameters(model)

            client.fit(
                initial_parameters,
                {
                    "algorithm": "ditto",
                    "local_epochs": 1,
                    "learning_rate": 0.1,
                    "ditto_lambda": 0.3,
                },
            )
            personal_path = (
                Path(output_dir)
                / "ditto_clients"
                / "ditto-2"
                / "personalized.pt"
            )
            first_state = torch.load(personal_path, map_location="cpu", weights_only=True)

            client.fit(
                initial_parameters,
                {
                    "algorithm": "ditto",
                    "local_epochs": 1,
                    "learning_rate": 0.1,
                    "ditto_lambda": 0.3,
                },
            )
            second_state = torch.load(personal_path, map_location="cpu", weights_only=True)

            self.assertEqual(len(first_state), len(second_state))
            self.assertTrue(
                any(
                    not torch.equal(first_state[key], second_state[key])
                    for key in first_state
                )
            )

    def test_pfedme_client_fit_persists_personalized_state(self) -> None:
        with tempfile.TemporaryDirectory() as output_dir:
            config = ExperimentConfig.from_run_config(
                {
                    "algorithm": "pfedme",
                    "output-dir": output_dir,
                    "pfedme-lambda": 15.0,
                    "pfedme-beta": 0.7,
                    "pfedme-personal-learning-rate": 0.02,
                    "pfedme-personal-steps": 3,
                }
            )
            model = torch.nn.Linear(3, 2)
            loader = DataLoader(
                TensorDataset(torch.ones(4, 3), torch.zeros(4, dtype=torch.long)),
                batch_size=2,
            )
            client = TorchFlowerClient(
                model,
                loaders=type("Loaders", (), {"train": loader, "test": loader})(),
                config=config,
                client_id="pfedme-1",
            )
            initial_parameters = get_model_parameters(model)

            updated_parameters, num_examples, metrics = client.fit(
                initial_parameters,
                {
                    "algorithm": "pfedme",
                    "local_epochs": 1,
                    "learning_rate": 0.1,
                    "pfedme_lambda": 15.0,
                    "pfedme_beta": 0.7,
                    "pfedme_personal_learning_rate": 0.02,
                    "pfedme_personal_steps": 3,
                },
            )

            self.assertEqual(num_examples, 4)
            self.assertEqual(len(updated_parameters), len(initial_parameters))
            self.assertIn("train_loss", metrics)
            self.assertTrue(
                (
                    Path(output_dir)
                    / "pfedme_clients"
                    / "pfedme-1"
                    / "personalized.pt"
                ).exists()
            )

    def test_pfedme_client_fit_reuses_personalized_state(self) -> None:
        with tempfile.TemporaryDirectory() as output_dir:
            config = ExperimentConfig.from_run_config(
                {
                    "algorithm": "pfedme",
                    "output-dir": output_dir,
                    "pfedme-lambda": 15.0,
                    "pfedme-beta": 0.7,
                    "pfedme-personal-learning-rate": 0.02,
                    "pfedme-personal-steps": 3,
                }
            )
            model = torch.nn.Linear(3, 2)
            loader = DataLoader(
                TensorDataset(torch.ones(4, 3), torch.zeros(4, dtype=torch.long)),
                batch_size=2,
            )
            client = TorchFlowerClient(
                model,
                loaders=type("Loaders", (), {"train": loader, "test": loader})(),
                config=config,
                client_id="pfedme-2",
            )
            initial_parameters = get_model_parameters(model)

            client.fit(
                initial_parameters,
                {
                    "algorithm": "pfedme",
                    "local_epochs": 1,
                    "learning_rate": 0.1,
                    "pfedme_lambda": 15.0,
                    "pfedme_beta": 0.7,
                    "pfedme_personal_learning_rate": 0.02,
                    "pfedme_personal_steps": 3,
                },
            )
            personal_path = (
                Path(output_dir)
                / "pfedme_clients"
                / "pfedme-2"
                / "personalized.pt"
            )
            first_state = torch.load(personal_path, map_location="cpu", weights_only=True)

            client.fit(
                initial_parameters,
                {
                    "algorithm": "pfedme",
                    "local_epochs": 1,
                    "learning_rate": 0.1,
                    "pfedme_lambda": 15.0,
                    "pfedme_beta": 0.7,
                    "pfedme_personal_learning_rate": 0.02,
                    "pfedme_personal_steps": 3,
                },
            )
            second_state = torch.load(personal_path, map_location="cpu", weights_only=True)

            self.assertEqual(len(first_state), len(second_state))
            self.assertTrue(
                any(
                    not torch.equal(first_state[key], second_state[key])
                    for key in first_state
                )
            )

    def test_apfl_client_fit_persists_personalized_state_and_alpha(self) -> None:
        with tempfile.TemporaryDirectory() as output_dir:
            config = ExperimentConfig.from_run_config(
                {
                    "algorithm": "apfl",
                    "output-dir": output_dir,
                    "apfl-alpha": 0.6,
                    "apfl-personal-learning-rate": 0.02,
                    "apfl-adaptive-alpha": True,
                    "apfl-alpha-learning-rate": 0.01,
                }
            )
            model = torch.nn.Linear(3, 2)
            loader = DataLoader(
                TensorDataset(torch.ones(4, 3), torch.zeros(4, dtype=torch.long)),
                batch_size=2,
            )
            client = TorchFlowerClient(
                model,
                loaders=type("Loaders", (), {"train": loader, "test": loader})(),
                config=config,
                client_id="apfl-1",
            )
            initial_parameters = get_model_parameters(model)

            updated_parameters, num_examples, metrics = client.fit(
                initial_parameters,
                {
                    "algorithm": "apfl",
                    "local_epochs": 1,
                    "learning_rate": 0.1,
                    "apfl_alpha": 0.6,
                    "apfl_personal_learning_rate": 0.02,
                    "apfl_adaptive_alpha": True,
                    "apfl_alpha_learning_rate": 0.01,
                },
            )

            self.assertEqual(num_examples, 4)
            self.assertEqual(len(updated_parameters), len(initial_parameters))
            self.assertIn("train_loss", metrics)
            self.assertIn("apfl_alpha", metrics)
            self.assertTrue(
                (Path(output_dir) / "apfl_clients" / "apfl-1" / "personalized.pt").exists()
            )
            self.assertTrue(
                (Path(output_dir) / "apfl_clients" / "apfl-1" / "alpha.json").exists()
            )

    def test_apfl_client_evaluate_uses_personalized_mixture(self) -> None:
        with tempfile.TemporaryDirectory() as output_dir:
            config = ExperimentConfig.from_run_config(
                {
                    "algorithm": "apfl",
                    "output-dir": output_dir,
                    "apfl-alpha": 0.75,
                    "apfl-personal-learning-rate": 0.02,
                    "apfl-adaptive-alpha": False,
                    "apfl-alpha-learning-rate": 0.01,
                }
            )
            model = torch.nn.Linear(2, 2)
            loader = DataLoader(
                TensorDataset(torch.tensor([[1.0, 0.0]], dtype=torch.float32), torch.tensor([0])),
                batch_size=1,
            )
            client = TorchFlowerClient(
                model,
                loaders=type("Loaders", (), {"train": loader, "test": loader})(),
                config=config,
                client_id="apfl-2",
            )
            global_model = torch.nn.Linear(2, 2)
            personalized_model = torch.nn.Linear(2, 2)
            with torch.no_grad():
                global_model.weight.copy_(torch.tensor([[0.0, 0.0], [1.0, 1.0]]))
                global_model.bias.copy_(torch.tensor([0.0, 0.0]))
                personalized_model.weight.copy_(torch.tensor([[2.0, 0.0], [0.0, 2.0]]))
                personalized_model.bias.copy_(torch.tensor([0.0, 0.0]))
            client._save_apfl_personalized_model(personalized_model)
            client._save_apfl_alpha(0.75)

            loss, num_examples, metrics = client.evaluate(
                get_model_parameters(global_model),
                {"algorithm": "apfl"},
            )

            self.assertEqual(num_examples, 1)
            self.assertLess(loss, 0.7)
            self.assertEqual(metrics["accuracy"], 1.0)

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

    def test_fedent_learning_rate_is_finite(self) -> None:
        parameter_vector = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float32)
        gradient_vector = torch.tensor([0.2, -0.1, 0.05], dtype=torch.float32)

        learning_rate = compute_fedent_learning_rate(
            parameter_vector=parameter_vector,
            gradient_vector=gradient_vector,
            phi1_vector=parameter_vector.clone(),
            phi2_scalar=14.0,
            aggregation_weight=0.99,
            epsilon=1e-8,
            max_learning_rate=1.0,
        )

        self.assertTrue(torch.isfinite(torch.tensor(learning_rate)))
        self.assertGreaterEqual(learning_rate, 0.0)
        self.assertLessEqual(learning_rate, 1.0)

    def test_fedent_decay_uses_previous_eta(self) -> None:
        decayed = apply_fedent_eta_decay(
            previous_eta=0.8,
            current_eta=0.2,
            gamma=0.9,
        )

        self.assertAlmostEqual(decayed, 0.74)

    def test_fedent_learning_rate_handles_small_phi2(self) -> None:
        learning_rate = compute_fedent_learning_rate(
            parameter_vector=torch.tensor([1.0], dtype=torch.float32),
            gradient_vector=torch.tensor([0.1], dtype=torch.float32),
            phi1_vector=torch.tensor([1.0], dtype=torch.float32),
            phi2_scalar=0.0,
            aggregation_weight=0.99,
            epsilon=1e-8,
            max_learning_rate=0.5,
        )

        self.assertLessEqual(learning_rate, 0.5)

    def test_fedent_local_training_updates_parameters(self) -> None:
        model = torch.nn.Linear(2, 2)
        loader = DataLoader(
            TensorDataset(
                torch.tensor(
                    [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]],
                    dtype=torch.float32,
                ),
                torch.tensor([0, 1, 0, 1], dtype=torch.long),
            ),
            batch_size=2,
        )
        initial_state = {
            key: value.detach().clone() for key, value in model.state_dict().items()
        }
        phi1_vector = torch.cat([parameter.detach().flatten() for parameter in model.parameters()])

        metrics = train_fedent_client(
            model,
            loader,
            epochs=1,
            learning_rate=0.01,
            device="cpu",
            phi1_vector=phi1_vector,
            phi2_scalar=1.0,
            fedent_beta=0.99,
            fedent_gamma=0.99,
            fedent_epsilon=1e-8,
            fedent_max_learning_rate=1.0,
            previous_eta=None,
        )

        self.assertIn("fedent_learning_rate", metrics)
        self.assertTrue(
            any(
                not torch.equal(initial_state[key], model.state_dict()[key])
                for key in initial_state
            )
        )

    def test_train_fedvck_client_returns_payloads(self) -> None:
        model = torch.nn.Linear(2, 2)
        loader = DataLoader(
            TensorDataset(
                torch.tensor(
                    [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]],
                    dtype=torch.float32,
                ),
                torch.tensor([0, 1, 0, 1], dtype=torch.long),
            ),
            batch_size=2,
        )

        metrics, condensed_inputs, condensed_labels, prototype_sums, prototype_counts = (
            train_fedvck_client(
                model,
                loader,
                epochs=1,
                learning_rate=0.05,
                device="cpu",
                condensed_ratio=0.5,
                condensed_steps=1,
                condensed_learning_rate=0.1,
                importance_alpha=0.5,
                enable_latent_constraints=False,
                previous_model_state=None,
            )
        )

        self.assertIn("fedvck_condensed_size", metrics)
        self.assertEqual(condensed_inputs.shape[0], condensed_labels.shape[0])
        self.assertTrue(np.isfinite(prototype_sums).all())
        self.assertTrue(np.isfinite(prototype_counts).all())

    def test_fedvck_client_persists_previous_state(self) -> None:
        with tempfile.TemporaryDirectory() as output_dir:
            config = ExperimentConfig.from_run_config(
                {
                    "algorithm": "fedvck",
                    "output-dir": output_dir,
                    "local-epochs": 1,
                    "learning-rate": 0.05,
                }
            )
            model = torch.nn.Linear(2, 2)
            loader = DataLoader(
                TensorDataset(
                    torch.tensor(
                        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]],
                        dtype=torch.float32,
                    ),
                    torch.tensor([0, 1, 0, 1], dtype=torch.long),
                ),
                batch_size=2,
            )
            client = TorchFlowerClient(
                model,
                loaders=type("Loaders", (), {"train": loader, "test": loader})(),
                config=config,
                client_id="fedvck-1",
            )

            client.fit(
                get_model_parameters(model),
                {
                    "algorithm": "fedvck",
                    "local_epochs": 1,
                    "learning_rate": 0.05,
                    "fedvck_condensed_ratio": 0.5,
                    "fedvck_condensed_steps": 1,
                    "fedvck_condensed_learning_rate": 0.1,
                    "fedvck_importance_alpha": 0.5,
                },
            )

            state_path = Path(output_dir) / "fedvck_clients" / "fedvck-1" / "state.pt"
            self.assertTrue(state_path.exists())

    def test_torch_flower_client_routes_fedvck_fit(self) -> None:
        config = ExperimentConfig.from_run_config(
            {
                "algorithm": "fedvck",
                "local-epochs": 1,
                "learning-rate": 0.05,
                "fedvck-condensed-ratio": 0.5,
                "fedvck-condensed-steps": 1,
                "fedvck-condensed-learning-rate": 0.1,
                "fedvck-importance-alpha": 0.5,
            }
        )
        model = torch.nn.Linear(2, 2)
        loader = DataLoader(
            TensorDataset(
                torch.tensor(
                    [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]],
                    dtype=torch.float32,
                ),
                torch.tensor([0, 1, 0, 1], dtype=torch.long),
            ),
            batch_size=2,
        )
        client = TorchFlowerClient(
            model,
            loaders=type("Loaders", (), {"train": loader, "test": loader})(),
            config=config,
            client_id="fedvck-route",
        )
        initial_parameters = get_model_parameters(model)

        with patch.object(
            torch_client_module,
            "train_fedvck_client",
            return_value=(
                {
                    "train_loss": 1.0,
                    "train_accuracy": 0.5,
                    "fedvck_condensed_size": 2,
                },
                np.zeros((2, 2), dtype=np.float32),
                np.array([0, 1], dtype=np.int64),
                np.ones((2, 2), dtype=np.float32),
                np.ones(2, dtype=np.float32),
            ),
        ) as mocked_trainer:
            updated_parameters, num_examples, metrics = client.fit(
                initial_parameters,
                {
                    "algorithm": "fedvck",
                    "local_epochs": 1,
                    "learning_rate": 0.05,
                    "fedvck_condensed_ratio": 0.5,
                    "fedvck_condensed_steps": 1,
                    "fedvck_condensed_learning_rate": 0.1,
                    "fedvck_importance_alpha": 0.5,
                },
            )

        mocked_trainer.assert_called_once()
        self.assertEqual(num_examples, len(loader.dataset))
        self.assertEqual(len(updated_parameters), len(initial_parameters) + 4)
        self.assertIn("fedvck_condensed_size", metrics)

    def test_torch_flower_client_routes_fedent_fit(self) -> None:
        config = ExperimentConfig.from_run_config(
            {
                "algorithm": "fedent",
                "local-epochs": 1,
                "learning-rate": 0.05,
                "fedent-beta": 0.99,
                "fedent-gamma": 0.99,
                "fedent-epsilon": 1e-8,
                "fedent-max-learning-rate": 1.0,
            }
        )
        model = torch.nn.Linear(2, 2)
        loader = DataLoader(
            TensorDataset(
                torch.tensor(
                    [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]],
                    dtype=torch.float32,
                ),
                torch.tensor([0, 1, 0, 1], dtype=torch.long),
            ),
            batch_size=2,
        )
        client = TorchFlowerClient(
            model,
            loaders=type("Loaders", (), {"train": loader, "test": loader})(),
            config=config,
            client_id="fedent-route",
        )
        initial_parameters = get_model_parameters(model)

        with patch.object(
            torch_client_module,
            "train_fedent_client",
            return_value={
                "train_loss": 1.0,
                "train_accuracy": 0.5,
                "fedent_learning_rate": 0.1,
            },
        ) as mocked_trainer:
            updated_parameters, num_examples, metrics = client.fit(
                initial_parameters,
                {
                    "algorithm": "fedent",
                    "local_epochs": 1,
                    "learning_rate": 0.05,
                    "fedent_beta": 0.99,
                    "fedent_gamma": 0.99,
                    "fedent_epsilon": 1e-8,
                    "fedent_max_learning_rate": 1.0,
                    "fedent_phi1": json.dumps([parameter.tolist() for parameter in initial_parameters]),
                    "fedent_phi2": 1.0,
                },
            )

        mocked_trainer.assert_called_once()
        self.assertEqual(num_examples, len(loader.dataset))
        self.assertEqual(len(updated_parameters), len(initial_parameters))
        self.assertIn("fedent_learning_rate", metrics)
        self.assertIn("fedent_weight_sq_norm", metrics)


if __name__ == "__main__":
    unittest.main()
