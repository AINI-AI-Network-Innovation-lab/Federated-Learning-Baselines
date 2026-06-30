"""Flower ServerApp entrypoint."""

from __future__ import annotations

from typing import cast

from flwr.app import Context
from flwr.server import ServerAppComponents, ServerConfig
from flwr.serverapp import ServerApp

from fl_baselines import register_default_components
from fl_baselines.algorithms.base import AlgorithmBuilder
from fl_baselines.clients.torch_client import set_model_parameters
from fl_baselines.core.config import ExperimentConfig
from fl_baselines.core.registry import ALGORITHMS, DATASETS, MODELS
from fl_baselines.datasets.base import DatasetBuilder
from fl_baselines.logging.artifacts import save_config
from fl_baselines.models.base import ModelBuilder
from fl_baselines.training.evaluate import evaluate_model


def server_fn(context: Context) -> ServerAppComponents:
    register_default_components()
    config = ExperimentConfig.from_run_config(context.run_config)
    save_config(config.output_dir, context.run_config)

    dataset = cast(DatasetBuilder, DATASETS.get(config.dataset))
    model_builder = cast(ModelBuilder, MODELS.get(config.model))
    algorithm = cast(AlgorithmBuilder, ALGORITHMS.get(config.algorithm))

    initial_model = model_builder.build_model(config)
    server_loader = dataset.build_server_loader(config)

    def evaluate_fn(server_round, parameters, eval_config):
        set_model_parameters(initial_model, parameters)
        return evaluate_model(initial_model, server_loader, config.device)

    strategy = algorithm.build_strategy(config, initial_model, evaluate_fn=evaluate_fn)
    server_config = ServerConfig(num_rounds=config.num_server_rounds)
    return ServerAppComponents(strategy=strategy, config=server_config)


app = ServerApp(server_fn=server_fn)
