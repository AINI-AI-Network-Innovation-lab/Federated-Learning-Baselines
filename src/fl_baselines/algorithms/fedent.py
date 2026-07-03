"""FedEnt algorithm builder."""

from __future__ import annotations

import json
import math

import numpy as np
import torch
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg

from fl_baselines.clients.torch_client import get_model_parameters
from fl_baselines.core.config import ExperimentConfig
from fl_baselines.core.types import ServerEvaluateFn
from fl_baselines.logging.checkpointing import CheckpointingStrategyMixin
from fl_baselines.logging.metrics import weighted_average


class FedEntStrategy(CheckpointingStrategyMixin, FedAvg):
    def __init__(
        self,
        *,
        checkpoint_model: torch.nn.Module,
        output_dir: str,
        fedent_beta: float,
        fedent_gamma: float,
        fedent_epsilon: float,
        fedent_fixed_point_steps: int,
        fedent_max_learning_rate: float,
        **kwargs: object,
    ) -> None:
        super().__init__(
            checkpoint_model=checkpoint_model,
            output_dir=output_dir,
            **kwargs,
        )
        self._fedent_beta = fedent_beta
        self._fedent_gamma = fedent_gamma
        self._fedent_epsilon = fedent_epsilon
        self._fedent_fixed_point_steps = fedent_fixed_point_steps
        self._fedent_max_learning_rate = fedent_max_learning_rate
        self._phi1 = get_model_parameters(checkpoint_model)
        self._phi2 = self._compute_phi2(self._phi1)

    def make_fit_config(self, config: ExperimentConfig):
        def fn(server_round: int) -> dict[str, bool | bytes | float | int | str]:
            return {
                "algorithm": "fedent",
                "server_round": server_round,
                "local_epochs": config.local_epochs,
                "learning_rate": config.learning_rate,
                "fedent_beta": self._fedent_beta,
                "fedent_gamma": self._fedent_gamma,
                "fedent_epsilon": self._fedent_epsilon,
                "fedent_fixed_point_steps": self._fedent_fixed_point_steps,
                "fedent_max_learning_rate": self._fedent_max_learning_rate,
                "fedent_enable_decay": config.fedent_enable_decay,
                "fedent_phi1": self._serialize_phi1(),
                "fedent_phi2": self._phi2,
            }

        return fn

    def _serialize_phi1(self) -> str:
        return json.dumps([parameter.tolist() for parameter in self._phi1])

    def _compute_phi2(self, parameters: list[np.ndarray]) -> float:
        squared_norm = 0.0
        for parameter in parameters:
            squared_norm += float(np.sum(np.square(parameter)))
        return max(squared_norm, self._fedent_epsilon)

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is None or not results:
            return aggregated_parameters, metrics

        self._phi1 = parameters_to_ndarrays(aggregated_parameters)
        total_examples = sum(fit_res.num_examples for _, fit_res in results)
        if total_examples > 0:
            weighted_phi2 = 0.0
            for _, fit_res in results:
                weight = fit_res.num_examples / total_examples
                weighted_phi2 += weight * float(fit_res.metrics.get("fedent_weight_sq_norm", 0.0))
            self._phi2 = max(weighted_phi2, self._fedent_epsilon)
        else:
            self._phi2 = self._compute_phi2(self._phi1)
        return aggregated_parameters, metrics


class FedEntBuilder:
    name = "fedent"

    def build_strategy(
        self,
        config: ExperimentConfig,
        initial_model: torch.nn.Module,
        evaluate_fn: ServerEvaluateFn | None,
    ) -> FedEntStrategy:
        min_fit_clients = max(1, math.ceil(config.num_supernodes * config.fraction_train))
        min_eval_clients = max(1, math.ceil(config.num_supernodes * config.fraction_evaluate))
        initial_parameters = ndarrays_to_parameters(get_model_parameters(initial_model))

        strategy = FedEntStrategy(
            fraction_fit=config.fraction_train,
            fraction_evaluate=config.fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_eval_clients,
            min_available_clients=config.num_supernodes,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=lambda _: {},
            fit_metrics_aggregation_fn=weighted_average,
            evaluate_metrics_aggregation_fn=weighted_average,
            initial_parameters=initial_parameters,
            checkpoint_model=initial_model,
            output_dir=config.output_dir,
            fedent_beta=config.fedent_beta,
            fedent_gamma=config.fedent_gamma,
            fedent_epsilon=config.fedent_epsilon,
            fedent_fixed_point_steps=config.fedent_fixed_point_steps,
            fedent_max_learning_rate=config.fedent_max_learning_rate,
        )
        strategy.on_fit_config_fn = strategy.make_fit_config(config)
        return strategy
