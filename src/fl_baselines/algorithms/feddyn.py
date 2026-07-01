"""FedDyn algorithm builder."""

from __future__ import annotations

import math

import numpy as np
import torch
from flwr.common import Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate

from fl_baselines.clients.torch_client import get_model_parameters
from fl_baselines.core.config import ExperimentConfig
from fl_baselines.core.types import ServerEvaluateFn
from fl_baselines.logging.checkpointing import save_round_checkpoints
from fl_baselines.logging.metrics import weighted_average


class FedDynStrategy(FedAvg):
    """FedDyn strategy with dynamic server-side correction state."""

    def __init__(
        self,
        *,
        alpha: float,
        num_total_clients: int,
        checkpoint_model: torch.nn.Module,
        output_dir: str,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        if self.initial_parameters is None:
            raise ValueError("FedDynStrategy requires initial_parameters")
        if alpha <= 0:
            raise ValueError("FedDyn alpha must be positive")
        self.alpha = alpha
        self.num_total_clients = num_total_clients
        self.global_parameters = parameters_to_ndarrays(self.initial_parameters)
        self.h = [np.zeros_like(array) for array in self.global_parameters]
        self._checkpoint_model = checkpoint_model
        self._output_dir = output_dir

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        total_examples = sum(fit_res.num_examples for _, fit_res in results)
        if total_examples <= 0:
            return None, {}

        previous_global = [array.copy() for array in self.global_parameters]
        client_models = [
            parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results
        ]
        weights = [
            fit_res.num_examples / total_examples for _, fit_res in results
        ]
        averaged_model = aggregate(list(zip(client_models, weights)))

        summed_updates = [np.zeros_like(array) for array in previous_global]
        for client_model in client_models:
            for index, (client_array, global_array) in enumerate(
                zip(client_model, previous_global)
            ):
                summed_updates[index] += client_array - global_array

        for index, update in enumerate(summed_updates):
            self.h[index] -= self.alpha * (update / self.num_total_clients)

        self.global_parameters = [
            averaged - (h_array / self.alpha)
            for averaged, h_array in zip(averaged_model, self.h)
        ]
        aggregated_parameters = ndarrays_to_parameters(self.global_parameters)

        metrics_aggregated: dict[str, Scalar] = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(fit_res.num_examples, fit_res.metrics) for _, fit_res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        save_round_checkpoints(
            self._checkpoint_model,
            aggregated_parameters,
            self._output_dir,
            server_round,
        )
        return aggregated_parameters, metrics_aggregated


class FedDynBuilder:
    name = "feddyn"

    def build_strategy(
        self,
        config: ExperimentConfig,
        initial_model: torch.nn.Module,
        evaluate_fn: ServerEvaluateFn | None,
    ) -> FedDynStrategy:
        min_fit_clients = max(1, math.ceil(config.num_supernodes * config.fraction_train))
        min_eval_clients = max(1, math.ceil(config.num_supernodes * config.fraction_evaluate))
        initial_parameters = ndarrays_to_parameters(get_model_parameters(initial_model))

        return FedDynStrategy(
            fraction_fit=config.fraction_train,
            fraction_evaluate=config.fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_eval_clients,
            min_available_clients=config.num_supernodes,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=self._fit_config(config),
            fit_metrics_aggregation_fn=weighted_average,
            evaluate_metrics_aggregation_fn=weighted_average,
            initial_parameters=initial_parameters,
            alpha=config.feddyn_alpha,
            num_total_clients=config.num_supernodes,
            checkpoint_model=initial_model,
            output_dir=config.output_dir,
        )

    def _fit_config(self, config: ExperimentConfig):
        def fn(server_round: int) -> dict[str, bool | bytes | float | int | str]:
            return {
                "algorithm": self.name,
                "server_round": server_round,
                "local_epochs": config.local_epochs,
                "learning_rate": config.learning_rate,
                "feddyn_alpha": config.feddyn_alpha,
            }

        return fn
