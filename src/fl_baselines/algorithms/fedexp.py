"""FedExP algorithm builder."""

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


class FedExPStrategy(FedAvg):
    """FedExP strategy using adaptive server extrapolation."""

    def __init__(
        self,
        *,
        epsilon: float,
        checkpoint_model: torch.nn.Module,
        output_dir: str,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        if self.initial_parameters is None:
            raise ValueError("FedExPStrategy requires initial_parameters")
        if epsilon < 0:
            raise ValueError("FedExP epsilon must be non-negative")
        self.global_parameters = parameters_to_ndarrays(self.initial_parameters)
        self.epsilon = epsilon
        self.last_server_step_size = 1.0
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

        client_models = [
            parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results
        ]
        sample_weights = [
            fit_res.num_examples / total_examples for _, fit_res in results
        ]
        deltas = [
            [
                global_parameter - local_parameter
                for global_parameter, local_parameter in zip(
                    self.global_parameters,
                    client_model,
                )
            ]
            for client_model in client_models
        ]
        delta_bar = aggregate(list(zip(deltas, sample_weights)))

        mean_squared_norm = sum(
            weight * _squared_norm(delta)
            for delta, weight in zip(deltas, sample_weights)
        )
        denominator = 2.0 * (_squared_norm(delta_bar) + self.epsilon)
        server_step_size = 1.0
        if denominator > 0:
            server_step_size = max(1.0, mean_squared_norm / denominator)
        self.last_server_step_size = float(server_step_size)

        self.global_parameters = [
            global_parameter - self.last_server_step_size * delta_component
            for global_parameter, delta_component in zip(
                self.global_parameters,
                delta_bar,
            )
        ]
        aggregated_parameters = ndarrays_to_parameters(self.global_parameters)

        metrics_aggregated: dict[str, Scalar] = {
            "fedexp_server_step_size": self.last_server_step_size
        }
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(fit_res.num_examples, fit_res.metrics) for _, fit_res in results]
            metrics_aggregated.update(self.fit_metrics_aggregation_fn(fit_metrics))

        save_round_checkpoints(
            self._checkpoint_model,
            aggregated_parameters,
            self._output_dir,
            server_round,
        )
        return aggregated_parameters, metrics_aggregated


def _squared_norm(arrays: list[np.ndarray]) -> float:
    return float(
        sum(
            np.sum(array.astype(np.float64, copy=False) ** 2)
            for array in arrays
        )
    )


class FedExPBuilder:
    name = "fedexp"

    def build_strategy(
        self,
        config: ExperimentConfig,
        initial_model: torch.nn.Module,
        evaluate_fn: ServerEvaluateFn | None,
    ) -> FedExPStrategy:
        min_fit_clients = max(1, math.ceil(config.num_supernodes * config.fraction_train))
        min_eval_clients = max(1, math.ceil(config.num_supernodes * config.fraction_evaluate))
        initial_parameters = ndarrays_to_parameters(get_model_parameters(initial_model))

        return FedExPStrategy(
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
            epsilon=config.fedexp_epsilon,
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
            }

        return fn
