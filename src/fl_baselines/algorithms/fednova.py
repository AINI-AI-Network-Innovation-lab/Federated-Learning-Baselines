"""FedNova algorithm builder."""

from __future__ import annotations

import math

import numpy as np
import torch
from flwr.common import (
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate

from fl_baselines.clients.torch_client import get_model_parameters
from fl_baselines.core.config import ExperimentConfig
from fl_baselines.core.types import ServerEvaluateFn
from fl_baselines.logging.checkpointing import save_round_checkpoints
from fl_baselines.logging.metrics import weighted_average


class FedNovaStrategy(FedAvg):
    """FedNova strategy using normalized averaging of client updates."""

    def __init__(
        self,
        *,
        server_momentum: float,
        checkpoint_model: torch.nn.Module,
        output_dir: str,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        if self.initial_parameters is None:
            raise ValueError("FedNovaStrategy requires initial_parameters")
        self.global_parameters = parameters_to_ndarrays(self.initial_parameters)
        self.server_momentum = server_momentum
        self.global_momentum_buffer: list[np.ndarray] = []
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

        tau_effective = 0.0
        client_stats: list[tuple[list[np.ndarray], int, float, float]] = []
        for _, fit_res in results:
            local_norm = float(fit_res.metrics.get("local_norm", 0.0))
            if local_norm <= 0:
                raise ValueError("FedNova client result requires positive local_norm")
            weight = fit_res.num_examples / total_examples
            tau_effective += local_norm * weight
            client_stats.append(
                (
                    parameters_to_ndarrays(fit_res.parameters),
                    fit_res.num_examples,
                    local_norm,
                    weight,
                )
            )

        normalized_updates = []
        for update_arrays, _, local_norm, weight in client_stats:
            scale = tau_effective / local_norm
            scale *= weight
            normalized_updates.append((update_arrays, scale))

        aggregated_update = aggregate(normalized_updates)
        self._update_server_parameters(aggregated_update)
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

    def _update_server_parameters(self, aggregated_update: list[np.ndarray]) -> None:
        if self.server_momentum == 0:
            for index, update in enumerate(aggregated_update):
                self.global_parameters[index] -= update
            return

        if not self.global_momentum_buffer:
            self.global_momentum_buffer = [update.copy() for update in aggregated_update]
        else:
            for index, update in enumerate(aggregated_update):
                self.global_momentum_buffer[index] *= self.server_momentum
                self.global_momentum_buffer[index] += update

        for index, update in enumerate(self.global_momentum_buffer):
            self.global_parameters[index] -= update


class FedNovaBuilder:
    name = "fednova"

    def build_strategy(
        self,
        config: ExperimentConfig,
        initial_model: torch.nn.Module,
        evaluate_fn: ServerEvaluateFn | None,
    ) -> FedNovaStrategy:
        min_fit_clients = max(1, math.ceil(config.num_supernodes * config.fraction_train))
        min_eval_clients = max(1, math.ceil(config.num_supernodes * config.fraction_evaluate))
        initial_parameters = ndarrays_to_parameters(get_model_parameters(initial_model))

        return FedNovaStrategy(
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
            server_momentum=config.fednova_server_momentum,
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
