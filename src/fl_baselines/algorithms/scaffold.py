"""SCAFFOLD algorithm builder."""

from __future__ import annotations

import math

import numpy as np
import torch
from flwr.common import (
    FitIns,
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


class ScaffoldStrategy(FedAvg):
    """FedAvg-compatible Flower strategy with SCAFFOLD control variates."""

    def __init__(
        self,
        *,
        num_total_clients: int,
        server_control: list[np.ndarray],
        checkpoint_model: torch.nn.Module,
        output_dir: str,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.num_total_clients = num_total_clients
        self.server_control = [control.copy() for control in server_control]
        self._checkpoint_model = checkpoint_model
        self._output_dir = output_dir

    def configure_fit(self, server_round, parameters, client_manager):
        client_fit_ins = super().configure_fit(server_round, parameters, client_manager)
        model_parameters = parameters_to_ndarrays(parameters)
        combined_parameters = ndarrays_to_parameters(
            model_parameters + [control.copy() for control in self.server_control]
        )

        return [
            (client, FitIns(combined_parameters, fit_ins.config))
            for client, fit_ins in client_fit_ins
        ]

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        num_controls = len(self.server_control)
        model_results = []
        control_deltas = []
        for _, fit_res in results:
            arrays = parameters_to_ndarrays(fit_res.parameters)
            if len(arrays) <= num_controls:
                raise ValueError("SCAFFOLD client result is missing model parameters")
            model_arrays = arrays[:-num_controls]
            delta_arrays = arrays[-num_controls:]
            model_results.append((model_arrays, fit_res.num_examples))
            control_deltas.append(delta_arrays)

        aggregated_model = aggregate(model_results)
        self._update_server_control(control_deltas)
        parameters_aggregated = ndarrays_to_parameters(aggregated_model)

        metrics_aggregated: dict[str, Scalar] = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(fit_res.num_examples, fit_res.metrics) for _, fit_res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        save_round_checkpoints(
            self._checkpoint_model,
            parameters_aggregated,
            self._output_dir,
            server_round,
        )
        return parameters_aggregated, metrics_aggregated

    def _update_server_control(self, control_deltas: list[list[np.ndarray]]) -> None:
        selected_clients = len(control_deltas)
        if selected_clients == 0:
            return

        update_scale = selected_clients / max(1, self.num_total_clients)
        for control_index, server_control in enumerate(self.server_control):
            mean_delta = sum(
                client_delta[control_index] for client_delta in control_deltas
            ) / selected_clients
            server_control += update_scale * mean_delta


class ScaffoldBuilder:
    name = "scaffold"

    def build_strategy(
        self,
        config: ExperimentConfig,
        initial_model: torch.nn.Module,
        evaluate_fn: ServerEvaluateFn | None,
    ) -> ScaffoldStrategy:
        min_fit_clients = max(1, math.ceil(config.num_supernodes * config.fraction_train))
        min_eval_clients = max(1, math.ceil(config.num_supernodes * config.fraction_evaluate))
        initial_parameters = ndarrays_to_parameters(get_model_parameters(initial_model))
        server_control = [
            np.zeros_like(parameter.detach().cpu().numpy())
            for parameter in initial_model.parameters()
        ]

        return ScaffoldStrategy(
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
            num_total_clients=config.num_supernodes,
            server_control=server_control,
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
