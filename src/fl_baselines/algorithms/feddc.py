"""FedDC algorithm builder."""

from __future__ import annotations

import math

import numpy as np
import torch
from flwr.common import FitIns, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate

from fl_baselines.clients.torch_client import get_model_parameters
from fl_baselines.core.config import ExperimentConfig
from fl_baselines.core.types import ServerEvaluateFn
from fl_baselines.logging.checkpointing import save_round_checkpoints
from fl_baselines.logging.metrics import weighted_average


class FedDCStrategy(FedAvg):
    """FedAvg-compatible Flower strategy with FedDC average update state."""

    def __init__(
        self,
        *,
        alpha: float,
        num_total_clients: int,
        average_update_state: list[np.ndarray],
        checkpoint_model: torch.nn.Module,
        output_dir: str,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        if self.initial_parameters is None:
            raise ValueError("FedDCStrategy requires initial_parameters")
        if alpha <= 0:
            raise ValueError("FedDC alpha must be positive")
        self.alpha = alpha
        self.num_total_clients = num_total_clients
        self.average_update_state = [state.copy() for state in average_update_state]
        self.global_parameters = parameters_to_ndarrays(self.initial_parameters)
        self._checkpoint_model = checkpoint_model
        self._output_dir = output_dir

    def configure_fit(self, server_round, parameters, client_manager):
        client_fit_ins = super().configure_fit(server_round, parameters, client_manager)
        model_parameters = parameters_to_ndarrays(parameters)
        combined_parameters = ndarrays_to_parameters(
            model_parameters + [state.copy() for state in self.average_update_state]
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

        num_state_arrays = len(self.average_update_state)
        model_results = []
        local_updates = []
        total_examples = sum(fit_res.num_examples for _, fit_res in results)
        if total_examples <= 0:
            return None, {}

        for _, fit_res in results:
            arrays = parameters_to_ndarrays(fit_res.parameters)
            if len(arrays) <= num_state_arrays:
                raise ValueError("FedDC client result is missing corrected model parameters")
            corrected_model = arrays[:-num_state_arrays]
            local_update = arrays[-num_state_arrays:]
            model_results.append((corrected_model, fit_res.num_examples))
            local_updates.append((local_update, fit_res.num_examples))

        aggregated_model = aggregate(model_results)
        self._update_average_update_state(local_updates, total_examples)
        self.global_parameters = [array.copy() for array in aggregated_model]
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

    def _update_average_update_state(
        self,
        local_updates: list[tuple[list[np.ndarray], int]],
        total_examples: int,
    ) -> None:
        self.average_update_state = aggregate(
            [
                (update_arrays, num_examples / total_examples)
                for update_arrays, num_examples in local_updates
            ]
        )


class FedDCBuilder:
    name = "feddc"

    def build_strategy(
        self,
        config: ExperimentConfig,
        initial_model: torch.nn.Module,
        evaluate_fn: ServerEvaluateFn | None,
    ) -> FedDCStrategy:
        min_fit_clients = max(1, math.ceil(config.num_supernodes * config.fraction_train))
        min_eval_clients = max(1, math.ceil(config.num_supernodes * config.fraction_evaluate))
        initial_parameters = ndarrays_to_parameters(get_model_parameters(initial_model))
        average_update_state = [
            np.zeros_like(parameter.detach().cpu().numpy())
            for parameter in initial_model.parameters()
        ]

        return FedDCStrategy(
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
            alpha=config.feddc_alpha,
            num_total_clients=config.num_supernodes,
            average_update_state=average_update_state,
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
                "feddc_alpha": config.feddc_alpha,
            }

        return fn
