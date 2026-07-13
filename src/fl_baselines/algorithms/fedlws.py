"""FedLWS algorithm builder."""

from __future__ import annotations

import math
from collections import OrderedDict

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


class FedLWSStrategy(FedAvg):
    """FedLWS strategy with adaptive layer-wise weight shrinking."""

    def __init__(
        self,
        *,
        beta: float,
        epsilon: float,
        checkpoint_model: torch.nn.Module,
        output_dir: str,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        if self.initial_parameters is None:
            raise ValueError("FedLWSStrategy requires initial_parameters")
        if beta <= 0:
            raise ValueError("FedLWS beta must be positive")
        if epsilon <= 0:
            raise ValueError("FedLWS epsilon must be positive")
        self.global_parameters = parameters_to_ndarrays(self.initial_parameters)
        self.beta = beta
        self.epsilon = epsilon
        self.last_layer_gammas: list[float] = []
        self.last_layer_taus: list[float] = []
        self._checkpoint_model = checkpoint_model
        self._output_dir = output_dir
        self._reference_state = OrderedDict(
            (name, tensor.detach().cpu().clone())
            for name, tensor in checkpoint_model.state_dict().items()
        )
        self._parameter_names = set(dict(checkpoint_model.named_parameters()).keys())

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
        aggregated = aggregate(list(zip(client_models, sample_weights)))
        shrunk = self._apply_layerwise_shrinking(client_models, aggregated)
        self.global_parameters = shrunk
        aggregated_parameters = ndarrays_to_parameters(shrunk)

        metrics_aggregated: dict[str, Scalar] = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(fit_res.num_examples, fit_res.metrics) for _, fit_res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        if self.last_layer_gammas:
            metrics_aggregated["fedlws_gamma_mean"] = float(np.mean(self.last_layer_gammas))
            metrics_aggregated["fedlws_gamma_min"] = float(np.min(self.last_layer_gammas))
            metrics_aggregated["fedlws_tau_mean"] = float(np.mean(self.last_layer_taus))

        save_round_checkpoints(
            self._checkpoint_model,
            aggregated_parameters,
            self._output_dir,
            server_round,
        )
        return aggregated_parameters, metrics_aggregated

    def _apply_layerwise_shrinking(
        self,
        client_models: list[list[np.ndarray]],
        aggregated: list[np.ndarray],
    ) -> list[np.ndarray]:
        shrunk: list[np.ndarray] = []
        gammas: list[float] = []
        taus: list[float] = []

        for index, (name, reference_tensor) in enumerate(self._reference_state.items()):
            reference_array = reference_tensor.numpy()
            aggregated_array = aggregated[index]
            if (
                name not in self._parameter_names
                or not np.issubdtype(reference_array.dtype, np.floating)
            ):
                if not np.issubdtype(reference_array.dtype, np.floating):
                    aggregated_array = np.rint(aggregated_array)
                shrunk.append(aggregated_array.astype(reference_array.dtype, copy=False))
                continue

            global_array = self.global_parameters[index].astype(np.float64, copy=False)
            aggregated_float = aggregated_array.astype(np.float64, copy=False)
            local_updates = np.stack(
                [
                    client_model[index].astype(np.float64, copy=False) - global_array
                    for client_model in client_models
                ],
                axis=0,
            )
            mean_update = np.mean(local_updates, axis=0)
            tau = float(
                np.mean(
                    [
                        np.linalg.norm(update - mean_update)
                        for update in local_updates
                    ]
                )
            )
            global_update_norm = float(np.linalg.norm(aggregated_float - global_array))
            global_norm = float(np.linalg.norm(global_array))
            denominator = global_norm + self.beta * tau * global_update_norm
            gamma = 1.0
            if denominator > self.epsilon:
                gamma = global_norm / denominator

            shrunk_array = gamma * aggregated_float
            shrunk.append(shrunk_array.astype(reference_array.dtype, copy=False))
            gammas.append(float(gamma))
            taus.append(tau)

        self.last_layer_gammas = gammas
        self.last_layer_taus = taus
        return shrunk


class FedLWSBuilder:
    name = "fedlws"

    def build_strategy(
        self,
        config: ExperimentConfig,
        initial_model: torch.nn.Module,
        evaluate_fn: ServerEvaluateFn | None,
    ) -> FedLWSStrategy:
        min_fit_clients = max(1, math.ceil(config.num_supernodes * config.fraction_train))
        min_eval_clients = max(1, math.ceil(config.num_supernodes * config.fraction_evaluate))
        initial_parameters = ndarrays_to_parameters(get_model_parameters(initial_model))

        return FedLWSStrategy(
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
            beta=config.fedlws_beta,
            epsilon=config.fedlws_epsilon,
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
                "fedlws_beta": config.fedlws_beta,
            }

        return fn
