"""FedCurv algorithm builder."""

from __future__ import annotations

import math

import numpy as np
import torch
from flwr.common import FitIns, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg

from fl_baselines.clients.torch_client import get_model_parameters
from fl_baselines.core.config import ExperimentConfig
from fl_baselines.core.types import ServerEvaluateFn
from fl_baselines.logging.checkpointing import save_round_checkpoints
from fl_baselines.logging.metrics import weighted_average


class FedCurvStrategy(FedAvg):
    """FedCurv strategy broadcasting global curvature aggregates."""

    def __init__(
        self,
        *,
        curvature_aggregate: list[np.ndarray],
        weighted_parameter_aggregate: list[np.ndarray],
        model_parameter_count: int,
        checkpoint_model: torch.nn.Module,
        output_dir: str,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.curvature_aggregate = [array.copy() for array in curvature_aggregate]
        self.weighted_parameter_aggregate = [
            array.copy() for array in weighted_parameter_aggregate
        ]
        self.model_parameter_count = model_parameter_count
        self._checkpoint_model = checkpoint_model
        self._output_dir = output_dir

    def configure_fit(self, server_round, parameters, client_manager):
        client_fit_ins = super().configure_fit(server_round, parameters, client_manager)
        model_parameters = parameters_to_ndarrays(parameters)
        payload = ndarrays_to_parameters(
            model_parameters
            + [array.copy() for array in self.curvature_aggregate]
            + [array.copy() for array in self.weighted_parameter_aggregate]
        )
        return [
            (client, FitIns(payload, fit_ins.config))
            for client, fit_ins in client_fit_ins
        ]

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        expected_payload_count = self.model_parameter_count * 3
        model_updates: list[list[np.ndarray]] = []
        curvature_updates: list[list[np.ndarray]] = []
        weighted_updates: list[list[np.ndarray]] = []
        for _, fit_res in results:
            arrays = parameters_to_ndarrays(fit_res.parameters)
            if len(arrays) != expected_payload_count:
                raise ValueError("FedCurv client result is missing curvature payload")
            model_updates.append(arrays[: self.model_parameter_count])
            curvature_updates.append(
                arrays[self.model_parameter_count : 2 * self.model_parameter_count]
            )
            weighted_updates.append(arrays[2 * self.model_parameter_count :])

        aggregated_model = _average_arrays(model_updates)
        self.curvature_aggregate = _sum_arrays(curvature_updates)
        self.weighted_parameter_aggregate = _sum_arrays(weighted_updates)
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


def _average_arrays(results: list[list[np.ndarray]]) -> list[np.ndarray]:
    client_count = len(results)
    return [
        sum(client_arrays[index] for client_arrays in results) / float(client_count)
        for index in range(len(results[0]))
    ]


def _sum_arrays(results: list[list[np.ndarray]]) -> list[np.ndarray]:
    return [
        sum(client_arrays[index] for client_arrays in results)
        for index in range(len(results[0]))
    ]


class FedCurvBuilder:
    name = "fedcurv"

    def build_strategy(
        self,
        config: ExperimentConfig,
        initial_model: torch.nn.Module,
        evaluate_fn: ServerEvaluateFn | None,
    ) -> FedCurvStrategy:
        min_fit_clients = max(1, math.ceil(config.num_supernodes * config.fraction_train))
        min_eval_clients = max(1, math.ceil(config.num_supernodes * config.fraction_evaluate))
        model_parameters = get_model_parameters(initial_model)
        initial_parameters = ndarrays_to_parameters(model_parameters)
        zero_curvature = [np.zeros_like(array, dtype=np.float32) for array in model_parameters]

        return FedCurvStrategy(
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
            curvature_aggregate=zero_curvature,
            weighted_parameter_aggregate=zero_curvature,
            model_parameter_count=len(model_parameters),
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
                "fedcurv_lambda": config.fedcurv_lambda,
                "fedcurv_fisher_batches": config.fedcurv_fisher_batches,
                "fedcurv_stability_eps": config.fedcurv_stability_eps,
            }

        return fn
