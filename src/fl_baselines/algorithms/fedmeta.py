"""FedMeta algorithm builder."""

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


class FedMetaStrategy(FedAvg):
    """FedMeta strategy updating MAML or Meta-SGD algorithm parameters."""

    def __init__(
        self,
        *,
        meta_sgd: bool,
        outer_learning_rate: float,
        checkpoint_model: torch.nn.Module,
        output_dir: str,
        alpha_parameters: list[np.ndarray] | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        if self.initial_parameters is None:
            raise ValueError("FedMetaStrategy requires initial_parameters")
        if outer_learning_rate <= 0:
            raise ValueError("FedMeta outer learning rate must be positive")
        self.meta_sgd = meta_sgd
        self.outer_learning_rate = outer_learning_rate
        self.algorithm_parameters = parameters_to_ndarrays(self.initial_parameters)
        self.alpha_parameters = [
            alpha.copy() for alpha in alpha_parameters
        ] if alpha_parameters is not None else []
        self._checkpoint_model = checkpoint_model
        self._output_dir = output_dir

    def configure_fit(self, server_round, parameters, client_manager):
        client_fit_ins = super().configure_fit(server_round, parameters, client_manager)
        payload = self.algorithm_parameters
        if self.meta_sgd:
            payload = payload + self.alpha_parameters
        payload_parameters = ndarrays_to_parameters([array.copy() for array in payload])
        return [
            (client, FitIns(payload_parameters, fit_ins.config))
            for client, fit_ins in client_fit_ins
        ]

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        parameter_count = len(self.algorithm_parameters)
        expected_count = parameter_count * (2 if self.meta_sgd else 1)
        gradient_results: list[tuple[list[np.ndarray], int]] = []
        for _, fit_res in results:
            gradients = parameters_to_ndarrays(fit_res.parameters)
            if len(gradients) != expected_count:
                raise ValueError("FedMeta client result has unexpected gradient payload")
            gradient_results.append((gradients, fit_res.num_examples))

        averaged_gradients = _weighted_average_arrays(gradient_results)
        model_gradients = averaged_gradients[:parameter_count]
        self.algorithm_parameters = [
            parameter - self.outer_learning_rate * gradient
            for parameter, gradient in zip(self.algorithm_parameters, model_gradients)
        ]
        if self.meta_sgd:
            alpha_gradients = averaged_gradients[parameter_count:]
            self.alpha_parameters = [
                alpha - self.outer_learning_rate * gradient
                for alpha, gradient in zip(self.alpha_parameters, alpha_gradients)
            ]

        parameters_aggregated = ndarrays_to_parameters(self.algorithm_parameters)
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


def _weighted_average_arrays(
    results: list[tuple[list[np.ndarray], int]],
) -> list[np.ndarray]:
    total_examples = sum(num_examples for _, num_examples in results)
    if total_examples <= 0:
        raise ValueError("FedMeta aggregation requires positive example counts")
    return [
        sum(
            arrays[index] * (num_examples / total_examples)
            for arrays, num_examples in results
        )
        for index in range(len(results[0][0]))
    ]


class FedMetaBuilder:
    name = "fedmeta"

    def build_strategy(
        self,
        config: ExperimentConfig,
        initial_model: torch.nn.Module,
        evaluate_fn: ServerEvaluateFn | None,
    ) -> FedMetaStrategy:
        min_fit_clients = max(1, math.ceil(config.num_supernodes * config.fraction_train))
        min_eval_clients = max(1, math.ceil(config.num_supernodes * config.fraction_evaluate))
        initial_arrays = get_model_parameters(initial_model)
        initial_parameters = ndarrays_to_parameters(initial_arrays)
        meta_sgd = config.fedmeta_method == "meta-sgd"
        alpha_parameters = None
        if meta_sgd:
            alpha_parameters = [
                np.full_like(array, config.fedmeta_alpha_init, dtype=np.float32)
                for array in initial_arrays
            ]

        return FedMetaStrategy(
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
            meta_sgd=meta_sgd,
            outer_learning_rate=config.fedmeta_outer_learning_rate,
            alpha_parameters=alpha_parameters,
            checkpoint_model=initial_model,
            output_dir=config.output_dir,
        )

    def _fit_config(self, config: ExperimentConfig):
        def fn(server_round: int) -> dict[str, bool | bytes | float | int | str]:
            return {
                "algorithm": self.name,
                "server_round": server_round,
                "fedmeta_method": config.fedmeta_method,
                "fedmeta_inner_learning_rate": config.fedmeta_inner_learning_rate,
                "fedmeta_support_fraction": config.fedmeta_support_fraction,
                "fedmeta_inner_steps": config.fedmeta_inner_steps,
                "fedmeta_first_order": config.fedmeta_first_order,
            }

        return fn
