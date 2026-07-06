"""FedLAA algorithm builder."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from flwr.common import Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg

from fl_baselines.clients.torch_client import get_model_parameters
from fl_baselines.core.config import ExperimentConfig
from fl_baselines.core.types import ServerEvaluateFn
from fl_baselines.logging.checkpointing import save_round_checkpoints
from fl_baselines.logging.metrics import weighted_average


class FedLAAStrategy(FedAvg):
    """FedLAA strategy using layer-wise gradient-aligned aggregation."""

    def __init__(
        self,
        *,
        beta: float,
        checkpoint_model: torch.nn.Module,
        output_dir: str,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        if self.initial_parameters is None:
            raise ValueError("FedLAAStrategy requires initial_parameters")
        if beta <= 0:
            raise ValueError("FedLAA beta must be positive")
        self.beta = beta
        self.global_parameters = parameters_to_ndarrays(self.initial_parameters)
        self.smoothed_angles: dict[str, list[float]] = {}
        self.angle_counts: dict[str, list[int]] = {}
        self.last_layer_weights: list[list[float]] = []
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
        local_updates = [
            [
                local_parameter - global_parameter
                for local_parameter, global_parameter in zip(
                    client_model, self.global_parameters
                )
            ]
            for client_model in client_models
        ]

        fit_config = self.on_fit_config_fn(server_round) if self.on_fit_config_fn else {}
        learning_rate = float(fit_config.get("learning_rate", 1.0))
        local_epochs = float(fit_config.get("local_epochs", 1.0))
        scale = learning_rate * local_epochs
        if scale <= 0:
            scale = 1.0
        local_gradients = [
            [-(layer_update / scale) for layer_update in client_update]
            for client_update in local_updates
        ]

        global_gradients = self._global_gradients(local_gradients, sample_weights)
        layer_weights = self._layer_weights(
            results,
            local_gradients,
            global_gradients,
            sample_weights,
        )
        self.last_layer_weights = layer_weights

        self.global_parameters = self._aggregate_layers(local_updates, layer_weights)
        aggregated_parameters = ndarrays_to_parameters(self.global_parameters)

        metrics_aggregated: dict[str, Scalar] = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(fit_res.num_examples, fit_res.metrics) for _, fit_res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        metrics_aggregated["fedlaa_layer_count"] = len(self.global_parameters)

        save_round_checkpoints(
            self._checkpoint_model,
            aggregated_parameters,
            self._output_dir,
            server_round,
        )
        return aggregated_parameters, metrics_aggregated

    def _global_gradients(
        self,
        local_gradients: list[list[np.ndarray]],
        sample_weights: list[float],
    ) -> list[np.ndarray]:
        return [
            sum(
                sample_weight * client_gradients[layer_index]
                for sample_weight, client_gradients in zip(sample_weights, local_gradients)
            )
            for layer_index in range(len(local_gradients[0]))
        ]

    def _layer_weights(
        self,
        results: list[tuple[Any, Any]],
        local_gradients: list[list[np.ndarray]],
        global_gradients: list[np.ndarray],
        sample_weights: list[float],
    ) -> list[list[float]]:
        num_layers = len(global_gradients)
        all_layer_weights: list[list[float]] = []

        for layer_index in range(num_layers):
            scores: list[float] = []
            for client_index, ((client_proxy, _), client_gradients) in enumerate(
                zip(results, local_gradients)
            ):
                client_key = _client_key(client_proxy, client_index)
                angle = _angle_between(
                    client_gradients[layer_index],
                    global_gradients[layer_index],
                )
                smoothed_angle = self._update_smoothed_angle(
                    client_key,
                    layer_index,
                    angle,
                )
                scores.append(self._mapped_contribution(smoothed_angle))

            all_layer_weights.append(_softmax_with_sample_weights(scores, sample_weights))
        return all_layer_weights

    def _update_smoothed_angle(
        self,
        client_key: str,
        layer_index: int,
        angle: float,
    ) -> float:
        angles = self.smoothed_angles.setdefault(client_key, [])
        counts = self.angle_counts.setdefault(client_key, [])
        while len(angles) <= layer_index:
            angles.append(angle)
            counts.append(0)
        count = counts[layer_index] + 1
        previous = angles[layer_index]
        smoothed = ((count - 1) / count) * previous + (angle / count)
        angles[layer_index] = smoothed
        counts[layer_index] = count
        return smoothed

    def _mapped_contribution(self, smoothed_angle: float) -> float:
        return self.beta * (
            1.0 - math.exp(-math.exp(-self.beta * (smoothed_angle - 1.0)))
        )

    def _aggregate_layers(
        self,
        local_updates: list[list[np.ndarray]],
        layer_weights: list[list[float]],
    ) -> list[np.ndarray]:
        aggregated_parameters: list[np.ndarray] = []
        for layer_index, global_parameter in enumerate(self.global_parameters):
            aggregated_update = sum(
                layer_weights[layer_index][client_index]
                * local_updates[client_index][layer_index]
                for client_index in range(len(local_updates))
            )
            aggregated_parameters.append(global_parameter + aggregated_update)
        return aggregated_parameters


def _client_key(client_proxy: object, index: int) -> str:
    cid = getattr(client_proxy, "cid", None)
    if cid is not None:
        return str(cid)
    return str(index)


def _angle_between(left: np.ndarray, right: np.ndarray) -> float:
    left_vector = left.astype(np.float64, copy=False).ravel()
    right_vector = right.astype(np.float64, copy=False).ravel()
    left_norm = float(np.linalg.norm(left_vector))
    right_norm = float(np.linalg.norm(right_vector))
    if left_norm == 0.0 or right_norm == 0.0:
        return math.pi / 2.0
    cosine = float(np.dot(left_vector, right_vector) / (left_norm * right_norm))
    return math.acos(float(np.clip(cosine, -1.0, 1.0)))


def _softmax_with_sample_weights(
    scores: list[float],
    sample_weights: list[float],
) -> list[float]:
    shifted_scores = np.asarray(scores, dtype=np.float64)
    shifted_scores -= float(np.max(shifted_scores))
    weighted_scores = np.exp(shifted_scores) * np.asarray(
        sample_weights,
        dtype=np.float64,
    )
    total = float(np.sum(weighted_scores))
    if total == 0.0 or not math.isfinite(total):
        return sample_weights
    return [float(weight / total) for weight in weighted_scores]


class FedLAABuilder:
    name = "fedlaa"

    def build_strategy(
        self,
        config: ExperimentConfig,
        initial_model: torch.nn.Module,
        evaluate_fn: ServerEvaluateFn | None,
    ) -> FedLAAStrategy:
        min_fit_clients = max(1, math.ceil(config.num_supernodes * config.fraction_train))
        min_eval_clients = max(1, math.ceil(config.num_supernodes * config.fraction_evaluate))
        initial_parameters = ndarrays_to_parameters(get_model_parameters(initial_model))

        return FedLAAStrategy(
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
            beta=config.fedlaa_beta,
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
