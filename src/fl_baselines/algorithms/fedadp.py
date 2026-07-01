"""FedAdp algorithm builder."""

from __future__ import annotations

import math
from typing import Any

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


class FedAdpStrategy(FedAvg):
    """FedAdp strategy using contribution-aware adaptive aggregation weights."""

    def __init__(
        self,
        *,
        alpha: float,
        checkpoint_model: torch.nn.Module,
        output_dir: str,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        if self.initial_parameters is None:
            raise ValueError("FedAdpStrategy requires initial_parameters")
        if alpha <= 0:
            raise ValueError("FedAdp alpha must be positive")
        self.global_parameters = parameters_to_ndarrays(self.initial_parameters)
        self.alpha = alpha
        self.smoothed_angles: dict[str, float] = {}
        self.angle_counts: dict[str, int] = {}
        self.last_aggregation_weights: list[float] = []
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
        global_update = aggregate(list(zip(local_updates, sample_weights)))

        adaptive_weights = self._adaptive_weights(
            results,
            local_updates,
            global_update,
            sample_weights,
        )
        self.last_aggregation_weights = adaptive_weights
        self.global_parameters = aggregate(list(zip(client_models, adaptive_weights)))
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

    def _adaptive_weights(
        self,
        results: list[tuple[Any, Any]],
        local_updates: list[list[np.ndarray]],
        global_update: list[np.ndarray],
        sample_weights: list[float],
    ) -> list[float]:
        global_vector = _flatten(global_update)
        global_norm = np.linalg.norm(global_vector)
        if global_norm == 0:
            return sample_weights

        scores = []
        for index, ((client_proxy, _), local_update) in enumerate(
            zip(results, local_updates)
        ):
            local_vector = _flatten(local_update)
            local_norm = np.linalg.norm(local_vector)
            angle = math.pi / 2
            if local_norm > 0:
                cosine = np.dot(global_vector, local_vector) / (
                    global_norm * local_norm
                )
                angle = math.acos(float(np.clip(cosine, -1.0, 1.0)))

            client_key = _client_key(client_proxy, index)
            smoothed_angle = self._update_smoothed_angle(client_key, angle)
            scores.append(self._mapped_contribution(smoothed_angle))

        return _softmax_with_sample_weights(scores, sample_weights)

    def _update_smoothed_angle(self, client_key: str, angle: float) -> float:
        count = self.angle_counts.get(client_key, 0) + 1
        previous = self.smoothed_angles.get(client_key, angle)
        smoothed = ((count - 1) / count) * previous + (angle / count)
        self.angle_counts[client_key] = count
        self.smoothed_angles[client_key] = smoothed
        return smoothed

    def _mapped_contribution(self, smoothed_angle: float) -> float:
        return self.alpha * (
            1.0 - math.exp(-math.exp(-self.alpha * (smoothed_angle - 1.0)))
        )


def _flatten(arrays: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(
        [array.astype(np.float64, copy=False).ravel() for array in arrays]
    )


def _client_key(client_proxy: object, index: int) -> str:
    cid = getattr(client_proxy, "cid", None)
    if cid is not None:
        return str(cid)
    return str(index)


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
    if total == 0:
        return sample_weights
    return [float(weight / total) for weight in weighted_scores]


class FedAdpBuilder:
    name = "fedadp"

    def build_strategy(
        self,
        config: ExperimentConfig,
        initial_model: torch.nn.Module,
        evaluate_fn: ServerEvaluateFn | None,
    ) -> FedAdpStrategy:
        min_fit_clients = max(1, math.ceil(config.num_supernodes * config.fraction_train))
        min_eval_clients = max(1, math.ceil(config.num_supernodes * config.fraction_evaluate))
        initial_parameters = ndarrays_to_parameters(get_model_parameters(initial_model))

        return FedAdpStrategy(
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
            alpha=config.fedadp_alpha,
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
