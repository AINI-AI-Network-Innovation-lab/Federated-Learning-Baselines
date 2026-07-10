"""FedMMD algorithm builder."""

from __future__ import annotations

import math

import numpy as np
import torch
from flwr.common import Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg

from fl_baselines.clients.torch_client import get_model_parameters
from fl_baselines.core.config import ExperimentConfig
from fl_baselines.core.types import ServerEvaluateFn
from fl_baselines.logging.checkpointing import save_round_checkpoints
from fl_baselines.logging.metrics import weighted_average


class FedMMDStrategy(FedAvg):
    """FedAvg-style strategy with parameter-space discrepancy weighting."""

    def __init__(
        self,
        *,
        sigma: float,
        sknq_threshold: float,
        min_clients: int,
        entropy_eps: float,
        checkpoint_model: torch.nn.Module,
        output_dir: str,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.sigma = sigma
        self.sknq_threshold = sknq_threshold
        self.min_clients = min_clients
        self.entropy_eps = entropy_eps
        self.last_client_weights: list[float] = []
        self.last_client_scores: list[float] = []
        self.last_selected_indices: list[int] = []
        self._checkpoint_model = checkpoint_model
        self._output_dir = output_dir

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        client_arrays = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
        client_examples = [fit_res.num_examples for _, fit_res in results]
        discrepancy_scores = _client_discrepancy_scores(client_arrays, self.sigma)
        selected_indices = _select_clients_by_sknq(
            discrepancy_scores,
            threshold=self.sknq_threshold,
            min_clients=min(self.min_clients, len(results)),
        )
        selected_scores = [discrepancy_scores[index] for index in selected_indices]
        selected_examples = [client_examples[index] for index in selected_indices]
        entropy_weights = _entropy_client_weights(selected_scores, self.entropy_eps)
        combined_weights = _combine_sample_and_entropy_weights(
            selected_examples,
            entropy_weights,
            self.entropy_eps,
        )

        aggregated_arrays = _weighted_parameter_average(
            [client_arrays[index] for index in selected_indices],
            combined_weights,
        )
        parameters_aggregated = ndarrays_to_parameters(aggregated_arrays)

        self.last_client_scores = selected_scores
        self.last_client_weights = combined_weights
        self.last_selected_indices = selected_indices

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


def _flatten_parameters(arrays: list[np.ndarray]) -> np.ndarray:
    return np.concatenate([array.astype(np.float32, copy=False).ravel() for array in arrays])


def _parameter_mmd(left: np.ndarray, right: np.ndarray, sigma: float) -> float:
    squared_distance = float(np.mean(np.square(left - right)))
    return float(2.0 - (2.0 * np.exp(-squared_distance / (2.0 * sigma * sigma))))


def _client_discrepancy_scores(
    client_arrays: list[list[np.ndarray]],
    sigma: float,
) -> list[float]:
    flattened = [_flatten_parameters(arrays) for arrays in client_arrays]
    scores = []
    for index, parameters in enumerate(flattened):
        pairwise = [
            _parameter_mmd(parameters, other, sigma)
            for other_index, other in enumerate(flattened)
            if other_index != index
        ]
        scores.append(float(np.mean(pairwise)) if pairwise else 0.0)
    return scores


def _select_clients_by_sknq(
    scores: list[float],
    *,
    threshold: float,
    min_clients: int,
) -> list[int]:
    if len(scores) <= min_clients:
        return list(range(len(scores)))
    minimum = min(scores)
    maximum = max(scores)
    if maximum <= minimum:
        return list(range(len(scores)))

    cutoff = minimum + threshold * (maximum - minimum)
    selected = [index for index, score in enumerate(scores) if score <= cutoff]
    if len(selected) >= min_clients:
        return selected
    ordered = sorted(range(len(scores)), key=scores.__getitem__)
    return ordered[:min_clients]


def _entropy_client_weights(scores: list[float], eps: float) -> list[float]:
    if len(scores) == 1:
        return [1.0]
    minimum = min(scores)
    maximum = max(scores)
    if maximum <= minimum + eps:
        return [1.0 / len(scores)] * len(scores)

    normalized = [
        (score - minimum) / (maximum - minimum + eps)
        for score in scores
    ]
    probabilities = np.asarray(normalized, dtype=np.float64) + eps
    probabilities = probabilities / probabilities.sum()
    information = 1.0 - probabilities
    information_sum = float(information.sum())
    if information_sum <= eps:
        return [1.0 / len(scores)] * len(scores)
    return [float(value / information_sum) for value in information]


def _combine_sample_and_entropy_weights(
    sample_counts: list[int],
    entropy_weights: list[float],
    eps: float,
) -> list[float]:
    total_examples = float(sum(sample_counts))
    if total_examples <= 0:
        return [1.0 / len(entropy_weights)] * len(entropy_weights)
    combined = [
        (sample_count / total_examples) * entropy_weight
        for sample_count, entropy_weight in zip(sample_counts, entropy_weights)
    ]
    combined_sum = float(sum(combined))
    if combined_sum <= eps:
        return [1.0 / len(entropy_weights)] * len(entropy_weights)
    return [float(weight / combined_sum) for weight in combined]


def _weighted_parameter_average(
    client_arrays: list[list[np.ndarray]],
    weights: list[float],
) -> list[np.ndarray]:
    return [
        sum(arrays[index] * weight for arrays, weight in zip(client_arrays, weights))
        for index in range(len(client_arrays[0]))
    ]


class FedMMDBuilder:
    name = "fedmmd"

    def build_strategy(
        self,
        config: ExperimentConfig,
        initial_model: torch.nn.Module,
        evaluate_fn: ServerEvaluateFn | None,
    ) -> FedMMDStrategy:
        min_fit_clients = max(1, math.ceil(config.num_supernodes * config.fraction_train))
        min_eval_clients = max(1, math.ceil(config.num_supernodes * config.fraction_evaluate))
        initial_parameters = ndarrays_to_parameters(get_model_parameters(initial_model))

        return FedMMDStrategy(
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
            sigma=config.fedmmd_sigma,
            sknq_threshold=config.fedmmd_sknq_threshold,
            min_clients=config.fedmmd_min_clients,
            entropy_eps=config.fedmmd_entropy_eps,
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
                "fedmmd_sigma": config.fedmmd_sigma,
                "fedmmd_sknq_threshold": config.fedmmd_sknq_threshold,
                "fedmmd_min_clients": config.fedmmd_min_clients,
                "fedmmd_entropy_eps": config.fedmmd_entropy_eps,
            }

        return fn
