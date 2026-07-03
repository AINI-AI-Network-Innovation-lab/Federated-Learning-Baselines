"""FedAAW algorithm builder."""

from __future__ import annotations

import math
from typing import Any

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


class FedAAWStrategy(FedAvg):
    """FedAAW strategy with gradient-norm-driven adaptive aggregation."""

    def __init__(
        self,
        *,
        beta: float,
        gamma: float,
        epsilon: float,
        checkpoint_model: torch.nn.Module,
        output_dir: str,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        if self.initial_parameters is None:
            raise ValueError("FedAAWStrategy requires initial_parameters")
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon
        self.global_parameters = parameters_to_ndarrays(self.initial_parameters)
        self.gradient_trackers: dict[str, float] = {}
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
        adaptive_weights = self._adaptive_weights(server_round, results, total_examples)
        self.last_aggregation_weights = adaptive_weights
        self.global_parameters = aggregate(list(zip(client_models, adaptive_weights)))
        aggregated_parameters = ndarrays_to_parameters(self.global_parameters)

        metrics_aggregated: dict[str, Scalar] = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [
                (fit_res.num_examples, fit_res.metrics) for _, fit_res in results
            ]
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
        server_round: int,
        results: list[tuple[Any, Any]],
        total_examples: int,
    ) -> list[float]:
        round_index = server_round - 1
        scores: list[float] = []
        sample_weights = [
            fit_res.num_examples / total_examples for _, fit_res in results
        ]

        for index, ((client_proxy, fit_res), sample_weight) in enumerate(
            zip(results, sample_weights)
        ):
            client_key = _client_key(client_proxy, index)
            grad_norm_sq = float(fit_res.metrics.get("fedaaw_grad_norm_sq", 0.0))
            if not math.isfinite(grad_norm_sq):
                return sample_weights
            tracker = self._update_tracker(client_key, grad_norm_sq, round_index)
            score = sample_weight + (self.beta / max(tracker, self.epsilon)) - self.gamma
            scores.append(score)

        weights = _softmax(scores)
        if not np.isfinite(np.asarray(weights)).all():
            return sample_weights
        return weights

    def _update_tracker(
        self,
        client_key: str,
        grad_norm_sq: float,
        round_index: int,
    ) -> float:
        if not math.isfinite(grad_norm_sq):
            grad_norm_sq = self.epsilon
        previous = self.gradient_trackers.get(client_key)
        if previous is None or round_index <= 0:
            tracker = grad_norm_sq
        else:
            tracker = ((round_index * previous) + grad_norm_sq) / (round_index + 1)
        self.gradient_trackers[client_key] = tracker
        return tracker


def _client_key(client_proxy: object, index: int) -> str:
    cid = getattr(client_proxy, "cid", None)
    if cid is not None:
        return str(cid)
    return str(index)


def _softmax(scores: list[float]) -> list[float]:
    shifted_scores = np.asarray(scores, dtype=np.float64)
    if shifted_scores.size == 0:
        return []
    shifted_scores -= float(np.nanmax(shifted_scores))
    weights = np.exp(shifted_scores)
    total = float(np.sum(weights))
    if not math.isfinite(total) or total <= 0:
        return [float("nan")] * len(scores)
    return [float(weight / total) for weight in weights]


class FedAAWBuilder:
    name = "fedaaw"

    def build_strategy(
        self,
        config: ExperimentConfig,
        initial_model: torch.nn.Module,
        evaluate_fn: ServerEvaluateFn | None,
    ) -> FedAAWStrategy:
        min_fit_clients = max(1, math.ceil(config.num_supernodes * config.fraction_train))
        min_eval_clients = max(1, math.ceil(config.num_supernodes * config.fraction_evaluate))
        initial_parameters = ndarrays_to_parameters(get_model_parameters(initial_model))

        return FedAAWStrategy(
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
            beta=config.fedaaw_beta,
            gamma=config.fedaaw_gamma,
            epsilon=config.fedaaw_epsilon,
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
                "fedaaw_beta": config.fedaaw_beta,
                "fedaaw_gamma": config.fedaaw_gamma,
                "fedaaw_epsilon": config.fedaaw_epsilon,
            }

        return fn
