"""FedDisco algorithm builder."""

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


class FedDiscoStrategy(FedAvg):
    """FedDisco strategy with discrepancy-aware client aggregation."""

    def __init__(
        self,
        *,
        discrepancy_weight: float,
        bias: float,
        checkpoint_model: torch.nn.Module,
        output_dir: str,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        if self.initial_parameters is None:
            raise ValueError("FedDiscoStrategy requires initial_parameters")
        self.discrepancy_weight = discrepancy_weight
        self.bias = bias
        self.client_discrepancies: dict[str, float] = {}
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
        aggregation_weights = self._aggregation_weights(results, total_examples)
        self.last_aggregation_weights = aggregation_weights
        aggregated_ndarrays = aggregate(list(zip(client_models, aggregation_weights)))
        aggregated_parameters = ndarrays_to_parameters(aggregated_ndarrays)

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

    def _aggregation_weights(
        self,
        results: list[tuple[Any, Any]],
        total_examples: int,
    ) -> list[float]:
        sample_weights = [
            fit_res.num_examples / total_examples for _, fit_res in results
        ]
        scores: list[float] = []

        for index, ((client_proxy, fit_res), sample_weight) in enumerate(
            zip(results, sample_weights)
        ):
            client_key = _client_key(client_proxy, index)
            discrepancy = self._discrepancy_for_client(client_key, fit_res.metrics)
            scores.append(
                sample_weight - (self.discrepancy_weight * discrepancy) + self.bias
            )

        weights = np.maximum(np.asarray(scores, dtype=np.float64), 0.0)
        total = float(np.sum(weights))
        if not math.isfinite(total) or total <= 0:
            return sample_weights
        normalized = [float(weight / total) for weight in weights]
        if not np.isfinite(np.asarray(normalized)).all():
            return sample_weights
        return normalized

    def _discrepancy_for_client(
        self,
        client_key: str,
        metrics: dict[str, Scalar],
    ) -> float:
        raw_discrepancy = metrics.get("feddisco_discrepancy")
        if raw_discrepancy is None:
            return self.client_discrepancies.get(client_key, 0.0)

        discrepancy = float(raw_discrepancy)
        if not math.isfinite(discrepancy):
            return self.client_discrepancies.get(client_key, 0.0)

        self.client_discrepancies[client_key] = discrepancy
        return discrepancy


def _client_key(client_proxy: object, index: int) -> str:
    cid = getattr(client_proxy, "cid", None)
    if cid is not None:
        return str(cid)
    return str(index)


class FedDiscoBuilder:
    name = "feddisco"

    def build_strategy(
        self,
        config: ExperimentConfig,
        initial_model: torch.nn.Module,
        evaluate_fn: ServerEvaluateFn | None,
    ) -> FedDiscoStrategy:
        min_fit_clients = max(1, math.ceil(config.num_supernodes * config.fraction_train))
        min_eval_clients = max(1, math.ceil(config.num_supernodes * config.fraction_evaluate))
        initial_parameters = ndarrays_to_parameters(get_model_parameters(initial_model))

        return FedDiscoStrategy(
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
            discrepancy_weight=config.feddisco_discrepancy_weight,
            bias=config.feddisco_bias,
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
                "num_classes": config.num_classes,
                "feddisco_discrepancy_weight": config.feddisco_discrepancy_weight,
                "feddisco_bias": config.feddisco_bias,
                "feddisco_metric": config.feddisco_metric,
                "feddisco_epsilon": config.feddisco_epsilon,
            }

        return fn
