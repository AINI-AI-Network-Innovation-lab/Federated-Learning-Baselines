"""FedAMP algorithm builder."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from flwr.common import (
    FitIns,
    Parameters,
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


class FedAMPStrategy(FedAvg):
    """FedAMP strategy with personalized cloud models per client."""

    def __init__(
        self,
        *,
        fedamp_lambda: float,
        alpha: float,
        sigma: float,
        checkpoint_model: torch.nn.Module,
        output_dir: str,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        if self.initial_parameters is None:
            raise ValueError("FedAMPStrategy requires initial_parameters")
        if fedamp_lambda <= 0:
            raise ValueError("FedAMP lambda must be positive")
        if alpha <= 0:
            raise ValueError("FedAMP alpha must be positive")
        if sigma <= 0:
            raise ValueError("FedAMP sigma must be positive")
        self.fedamp_lambda = fedamp_lambda
        self.alpha = alpha
        self.sigma = sigma
        self.client_models: dict[str, list[np.ndarray]] = {}
        self.personalized_cloud_models: dict[str, list[np.ndarray]] = {}
        self.last_message_weights: dict[str, dict[str, float]] = {}
        self._checkpoint_model = checkpoint_model
        self._output_dir = output_dir

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: Any,
    ) -> list[tuple[Any, FitIns]]:
        client_fit_ins = super().configure_fit(server_round, parameters, client_manager)
        personalized_fit_ins: list[tuple[Any, FitIns]] = []

        for index, (client_proxy, fit_ins) in enumerate(client_fit_ins):
            client_key = _client_key(client_proxy, index)
            cloud_parameters = self.personalized_cloud_models.get(client_key)
            if cloud_parameters is None:
                personalized_fit_ins.append((client_proxy, fit_ins))
                continue
            personalized_fit_ins.append(
                (
                    client_proxy,
                    FitIns(
                        ndarrays_to_parameters(cloud_parameters),
                        dict(fit_ins.config),
                    ),
                )
            )
        return personalized_fit_ins

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        participating_client_ids: list[str] = []
        for index, (client_proxy, fit_res) in enumerate(results):
            client_id = _client_key(client_proxy, index)
            participating_client_ids.append(client_id)
            self.client_models[client_id] = parameters_to_ndarrays(fit_res.parameters)

        for client_id in participating_client_ids:
            self.personalized_cloud_models[client_id] = self._personalized_cloud_model(
                client_id
            )

        aggregated_ndarrays = aggregate(
            [
                (self.personalized_cloud_models[client_id], 1.0 / len(participating_client_ids))
                for client_id in participating_client_ids
            ]
        )
        aggregated_parameters = ndarrays_to_parameters(aggregated_ndarrays)

        metrics_aggregated: dict[str, Scalar] = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [
                (fit_res.num_examples, fit_res.metrics) for _, fit_res in results
            ]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        metrics_aggregated["fedamp_client_count"] = len(self.client_models)

        save_round_checkpoints(
            self._checkpoint_model,
            aggregated_parameters,
            self._output_dir,
            server_round,
        )
        return aggregated_parameters, metrics_aggregated

    def _personalized_cloud_model(self, client_id: str) -> list[np.ndarray]:
        client_ids = sorted(self.client_models.keys())
        weights = self._message_weights(client_id, client_ids)
        self.last_message_weights[client_id] = weights
        return aggregate(
            [
                (self.client_models[source_client_id], weights[source_client_id])
                for source_client_id in client_ids
            ]
        )

    def _message_weights(
        self,
        client_id: str,
        client_ids: list[str],
    ) -> dict[str, float]:
        if len(client_ids) == 1:
            return {client_id: 1.0}

        target_model = self.client_models[client_id]
        raw_weights: dict[str, float] = {}
        off_diagonal_sum = 0.0

        for source_client_id in client_ids:
            if source_client_id == client_id:
                continue
            distance_sq = _squared_distance(
                target_model,
                self.client_models[source_client_id],
            )
            weight = self.alpha * math.exp(-distance_sq / self.sigma) / self.sigma
            raw_weights[source_client_id] = weight
            off_diagonal_sum += weight

        self_weight = max(0.0, 1.0 - off_diagonal_sum)
        raw_weights[client_id] = self_weight
        total = sum(raw_weights.values())
        if total <= 0 or not math.isfinite(total):
            uniform = 1.0 / len(client_ids)
            return {source_client_id: uniform for source_client_id in client_ids}
        return {
            source_client_id: raw_weights[source_client_id] / total
            for source_client_id in client_ids
        }


def _client_key(client_proxy: object, index: int) -> str:
    cid = getattr(client_proxy, "cid", None)
    if cid is not None:
        return str(cid)
    return str(index)


def _squared_distance(
    left_parameters: list[np.ndarray],
    right_parameters: list[np.ndarray],
) -> float:
    return float(
        sum(
            float(
                np.sum(
                    np.square(
                        left.astype(np.float64, copy=False)
                        - right.astype(np.float64, copy=False)
                    )
                )
            )
            for left, right in zip(left_parameters, right_parameters)
        )
    )


class FedAMPBuilder:
    name = "fedamp"

    def build_strategy(
        self,
        config: ExperimentConfig,
        initial_model: torch.nn.Module,
        evaluate_fn: ServerEvaluateFn | None,
    ) -> FedAMPStrategy:
        min_fit_clients = max(1, math.ceil(config.num_supernodes * config.fraction_train))
        min_eval_clients = max(1, math.ceil(config.num_supernodes * config.fraction_evaluate))
        initial_parameters = ndarrays_to_parameters(get_model_parameters(initial_model))

        return FedAMPStrategy(
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
            fedamp_lambda=config.fedamp_lambda,
            alpha=config.fedamp_alpha,
            sigma=config.fedamp_sigma,
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
                "fedamp_proximal_mu": config.fedamp_lambda / config.fedamp_alpha,
            }

        return fn
