"""FedCDA algorithm builder."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import product
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


@dataclass
class _CachedModel:
    client_id: str
    round_number: int
    parameters: list[np.ndarray]
    train_loss: float
    norm_sq: float


class FedCDAStrategy(FedAvg):
    """FedCDA cross-round divergence-aware aggregation."""

    def __init__(
        self,
        *,
        memory_size: int,
        num_batches: int,
        warmup_rounds: int,
        loss_weight: float,
        checkpoint_model: torch.nn.Module,
        output_dir: str,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        if self.initial_parameters is None:
            raise ValueError("FedCDAStrategy requires initial_parameters")
        self.memory_size = memory_size
        self.num_batches = num_batches
        self.warmup_rounds = warmup_rounds
        self.loss_weight = loss_weight
        self.client_histories: dict[str, deque[_CachedModel]] = {}
        self.selected_models: dict[str, _CachedModel] = {}
        self.last_aggregation_weights: list[float] = []
        self.last_selected_client_ids: list[str] = []
        self.last_selected_rounds: dict[str, int] = {}
        self._checkpoint_model = checkpoint_model
        self._output_dir = output_dir

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        latest_models = self._update_histories(server_round, results)
        participating_client_ids = [client_id for client_id, _ in latest_models]

        if server_round <= self.warmup_rounds:
            for client_id, cached_model in latest_models:
                self.selected_models[client_id] = cached_model
        else:
            self._select_cross_round_models(participating_client_ids)

        selected_client_ids = sorted(self.selected_models.keys())
        selected_models = [self.selected_models[client_id] for client_id in selected_client_ids]
        if not selected_models:
            return None, {}

        uniform_weight = 1.0 / len(selected_models)
        self.last_aggregation_weights = [uniform_weight] * len(selected_models)
        self.last_selected_client_ids = selected_client_ids
        self.last_selected_rounds = {
            client_id: self.selected_models[client_id].round_number
            for client_id in selected_client_ids
        }

        aggregated_ndarrays = aggregate(
            [(cached_model.parameters, uniform_weight) for cached_model in selected_models]
        )
        aggregated_parameters = ndarrays_to_parameters(aggregated_ndarrays)

        metrics_aggregated: dict[str, Scalar] = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [
                (fit_res.num_examples, fit_res.metrics) for _, fit_res in results
            ]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        metrics_aggregated["fedcda_cache_size"] = len(self.client_histories)
        metrics_aggregated["fedcda_selected_client_count"] = len(selected_models)
        metrics_aggregated["fedcda_used_cross_round"] = float(
            any(
                cached_model.round_number != server_round
                for cached_model in selected_models
            )
        )

        save_round_checkpoints(
            self._checkpoint_model,
            aggregated_parameters,
            self._output_dir,
            server_round,
        )
        return aggregated_parameters, metrics_aggregated

    def _update_histories(
        self,
        server_round: int,
        results: list[tuple[Any, Any]],
    ) -> list[tuple[str, _CachedModel]]:
        latest_models: list[tuple[str, _CachedModel]] = []

        for index, (client_proxy, fit_res) in enumerate(results):
            client_id = _client_key(client_proxy, index)
            parameters = parameters_to_ndarrays(fit_res.parameters)
            train_loss = float(fit_res.metrics.get("train_loss", 0.0))
            cached_model = _CachedModel(
                client_id=client_id,
                round_number=server_round,
                parameters=parameters,
                train_loss=train_loss,
                norm_sq=_parameter_norm_sq(parameters),
            )

            history = self.client_histories.setdefault(
                client_id,
                deque(maxlen=self.memory_size),
            )
            history.append(cached_model)
            latest_models.append((client_id, cached_model))

            if client_id not in self.selected_models:
                self.selected_models[client_id] = cached_model

        return latest_models

    def _select_cross_round_models(self, participating_client_ids: Iterable[str]) -> None:
        participating = sorted(set(participating_client_ids))
        fixed_models = {
            client_id: cached_model
            for client_id, cached_model in self.selected_models.items()
            if client_id not in participating
        }
        if not participating:
            return

        batch_size = max(1, math.ceil(len(participating) / self.num_batches))
        for start in range(0, len(participating), batch_size):
            batch_client_ids = participating[start : start + batch_size]
            best_models = self._best_batch_selection(
                batch_client_ids=batch_client_ids,
                fixed_models=fixed_models,
            )
            fixed_models.update(best_models)

        self.selected_models = fixed_models

    def _best_batch_selection(
        self,
        *,
        batch_client_ids: list[str],
        fixed_models: dict[str, _CachedModel],
    ) -> dict[str, _CachedModel]:
        candidate_lists = [
            list(self.client_histories[client_id]) for client_id in batch_client_ids
        ]
        best_selection = {
            client_id: candidates[-1]
            for client_id, candidates in zip(batch_client_ids, candidate_lists)
        }
        best_score = float("inf")

        for candidate_tuple in product(*candidate_lists):
            selected_models = list(fixed_models.values()) + list(candidate_tuple)
            score = self._selection_objective(selected_models)
            if score < best_score:
                best_score = score
                best_selection = {
                    client_id: candidate
                    for client_id, candidate in zip(batch_client_ids, candidate_tuple)
                }

        return best_selection

    def _selection_objective(self, cached_models: list[_CachedModel]) -> float:
        if not cached_models:
            return float("inf")
        num_models = len(cached_models)
        average_loss = sum(cached_model.train_loss for cached_model in cached_models) / num_models
        average_norm_sq = sum(cached_model.norm_sq for cached_model in cached_models) / num_models
        mean_parameters = aggregate(
            [(cached_model.parameters, 1.0 / num_models) for cached_model in cached_models]
        )
        mean_norm_sq = _parameter_norm_sq(mean_parameters)
        return average_loss + (self.loss_weight / 2.0) * (average_norm_sq - mean_norm_sq)


def _client_key(client_proxy: object, index: int) -> str:
    cid = getattr(client_proxy, "cid", None)
    if cid is not None:
        return str(cid)
    return str(index)


def _parameter_norm_sq(parameters: list[np.ndarray]) -> float:
    return float(
        sum(
            float(np.sum(np.square(parameter.astype(np.float64, copy=False))))
            for parameter in parameters
        )
    )


class FedCDABuilder:
    name = "fedcda"

    def build_strategy(
        self,
        config: ExperimentConfig,
        initial_model: torch.nn.Module,
        evaluate_fn: ServerEvaluateFn | None,
    ) -> FedCDAStrategy:
        min_fit_clients = max(1, math.ceil(config.num_supernodes * config.fraction_train))
        min_eval_clients = max(1, math.ceil(config.num_supernodes * config.fraction_evaluate))
        initial_parameters = ndarrays_to_parameters(get_model_parameters(initial_model))

        return FedCDAStrategy(
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
            memory_size=config.fedcda_memory_size,
            num_batches=config.fedcda_num_batches,
            warmup_rounds=config.fedcda_warmup_rounds,
            loss_weight=config.fedcda_loss_weight,
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
