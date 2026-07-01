"""FedProto algorithm builder."""

from __future__ import annotations

import math

import numpy as np
import torch
from flwr.common import FitIns, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate

from fl_baselines.clients.torch_client import get_model_parameters
from fl_baselines.core.config import ExperimentConfig
from fl_baselines.core.types import ServerEvaluateFn
from fl_baselines.logging.checkpointing import save_round_checkpoints
from fl_baselines.logging.metrics import weighted_average
from fl_baselines.training.features import infer_feature_dim


class FedProtoStrategy(FedAvg):
    """FedAvg-compatible Flower strategy with class prototype aggregation."""

    def __init__(
        self,
        *,
        num_classes: int,
        global_prototypes: np.ndarray,
        checkpoint_model: torch.nn.Module,
        output_dir: str,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.global_prototypes = global_prototypes.copy()
        self._checkpoint_model = checkpoint_model
        self._output_dir = output_dir

    def configure_fit(self, server_round, parameters, client_manager):
        client_fit_ins = super().configure_fit(server_round, parameters, client_manager)
        model_parameters = parameters_to_ndarrays(parameters)
        combined_parameters = ndarrays_to_parameters(
            model_parameters + [self.global_prototypes.copy()]
        )
        return [
            (client, FitIns(combined_parameters, fit_ins.config))
            for client, fit_ins in client_fit_ins
        ]

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        model_results = []
        prototype_sums = []
        prototype_counts = []
        for _, fit_res in results:
            arrays = parameters_to_ndarrays(fit_res.parameters)
            if len(arrays) < 3:
                raise ValueError("FedProto client result is missing prototype payload")
            model_arrays = arrays[:-2]
            prototype_sum = arrays[-2]
            prototype_count = arrays[-1]
            model_results.append((model_arrays, fit_res.num_examples))
            prototype_sums.append(prototype_sum)
            prototype_counts.append(prototype_count)

        aggregated_model = aggregate(model_results)
        self._aggregate_prototypes(prototype_sums, prototype_counts)
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

    def _aggregate_prototypes(
        self,
        prototype_sums: list[np.ndarray],
        prototype_counts: list[np.ndarray],
    ) -> None:
        total_sums = np.sum(np.stack(prototype_sums, axis=0), axis=0)
        total_counts = np.sum(np.stack(prototype_counts, axis=0), axis=0)
        for class_index in range(self.num_classes):
            count = total_counts[class_index]
            if count > 0:
                self.global_prototypes[class_index] = total_sums[class_index] / count


class FedProtoBuilder:
    name = "fedproto"

    def build_strategy(
        self,
        config: ExperimentConfig,
        initial_model: torch.nn.Module,
        evaluate_fn: ServerEvaluateFn | None,
    ) -> FedProtoStrategy:
        min_fit_clients = max(1, math.ceil(config.num_supernodes * config.fraction_train))
        min_eval_clients = max(1, math.ceil(config.num_supernodes * config.fraction_evaluate))
        initial_parameters = ndarrays_to_parameters(get_model_parameters(initial_model))
        feature_dim = infer_feature_dim(initial_model, config)
        global_prototypes = np.zeros((config.num_classes, feature_dim), dtype=np.float32)

        return FedProtoStrategy(
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
            num_classes=config.num_classes,
            global_prototypes=global_prototypes,
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
                "fedproto_lambda": config.fedproto_lambda,
            }

        return fn
