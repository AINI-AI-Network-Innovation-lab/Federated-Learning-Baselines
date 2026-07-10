"""FedLAMA algorithm builder."""

from __future__ import annotations

import json
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
from fl_baselines.training.fedlama import fedlama_sync_mask_for_round


class FedLAMAStrategy(FedAvg):
    """FedAvg-compatible Flower strategy with layer-wise adaptive aggregation."""

    def __init__(
        self,
        *,
        base_interval: int,
        interval_factor: float,
        checkpoint_model: torch.nn.Module,
        output_dir: str,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        if self.initial_parameters is None:
            raise ValueError("FedLAMAStrategy requires initial_parameters")
        if base_interval <= 0:
            raise ValueError("FedLAMA base interval must be positive")
        if interval_factor < 1:
            raise ValueError("FedLAMA interval factor must be at least 1")

        self.base_interval = base_interval
        self.interval_factor = interval_factor
        self._checkpoint_model = checkpoint_model
        self._output_dir = output_dir
        self.global_parameters = parameters_to_ndarrays(self.initial_parameters)
        self.layer_intervals = [self.base_interval] * len(self.global_parameters)
        self._current_sync_mask = [True] * len(self.global_parameters)
        self._current_sync_indices = list(range(len(self.global_parameters)))

    def configure_fit(self, server_round, parameters, client_manager):
        client_fit_ins = super().configure_fit(server_round, parameters, client_manager)
        sync_mask = fedlama_sync_mask_for_round(server_round, self.layer_intervals)
        self._current_sync_mask = sync_mask
        self._current_sync_indices = [
            index for index, should_sync in enumerate(sync_mask) if should_sync
        ]
        fit_config = self._fit_config(server_round, sync_mask)
        return [
            (client, FitIns(fit_ins.parameters, fit_config.copy()))
            for client, fit_ins in client_fit_ins
        ]

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        total_examples = sum(fit_res.num_examples for _, fit_res in results)
        if total_examples <= 0:
            return None, {}

        sync_indices = self._current_sync_indices
        updated_global = [parameter.copy() for parameter in self.global_parameters]
        synced_payloads: dict[int, list[tuple[np.ndarray, int]]] = {
            index: [] for index in sync_indices
        }
        layer_discrepancies: list[list[float]] = []

        for _, fit_res in results:
            client_arrays = parameters_to_ndarrays(fit_res.parameters)
            if len(client_arrays) != len(sync_indices):
                raise ValueError("FedLAMA client result is missing synced layer payload")
            for layer_index, layer_array in zip(sync_indices, client_arrays):
                synced_payloads[layer_index].append((layer_array, fit_res.num_examples))

            raw_discrepancies = fit_res.metrics.get("fedlama_layer_discrepancies")
            if raw_discrepancies is not None:
                if isinstance(raw_discrepancies, bytes):
                    raw_discrepancies = raw_discrepancies.decode("utf-8")
                layer_values = json.loads(str(raw_discrepancies))
                if len(layer_values) != len(self.global_parameters):
                    raise ValueError(
                        "FedLAMA client discrepancy payload has unexpected length"
                    )
                layer_discrepancies.append([float(value) for value in layer_values])

        for layer_index in sync_indices:
            layer_results = synced_payloads[layer_index]
            if not layer_results:
                continue
            total_layer_examples = sum(num_examples for _, num_examples in layer_results)
            updated_global[layer_index] = sum(
                layer_array * (num_examples / total_layer_examples)
                for layer_array, num_examples in layer_results
            )

        self.global_parameters = updated_global
        aggregated_parameters = ndarrays_to_parameters(updated_global)

        if layer_discrepancies:
            mean_discrepancies = [
                sum(client_values[index] for client_values in layer_discrepancies)
                / len(layer_discrepancies)
                for index in range(len(self.global_parameters))
            ]
            self.layer_intervals = self._adjust_intervals(mean_discrepancies)

        metrics_aggregated: dict[str, Scalar] = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(fit_res.num_examples, fit_res.metrics) for _, fit_res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        metrics_aggregated["fedlama_sync_layer_count"] = len(sync_indices)
        metrics_aggregated["fedlama_long_interval_layers"] = sum(
            1 for interval in self.layer_intervals if interval > self.base_interval
        )

        save_round_checkpoints(
            self._checkpoint_model,
            aggregated_parameters,
            self._output_dir,
            server_round,
        )
        return aggregated_parameters, metrics_aggregated

    def _adjust_intervals(self, discrepancies: list[float]) -> list[int]:
        total_discrepancy = sum(
            discrepancy * size
            for discrepancy, size in zip(discrepancies, self._layer_sizes())
        )
        long_interval = max(
            self.base_interval, int(round(self.base_interval * self.interval_factor))
        )
        if total_discrepancy <= 0.0:
            return [long_interval] * len(discrepancies)

        layer_sizes = self._layer_sizes()
        total_size = sum(layer_sizes)
        ranked_layers = sorted(enumerate(discrepancies), key=lambda item: item[1])
        new_intervals = [self.base_interval] * len(discrepancies)
        cumulative_discrepancy = 0.0
        cumulative_size = 0.0

        for layer_index, discrepancy in ranked_layers:
            layer_size = layer_sizes[layer_index]
            cumulative_discrepancy += discrepancy * layer_size
            cumulative_size += layer_size
            if cumulative_discrepancy / total_discrepancy < cumulative_size / total_size:
                new_intervals[layer_index] = long_interval
            else:
                new_intervals[layer_index] = self.base_interval
        return new_intervals

    def _layer_sizes(self) -> list[int]:
        return [int(parameter.size) for parameter in self.global_parameters]

    def _fit_config(
        self,
        server_round: int,
        sync_mask: list[bool],
    ) -> dict[str, bool | bytes | float | int | str]:
        return {
            "algorithm": self.name,
            "server_round": server_round,
            "local_epochs": self._resolve_local_epochs(),
            "learning_rate": self._resolve_learning_rate(),
            "fedlama_base_interval": self.base_interval,
            "fedlama_interval_factor": self.interval_factor,
            "fedlama_sync_mask": json.dumps(sync_mask),
            "fedlama_layer_intervals": json.dumps(self.layer_intervals),
        }

    def _resolve_local_epochs(self) -> int:
        if self.on_fit_config_fn is None:
            return 1
        fit_config = self.on_fit_config_fn(1)
        return int(fit_config.get("local_epochs", 1))

    def _resolve_learning_rate(self) -> float:
        if self.on_fit_config_fn is None:
            return 0.01
        fit_config = self.on_fit_config_fn(1)
        return float(fit_config.get("learning_rate", 0.01))


class FedLAMABuilder:
    name = "fedlama"

    def build_strategy(
        self,
        config: ExperimentConfig,
        initial_model: torch.nn.Module,
        evaluate_fn: ServerEvaluateFn | None,
    ) -> FedLAMAStrategy:
        min_fit_clients = max(1, math.ceil(config.num_supernodes * config.fraction_train))
        min_eval_clients = max(1, math.ceil(config.num_supernodes * config.fraction_evaluate))
        initial_parameters = ndarrays_to_parameters(get_model_parameters(initial_model))

        return FedLAMAStrategy(
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
            base_interval=config.fedlama_base_interval,
            interval_factor=config.fedlama_interval_factor,
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
                "fedlama_base_interval": config.fedlama_base_interval,
                "fedlama_interval_factor": config.fedlama_interval_factor,
            }

        return fn
