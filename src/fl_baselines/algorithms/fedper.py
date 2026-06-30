"""FedPer algorithm builder."""

from __future__ import annotations

import math
from collections import OrderedDict

import numpy as np
import torch
from flwr.common import (
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate

from fl_baselines.core.config import ExperimentConfig
from fl_baselines.core.types import ServerEvaluateFn
from fl_baselines.logging.artifacts import save_model
from fl_baselines.logging.metrics import weighted_average


def _state_keys(model: torch.nn.Module) -> list[str]:
    return list(model.state_dict().keys())


def get_model_state_arrays(model: torch.nn.Module) -> list[np.ndarray]:
    return [value.detach().cpu().numpy() for value in model.state_dict().values()]


def _parameterized_module_names(model: torch.nn.Module) -> list[str]:
    module_names: list[str] = []
    for name, module in model.named_modules():
        if not name:
            continue
        if any(True for _ in module.parameters(recurse=False)):
            module_names.append(name)
    return module_names


def split_fedper_parameter_indices(
    model: torch.nn.Module,
    personal_layers: int,
) -> tuple[list[int], list[int]]:
    """Split state_dict indices into shared/base and personal/head parts."""

    parameterized_modules = _parameterized_module_names(model)
    if personal_layers <= 0:
        raise ValueError("fedper-personal-layers must be positive")
    if personal_layers > len(parameterized_modules):
        raise ValueError(
            "fedper-personal-layers cannot exceed the number of parameterized modules"
        )

    personal_module_names = tuple(parameterized_modules[-personal_layers:])
    personal_indices: list[int] = []
    shared_indices: list[int] = []
    for index, key in enumerate(_state_keys(model)):
        if key.startswith(tuple(f"{name}." for name in personal_module_names)):
            personal_indices.append(index)
        else:
            shared_indices.append(index)

    if not shared_indices or not personal_indices:
        raise ValueError("FedPer requires both shared and personal parameters")
    return shared_indices, personal_indices


def get_indexed_model_parameters(
    model: torch.nn.Module,
    indices: list[int],
) -> list[np.ndarray]:
    arrays = get_model_state_arrays(model)
    return [arrays[index] for index in indices]


def set_indexed_model_parameters(
    model: torch.nn.Module,
    indices: list[int],
    parameters: list[np.ndarray],
) -> None:
    if len(indices) != len(parameters):
        raise ValueError("parameter count does not match selected FedPer indices")

    state = OrderedDict(model.state_dict())
    keys = _state_keys(model)
    for index, value in zip(indices, parameters):
        key = keys[index]
        current = state[key]
        state[key] = torch.as_tensor(value, dtype=current.dtype)
    model.load_state_dict(state, strict=True)


def combine_indexed_parameters(
    model: torch.nn.Module,
    indices: list[int],
    parameters: list[np.ndarray],
) -> list[np.ndarray]:
    arrays = get_model_state_arrays(model)
    for index, value in zip(indices, parameters):
        arrays[index] = value
    return arrays


class FedPerStrategy(FedAvg):
    """FedAvg-compatible strategy that aggregates only shared/base parameters."""

    def __init__(
        self,
        *,
        shared_parameter_indices: list[int],
        personal_parameter_indices: list[int],
        checkpoint_model: torch.nn.Module,
        output_dir: str,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.shared_parameter_indices = shared_parameter_indices
        self.personal_parameter_indices = personal_parameter_indices
        self._checkpoint_model = checkpoint_model
        self._output_dir = output_dir

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        weighted_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        aggregated_shared = aggregate(weighted_results)
        set_indexed_model_parameters(
            self._checkpoint_model,
            self.shared_parameter_indices,
            aggregated_shared,
        )
        save_model(self._checkpoint_model, self._output_dir, "final_model.pt")
        save_model(self._checkpoint_model, self._output_dir, f"round_{server_round}_model.pt")

        metrics_aggregated: dict[str, Scalar] = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(fit_res.num_examples, fit_res.metrics) for _, fit_res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        return ndarrays_to_parameters(aggregated_shared), metrics_aggregated

    def evaluate(
        self,
        server_round: int,
        parameters: Parameters,
    ) -> tuple[float, dict[str, Scalar]] | None:
        if self.evaluate_fn is None:
            return None

        shared_parameters = parameters_to_ndarrays(parameters)
        full_parameters = combine_indexed_parameters(
            self._checkpoint_model,
            self.shared_parameter_indices,
            shared_parameters,
        )
        return self.evaluate_fn(server_round, full_parameters, {})


class FedPerBuilder:
    name = "fedper"

    def build_strategy(
        self,
        config: ExperimentConfig,
        initial_model: torch.nn.Module,
        evaluate_fn: ServerEvaluateFn | None,
    ) -> FedPerStrategy:
        min_fit_clients = max(1, math.ceil(config.num_supernodes * config.fraction_train))
        min_eval_clients = max(1, math.ceil(config.num_supernodes * config.fraction_evaluate))
        shared_indices, personal_indices = split_fedper_parameter_indices(
            initial_model,
            config.fedper_personal_layers,
        )
        initial_parameters = ndarrays_to_parameters(
            get_indexed_model_parameters(initial_model, shared_indices)
        )

        return FedPerStrategy(
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
            shared_parameter_indices=shared_indices,
            personal_parameter_indices=personal_indices,
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
                "fedper_personal_layers": config.fedper_personal_layers,
            }

        return fn
