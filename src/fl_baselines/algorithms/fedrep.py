"""FedRep algorithm builder."""

from __future__ import annotations

import math

import torch
from flwr.common import ndarrays_to_parameters

from fl_baselines.algorithms.fedper import (
    FedPerStrategy,
    get_indexed_model_parameters,
    split_fedper_parameter_indices,
)
from fl_baselines.core.config import ExperimentConfig
from fl_baselines.core.types import ServerEvaluateFn
from fl_baselines.logging.metrics import weighted_average


class FedRepStrategy(FedPerStrategy):
    """FedPer-style shared representation aggregation for FedRep."""

    def __repr__(self) -> str:
        return f"FedRepStrategy(accept_failures={self.accept_failures})"


class FedRepBuilder:
    name = "fedrep"

    def build_strategy(
        self,
        config: ExperimentConfig,
        initial_model: torch.nn.Module,
        evaluate_fn: ServerEvaluateFn | None,
    ) -> FedRepStrategy:
        min_fit_clients = max(1, math.ceil(config.num_supernodes * config.fraction_train))
        min_eval_clients = max(1, math.ceil(config.num_supernodes * config.fraction_evaluate))
        shared_indices, personal_indices = split_fedper_parameter_indices(
            initial_model,
            config.fedrep_personal_layers,
        )
        initial_parameters = ndarrays_to_parameters(
            get_indexed_model_parameters(initial_model, shared_indices)
        )

        return FedRepStrategy(
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
                "fedrep_personal_layers": config.fedrep_personal_layers,
                "fedrep_representation_epochs": config.fedrep_representation_epochs,
            }

        return fn
