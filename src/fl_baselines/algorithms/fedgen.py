"""FedGen algorithm builder."""

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
from fl_baselines.training.features import infer_feature_dim


class FedGENStrategy(FedAvg):
    """FedAvg-compatible strategy that aggregates both model parameters and feature masks."""

    def __init__(
        self,
        *,
        model_parameter_count: int,
        initial_mask: np.ndarray,
        checkpoint_model: torch.nn.Module,
        output_dir: str,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.model_parameter_count = model_parameter_count
        self.global_mask = initial_mask.copy()
        self._checkpoint_model = checkpoint_model
        self._output_dir = output_dir

    def aggregate_fit(self, server_round, results, failures):
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round,
            results,
            failures,
        )
        if parameters_aggregated is None:
            return None, metrics_aggregated

        arrays = parameters_to_ndarrays(parameters_aggregated)
        if len(arrays) <= self.model_parameter_count:
            raise ValueError("FedGEN aggregated payload is missing the global mask")
        self.global_mask = arrays[self.model_parameter_count].copy()

        if metrics_aggregated is None:
            metrics_aggregated = {}
        metrics_aggregated = dict(metrics_aggregated)
        metrics_aggregated["fedgen_mask_mean"] = float(
            torch.sigmoid(torch.as_tensor(self.global_mask, dtype=torch.float32)).mean().item()
        )

        save_round_checkpoints(
            self._checkpoint_model,
            parameters_aggregated,
            self._output_dir,
            server_round,
        )
        return parameters_aggregated, metrics_aggregated


class FedGENBuilder:
    name = "fedgen"

    def build_strategy(
        self,
        config: ExperimentConfig,
        initial_model: torch.nn.Module,
        evaluate_fn: ServerEvaluateFn | None,
    ) -> FedGENStrategy:
        min_fit_clients = max(1, math.ceil(config.num_supernodes * config.fraction_train))
        min_eval_clients = max(1, math.ceil(config.num_supernodes * config.fraction_evaluate))
        model_parameters = get_model_parameters(initial_model)
        feature_dim = infer_feature_dim(initial_model, config)
        initial_mask = np.ones(feature_dim, dtype=np.float32)
        initial_parameters = ndarrays_to_parameters(model_parameters + [initial_mask.copy()])

        return FedGENStrategy(
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
            model_parameter_count=len(model_parameters),
            initial_mask=initial_mask,
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
                "fedgen_alpha": config.fedgen_alpha,
                "fedgen_lambda": config.fedgen_lambda,
                "fedgen_beta": config.fedgen_beta,
                "fedgen_delta": config.fedgen_delta,
                "fedgen_warmup_epochs": config.fedgen_warmup_epochs,
                "fedgen_l1_weight": config.fedgen_l1_weight,
            }

        return fn
