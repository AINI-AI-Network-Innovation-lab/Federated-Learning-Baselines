"""FedNP algorithm builder."""

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


class FedNPStrategy(FedAvg):
    """FedAvg-compatible strategy with latent Gaussian statistics aggregation."""

    def __init__(
        self,
        *,
        latent_mean: np.ndarray,
        latent_var: np.ndarray,
        prior_variance: float,
        model_parameter_count: int,
        checkpoint_model: torch.nn.Module,
        output_dir: str,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.latent_mean = latent_mean.copy()
        self.latent_var = latent_var.copy()
        self.prior_variance = prior_variance
        self.model_parameter_count = model_parameter_count
        self._checkpoint_model = checkpoint_model
        self._output_dir = output_dir

    def configure_fit(self, server_round, parameters, client_manager):
        client_fit_ins = super().configure_fit(server_round, parameters, client_manager)
        model_parameters = parameters_to_ndarrays(parameters)
        payload = ndarrays_to_parameters(
            model_parameters + [self.latent_mean.copy(), self.latent_var.copy()]
        )
        return [
            (client, FitIns(payload, fit_ins.config))
            for client, fit_ins in client_fit_ins
        ]

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        model_results = []
        latent_sums = []
        latent_square_sums = []
        latent_counts = []
        for _, fit_res in results:
            arrays = parameters_to_ndarrays(fit_res.parameters)
            if len(arrays) != self.model_parameter_count + 3:
                raise ValueError("FedNP client result is missing latent statistics payload")
            model_results.append((arrays[: self.model_parameter_count], fit_res.num_examples))
            latent_sums.append(arrays[-3])
            latent_square_sums.append(arrays[-2])
            latent_counts.append(float(arrays[-1][0]))

        aggregated_model = aggregate(model_results)
        self._aggregate_latent_statistics(latent_sums, latent_square_sums, latent_counts)
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

    def _aggregate_latent_statistics(
        self,
        latent_sums: list[np.ndarray],
        latent_square_sums: list[np.ndarray],
        latent_counts: list[float],
    ) -> None:
        total_count = float(sum(latent_counts))
        if total_count <= 0:
            return
        total_sum = np.sum(np.stack(latent_sums, axis=0), axis=0)
        total_square_sum = np.sum(np.stack(latent_square_sums, axis=0), axis=0)
        self.latent_mean = total_sum / total_count
        second_moment = total_square_sum / total_count
        self.latent_var = np.maximum(
            second_moment - np.square(self.latent_mean),
            np.full_like(self.latent_mean, self.prior_variance * 1e-6),
        )


class FedNPBuilder:
    name = "fednp"

    def build_strategy(
        self,
        config: ExperimentConfig,
        initial_model: torch.nn.Module,
        evaluate_fn: ServerEvaluateFn | None,
    ) -> FedNPStrategy:
        min_fit_clients = max(1, math.ceil(config.num_supernodes * config.fraction_train))
        min_eval_clients = max(1, math.ceil(config.num_supernodes * config.fraction_evaluate))
        model_parameters = get_model_parameters(initial_model)
        initial_parameters = ndarrays_to_parameters(model_parameters)
        feature_dim = infer_feature_dim(initial_model, config)

        return FedNPStrategy(
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
            latent_mean=np.zeros(feature_dim, dtype=np.float32),
            latent_var=np.full(feature_dim, config.fednp_prior_variance, dtype=np.float32),
            prior_variance=config.fednp_prior_variance,
            model_parameter_count=len(model_parameters),
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
                "fednp_lambda": config.fednp_lambda,
                "fednp_prior_variance": config.fednp_prior_variance,
                "fednp_stability_eps": config.fednp_stability_eps,
            }

        return fn
