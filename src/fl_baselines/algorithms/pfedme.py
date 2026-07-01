"""pFedMe algorithm builder."""

from __future__ import annotations

import math

import torch
from flwr.common import Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate

from fl_baselines.clients.torch_client import get_model_parameters
from fl_baselines.core.config import ExperimentConfig
from fl_baselines.core.types import ServerEvaluateFn
from fl_baselines.logging.checkpointing import save_round_checkpoints
from fl_baselines.logging.metrics import weighted_average


class PFedMeStrategy(FedAvg):
    def __init__(
        self,
        *,
        beta: float,
        checkpoint_model: torch.nn.Module,
        output_dir: str,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        if self.initial_parameters is None:
            raise ValueError("PFedMeStrategy requires initial_parameters")
        if not 0 < beta <= 1:
            raise ValueError("PFedMe beta must be in (0, 1]")
        self.beta = beta
        self.global_parameters = parameters_to_ndarrays(self.initial_parameters)
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

        previous_global = [array.copy() for array in self.global_parameters]
        client_models = [
            parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results
        ]
        weights = [fit_res.num_examples / total_examples for _, fit_res in results]
        averaged_model = aggregate(list(zip(client_models, weights)))

        self.global_parameters = [
            (1.0 - self.beta) * global_array + self.beta * averaged_array
            for global_array, averaged_array in zip(previous_global, averaged_model)
        ]
        aggregated_parameters = ndarrays_to_parameters(self.global_parameters)

        metrics_aggregated: dict[str, Scalar] = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(fit_res.num_examples, fit_res.metrics) for _, fit_res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        save_round_checkpoints(
            self._checkpoint_model,
            aggregated_parameters,
            self._output_dir,
            server_round,
        )
        return aggregated_parameters, metrics_aggregated


class PFedMeBuilder:
    name = "pfedme"

    def build_strategy(
        self,
        config: ExperimentConfig,
        initial_model: torch.nn.Module,
        evaluate_fn: ServerEvaluateFn | None,
    ) -> PFedMeStrategy:
        min_fit_clients = max(1, math.ceil(config.num_supernodes * config.fraction_train))
        min_eval_clients = max(1, math.ceil(config.num_supernodes * config.fraction_evaluate))
        initial_parameters = ndarrays_to_parameters(get_model_parameters(initial_model))

        return PFedMeStrategy(
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
            checkpoint_model=initial_model,
            output_dir=config.output_dir,
            beta=config.pfedme_beta,
        )

    def _fit_config(self, config: ExperimentConfig):
        def fn(server_round: int) -> dict[str, bool | bytes | float | int | str]:
            return {
                "algorithm": self.name,
                "server_round": server_round,
                "local_epochs": config.local_epochs,
                "learning_rate": config.learning_rate,
                "pfedme_lambda": config.pfedme_lambda,
                "pfedme_beta": config.pfedme_beta,
                "pfedme_personal_learning_rate": config.pfedme_personal_learning_rate,
                "pfedme_personal_steps": config.pfedme_personal_steps,
            }

        return fn
