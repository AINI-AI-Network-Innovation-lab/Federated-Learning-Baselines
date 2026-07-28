"""Faithful FedADMM strategy and builder."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from flwr.common import Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg

from fl_baselines.clients.torch_client import get_model_parameters
from fl_baselines.core.config import ExperimentConfig
from fl_baselines.core.types import ServerEvaluateFn
from fl_baselines.logging.checkpointing import save_round_checkpoints
from fl_baselines.logging.metrics import weighted_average
from fl_baselines.training.proximal import ProxOperator, build_prox_operator


class FedADMMStrategy(FedAvg):
    """FedADMM Algorithm 2 with partial participation and server prox."""

    def __init__(
        self,
        *,
        alpha: float,
        num_total_clients: int,
        prox_operator: ProxOperator,
        checkpoint_model: torch.nn.Module,
        output_dir: str,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        if self.initial_parameters is None:
            raise ValueError("FedADMMStrategy requires initial_parameters")
        if alpha <= 0:
            raise ValueError("FedADMM penalty must be positive")
        if num_total_clients <= 0:
            raise ValueError("FedADMM num_total_clients must be positive")

        self.alpha = alpha
        self.num_total_clients = num_total_clients
        self.prox_operator = prox_operator
        self._checkpoint_model = checkpoint_model
        self._output_dir = output_dir
        self.model_parameter_count = len(parameters_to_ndarrays(self.initial_parameters))
        initial_model = parameters_to_ndarrays(self.initial_parameters)
        initial_bar = prox_operator(initial_model, alpha)
        self.tilde_parameters = [array.copy() for array in initial_bar]
        self.client_hat_models: dict[str, list[np.ndarray]] = {
            str(client_index): [array.copy() for array in initial_bar]
            for client_index in range(num_total_clients)
        }
        self.bar_parameters = [array.copy() for array in initial_bar]
        self.initial_parameters = ndarrays_to_parameters(self.bar_parameters)
        self._load_server_state()
        self.initial_parameters = ndarrays_to_parameters(self.bar_parameters)

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        seen_client_ids: set[str] = set()
        delta_sum = [np.zeros_like(value) for value in self.tilde_parameters]
        for _, fit_res in results:
            raw_client_id = fit_res.metrics.get("fedadmm_client_id")
            if not isinstance(raw_client_id, str) or not raw_client_id:
                raise ValueError("FedADMM client result is missing fedadmm_client_id")
            client_id = raw_client_id
            if client_id in seen_client_ids:
                raise ValueError(f"Duplicate FedADMM client id '{client_id}'")
            if client_id not in self.client_hat_models:
                raise ValueError(
                    f"Unknown FedADMM client id '{client_id}'; expected configured client ids"
                )
            seen_client_ids.add(client_id)

            delta_arrays = parameters_to_ndarrays(fit_res.parameters)
            self._validate_payload(delta_arrays)
            for index, delta in enumerate(delta_arrays):
                self.client_hat_models[client_id][index] += delta
                delta_sum[index] += delta

        scale = 1.0 / float(self.num_total_clients)
        self.tilde_parameters = [
            value + scale * delta
            for value, delta in zip(self.tilde_parameters, delta_sum)
        ]
        self.bar_parameters = self.prox_operator(self.tilde_parameters, self.alpha)
        aggregated_parameters = ndarrays_to_parameters(self.bar_parameters)

        metrics_aggregated: dict[str, Scalar] = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(fit_res.num_examples, fit_res.metrics) for _, fit_res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        metrics_aggregated["fedadmm_selected_clients"] = len(seen_client_ids)
        metrics_aggregated["fedadmm_consensus_norm"] = float(
            np.sqrt(sum(np.sum(array.astype(np.float64) ** 2) for array in self.bar_parameters))
        )

        self._save_server_state()
        save_round_checkpoints(
            self._checkpoint_model,
            aggregated_parameters,
            self._output_dir,
            server_round,
        )
        return aggregated_parameters, metrics_aggregated

    def _validate_payload(self, arrays: list[np.ndarray]) -> None:
        if len(arrays) != self.model_parameter_count:
            raise ValueError(
                "FedADMM client result must contain exactly one delta-hat tensor per model tensor"
            )
        reference = self.tilde_parameters
        for index, (array, expected) in enumerate(zip(arrays, reference)):
            if array.shape != expected.shape:
                raise ValueError(
                    f"FedADMM payload tensor {index} has shape {array.shape}; "
                    f"expected {expected.shape}"
                )

    def _server_state_path(self) -> Path:
        return Path(self._output_dir) / "fedadmm_server" / "state.pt"

    def _save_server_state(self) -> None:
        state_path = self._server_state_path()
        state_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "version": 1,
                "tilde": [torch.from_numpy(value.copy()) for value in self.tilde_parameters],
                "bar": [torch.from_numpy(value.copy()) for value in self.bar_parameters],
                "client_hat_models": {
                    client_id: [torch.from_numpy(value.copy()) for value in values]
                    for client_id, values in self.client_hat_models.items()
                },
            },
            state_path,
        )

    def _load_server_state(self) -> None:
        state_path = self._server_state_path()
        if not state_path.exists():
            return
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or state.get("version") != 1:
            raise ValueError("FedADMM server state file has an invalid schema")
        client_hat_models = state.get("client_hat_models")
        if not isinstance(client_hat_models, dict) or set(client_hat_models) != set(self.client_hat_models):
            raise ValueError("FedADMM server state client set does not match configuration")
        tilde = state.get("tilde")
        bar = state.get("bar")
        if not isinstance(tilde, list) or not isinstance(bar, list):
            raise ValueError("FedADMM server state is missing model tensors")
        tilde_arrays = [value.detach().cpu().numpy() for value in tilde]
        bar_arrays = [value.detach().cpu().numpy() for value in bar]
        self._validate_payload(tilde_arrays)
        self._validate_payload(bar_arrays)
        self.tilde_parameters = [value.copy() for value in tilde_arrays]
        self.bar_parameters = [value.copy() for value in bar_arrays]
        self.client_hat_models = {
            client_id: [value.detach().cpu().numpy().copy() for value in values]
            for client_id, values in client_hat_models.items()
        }


class FedADMMBuilder:
    name = "fedadmm"

    def build_strategy(
        self,
        config: ExperimentConfig,
        initial_model: torch.nn.Module,
        evaluate_fn: ServerEvaluateFn | None,
    ) -> FedADMMStrategy:
        min_fit_clients = max(1, math.ceil(config.num_supernodes * config.fraction_train))
        min_eval_clients = max(1, math.ceil(config.num_supernodes * config.fraction_evaluate))
        initial_parameters = ndarrays_to_parameters(get_model_parameters(initial_model))
        prox_operator = build_prox_operator(
            config.fedadmm_prox,
            l1_weight=config.fedadmm_l1_weight,
            box_min=config.fedadmm_box_min,
            box_max=config.fedadmm_box_max,
        )

        return FedADMMStrategy(
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
            alpha=config.fedadmm_penalty,
            num_total_clients=config.num_supernodes,
            prox_operator=prox_operator,
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
                "fedadmm_alpha": config.fedadmm_alpha,
                "fedadmm_penalty": config.fedadmm_penalty,
                "fedadmm_local_steps": config.fedadmm_local_steps,
                "fedadmm_tolerance": config.fedadmm_tolerance,
                "fedadmm_prox": config.fedadmm_prox,
            }

        return fn
