"""FedLAW algorithm builder."""

from __future__ import annotations

import copy
import math
from collections import OrderedDict

import numpy as np
import torch
from flwr.common import Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg
from torch import nn
from torch.func import functional_call
from torch.utils.data import DataLoader

from fl_baselines.clients.torch_client import get_model_parameters
from fl_baselines.core.config import ExperimentConfig
from fl_baselines.core.types import ServerEvaluateFn
from fl_baselines.logging.checkpointing import save_round_checkpoints
from fl_baselines.logging.metrics import weighted_average
from fl_baselines.training.evaluate import _logits_from_output


class FedLAWStrategy(FedAvg):
    """FedLAW strategy with proxy-data-learned aggregation weights."""

    def __init__(
        self,
        *,
        server_epochs: int,
        server_learning_rate: float,
        gamma_init: float,
        checkpoint_model: torch.nn.Module,
        output_dir: str,
        device: str,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        if self.initial_parameters is None:
            raise ValueError("FedLAWStrategy requires initial_parameters")
        self.server_epochs = server_epochs
        self.server_learning_rate = server_learning_rate
        self.gamma_init = gamma_init
        self.device = device
        self.proxy_loader: DataLoader | None = None
        self.last_lambda: list[float] = []
        self.last_gamma = gamma_init
        self.last_proxy_loss = 0.0
        self._checkpoint_model = checkpoint_model
        self._output_dir = output_dir
        self._reference_state = OrderedDict(
            (name, tensor.detach().cpu().clone())
            for name, tensor in checkpoint_model.state_dict().items()
        )
        self._parameter_names = set(dict(checkpoint_model.named_parameters()).keys())
        self._proxy_model = copy.deepcopy(checkpoint_model).to(device)

    def set_proxy_loader(self, proxy_loader: DataLoader) -> None:
        self.proxy_loader = proxy_loader

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        total_examples = sum(fit_res.num_examples for _, fit_res in results)
        if total_examples <= 0:
            return None, {}

        client_models = [
            parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results
        ]
        sample_weights = [
            fit_res.num_examples / total_examples for _, fit_res in results
        ]
        lambda_weights = sample_weights
        gamma = self.gamma_init

        learned = self._learn_proxy_weights(client_models, sample_weights)
        if learned is not None:
            lambda_weights, gamma, self.last_proxy_loss = learned
        else:
            self.last_proxy_loss = 0.0

        self.last_lambda = lambda_weights
        self.last_gamma = gamma
        aggregated_ndarrays = self._aggregate_models(client_models, lambda_weights, gamma)
        aggregated_parameters = ndarrays_to_parameters(aggregated_ndarrays)

        metrics_aggregated: dict[str, Scalar] = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [
                (fit_res.num_examples, fit_res.metrics) for _, fit_res in results
            ]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        metrics_aggregated["fedlaw_gamma"] = float(gamma)
        metrics_aggregated["fedlaw_proxy_loss"] = float(self.last_proxy_loss)
        metrics_aggregated["fedlaw_lambda_entropy"] = float(
            -sum(weight * math.log(max(weight, 1e-12)) for weight in lambda_weights)
        )

        save_round_checkpoints(
            self._checkpoint_model,
            aggregated_parameters,
            self._output_dir,
            server_round,
        )
        return aggregated_parameters, metrics_aggregated

    def _learn_proxy_weights(
        self,
        client_models: list[list[np.ndarray]],
        sample_weights: list[float],
    ) -> tuple[list[float], float, float] | None:
        if self.proxy_loader is None:
            return None

        client_tensors = [
            [torch.as_tensor(array, device=self.device) for array in client_model]
            for client_model in client_models
        ]
        lambda_logits = nn.Parameter(
            torch.log(
                torch.as_tensor(sample_weights, dtype=torch.float32, device=self.device)
                + 1e-12
            )
        )
        gamma_raw = nn.Parameter(
            torch.log(torch.tensor(self.gamma_init, dtype=torch.float32, device=self.device))
        )
        optimizer = torch.optim.Adam(
            [lambda_logits, gamma_raw],
            lr=self.server_learning_rate,
        )
        criterion = nn.CrossEntropyLoss()
        saw_batch = False
        last_loss = 0.0

        self._proxy_model.train()
        for _ in range(self.server_epochs):
            for inputs, targets in self.proxy_loader:
                saw_batch = True
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                optimizer.zero_grad()
                lambda_weights = torch.softmax(lambda_logits, dim=0)
                gamma = torch.exp(gamma_raw)
                state = self._aggregate_state_tensors(
                    client_tensors,
                    lambda_weights,
                    gamma,
                )
                outputs = _logits_from_output(
                    functional_call(self._proxy_model, state, (inputs,))
                )
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                last_loss = float(loss.detach().item())

        if not saw_batch:
            return None

        with torch.no_grad():
            lambda_weights = torch.softmax(lambda_logits, dim=0).cpu().tolist()
            gamma = float(torch.exp(gamma_raw).cpu().item())
        return lambda_weights, gamma, last_loss

    def _aggregate_state_tensors(
        self,
        client_tensors: list[list[torch.Tensor]],
        lambda_weights: torch.Tensor,
        gamma: torch.Tensor,
    ) -> OrderedDict[str, torch.Tensor]:
        aggregated_state: OrderedDict[str, torch.Tensor] = OrderedDict()

        for index, (name, reference_tensor) in enumerate(self._reference_state.items()):
            stacked = torch.stack(
                [client_tensor[index].to(device=self.device) for client_tensor in client_tensors],
                dim=0,
            )
            if reference_tensor.is_floating_point():
                weighted = torch.tensordot(lambda_weights, stacked, dims=([0], [0]))
                if name in self._parameter_names:
                    weighted = gamma * weighted
                aggregated_state[name] = weighted.to(dtype=reference_tensor.dtype)
                continue

            weighted = torch.tensordot(
                lambda_weights.detach(),
                stacked.to(dtype=torch.float32),
                dims=([0], [0]),
            )
            aggregated_state[name] = weighted.round().to(dtype=reference_tensor.dtype)

        return aggregated_state

    def _aggregate_models(
        self,
        client_models: list[list[np.ndarray]],
        lambda_weights: list[float],
        gamma: float,
    ) -> list[np.ndarray]:
        aggregated: list[np.ndarray] = []
        weight_array = np.asarray(lambda_weights, dtype=np.float64)

        for index, (name, reference_tensor) in enumerate(self._reference_state.items()):
            stacked = np.stack([client_model[index] for client_model in client_models], axis=0)
            weighted = np.tensordot(weight_array, stacked, axes=(0, 0))
            if np.issubdtype(reference_tensor.numpy().dtype, np.floating):
                if name in self._parameter_names:
                    weighted = gamma * weighted
                aggregated.append(weighted.astype(reference_tensor.numpy().dtype, copy=False))
                continue
            aggregated.append(np.rint(weighted).astype(reference_tensor.numpy().dtype, copy=False))

        return aggregated


class FedLAWBuilder:
    name = "fedlaw"

    def build_strategy(
        self,
        config: ExperimentConfig,
        initial_model: torch.nn.Module,
        evaluate_fn: ServerEvaluateFn | None,
    ) -> FedLAWStrategy:
        min_fit_clients = max(1, math.ceil(config.num_supernodes * config.fraction_train))
        min_eval_clients = max(1, math.ceil(config.num_supernodes * config.fraction_evaluate))
        initial_parameters = ndarrays_to_parameters(get_model_parameters(initial_model))

        return FedLAWStrategy(
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
            server_epochs=config.fedlaw_server_epochs,
            server_learning_rate=config.fedlaw_server_learning_rate,
            gamma_init=config.fedlaw_gamma_init,
            checkpoint_model=initial_model,
            output_dir=config.output_dir,
            device=config.device,
        )

    def _fit_config(self, config: ExperimentConfig):
        def fn(server_round: int) -> dict[str, bool | bytes | float | int | str]:
            return {
                "algorithm": self.name,
                "server_round": server_round,
                "local_epochs": config.local_epochs,
                "learning_rate": config.learning_rate,
                "fedlaw_server_epochs": config.fedlaw_server_epochs,
                "fedlaw_server_learning_rate": config.fedlaw_server_learning_rate,
                "fedlaw_gamma_init": config.fedlaw_gamma_init,
            }

        return fn
