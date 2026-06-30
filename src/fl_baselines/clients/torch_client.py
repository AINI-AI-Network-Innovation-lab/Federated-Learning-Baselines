"""Flower NumPyClient wrapper around PyTorch training code."""

from __future__ import annotations

import copy
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from flwr.client import NumPyClient
from torch import nn

from fl_baselines.core.config import ExperimentConfig
from fl_baselines.core.types import ClientDataLoaders
from fl_baselines.training.evaluate import evaluate_model
from fl_baselines.training.moon import train_moon_client
from fl_baselines.training.scaffold import train_scaffold_client
from fl_baselines.training.train import train_one_client


_SCAFFOLD_CLIENT_CONTROLS: dict[str, list[np.ndarray]] = {}


def get_model_parameters(model: nn.Module) -> list[np.ndarray]:
    return [value.detach().cpu().numpy() for value in model.state_dict().values()]


def set_model_parameters(model: nn.Module, parameters: list[np.ndarray]) -> None:
    state_dict = OrderedDict(
        (key, torch.tensor(value)) for key, value in zip(model.state_dict().keys(), parameters)
    )
    model.load_state_dict(state_dict, strict=True)


class TorchFlowerClient(NumPyClient):
    def __init__(
        self,
        model: nn.Module,
        loaders: ClientDataLoaders,
        config: ExperimentConfig,
        client_id: str = "default",
    ) -> None:
        self.model = model
        self.loaders = loaders
        self.config = config
        self.client_id = client_id

    def get_parameters(
        self,
        config: dict[str, bool | bytes | float | int | str],
    ) -> list[np.ndarray]:
        return get_model_parameters(self.model)

    def fit(
        self,
        parameters: list[np.ndarray],
        config: dict[str, bool | bytes | float | int | str],
    ) -> tuple[list[np.ndarray], int, dict[str, bool | bytes | float | int | str]]:
        algorithm = str(config.get("algorithm", self.config.algorithm))
        if algorithm == "scaffold":
            return self._fit_scaffold(parameters, config)
        if algorithm == "moon":
            return self._fit_moon(parameters, config)

        set_model_parameters(self.model, parameters)
        local_epochs = int(config.get("local_epochs", self.config.local_epochs))
        learning_rate = float(config.get("learning_rate", self.config.learning_rate))
        proximal_mu = float(config.get("proximal_mu", 0.0))
        metrics = train_one_client(
            self.model,
            self.loaders.train,
            epochs=local_epochs,
            learning_rate=learning_rate,
            device=self.config.device,
            proximal_mu=proximal_mu,
        )
        return get_model_parameters(self.model), len(self.loaders.train.dataset), metrics

    def _fit_scaffold(
        self,
        parameters: list[np.ndarray],
        config: dict[str, bool | bytes | float | int | str],
    ) -> tuple[list[np.ndarray], int, dict[str, bool | bytes | float | int | str]]:
        model_parameter_count = len(get_model_parameters(self.model))
        if len(parameters) <= model_parameter_count:
            raise ValueError("SCAFFOLD fit requires model parameters and server controls")

        model_parameters = parameters[:model_parameter_count]
        server_control = parameters[model_parameter_count:]
        set_model_parameters(self.model, model_parameters)

        client_control = _SCAFFOLD_CLIENT_CONTROLS.get(self.client_id)
        if client_control is None:
            client_control = [np.zeros_like(control) for control in server_control]

        local_epochs = int(config.get("local_epochs", self.config.local_epochs))
        learning_rate = float(config.get("learning_rate", self.config.learning_rate))
        metrics, new_control, control_delta = train_scaffold_client(
            self.model,
            self.loaders.train,
            epochs=local_epochs,
            learning_rate=learning_rate,
            device=self.config.device,
            server_control=server_control,
            client_control=client_control,
        )
        _SCAFFOLD_CLIENT_CONTROLS[self.client_id] = [
            control.copy() for control in new_control
        ]
        return (
            get_model_parameters(self.model) + control_delta,
            len(self.loaders.train.dataset),
            metrics,
        )

    def _fit_moon(
        self,
        parameters: list[np.ndarray],
        config: dict[str, bool | bytes | float | int | str],
    ) -> tuple[list[np.ndarray], int, dict[str, bool | bytes | float | int | str]]:
        set_model_parameters(self.model, parameters)
        global_model = copy.deepcopy(self.model)
        previous_model = copy.deepcopy(self.model)
        previous_model_path = self._moon_previous_model_path()
        if previous_model_path.exists():
            previous_state = torch.load(
                previous_model_path,
                map_location="cpu",
                weights_only=True,
            )
            previous_model.load_state_dict(previous_state, strict=True)

        local_epochs = int(config.get("local_epochs", self.config.local_epochs))
        learning_rate = float(config.get("learning_rate", self.config.learning_rate))
        moon_mu = float(config.get("moon_mu", self.config.moon_mu))
        temperature = float(
            config.get("moon_temperature", self.config.moon_temperature)
        )
        metrics = train_moon_client(
            self.model,
            global_model,
            previous_model,
            self.loaders.train,
            epochs=local_epochs,
            learning_rate=learning_rate,
            device=self.config.device,
            moon_mu=moon_mu,
            temperature=temperature,
        )

        previous_model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), previous_model_path)
        return get_model_parameters(self.model), len(self.loaders.train.dataset), metrics

    def _moon_previous_model_path(self) -> Path:
        return (
            Path(self.config.output_dir)
            / "moon_clients"
            / self.client_id
            / "previous_model.pt"
        )

    def evaluate(
        self,
        parameters: list[np.ndarray],
        config: dict[str, bool | bytes | float | int | str],
    ) -> tuple[float, int, dict[str, bool | bytes | float | int | str]]:
        set_model_parameters(self.model, parameters)
        loss, metrics = evaluate_model(
            self.model,
            self.loaders.validation,
            device=self.config.device,
        )
        return loss, len(self.loaders.validation.dataset), metrics
