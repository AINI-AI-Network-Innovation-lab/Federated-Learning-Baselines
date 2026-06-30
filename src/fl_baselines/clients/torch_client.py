"""Flower NumPyClient wrapper around PyTorch training code."""

from __future__ import annotations

import copy
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from flwr.client import NumPyClient
from torch import nn

from fl_baselines.algorithms.fedper import (
    get_indexed_model_parameters,
    set_indexed_model_parameters,
    split_fedper_parameter_indices,
)
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
        if algorithm == "fednova":
            return self._fit_fednova(parameters, config)
        if algorithm == "fedper":
            return self._fit_fedper(parameters, config)
        if algorithm == "fedrep":
            return self._fit_fedrep(parameters, config)
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

    def _fit_fednova(
        self,
        parameters: list[np.ndarray],
        config: dict[str, bool | bytes | float | int | str],
    ) -> tuple[list[np.ndarray], int, dict[str, bool | bytes | float | int | str]]:
        set_model_parameters(self.model, parameters)
        initial_parameters = [parameter.copy() for parameter in parameters]

        local_epochs = int(config.get("local_epochs", self.config.local_epochs))
        learning_rate = float(config.get("learning_rate", self.config.learning_rate))
        metrics = train_one_client(
            self.model,
            self.loaders.train,
            epochs=local_epochs,
            learning_rate=learning_rate,
            device=self.config.device,
        )
        updated_parameters = get_model_parameters(self.model)
        updates = [
            initial_parameter - updated_parameter
            for initial_parameter, updated_parameter in zip(
                initial_parameters,
                updated_parameters,
            )
        ]
        local_norm = float(local_epochs * len(self.loaders.train))
        metrics["local_norm"] = local_norm
        metrics["tau"] = local_norm
        return updates, len(self.loaders.train.dataset), metrics

    def _fit_fedper(
        self,
        parameters: list[np.ndarray],
        config: dict[str, bool | bytes | float | int | str],
    ) -> tuple[list[np.ndarray], int, dict[str, bool | bytes | float | int | str]]:
        shared_indices, personal_indices = self._fedper_indices(config)
        set_indexed_model_parameters(self.model, shared_indices, parameters)
        self._load_fedper_personal_parameters(personal_indices)

        local_epochs = int(config.get("local_epochs", self.config.local_epochs))
        learning_rate = float(config.get("learning_rate", self.config.learning_rate))
        metrics = train_one_client(
            self.model,
            self.loaders.train,
            epochs=local_epochs,
            learning_rate=learning_rate,
            device=self.config.device,
        )
        self._save_fedper_personal_parameters(personal_indices)
        shared_parameters = get_indexed_model_parameters(self.model, shared_indices)
        return shared_parameters, len(self.loaders.train.dataset), metrics

    def _fedper_indices(
        self,
        config: dict[str, bool | bytes | float | int | str],
    ) -> tuple[list[int], list[int]]:
        personal_layers = int(
            config.get(
                "fedper_personal_layers",
                self.config.fedper_personal_layers,
            )
        )
        return split_fedper_parameter_indices(self.model, personal_layers)

    def _fedper_personal_model_path(self) -> Path:
        return (
            Path(self.config.output_dir)
            / "fedper_clients"
            / self.client_id
            / "personal.pt"
        )

    def _load_fedper_personal_parameters(self, personal_indices: list[int]) -> None:
        personal_path = self._fedper_personal_model_path()
        if not personal_path.exists():
            return
        personal_parameters = torch.load(
            personal_path,
            map_location="cpu",
            weights_only=True,
        )
        set_indexed_model_parameters(self.model, personal_indices, personal_parameters)

    def _save_fedper_personal_parameters(self, personal_indices: list[int]) -> None:
        personal_path = self._fedper_personal_model_path()
        personal_path.parent.mkdir(parents=True, exist_ok=True)
        state_values = list(self.model.state_dict().values())
        torch.save(
            [
                state_values[index].detach().cpu().clone()
                for index in personal_indices
            ],
            personal_path,
        )

    def _fit_fedrep(
        self,
        parameters: list[np.ndarray],
        config: dict[str, bool | bytes | float | int | str],
    ) -> tuple[list[np.ndarray], int, dict[str, bool | bytes | float | int | str]]:
        shared_indices, personal_indices = self._fedrep_indices(config)
        set_indexed_model_parameters(self.model, shared_indices, parameters)
        self._load_fedrep_personal_parameters(personal_indices)

        head_epochs = int(config.get("local_epochs", self.config.local_epochs))
        representation_epochs = int(
            config.get(
                "fedrep_representation_epochs",
                self.config.fedrep_representation_epochs,
            )
        )
        learning_rate = float(config.get("learning_rate", self.config.learning_rate))

        head_metrics = self._train_fedrep_phase(
            train_shared=False,
            shared_indices=shared_indices,
            epochs=head_epochs,
            learning_rate=learning_rate,
        )
        representation_metrics = self._train_fedrep_phase(
            train_shared=True,
            shared_indices=shared_indices,
            epochs=representation_epochs,
            learning_rate=learning_rate,
        )
        self._save_fedrep_personal_parameters(personal_indices)
        shared_parameters = get_indexed_model_parameters(self.model, shared_indices)
        metrics = {
            "fedrep_head_train_loss": float(head_metrics["train_loss"]),
            "fedrep_head_train_accuracy": float(head_metrics["train_accuracy"]),
            "fedrep_representation_train_loss": float(representation_metrics["train_loss"]),
            "fedrep_representation_train_accuracy": float(
                representation_metrics["train_accuracy"]
            ),
            "train_loss": float(representation_metrics["train_loss"]),
            "train_accuracy": float(representation_metrics["train_accuracy"]),
        }
        return shared_parameters, len(self.loaders.train.dataset), metrics

    def _train_fedrep_phase(
        self,
        *,
        train_shared: bool,
        shared_indices: list[int],
        epochs: int,
        learning_rate: float,
    ) -> dict[str, float]:
        self._set_fedrep_requires_grad(shared_indices, train_shared=train_shared)
        metrics = train_one_client(
            self.model,
            self.loaders.train,
            epochs=epochs,
            learning_rate=learning_rate,
            device=self.config.device,
        )
        for parameter in self.model.parameters():
            parameter.requires_grad = True
        return metrics

    def _set_fedrep_requires_grad(
        self,
        shared_indices: list[int],
        *,
        train_shared: bool,
    ) -> None:
        state_keys = list(self.model.state_dict().keys())
        shared_key_prefixes = tuple(state_keys[index] for index in shared_indices)
        for name, parameter in self.model.named_parameters():
            is_shared = name in shared_key_prefixes
            parameter.requires_grad = is_shared if train_shared else not is_shared

    def _fedrep_indices(
        self,
        config: dict[str, bool | bytes | float | int | str],
    ) -> tuple[list[int], list[int]]:
        personal_layers = int(
            config.get(
                "fedrep_personal_layers",
                self.config.fedrep_personal_layers,
            )
        )
        return split_fedper_parameter_indices(self.model, personal_layers)

    def _fedrep_personal_model_path(self) -> Path:
        return (
            Path(self.config.output_dir)
            / "fedrep_clients"
            / self.client_id
            / "personal.pt"
        )

    def _load_fedrep_personal_parameters(self, personal_indices: list[int]) -> None:
        personal_path = self._fedrep_personal_model_path()
        if not personal_path.exists():
            return
        personal_parameters = torch.load(
            personal_path,
            map_location="cpu",
            weights_only=True,
        )
        set_indexed_model_parameters(self.model, personal_indices, personal_parameters)

    def _save_fedrep_personal_parameters(self, personal_indices: list[int]) -> None:
        personal_path = self._fedrep_personal_model_path()
        personal_path.parent.mkdir(parents=True, exist_ok=True)
        state_values = list(self.model.state_dict().values())
        torch.save(
            [
                state_values[index].detach().cpu().clone()
                for index in personal_indices
            ],
            personal_path,
        )

    def evaluate(
        self,
        parameters: list[np.ndarray],
        config: dict[str, bool | bytes | float | int | str],
    ) -> tuple[float, int, dict[str, bool | bytes | float | int | str]]:
        algorithm = str(config.get("algorithm", self.config.algorithm))
        if algorithm == "fedper":
            shared_indices, personal_indices = self._fedper_indices(config)
            set_indexed_model_parameters(self.model, shared_indices, parameters)
            self._load_fedper_personal_parameters(personal_indices)
        elif algorithm == "fedrep":
            shared_indices, personal_indices = self._fedrep_indices(config)
            set_indexed_model_parameters(self.model, shared_indices, parameters)
            self._load_fedrep_personal_parameters(personal_indices)
        else:
            set_model_parameters(self.model, parameters)
        loss, metrics = evaluate_model(
            self.model,
            self.loaders.validation,
            device=self.config.device,
        )
        return loss, len(self.loaders.validation.dataset), metrics
