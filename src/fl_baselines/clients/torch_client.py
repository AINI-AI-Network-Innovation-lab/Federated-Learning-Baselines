"""Flower NumPyClient wrapper around PyTorch training code."""

from __future__ import annotations

import copy
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from flwr.client import NumPyClient
from torch import nn

from fl_baselines.algorithms.fedper import (
    get_indexed_model_parameters,
    split_fedper_parameter_indices,
    set_indexed_model_parameters,
)
from fl_baselines.core.config import ExperimentConfig
from fl_baselines.core.types import ClientDataLoaders
from fl_baselines.training.ditto import train_ditto_personalized
from fl_baselines.training.evaluate import evaluate_model
from fl_baselines.training.fedaaw import train_fedaaw_client
from fl_baselines.training.fedala import adaptive_local_aggregation
from fl_baselines.training.feddc import train_feddc_client
from fl_baselines.training.feddecorr import train_feddecorr_client
from fl_baselines.training.fedma import train_fedma_client
from fl_baselines.training.fedgen import evaluate_fedgen_model, train_fedgen_client
from fl_baselines.training.feddrl import train_feddrl_client
from fl_baselines.training.feddisco import train_feddisco_client
from fl_baselines.training.fedent import train_fedent_client
from fl_baselines.training.fedvck import train_fedvck_client
from fl_baselines.training.feddyn import train_feddyn_client, update_feddyn_state
from fl_baselines.training.fedsam import train_fedsam_client
from fl_baselines.training.fedspeed import train_fedspeed_client
from fl_baselines.training.fedntd import train_fedntd_client
from fl_baselines.training.fedlc import train_fedlc_client
from fl_baselines.training.fedrs import train_fedrs_client
from fl_baselines.training.fedproto import train_fedproto_client
from fl_baselines.training.fedmeta import train_fedmeta_client
from fl_baselines.training.fednp import train_fednp_client
from fl_baselines.training.fedcurv import train_fedcurv_client
from fl_baselines.training.apfl import train_apfl_client
from fl_baselines.training.moon import train_moon_client
from fl_baselines.training.pfedme import train_pfedme_client
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
        if algorithm == "fedala":
            return self._fit_fedala(parameters, config)
        if algorithm == "fedamp":
            return self._fit_fedamp(parameters, config)
        if algorithm == "scaffold":
            return self._fit_scaffold(parameters, config)
        if algorithm == "moon":
            return self._fit_moon(parameters, config)
        if algorithm == "feddyn":
            return self._fit_feddyn(parameters, config)
        if algorithm == "feddc":
            return self._fit_feddc(parameters, config)
        if algorithm == "feddecorr":
            return self._fit_feddecorr(parameters, config)
        if algorithm == "fedma":
            return self._fit_fedma(parameters, config)
        if algorithm == "fedaaw":
            return self._fit_fedaaw(parameters, config)
        if algorithm == "fedgen":
            return self._fit_fedgen(parameters, config)
        if algorithm in {"feddrl", "feddrrl"}:
            return self._fit_feddrl(parameters, config)
        if algorithm == "feddisco":
            return self._fit_feddisco(parameters, config)
        if algorithm == "fedent":
            return self._fit_fedent(parameters, config)
        if algorithm == "fedvck":
            return self._fit_fedvck(parameters, config)
        if algorithm == "fedsam":
            return self._fit_fedsam(parameters, config)
        if algorithm == "fedspeed":
            return self._fit_fedspeed(parameters, config)
        if algorithm == "fedproto":
            return self._fit_fedproto(parameters, config)
        if algorithm == "fedmeta":
            return self._fit_fedmeta(parameters, config)
        if algorithm == "fednp":
            return self._fit_fednp(parameters, config)
        if algorithm == "fedcurv":
            return self._fit_fedcurv(parameters, config)
        if algorithm == "fedntd":
            return self._fit_fedntd(parameters, config)
        if algorithm == "fedlc":
            return self._fit_fedlc(parameters, config)
        if algorithm == "fedrs":
            return self._fit_fedrs(parameters, config)
        if algorithm == "apfl":
            return self._fit_apfl(parameters, config)
        if algorithm == "ditto":
            return self._fit_ditto(parameters, config)
        if algorithm == "pfedme":
            return self._fit_pfedme(parameters, config)

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

    def _fit_fedma(
        self,
        parameters: list[np.ndarray],
        config: dict[str, bool | bytes | float | int | str],
    ) -> tuple[list[np.ndarray], int, dict[str, bool | bytes | float | int | str]]:
        set_model_parameters(self.model, parameters)
        local_epochs = int(config.get("local_epochs", self.config.local_epochs))
        learning_rate = float(config.get("learning_rate", self.config.learning_rate))
        stage = int(config.get("fedma_stage", 0))
        from fl_baselines.algorithms.fedma import fedma_frozen_prefixes

        metrics = train_fedma_client(
            self.model,
            self.loaders.train,
            epochs=local_epochs,
            learning_rate=learning_rate,
            device=self.config.device,
            frozen_layer_prefixes=fedma_frozen_prefixes(self.model, stage),
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

    def _fit_feddyn(
        self,
        parameters: list[np.ndarray],
        config: dict[str, bool | bytes | float | int | str],
    ) -> tuple[list[np.ndarray], int, dict[str, bool | bytes | float | int | str]]:
        set_model_parameters(self.model, parameters)
        global_model = copy.deepcopy(self.model)
        state = self._load_feddyn_state()

        local_epochs = int(config.get("local_epochs", self.config.local_epochs))
        learning_rate = float(config.get("learning_rate", self.config.learning_rate))
        alpha = float(config.get("feddyn_alpha", self.config.feddyn_alpha))
        metrics = train_feddyn_client(
            self.model,
            global_model,
            self.loaders.train,
            epochs=local_epochs,
            learning_rate=learning_rate,
            device=self.config.device,
            alpha=alpha,
            state=state,
        )
        new_state = update_feddyn_state(state, self.model, global_model, alpha)
        self._save_feddyn_state(new_state)
        return get_model_parameters(self.model), len(self.loaders.train.dataset), metrics

    def _fit_feddc(
        self,
        parameters: list[np.ndarray],
        config: dict[str, bool | bytes | float | int | str],
    ) -> tuple[list[np.ndarray], int, dict[str, bool | bytes | float | int | str]]:
        model_parameter_count = len(get_model_parameters(self.model))
        if len(parameters) <= model_parameter_count:
            raise ValueError("FedDC fit requires model parameters and server update state")

        model_parameters = parameters[:model_parameter_count]
        server_update_state = parameters[model_parameter_count:]
        set_model_parameters(self.model, model_parameters)
        global_model = copy.deepcopy(self.model)
        drift_state, local_update_state = self._load_feddc_state()

        local_epochs = int(config.get("local_epochs", self.config.local_epochs))
        learning_rate = float(config.get("learning_rate", self.config.learning_rate))
        alpha = float(config.get("feddc_alpha", self.config.feddc_alpha))
        metrics, updated_drift_state, updated_local_state = train_feddc_client(
            self.model,
            global_model,
            self.loaders.train,
            epochs=local_epochs,
            learning_rate=learning_rate,
            device=self.config.device,
            alpha=alpha,
            drift_state=drift_state,
            local_update_state=local_update_state,
            server_update_state=[
                torch.as_tensor(state.copy())
                for state in server_update_state
            ],
        )
        self._save_feddc_state(updated_drift_state, updated_local_state)
        corrected_parameters = [
            local_parameter + drift_tensor.detach().cpu().numpy()
            for local_parameter, drift_tensor in zip(
                get_model_parameters(self.model),
                updated_drift_state,
            )
        ]
        local_update_arrays = [
            tensor.detach().cpu().numpy() for tensor in updated_local_state
        ]
        return (
            corrected_parameters + local_update_arrays,
            len(self.loaders.train.dataset),
            metrics,
        )

    def _fit_fedsam(
        self,
        parameters: list[np.ndarray],
        config: dict[str, bool | bytes | float | int | str],
    ) -> tuple[list[np.ndarray], int, dict[str, bool | bytes | float | int | str]]:
        set_model_parameters(self.model, parameters)

        local_epochs = int(config.get("local_epochs", self.config.local_epochs))
        learning_rate = float(config.get("learning_rate", self.config.learning_rate))
        fedsam_rho = float(config.get("fedsam_rho", self.config.fedsam_rho))
        metrics = train_fedsam_client(
            self.model,
            self.loaders.train,
            epochs=local_epochs,
            learning_rate=learning_rate,
            device=self.config.device,
            fedsam_rho=fedsam_rho,
        )
        return get_model_parameters(self.model), len(self.loaders.train.dataset), metrics

    def _fit_fedaaw(
        self,
        parameters: list[np.ndarray],
        config: dict[str, bool | bytes | float | int | str],
    ) -> tuple[list[np.ndarray], int, dict[str, bool | bytes | float | int | str]]:
        set_model_parameters(self.model, parameters)

        local_epochs = int(config.get("local_epochs", self.config.local_epochs))
        learning_rate = float(config.get("learning_rate", self.config.learning_rate))
        metrics = train_fedaaw_client(
            self.model,
            self.loaders.train,
            epochs=local_epochs,
            learning_rate=learning_rate,
            device=self.config.device,
        )
        return get_model_parameters(self.model), len(self.loaders.train.dataset), metrics

    def _fit_fedgen(
        self,
        parameters: list[np.ndarray],
        config: dict[str, bool | bytes | float | int | str],
    ) -> tuple[list[np.ndarray], int, dict[str, bool | bytes | float | int | str]]:
        model_parameter_count = len(get_model_parameters(self.model))
        if len(parameters) <= model_parameter_count:
            raise ValueError("FedGEN fit requires model parameters and global mask")

        model_parameters = parameters[:model_parameter_count]
        global_mask = parameters[model_parameter_count]
        set_model_parameters(self.model, model_parameters)

        local_epochs = int(config.get("local_epochs", self.config.local_epochs))
        learning_rate = float(config.get("learning_rate", self.config.learning_rate))
        fedgen_alpha = float(config.get("fedgen_alpha", self.config.fedgen_alpha))
        fedgen_lambda = float(config.get("fedgen_lambda", self.config.fedgen_lambda))
        fedgen_beta = float(config.get("fedgen_beta", self.config.fedgen_beta))
        fedgen_delta = float(config.get("fedgen_delta", self.config.fedgen_delta))
        fedgen_warmup_epochs = int(
            config.get("fedgen_warmup_epochs", self.config.fedgen_warmup_epochs)
        )
        fedgen_l1_weight = float(
            config.get("fedgen_l1_weight", self.config.fedgen_l1_weight)
        )
        metrics, local_mask = train_fedgen_client(
            self.model,
            self.loaders.train,
            epochs=local_epochs,
            learning_rate=learning_rate,
            device=self.config.device,
            global_mask=global_mask,
            fedgen_alpha=fedgen_alpha,
            fedgen_lambda=fedgen_lambda,
            fedgen_beta=fedgen_beta,
            fedgen_delta=fedgen_delta,
            fedgen_warmup_epochs=fedgen_warmup_epochs,
            fedgen_l1_weight=fedgen_l1_weight,
        )
        return (
            get_model_parameters(self.model) + [local_mask],
            len(self.loaders.train.dataset),
            metrics,
        )

    def _fit_feddrl(
        self,
        parameters: list[np.ndarray],
        config: dict[str, bool | bytes | float | int | str],
    ) -> tuple[list[np.ndarray], int, dict[str, bool | bytes | float | int | str]]:
        set_model_parameters(self.model, parameters)

        local_epochs = int(config.get("local_epochs", self.config.local_epochs))
        learning_rate = float(config.get("learning_rate", self.config.learning_rate))
        metrics = train_feddrl_client(
            self.model,
            self.loaders.train,
            epochs=local_epochs,
            learning_rate=learning_rate,
            device=self.config.device,
        )
        return get_model_parameters(self.model), len(self.loaders.train.dataset), metrics

    def _fit_feddisco(
        self,
        parameters: list[np.ndarray],
        config: dict[str, bool | bytes | float | int | str],
    ) -> tuple[list[np.ndarray], int, dict[str, bool | bytes | float | int | str]]:
        set_model_parameters(self.model, parameters)

        local_epochs = int(config.get("local_epochs", self.config.local_epochs))
        learning_rate = float(config.get("learning_rate", self.config.learning_rate))
        num_classes = int(config.get("num_classes", self.config.num_classes))
        metric = str(config.get("feddisco_metric", self.config.feddisco_metric))
        epsilon = float(config.get("feddisco_epsilon", self.config.feddisco_epsilon))
        metrics = train_feddisco_client(
            self.model,
            self.loaders.train,
            epochs=local_epochs,
            learning_rate=learning_rate,
            device=self.config.device,
            num_classes=num_classes,
            metric=metric,
            epsilon=epsilon,
        )
        return get_model_parameters(self.model), len(self.loaders.train.dataset), metrics

    def _fit_fedent(
        self,
        parameters: list[np.ndarray],
        config: dict[str, bool | bytes | float | int | str],
    ) -> tuple[list[np.ndarray], int, dict[str, bool | bytes | float | int | str]]:
        set_model_parameters(self.model, parameters)

        local_epochs = int(config.get("local_epochs", self.config.local_epochs))
        learning_rate = float(config.get("learning_rate", self.config.learning_rate))
        fedent_beta = float(config.get("fedent_beta", self.config.fedent_beta))
        fedent_gamma = float(config.get("fedent_gamma", self.config.fedent_gamma))
        fedent_epsilon = float(config.get("fedent_epsilon", self.config.fedent_epsilon))
        fedent_max_learning_rate = float(
            config.get(
                "fedent_max_learning_rate",
                self.config.fedent_max_learning_rate,
            )
        )
        fedent_enable_decay = bool(
            config.get("fedent_enable_decay", self.config.fedent_enable_decay)
        )
        phi1_vector = self._deserialize_fedent_phi1(str(config.get("fedent_phi1", "[]")))
        phi2_scalar = float(config.get("fedent_phi2", 0.0))
        previous_eta = self._load_fedent_previous_eta() if fedent_enable_decay else None
        metrics = train_fedent_client(
            self.model,
            self.loaders.train,
            epochs=local_epochs,
            learning_rate=learning_rate,
            device=self.config.device,
            phi1_vector=phi1_vector,
            phi2_scalar=phi2_scalar,
            fedent_beta=fedent_beta,
            fedent_gamma=fedent_gamma,
            fedent_epsilon=fedent_epsilon,
            fedent_max_learning_rate=fedent_max_learning_rate,
            previous_eta=previous_eta,
        )
        if fedent_enable_decay:
            self._save_fedent_previous_eta(float(metrics["fedent_learning_rate"]))
        metrics["fedent_weight_sq_norm"] = float(
            sum(np.sum(np.square(parameter)) for parameter in get_model_parameters(self.model))
        )
        return get_model_parameters(self.model), len(self.loaders.train.dataset), metrics

    def _fit_fedvck(
        self,
        parameters: list[np.ndarray],
        config: dict[str, bool | bytes | float | int | str],
    ) -> tuple[list[np.ndarray], int, dict[str, bool | bytes | float | int | str]]:
        set_model_parameters(self.model, parameters)

        local_epochs = int(config.get("local_epochs", self.config.local_epochs))
        learning_rate = float(config.get("learning_rate", self.config.learning_rate))
        condensed_ratio = float(
            config.get("fedvck_condensed_ratio", self.config.fedvck_condensed_ratio)
        )
        condensed_steps = int(
            config.get("fedvck_condensed_steps", self.config.fedvck_condensed_steps)
        )
        condensed_learning_rate = float(
            config.get(
                "fedvck_condensed_learning_rate",
                self.config.fedvck_condensed_learning_rate,
            )
        )
        importance_alpha = float(
            config.get("fedvck_importance_alpha", self.config.fedvck_importance_alpha)
        )
        enable_latent_constraints = bool(
            config.get(
                "fedvck_enable_latent_constraints",
                self.config.fedvck_enable_latent_constraints,
            )
        )
        previous_model_state = self._load_fedvck_previous_model_state()
        metrics, condensed_inputs, condensed_labels, prototype_sums, prototype_counts = (
            train_fedvck_client(
                self.model,
                self.loaders.train,
                epochs=local_epochs,
                learning_rate=learning_rate,
                device=self.config.device,
                condensed_ratio=condensed_ratio,
                condensed_steps=condensed_steps,
                condensed_learning_rate=condensed_learning_rate,
                importance_alpha=importance_alpha,
                enable_latent_constraints=enable_latent_constraints,
                previous_model_state=previous_model_state,
            )
        )
        self._save_fedvck_previous_model_state()
        return (
            get_model_parameters(self.model)
            + [condensed_inputs, condensed_labels, prototype_sums, prototype_counts],
            len(self.loaders.train.dataset),
            metrics,
        )

    def _deserialize_fedent_phi1(self, serialized_phi1: str) -> torch.Tensor:
        values = json.loads(serialized_phi1)
        if not values:
            return torch.zeros(0, dtype=torch.float32)
        arrays = [torch.tensor(value, dtype=torch.float32).flatten() for value in values]
        return torch.cat(arrays)

    def _fedent_state_path(self) -> Path:
        return (
            Path(self.config.output_dir)
            / "fedent_clients"
            / self.client_id
            / "state.pt"
        )

    def _load_fedent_previous_eta(self) -> float | None:
        state_path = self._fedent_state_path()
        if not state_path.exists():
            return None
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        return float(state["previous_eta"])

    def _save_fedent_previous_eta(self, previous_eta: float) -> None:
        state_path = self._fedent_state_path()
        state_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"previous_eta": float(previous_eta)}, state_path)

    def _fedvck_state_path(self) -> Path:
        return (
            Path(self.config.output_dir)
            / "fedvck_clients"
            / self.client_id
            / "state.pt"
        )

    def _load_fedvck_previous_model_state(self) -> dict[str, torch.Tensor] | None:
        state_path = self._fedvck_state_path()
        if not state_path.exists():
            return None
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        return state.get("previous_model_state")

    def _save_fedvck_previous_model_state(self) -> None:
        state_path = self._fedvck_state_path()
        state_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "previous_model_state": {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }
            },
            state_path,
        )

    def _fit_feddecorr(
        self,
        parameters: list[np.ndarray],
        config: dict[str, bool | bytes | float | int | str],
    ) -> tuple[list[np.ndarray], int, dict[str, bool | bytes | float | int | str]]:
        set_model_parameters(self.model, parameters)

        local_epochs = int(config.get("local_epochs", self.config.local_epochs))
        learning_rate = float(config.get("learning_rate", self.config.learning_rate))
        feddecorr_beta = float(
            config.get("feddecorr_beta", self.config.feddecorr_beta)
        )
        metrics = train_feddecorr_client(
            self.model,
            self.loaders.train,
            epochs=local_epochs,
            learning_rate=learning_rate,
            device=self.config.device,
            feddecorr_beta=feddecorr_beta,
        )
        return get_model_parameters(self.model), len(self.loaders.train.dataset), metrics

    def _fit_fedspeed(
        self,
        parameters: list[np.ndarray],
        config: dict[str, bool | bytes | float | int | str],
    ) -> tuple[list[np.ndarray], int, dict[str, bool | bytes | float | int | str]]:
        set_model_parameters(self.model, parameters)
        global_model = copy.deepcopy(self.model)
        state = self._load_fedspeed_state()

        local_epochs = int(config.get("local_epochs", self.config.local_epochs))
        learning_rate = float(config.get("learning_rate", self.config.learning_rate))
        fedspeed_lambda = float(
            config.get("fedspeed_lambda", self.config.fedspeed_lambda)
        )
        fedspeed_alpha = float(
            config.get("fedspeed_alpha", self.config.fedspeed_alpha)
        )
        fedspeed_rho = float(config.get("fedspeed_rho", self.config.fedspeed_rho))
        metrics, new_state, payload = train_fedspeed_client(
            self.model,
            global_model,
            self.loaders.train,
            epochs=local_epochs,
            learning_rate=learning_rate,
            device=self.config.device,
            fedspeed_lambda=fedspeed_lambda,
            fedspeed_alpha=fedspeed_alpha,
            fedspeed_rho=fedspeed_rho,
            state=state,
        )
        self._save_fedspeed_state(new_state)
        return (
            [tensor.detach().cpu().numpy() for tensor in payload],
            len(self.loaders.train.dataset),
            metrics,
        )

    def _fit_ditto(
        self,
        parameters: list[np.ndarray],
        config: dict[str, bool | bytes | float | int | str],
    ) -> tuple[list[np.ndarray], int, dict[str, bool | bytes | float | int | str]]:
        set_model_parameters(self.model, parameters)
        global_model = copy.deepcopy(self.model)

        local_epochs = int(config.get("local_epochs", self.config.local_epochs))
        learning_rate = float(config.get("learning_rate", self.config.learning_rate))
        ditto_lambda = float(config.get("ditto_lambda", self.config.ditto_lambda))

        metrics = train_one_client(
            self.model,
            self.loaders.train,
            epochs=local_epochs,
            learning_rate=learning_rate,
            device=self.config.device,
        )

        personalized_model = self._load_ditto_personalized_model(global_model)
        train_ditto_personalized(
            personalized_model,
            global_model,
            self.loaders.train,
            epochs=local_epochs,
            learning_rate=learning_rate,
            device=self.config.device,
            ditto_lambda=ditto_lambda,
        )
        self._save_ditto_personalized_model(personalized_model)

        return get_model_parameters(self.model), len(self.loaders.train.dataset), metrics

    def _fit_apfl(
        self,
        parameters: list[np.ndarray],
        config: dict[str, bool | bytes | float | int | str],
    ) -> tuple[list[np.ndarray], int, dict[str, bool | bytes | float | int | str]]:
        set_model_parameters(self.model, parameters)
        global_model = copy.deepcopy(self.model)
        personalized_model = self._load_apfl_personalized_model(global_model)
        alpha = self._load_apfl_alpha(float(config.get("apfl_alpha", self.config.apfl_alpha)))

        metrics, alpha_value = train_apfl_client(
            self.model,
            personalized_model,
            self.loaders.train,
            epochs=int(config.get("local_epochs", self.config.local_epochs)),
            learning_rate=float(config.get("learning_rate", self.config.learning_rate)),
            personal_learning_rate=float(
                config.get(
                    "apfl_personal_learning_rate",
                    self.config.apfl_personal_learning_rate,
                )
            ),
            device=self.config.device,
            alpha=alpha,
            adaptive_alpha=bool(
                config.get("apfl_adaptive_alpha", self.config.apfl_adaptive_alpha)
            ),
            alpha_learning_rate=float(
                config.get(
                    "apfl_alpha_learning_rate",
                    self.config.apfl_alpha_learning_rate,
                )
            ),
        )
        self._save_apfl_personalized_model(personalized_model)
        self._save_apfl_alpha(alpha_value)

        return get_model_parameters(self.model), len(self.loaders.train.dataset), metrics

    def _fit_fedntd(
        self,
        parameters: list[np.ndarray],
        config: dict[str, bool | bytes | float | int | str],
    ) -> tuple[list[np.ndarray], int, dict[str, bool | bytes | float | int | str]]:
        set_model_parameters(self.model, parameters)
        teacher_model = copy.deepcopy(self.model)

        local_epochs = int(config.get("local_epochs", self.config.local_epochs))
        learning_rate = float(config.get("learning_rate", self.config.learning_rate))
        fedntd_beta = float(config.get("fedntd_beta", self.config.fedntd_beta))
        fedntd_temperature = float(
            config.get("fedntd_temperature", self.config.fedntd_temperature)
        )
        metrics = train_fedntd_client(
            self.model,
            teacher_model,
            self.loaders.train,
            epochs=local_epochs,
            learning_rate=learning_rate,
            device=self.config.device,
            fedntd_beta=fedntd_beta,
            fedntd_temperature=fedntd_temperature,
        )
        return get_model_parameters(self.model), len(self.loaders.train.dataset), metrics

    def _fit_fedlc(
        self,
        parameters: list[np.ndarray],
        config: dict[str, bool | bytes | float | int | str],
    ) -> tuple[list[np.ndarray], int, dict[str, bool | bytes | float | int | str]]:
        set_model_parameters(self.model, parameters)

        local_epochs = int(config.get("local_epochs", self.config.local_epochs))
        learning_rate = float(config.get("learning_rate", self.config.learning_rate))
        num_classes = int(config.get("num_classes", self.config.num_classes))
        fedlc_tau = float(config.get("fedlc_tau", self.config.fedlc_tau))
        fedlc_epsilon = float(config.get("fedlc_epsilon", self.config.fedlc_epsilon))
        metrics = train_fedlc_client(
            self.model,
            self.loaders.train,
            epochs=local_epochs,
            learning_rate=learning_rate,
            device=self.config.device,
            num_classes=num_classes,
            fedlc_tau=fedlc_tau,
            fedlc_epsilon=fedlc_epsilon,
        )
        return get_model_parameters(self.model), len(self.loaders.train.dataset), metrics

    def _fit_fedrs(
        self,
        parameters: list[np.ndarray],
        config: dict[str, bool | bytes | float | int | str],
    ) -> tuple[list[np.ndarray], int, dict[str, bool | bytes | float | int | str]]:
        set_model_parameters(self.model, parameters)
        local_epochs = int(config.get("local_epochs", self.config.local_epochs))
        learning_rate = float(config.get("learning_rate", self.config.learning_rate))
        num_classes = int(config.get("num_classes", self.config.num_classes))
        fedrs_alpha = float(config.get("fedrs_alpha", self.config.fedrs_alpha))
        metrics = train_fedrs_client(
            self.model,
            self.loaders.train,
            epochs=local_epochs,
            learning_rate=learning_rate,
            device=self.config.device,
            num_classes=num_classes,
            fedrs_alpha=fedrs_alpha,
        )
        return get_model_parameters(self.model), len(self.loaders.train.dataset), metrics

    def _fit_fedproto(
        self,
        parameters: list[np.ndarray],
        config: dict[str, bool | bytes | float | int | str],
    ) -> tuple[list[np.ndarray], int, dict[str, bool | bytes | float | int | str]]:
        model_parameter_count = len(get_model_parameters(self.model))
        if len(parameters) <= model_parameter_count:
            raise ValueError("FedProto fit requires model parameters and global prototypes")

        model_parameters = parameters[:model_parameter_count]
        global_prototypes = parameters[model_parameter_count]
        local_model_path = self._fedproto_local_model_path()
        if local_model_path.exists():
            state = torch.load(local_model_path, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state, strict=True)
        else:
            set_model_parameters(self.model, model_parameters)

        local_epochs = int(config.get("local_epochs", self.config.local_epochs))
        learning_rate = float(config.get("learning_rate", self.config.learning_rate))
        fedproto_lambda = float(
            config.get("fedproto_lambda", self.config.fedproto_lambda)
        )
        metrics, prototype_sums, prototype_counts = train_fedproto_client(
            self.model,
            self.loaders.train,
            epochs=local_epochs,
            learning_rate=learning_rate,
            device=self.config.device,
            fedproto_lambda=fedproto_lambda,
            global_prototypes=global_prototypes,
        )
        local_model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), local_model_path)
        return (
            get_model_parameters(self.model) + [prototype_sums, prototype_counts],
            len(self.loaders.train.dataset),
            metrics,
        )

    def _fedproto_local_model_path(self) -> Path:
        return (
            Path(self.config.output_dir)
            / "fedproto_clients"
            / self.client_id
            / "local_model.pt"
        )

    def _fit_fedmeta(
        self,
        parameters: list[np.ndarray],
        config: dict[str, bool | bytes | float | int | str],
    ) -> tuple[list[np.ndarray], int, dict[str, bool | bytes | float | int | str]]:
        model_parameter_count = len(get_model_parameters(self.model))
        method = str(config.get("fedmeta_method", self.config.fedmeta_method))
        meta_sgd = method == "meta-sgd"
        expected_count = model_parameter_count * (2 if meta_sgd else 1)
        if len(parameters) != expected_count:
            raise ValueError("FedMeta fit requires algorithm parameters matching the method")

        model_parameters = parameters[:model_parameter_count]
        alpha_parameters = parameters[model_parameter_count:] if meta_sgd else None
        set_model_parameters(self.model, model_parameters)

        metrics, gradients = train_fedmeta_client(
            self.model,
            self.loaders.train,
            device=self.config.device,
            inner_learning_rate=float(
                config.get(
                    "fedmeta_inner_learning_rate",
                    self.config.fedmeta_inner_learning_rate,
                )
            ),
            support_fraction=float(
                config.get(
                    "fedmeta_support_fraction",
                    self.config.fedmeta_support_fraction,
                )
            ),
            inner_steps=int(
                config.get("fedmeta_inner_steps", self.config.fedmeta_inner_steps)
            ),
            first_order=bool(
                config.get("fedmeta_first_order", self.config.fedmeta_first_order)
            ),
            alpha_parameters=alpha_parameters,
        )
        return gradients, len(self.loaders.train.dataset), metrics

    def _fit_fednp(
        self,
        parameters: list[np.ndarray],
        config: dict[str, bool | bytes | float | int | str],
    ) -> tuple[list[np.ndarray], int, dict[str, bool | bytes | float | int | str]]:
        model_parameter_count = len(get_model_parameters(self.model))
        if len(parameters) != model_parameter_count + 2:
            raise ValueError("FedNP fit requires model parameters plus latent Gaussian statistics")

        model_parameters = parameters[:model_parameter_count]
        latent_mean = parameters[-2]
        latent_var = parameters[-1]
        set_model_parameters(self.model, model_parameters)

        metrics, latent_sum, latent_square_sum, latent_count = train_fednp_client(
            self.model,
            self.loaders.train,
            epochs=int(config.get("local_epochs", self.config.local_epochs)),
            learning_rate=float(config.get("learning_rate", self.config.learning_rate)),
            device=self.config.device,
            latent_mean=latent_mean,
            latent_var=latent_var,
            fednp_lambda=float(config.get("fednp_lambda", self.config.fednp_lambda)),
            fednp_stability_eps=float(
                config.get("fednp_stability_eps", self.config.fednp_stability_eps)
            ),
        )
        return (
            get_model_parameters(self.model) + [latent_sum, latent_square_sum, latent_count],
            len(self.loaders.train.dataset),
            metrics,
        )

    def _fit_fedcurv(
        self,
        parameters: list[np.ndarray],
        config: dict[str, bool | bytes | float | int | str],
    ) -> tuple[list[np.ndarray], int, dict[str, bool | bytes | float | int | str]]:
        model_parameter_count = len(get_model_parameters(self.model))
        if len(parameters) != model_parameter_count * 3:
            raise ValueError("FedCurv fit requires model parameters plus global curvature aggregates")

        model_parameters = parameters[:model_parameter_count]
        global_curvature = parameters[model_parameter_count : 2 * model_parameter_count]
        global_weighted = parameters[2 * model_parameter_count :]
        set_model_parameters(self.model, model_parameters)

        local_curvature, local_weighted = self._load_fedcurv_state(model_parameters)
        metrics, curvature_payload, weighted_payload = train_fedcurv_client(
            self.model,
            self.loaders.train,
            epochs=int(config.get("local_epochs", self.config.local_epochs)),
            learning_rate=float(config.get("learning_rate", self.config.learning_rate)),
            device=self.config.device,
            global_curvature=global_curvature,
            global_weighted=global_weighted,
            local_curvature=local_curvature,
            local_weighted=local_weighted,
            fedcurv_lambda=float(config.get("fedcurv_lambda", self.config.fedcurv_lambda)),
            fisher_batches=int(
                config.get("fedcurv_fisher_batches", self.config.fedcurv_fisher_batches)
            ),
            fedcurv_stability_eps=float(
                config.get(
                    "fedcurv_stability_eps",
                    self.config.fedcurv_stability_eps,
                )
            ),
        )
        self._save_fedcurv_state(curvature_payload, weighted_payload)
        return (
            get_model_parameters(self.model) + curvature_payload + weighted_payload,
            len(self.loaders.train.dataset),
            metrics,
        )

    def _fedcurv_state_path(self) -> Path:
        return (
            Path(self.config.output_dir)
            / "fedcurv_clients"
            / self.client_id
            / "curvature_state.pt"
        )

    def _load_fedcurv_state(
        self,
        model_parameters: list[np.ndarray],
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        state_path = self._fedcurv_state_path()
        zero_state = [np.zeros_like(array, dtype=np.float32) for array in model_parameters]
        if not state_path.exists():
            return zero_state, [array.copy() for array in zero_state]

        state = torch.load(state_path, map_location="cpu", weights_only=True)
        curvature = [
            tensor.detach().cpu().numpy().astype(np.float32)
            for tensor in state["curvature"]
        ]
        weighted = [
            tensor.detach().cpu().numpy().astype(np.float32)
            for tensor in state["weighted"]
        ]
        return curvature, weighted

    def _save_fedcurv_state(
        self,
        curvature_payload: list[np.ndarray],
        weighted_payload: list[np.ndarray],
    ) -> None:
        state_path = self._fedcurv_state_path()
        state_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "curvature": [torch.from_numpy(array.copy()) for array in curvature_payload],
                "weighted": [torch.from_numpy(array.copy()) for array in weighted_payload],
            },
            state_path,
        )

    def _fit_pfedme(
        self,
        parameters: list[np.ndarray],
        config: dict[str, bool | bytes | float | int | str],
    ) -> tuple[list[np.ndarray], int, dict[str, bool | bytes | float | int | str]]:
        set_model_parameters(self.model, parameters)
        reference_model = copy.deepcopy(self.model)
        personalized_model = self._load_pfedme_personalized_model(reference_model)

        local_epochs = int(config.get("local_epochs", self.config.local_epochs))
        learning_rate = float(config.get("learning_rate", self.config.learning_rate))
        pfedme_lambda = float(config.get("pfedme_lambda", self.config.pfedme_lambda))
        personal_learning_rate = float(
            config.get(
                "pfedme_personal_learning_rate",
                self.config.pfedme_personal_learning_rate,
            )
        )
        personal_steps = int(
            config.get("pfedme_personal_steps", self.config.pfedme_personal_steps)
        )
        metrics = train_pfedme_client(
            reference_model,
            personalized_model,
            self.loaders.train,
            epochs=local_epochs,
            learning_rate=learning_rate,
            personal_learning_rate=personal_learning_rate,
            personal_steps=personal_steps,
            device=self.config.device,
            pfedme_lambda=pfedme_lambda,
        )
        self._save_pfedme_personalized_model(personalized_model)
        self.model.load_state_dict(reference_model.state_dict(), strict=True)
        return get_model_parameters(self.model), len(self.loaders.train.dataset), metrics

    def _fit_fedala(
        self,
        parameters: list[np.ndarray],
        config: dict[str, bool | bytes | float | int | str],
    ) -> tuple[list[np.ndarray], int, dict[str, bool | bytes | float | int | str]]:
        global_model = copy.deepcopy(self.model)
        set_model_parameters(global_model, parameters)

        state = self._load_fedala_state()
        if state is None:
            set_model_parameters(self.model, parameters)
            weights = None
            start_phase = True
        else:
            self.model.load_state_dict(state["model"], strict=True)
            weights = state["weights"]
            start_phase = bool(state["start_phase"])

        server_round = int(config.get("server_round", 1))
        if server_round > 1:
            fedala_state = adaptive_local_aggregation(
                self.model,
                global_model,
                self.loaders.train,
                weights=weights,
                layer_count=int(
                    config.get("fedala_layer_count", self.config.fedala_layer_count)
                ),
                eta=float(config.get("fedala_eta", self.config.fedala_eta)),
                rand_percent=int(
                    config.get("fedala_rand_percent", self.config.fedala_rand_percent)
                ),
                threshold=float(
                    config.get("fedala_threshold", self.config.fedala_threshold)
                ),
                num_pre_loss=int(
                    config.get("fedala_num_pre_loss", self.config.fedala_num_pre_loss)
                ),
                start_max_steps=int(
                    config.get(
                        "fedala_start_max_steps",
                        self.config.fedala_start_max_steps,
                    )
                ),
                device=self.config.device,
                start_phase=start_phase,
            )
            weights = fedala_state.weights
            start_phase = fedala_state.start_phase

        local_epochs = int(config.get("local_epochs", self.config.local_epochs))
        learning_rate = float(config.get("learning_rate", self.config.learning_rate))
        metrics = train_one_client(
            self.model,
            self.loaders.train,
            epochs=local_epochs,
            learning_rate=learning_rate,
            device=self.config.device,
        )
        self._save_fedala_state(weights, start_phase)
        return get_model_parameters(self.model), len(self.loaders.train.dataset), metrics

    def _fit_fedamp(
        self,
        parameters: list[np.ndarray],
        config: dict[str, bool | bytes | float | int | str],
    ) -> tuple[list[np.ndarray], int, dict[str, bool | bytes | float | int | str]]:
        set_model_parameters(self.model, parameters)
        local_epochs = int(config.get("local_epochs", self.config.local_epochs))
        learning_rate = float(config.get("learning_rate", self.config.learning_rate))
        proximal_mu = float(
            config.get(
                "fedamp_proximal_mu",
                self.config.fedamp_lambda / self.config.fedamp_alpha,
            )
        )
        metrics = train_one_client(
            self.model,
            self.loaders.train,
            epochs=local_epochs,
            learning_rate=learning_rate,
            device=self.config.device,
            proximal_mu=proximal_mu,
        )
        return get_model_parameters(self.model), len(self.loaders.train.dataset), metrics

    def _feddyn_state_path(self) -> Path:
        return (
            Path(self.config.output_dir)
            / "feddyn_clients"
            / self.client_id
            / "state.pt"
        )

    def _load_feddyn_state(self) -> list[torch.Tensor]:
        state_path = self._feddyn_state_path()
        if state_path.exists():
            return torch.load(
                state_path,
                map_location="cpu",
                weights_only=True,
            )
        return [
            torch.zeros_like(value.detach().cpu())
            for value in self.model.state_dict().values()
        ]

    def _save_feddyn_state(self, state: list[torch.Tensor]) -> None:
        state_path = self._feddyn_state_path()
        state_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            [tensor.detach().cpu().clone() for tensor in state],
            state_path,
        )

    def _feddc_state_path(self) -> Path:
        return (
            Path(self.config.output_dir)
            / "feddc_clients"
            / self.client_id
            / "state.pt"
        )

    def _load_feddc_state(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        state_path = self._feddc_state_path()
        if state_path.exists():
            state = torch.load(
                state_path,
                map_location="cpu",
                weights_only=True,
            )
            return state["drift"], state["local_update"]
        zeros = [
            torch.zeros_like(value.detach().cpu())
            for value in self.model.state_dict().values()
        ]
        return zeros, [tensor.clone() for tensor in zeros]

    def _save_feddc_state(
        self,
        drift_state: list[torch.Tensor],
        local_update_state: list[torch.Tensor],
    ) -> None:
        state_path = self._feddc_state_path()
        state_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "drift": [tensor.detach().cpu().clone() for tensor in drift_state],
                "local_update": [
                    tensor.detach().cpu().clone() for tensor in local_update_state
                ],
            },
            state_path,
        )

    def _fedspeed_state_path(self) -> Path:
        return (
            Path(self.config.output_dir)
            / "fedspeed_clients"
            / self.client_id
            / "state.pt"
        )

    def _load_fedspeed_state(self) -> list[torch.Tensor]:
        state_path = self._fedspeed_state_path()
        if state_path.exists():
            return torch.load(
                state_path,
                map_location="cpu",
                weights_only=True,
            )
        return [
            torch.zeros_like(value.detach().cpu())
            for value in self.model.state_dict().values()
        ]

    def _save_fedspeed_state(self, state: list[torch.Tensor]) -> None:
        state_path = self._fedspeed_state_path()
        state_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            [tensor.detach().cpu().clone() for tensor in state],
            state_path,
        )

    def _ditto_personalized_model_path(self) -> Path:
        return (
            Path(self.config.output_dir)
            / "ditto_clients"
            / self.client_id
            / "personalized.pt"
        )

    def _load_ditto_personalized_model(self, global_model: nn.Module) -> nn.Module:
        personalized_model = copy.deepcopy(global_model)
        personal_path = self._ditto_personalized_model_path()
        if personal_path.exists():
            personal_state = torch.load(
                personal_path,
                map_location="cpu",
                weights_only=True,
            )
            personalized_model.load_state_dict(personal_state, strict=True)
        return personalized_model

    def _save_ditto_personalized_model(self, model: nn.Module) -> None:
        personal_path = self._ditto_personalized_model_path()
        personal_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), personal_path)

    def _pfedme_personalized_model_path(self) -> Path:
        return (
            Path(self.config.output_dir)
            / "pfedme_clients"
            / self.client_id
            / "personalized.pt"
        )

    def _load_pfedme_personalized_model(self, global_model: nn.Module) -> nn.Module:
        personalized_model = copy.deepcopy(global_model)
        personal_path = self._pfedme_personalized_model_path()
        if personal_path.exists():
            personal_state = torch.load(
                personal_path,
                map_location="cpu",
                weights_only=True,
            )
            personalized_model.load_state_dict(personal_state, strict=True)
        return personalized_model

    def _save_pfedme_personalized_model(self, model: nn.Module) -> None:
        personal_path = self._pfedme_personalized_model_path()
        personal_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), personal_path)

    def _apfl_personalized_model_path(self) -> Path:
        return (
            Path(self.config.output_dir)
            / "apfl_clients"
            / self.client_id
            / "personalized.pt"
        )

    def _apfl_alpha_path(self) -> Path:
        return (
            Path(self.config.output_dir)
            / "apfl_clients"
            / self.client_id
            / "alpha.json"
        )

    def _load_apfl_personalized_model(self, global_model: nn.Module) -> nn.Module:
        personalized_model = copy.deepcopy(global_model)
        personal_path = self._apfl_personalized_model_path()
        if personal_path.exists():
            personal_state = torch.load(
                personal_path,
                map_location="cpu",
                weights_only=True,
            )
            personalized_model.load_state_dict(personal_state, strict=True)
        return personalized_model

    def _save_apfl_personalized_model(self, model: nn.Module) -> None:
        personal_path = self._apfl_personalized_model_path()
        personal_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), personal_path)

    def _load_apfl_alpha(self, default_alpha: float) -> float:
        alpha_path = self._apfl_alpha_path()
        if not alpha_path.exists():
            return float(default_alpha)
        return float(json.loads(alpha_path.read_text())["alpha"])

    def _save_apfl_alpha(self, alpha: float) -> None:
        alpha_path = self._apfl_alpha_path()
        alpha_path.parent.mkdir(parents=True, exist_ok=True)
        alpha_path.write_text(json.dumps({"alpha": float(alpha)}))

    def _build_apfl_mixed_model(
        self,
        global_model: nn.Module,
        personalized_model: nn.Module,
        alpha: float,
    ) -> nn.Module:
        mixed_model = copy.deepcopy(global_model)
        global_state = global_model.state_dict()
        personal_state = personalized_model.state_dict()
        mixed_state = OrderedDict()
        for key in global_state:
            global_value = global_state[key].detach().cpu()
            personal_value = personal_state[key].detach().cpu()
            if torch.is_floating_point(global_value):
                mixed_state[key] = alpha * personal_value + (1.0 - alpha) * global_value
            else:
                mixed_state[key] = personal_value.clone()
        mixed_model.load_state_dict(mixed_state, strict=True)
        return mixed_model

    def _fedala_state_path(self) -> Path:
        return (
            Path(self.config.output_dir)
            / "fedala_clients"
            / self.client_id
            / "state.pt"
        )

    def _load_fedala_state(self) -> dict[str, object] | None:
        state_path = self._fedala_state_path()
        if not state_path.exists():
            return None
        return torch.load(
            state_path,
            map_location="cpu",
            weights_only=True,
        )

    def _save_fedala_state(
        self,
        weights: list[torch.Tensor] | None,
        start_phase: bool,
    ) -> None:
        state_path = self._fedala_state_path()
        state_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                },
                "weights": (
                    None
                    if weights is None
                    else [weight.detach().cpu().clone() for weight in weights]
                ),
                "start_phase": start_phase,
            },
            state_path,
        )

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
        elif algorithm == "fedgen":
            model_parameter_count = len(get_model_parameters(self.model))
            model_parameters = parameters[:model_parameter_count]
            global_mask = parameters[model_parameter_count]
            set_model_parameters(self.model, model_parameters)
            loss, metrics = evaluate_fedgen_model(
                self.model,
                self.loaders.test,
                device=self.config.device,
                global_mask=global_mask,
            )
            return loss, len(self.loaders.test.dataset), metrics
        elif algorithm == "fedala":
            state = self._load_fedala_state()
            if state is None:
                set_model_parameters(self.model, parameters)
            else:
                self.model.load_state_dict(state["model"], strict=True)
        elif algorithm == "apfl":
            set_model_parameters(self.model, parameters)
            global_model = copy.deepcopy(self.model)
            personalized_model = self._load_apfl_personalized_model(global_model)
            alpha = self._load_apfl_alpha(self.config.apfl_alpha)
            mixed_model = self._build_apfl_mixed_model(global_model, personalized_model, alpha)
            loss, metrics = evaluate_model(
                mixed_model,
                self.loaders.test,
                device=self.config.device,
            )
            return loss, len(self.loaders.test.dataset), metrics
        else:
            set_model_parameters(self.model, parameters)
        loss, metrics = evaluate_model(
            self.model,
            self.loaders.test,
            device=self.config.device,
        )
        return loss, len(self.loaders.test.dataset), metrics
