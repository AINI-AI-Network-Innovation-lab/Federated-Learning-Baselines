"""FedVCK algorithm builder."""

from __future__ import annotations

import math

import numpy as np
import torch
from flwr.common import Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate

from fl_baselines.clients.torch_client import get_model_parameters, set_model_parameters
from fl_baselines.core.config import ExperimentConfig
from fl_baselines.core.types import ServerEvaluateFn
from fl_baselines.logging.checkpointing import save_round_checkpoints
from fl_baselines.logging.metrics import weighted_average
from fl_baselines.training.fedvck import replay_fedvck_server


class FedVCKStrategy(FedAvg):
    """FedAvg-compatible strategy with condensed-knowledge replay."""

    def __init__(
        self,
        *,
        num_classes: int,
        checkpoint_model: torch.nn.Module,
        output_dir: str,
        server_replay_epochs: int,
        server_replay_learning_rate: float,
        contrastive_temperature: float,
        hard_negative_k: int,
        max_memory_rounds: int,
        device: str,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.condensed_memory: list[tuple[np.ndarray, np.ndarray]] = []
        self.global_logit_prototypes = np.zeros((num_classes, num_classes), dtype=np.float32)
        self.server_replay_epochs = server_replay_epochs
        self.server_replay_learning_rate = server_replay_learning_rate
        self.contrastive_temperature = contrastive_temperature
        self.hard_negative_k = hard_negative_k
        self.max_memory_rounds = max_memory_rounds
        self.device = device
        self._checkpoint_model = checkpoint_model
        self._output_dir = output_dir

    def make_fit_config(self, config: ExperimentConfig):
        def fn(server_round: int) -> dict[str, bool | bytes | float | int | str]:
            return {
                "algorithm": "fedvck",
                "server_round": server_round,
                "local_epochs": config.local_epochs,
                "learning_rate": config.learning_rate,
                "fedvck_condensed_ratio": config.fedvck_condensed_ratio,
                "fedvck_condensed_steps": config.fedvck_condensed_steps,
                "fedvck_condensed_learning_rate": config.fedvck_condensed_learning_rate,
                "fedvck_importance_alpha": config.fedvck_importance_alpha,
                "fedvck_server_replay_epochs": config.fedvck_server_replay_epochs,
                "fedvck_server_replay_learning_rate": config.fedvck_server_replay_learning_rate,
                "fedvck_contrastive_temperature": config.fedvck_contrastive_temperature,
                "fedvck_hard_negative_k": config.fedvck_hard_negative_k,
                "fedvck_enable_latent_constraints": config.fedvck_enable_latent_constraints,
                "fedvck_max_memory_rounds": config.fedvck_max_memory_rounds,
            }

        return fn

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        model_results = []
        prototype_sums = []
        prototype_counts = []
        for _, fit_res in results:
            arrays = parameters_to_ndarrays(fit_res.parameters)
            if len(arrays) < 5:
                raise ValueError("FedVCK client result is missing condensed payload")
            model_arrays = arrays[:-4]
            condensed_inputs = arrays[-4]
            condensed_labels = arrays[-3]
            class_prototype_sums = arrays[-2]
            class_prototype_counts = arrays[-1]
            model_results.append((model_arrays, fit_res.num_examples))
            prototype_sums.append(class_prototype_sums)
            prototype_counts.append(class_prototype_counts)
            self._append_condensed_memory(condensed_inputs, condensed_labels)

        aggregated_model = aggregate(model_results)
        self._aggregate_logit_prototypes(prototype_sums, prototype_counts)
        aggregated_parameters = ndarrays_to_parameters(aggregated_model)
        replayed_parameters, replay_metrics = self._run_server_replay(aggregated_parameters)

        metrics_aggregated: dict[str, Scalar] = {
            "fedvck_memory_size": len(self.condensed_memory),
            **replay_metrics,
        }
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(fit_res.num_examples, fit_res.metrics) for _, fit_res in results]
            metrics_aggregated.update(self.fit_metrics_aggregation_fn(fit_metrics))

        save_round_checkpoints(
            self._checkpoint_model,
            replayed_parameters,
            self._output_dir,
            server_round,
        )
        return replayed_parameters, metrics_aggregated

    def _append_condensed_memory(
        self,
        condensed_inputs: np.ndarray,
        condensed_labels: np.ndarray,
    ) -> None:
        self.condensed_memory.append((condensed_inputs.copy(), condensed_labels.copy()))
        if len(self.condensed_memory) > self.max_memory_rounds:
            self.condensed_memory = self.condensed_memory[-self.max_memory_rounds :]

    def _aggregate_logit_prototypes(
        self,
        prototype_sums: list[np.ndarray],
        prototype_counts: list[np.ndarray],
    ) -> None:
        total_sums = np.sum(np.stack(prototype_sums, axis=0), axis=0)
        total_counts = np.sum(np.stack(prototype_counts, axis=0), axis=0)
        if self.global_logit_prototypes.shape != total_sums.shape:
            self.global_logit_prototypes = np.zeros_like(total_sums, dtype=np.float32)
            self.num_classes = int(total_sums.shape[0])
        for class_index in range(self.num_classes):
            count = float(total_counts[class_index])
            if count > 0:
                self.global_logit_prototypes[class_index] = total_sums[class_index] / count

    def _run_server_replay(self, aggregated_parameters):
        set_model_parameters(self._checkpoint_model, parameters_to_ndarrays(aggregated_parameters))
        replay_metrics = replay_fedvck_server(
            self._checkpoint_model,
            self.condensed_memory,
            self.global_logit_prototypes,
            epochs=self.server_replay_epochs,
            learning_rate=self.server_replay_learning_rate,
            temperature=self.contrastive_temperature,
            hard_negative_k=self.hard_negative_k,
            device=self.device,
        )
        replayed_parameters = ndarrays_to_parameters(get_model_parameters(self._checkpoint_model))
        return replayed_parameters, replay_metrics


class FedVCKBuilder:
    name = "fedvck"

    def build_strategy(
        self,
        config: ExperimentConfig,
        initial_model: torch.nn.Module,
        evaluate_fn: ServerEvaluateFn | None,
    ) -> FedVCKStrategy:
        min_fit_clients = max(1, math.ceil(config.num_supernodes * config.fraction_train))
        min_eval_clients = max(1, math.ceil(config.num_supernodes * config.fraction_evaluate))
        initial_parameters = ndarrays_to_parameters(get_model_parameters(initial_model))

        strategy = FedVCKStrategy(
            fraction_fit=config.fraction_train,
            fraction_evaluate=config.fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_eval_clients,
            min_available_clients=config.num_supernodes,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=lambda _: {},
            fit_metrics_aggregation_fn=weighted_average,
            evaluate_metrics_aggregation_fn=weighted_average,
            initial_parameters=initial_parameters,
            num_classes=config.num_classes,
            checkpoint_model=initial_model,
            output_dir=config.output_dir,
            server_replay_epochs=config.fedvck_server_replay_epochs,
            server_replay_learning_rate=config.fedvck_server_replay_learning_rate,
            contrastive_temperature=config.fedvck_contrastive_temperature,
            hard_negative_k=min(config.fedvck_hard_negative_k, max(config.num_classes - 1, 1)),
            max_memory_rounds=config.fedvck_max_memory_rounds,
            device=config.device,
        )
        strategy.on_fit_config_fn = strategy.make_fit_config(config)
        return strategy
