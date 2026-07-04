"""FedMA algorithm builder and layer-wise matching strategy."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np
import torch
from flwr.common import Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate
from scipy.optimize import linear_sum_assignment

from fl_baselines.core.config import ExperimentConfig
from fl_baselines.core.types import ServerEvaluateFn
from fl_baselines.logging.checkpointing import save_round_checkpoints
from fl_baselines.logging.metrics import weighted_average
from fl_baselines.models.lenet import LeNet
from fl_baselines.models.mnist_cnn import MnistCnn


@dataclass(frozen=True)
class FedMALayerSpec:
    prefix: str
    weight_index: int
    bias_index: int
    next_weight_index: int | None = None
    next_kind: str | None = None
    block_size: int = 1
    is_final: bool = False


def fedma_layer_plan(model: torch.nn.Module) -> list[FedMALayerSpec]:
    if isinstance(model, MnistCnn):
        return [
            FedMALayerSpec("conv1", 0, 1, next_weight_index=2, next_kind="conv"),
            FedMALayerSpec("conv2", 2, 3, next_weight_index=4, next_kind="flatten", block_size=49),
            FedMALayerSpec("fc1", 4, 5, next_weight_index=6, next_kind="linear"),
            FedMALayerSpec("fc2", 6, 7, is_final=True),
        ]
    if isinstance(model, LeNet):
        return [
            FedMALayerSpec("conv1", 0, 1, next_weight_index=2, next_kind="conv"),
            FedMALayerSpec("conv2", 2, 3, next_weight_index=4, next_kind="flatten", block_size=25),
            FedMALayerSpec("fc1", 4, 5, next_weight_index=6, next_kind="linear"),
            FedMALayerSpec("fc2", 6, 7, next_weight_index=8, next_kind="linear"),
            FedMALayerSpec("fc3", 8, 9, is_final=True),
        ]
    raise ValueError(
        "FedMA currently supports only mnist_cnn and lenet because residual and batchnorm "
        "architectures are outside the paper's supported matching scope in this codebase."
    )


def fedma_frozen_prefixes(model: torch.nn.Module, stage: int) -> list[str]:
    plan = fedma_layer_plan(model)
    clamped_stage = max(0, min(stage, len(plan) - 1))
    return [spec.prefix for spec in plan[:clamped_stage]]


class FedMAStrategy(FedAvg):
    """Layer-wise matched averaging for fixed-width CNN/MLP baselines."""

    def __init__(
        self,
        *,
        checkpoint_model: torch.nn.Module,
        output_dir: str,
        matching_epsilon: float,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self._checkpoint_model = checkpoint_model
        self._output_dir = output_dir
        self.matching_epsilon = matching_epsilon
        self.layer_plan = fedma_layer_plan(checkpoint_model)

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        stage = min(server_round - 1, len(self.layer_plan) - 1)
        spec = self.layer_plan[stage]
        total_examples = sum(fit_res.num_examples for _, fit_res in results)
        if total_examples <= 0:
            return None, {}

        client_payloads = [
            (
                parameters_to_ndarrays(fit_res.parameters),
                fit_res.num_examples,
                fit_res.metrics,
            )
            for _, fit_res in results
        ]
        aligned_payloads = self._align_stage_payloads(client_payloads, spec)
        aggregated_ndarrays = aggregate([(payload, num_examples) for payload, num_examples, _ in aligned_payloads])

        if spec.is_final:
            self._aggregate_final_classifier(aggregated_ndarrays, aligned_payloads, spec, total_examples)

        aggregated_parameters = ndarrays_to_parameters(aggregated_ndarrays)
        metrics_aggregated: dict[str, Scalar] = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(fit_res.num_examples, fit_res.metrics) for _, fit_res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        metrics_aggregated = dict(metrics_aggregated)
        metrics_aggregated["fedma_stage"] = stage

        save_round_checkpoints(
            self._checkpoint_model,
            aggregated_parameters,
            self._output_dir,
            server_round,
        )
        return aggregated_parameters, metrics_aggregated

    def _align_stage_payloads(
        self,
        payloads: list[tuple[list[np.ndarray], int, dict[str, Scalar]]],
        spec: FedMALayerSpec,
    ) -> list[tuple[list[np.ndarray], int, dict[str, Scalar]]]:
        reference_parameters = payloads[0][0]
        reference_signature = _layer_signature(reference_parameters, spec)
        aligned: list[tuple[list[np.ndarray], int, dict[str, Scalar]]] = []

        for parameters, num_examples, metrics in payloads:
            copied = [parameter.copy() for parameter in parameters]
            permutation = _match_permutation(reference_signature, _layer_signature(copied, spec))
            _apply_output_permutation(copied, spec, permutation)
            if spec.next_weight_index is not None and spec.next_kind is not None:
                _apply_input_permutation(
                    copied[spec.next_weight_index],
                    permutation,
                    kind=spec.next_kind,
                    block_size=spec.block_size,
                )
            aligned.append((copied, num_examples, metrics))
        return aligned

    def _aggregate_final_classifier(
        self,
        aggregated_ndarrays: list[np.ndarray],
        payloads: list[tuple[list[np.ndarray], int, dict[str, Scalar]]],
        spec: FedMALayerSpec,
        total_examples: int,
    ) -> None:
        weight = aggregated_ndarrays[spec.weight_index]
        bias = aggregated_ndarrays[spec.bias_index]
        num_classes = weight.shape[0]
        class_totals = np.zeros(num_classes, dtype=np.float64)
        weight_accumulator = np.zeros_like(weight, dtype=np.float64)
        bias_accumulator = np.zeros_like(bias, dtype=np.float64)

        for parameters, num_examples, metrics in payloads:
            counts = _label_counts(metrics, num_classes)
            if counts is None:
                counts = np.full(num_classes, num_examples / max(num_classes, 1), dtype=np.float64)
            class_totals += counts
            weight_accumulator += parameters[spec.weight_index].astype(np.float64, copy=False) * counts[:, None]
            bias_accumulator += parameters[spec.bias_index].astype(np.float64, copy=False) * counts

        fallback_weights = np.full(num_classes, total_examples / max(num_classes, 1), dtype=np.float64)
        class_totals = np.where(class_totals > 0, class_totals, fallback_weights)
        aggregated_ndarrays[spec.weight_index] = (weight_accumulator / class_totals[:, None]).astype(weight.dtype)
        aggregated_ndarrays[spec.bias_index] = (bias_accumulator / class_totals).astype(bias.dtype)


def _layer_signature(parameters: list[np.ndarray], spec: FedMALayerSpec) -> np.ndarray:
    weight = parameters[spec.weight_index].reshape(parameters[spec.weight_index].shape[0], -1)
    bias = parameters[spec.bias_index].reshape(-1, 1)
    return np.concatenate([weight, bias], axis=1)


def _match_permutation(reference: np.ndarray, local: np.ndarray) -> np.ndarray:
    cost = np.linalg.norm(local[:, None, :] - reference[None, :, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(cost)
    permutation = np.empty(local.shape[0], dtype=np.int64)
    permutation[col_ind] = row_ind
    return permutation


def _apply_output_permutation(
    parameters: list[np.ndarray],
    spec: FedMALayerSpec,
    permutation: np.ndarray,
) -> None:
    parameters[spec.weight_index] = parameters[spec.weight_index][permutation]
    parameters[spec.bias_index] = parameters[spec.bias_index][permutation]


def _apply_input_permutation(
    weight: np.ndarray,
    permutation: np.ndarray,
    *,
    kind: str,
    block_size: int,
) -> None:
    if kind == "conv":
        weight[...] = weight[:, permutation, ...]
        return
    if kind == "linear":
        weight[...] = weight[:, permutation]
        return
    if kind == "flatten":
        out_features = weight.shape[0]
        reshaped = weight.reshape(out_features, len(permutation), block_size)
        reshaped = reshaped[:, permutation, :]
        weight[...] = reshaped.reshape(weight.shape)
        return
    raise ValueError(f"Unsupported FedMA input permutation kind: {kind}")


def _label_counts(metrics: dict[str, Scalar], num_classes: int) -> np.ndarray | None:
    raw_counts = metrics.get("fedma_label_counts")
    if raw_counts is None:
        return None
    try:
        counts = np.asarray(json.loads(str(raw_counts)), dtype=np.float64)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if counts.shape != (num_classes,):
        return None
    return counts


class FedMABuilder:
    name = "fedma"

    def build_strategy(
        self,
        config: ExperimentConfig,
        initial_model: torch.nn.Module,
        evaluate_fn: ServerEvaluateFn | None,
    ) -> FedMAStrategy:
        min_fit_clients = max(1, math.ceil(config.num_supernodes * config.fraction_train))
        min_eval_clients = max(1, math.ceil(config.num_supernodes * config.fraction_evaluate))
        initial_parameters = ndarrays_to_parameters(_model_parameters(initial_model))
        fedma_layer_plan(initial_model)

        return FedMAStrategy(
            fraction_fit=config.fraction_train,
            fraction_evaluate=config.fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_eval_clients,
            min_available_clients=config.num_supernodes,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=self._fit_config(config, initial_model),
            fit_metrics_aggregation_fn=weighted_average,
            evaluate_metrics_aggregation_fn=weighted_average,
            initial_parameters=initial_parameters,
            checkpoint_model=initial_model,
            output_dir=config.output_dir,
            matching_epsilon=config.fedma_matching_epsilon,
        )

    def _fit_config(self, config: ExperimentConfig, model: torch.nn.Module):
        num_stages = len(fedma_layer_plan(model))

        def fn(server_round: int) -> dict[str, bool | bytes | float | int | str]:
            return {
                "algorithm": self.name,
                "server_round": server_round,
                "local_epochs": config.local_epochs,
                "learning_rate": config.learning_rate,
                "fedma_stage": min(server_round - 1, num_stages - 1),
                "fedma_matching_epsilon": config.fedma_matching_epsilon,
            }

        return fn


def _model_parameters(model: torch.nn.Module) -> list[np.ndarray]:
    return [value.detach().cpu().numpy() for value in model.state_dict().values()]
