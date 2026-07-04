"""GAMF algorithm builder and graph-matching aggregation strategy."""

from __future__ import annotations

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
class GAMFLayerSpec:
    weight_index: int
    bias_index: int
    next_weight_index: int | None = None
    next_kind: str | None = None
    block_size: int = 1


def gamf_layer_plan(model: torch.nn.Module) -> list[GAMFLayerSpec]:
    if isinstance(model, MnistCnn):
        return [
            GAMFLayerSpec(0, 1, next_weight_index=2, next_kind="conv"),
            GAMFLayerSpec(2, 3, next_weight_index=4, next_kind="flatten", block_size=49),
            GAMFLayerSpec(4, 5, next_weight_index=6, next_kind="linear"),
        ]
    if isinstance(model, LeNet):
        return [
            GAMFLayerSpec(0, 1, next_weight_index=2, next_kind="conv"),
            GAMFLayerSpec(2, 3, next_weight_index=4, next_kind="flatten", block_size=25),
            GAMFLayerSpec(4, 5, next_weight_index=6, next_kind="linear"),
            GAMFLayerSpec(6, 7, next_weight_index=8, next_kind="linear"),
        ]
    raise ValueError(
        "GAMF currently supports only mnist_cnn and lenet because the repository does not "
        "yet expose a safe graph-matching path for residual/batchnorm architectures."
    )


class GAMFStrategy(FedAvg):
    """Graph-matching-based server aggregation for FL."""

    def __init__(
        self,
        *,
        checkpoint_model: torch.nn.Module,
        output_dir: str,
        sigma: float,
        initial_tau: float,
        descent_factor: float,
        min_tau: float,
        max_iters: int,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self._checkpoint_model = checkpoint_model
        self._output_dir = output_dir
        self.sigma = sigma
        self.initial_tau = initial_tau
        self.descent_factor = descent_factor
        self.min_tau = min_tau
        self.max_iters = max_iters
        self.layer_plan = gamf_layer_plan(checkpoint_model)

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        client_payloads = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        aligned_payloads = self._align_clients(client_payloads)
        aggregated_ndarrays = aggregate(aligned_payloads)
        aggregated_parameters = ndarrays_to_parameters(aggregated_ndarrays)

        metrics_aggregated: dict[str, Scalar] = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(fit_res.num_examples, fit_res.metrics) for _, fit_res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        metrics_aggregated = dict(metrics_aggregated)
        metrics_aggregated["gamf_num_matched_layers"] = len(self.layer_plan)

        save_round_checkpoints(
            self._checkpoint_model,
            aggregated_parameters,
            self._output_dir,
            server_round,
        )
        return aggregated_parameters, metrics_aggregated

    def _align_clients(
        self,
        payloads: list[tuple[list[np.ndarray], int]],
    ) -> list[tuple[list[np.ndarray], int]]:
        aligned_parameters = [[parameter.copy() for parameter in parameters] for parameters, _ in payloads]
        client_weights = [num_examples for _, num_examples in payloads]

        for spec in self.layer_plan:
            permutations = _solve_multi_client_permutations(
                aligned_parameters,
                spec,
                sigma=self.sigma,
                initial_tau=self.initial_tau,
                descent_factor=self.descent_factor,
                min_tau=self.min_tau,
                max_iters=self.max_iters,
            )
            for parameters, permutation in zip(aligned_parameters, permutations):
                _apply_output_permutation(parameters, spec, permutation)
                if spec.next_weight_index is not None and spec.next_kind is not None:
                    _apply_input_permutation(
                        parameters[spec.next_weight_index],
                        permutation,
                        kind=spec.next_kind,
                        block_size=spec.block_size,
                    )

        return list(zip(aligned_parameters, client_weights))


def _solve_multi_client_permutations(
    payloads: list[list[np.ndarray]],
    spec: GAMFLayerSpec,
    *,
    sigma: float,
    initial_tau: float,
    descent_factor: float,
    min_tau: float,
    max_iters: int,
) -> list[np.ndarray]:
    num_clients = len(payloads)
    num_channels = payloads[0][spec.weight_index].shape[0]
    permutations = [np.arange(num_channels, dtype=np.int64) for _ in range(num_clients)]
    if num_clients <= 1:
        return permutations
    tau = initial_tau

    for _ in range(max_iters):
        updated = False
        for client_index, parameters in enumerate(payloads[1:], start=1):
            consensus_incoming, consensus_outgoing = _consensus_signatures(
                payloads,
                permutations,
                spec,
                skip_index=client_index,
            )
            incoming, outgoing = _incoming_outgoing(parameters, spec)
            scores = _similarity_scores(
                consensus_incoming,
                consensus_outgoing,
                incoming,
                outgoing,
                sigma=sigma,
            )
            soft_scores = _sinkhorn(scores, tau=tau)
            permutation = _hungarian_from_scores(soft_scores)
            if not np.array_equal(permutation, permutations[client_index]):
                updated = True
            permutations[client_index] = permutation
        if not updated and tau <= min_tau:
            break
        tau = max(min_tau, tau * descent_factor)

    return permutations


def _consensus_signatures(
    payloads: list[list[np.ndarray]],
    permutations: list[np.ndarray],
    spec: GAMFLayerSpec,
    *,
    skip_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    incoming_accumulator = None
    outgoing_accumulator = None
    count = 0

    for index, (parameters, permutation) in enumerate(zip(payloads, permutations)):
        if index == skip_index:
            continue
        incoming, outgoing = _incoming_outgoing(parameters, spec)
        aligned_incoming = incoming[:, permutation]
        aligned_outgoing = outgoing[permutation, :]
        incoming_accumulator = (
            aligned_incoming if incoming_accumulator is None else incoming_accumulator + aligned_incoming
        )
        outgoing_accumulator = (
            aligned_outgoing if outgoing_accumulator is None else outgoing_accumulator + aligned_outgoing
        )
        count += 1

    if count == 0:
        incoming, outgoing = _incoming_outgoing(payloads[skip_index], spec)
        return incoming, outgoing
    return incoming_accumulator / count, outgoing_accumulator / count


def _incoming_outgoing(
    parameters: list[np.ndarray],
    spec: GAMFLayerSpec,
) -> tuple[np.ndarray, np.ndarray]:
    incoming = parameters[spec.weight_index].reshape(parameters[spec.weight_index].shape[0], -1).T
    if spec.next_weight_index is None or spec.next_kind is None:
        raise ValueError("GAMF hidden layer spec requires next-layer structure")
    next_weight = parameters[spec.next_weight_index]
    if spec.next_kind == "conv":
        outgoing = np.transpose(next_weight, (1, 0, 2, 3)).reshape(next_weight.shape[1], -1)
    elif spec.next_kind == "linear":
        outgoing = next_weight.T
    elif spec.next_kind == "flatten":
        outgoing = next_weight.reshape(next_weight.shape[0], incoming.shape[1], spec.block_size)
        outgoing = np.transpose(outgoing, (1, 0, 2)).reshape(incoming.shape[1], -1)
    else:
        raise ValueError(f"Unsupported GAMF next-layer kind: {spec.next_kind}")
    return incoming, outgoing


def _similarity_scores(
    consensus_incoming: np.ndarray,
    consensus_outgoing: np.ndarray,
    incoming: np.ndarray,
    outgoing: np.ndarray,
    *,
    sigma: float,
) -> np.ndarray:
    incoming_distance = np.sum(
        (consensus_incoming.T[:, None, :] - incoming.T[None, :, :]) ** 2,
        axis=2,
    )
    outgoing_distance = np.sum(
        (consensus_outgoing[:, None, :] - outgoing[None, :, :]) ** 2,
        axis=2,
    )
    return np.exp(-incoming_distance / sigma) + np.exp(-outgoing_distance / sigma)


def _sinkhorn(scores: np.ndarray, *, tau: float, rounds: int = 20) -> np.ndarray:
    scaled = scores / max(tau, 1e-12)
    scaled = scaled - np.max(scaled)
    matrix = np.exp(scaled)
    matrix = np.maximum(matrix, 1e-12)
    for _ in range(rounds):
        matrix /= np.maximum(matrix.sum(axis=1, keepdims=True), 1e-12)
        matrix /= np.maximum(matrix.sum(axis=0, keepdims=True), 1e-12)
    return matrix


def _hungarian_from_scores(scores: np.ndarray) -> np.ndarray:
    if not np.isfinite(scores).all():
        scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
    row_ind, col_ind = linear_sum_assignment(-scores)
    permutation = np.empty(scores.shape[0], dtype=np.int64)
    permutation[col_ind] = row_ind
    return permutation


def _apply_output_permutation(
    parameters: list[np.ndarray],
    spec: GAMFLayerSpec,
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
    raise ValueError(f"Unsupported GAMF input permutation kind: {kind}")


class GAMFBuilder:
    name = "gamf"

    def build_strategy(
        self,
        config: ExperimentConfig,
        initial_model: torch.nn.Module,
        evaluate_fn: ServerEvaluateFn | None,
    ) -> GAMFStrategy:
        min_fit_clients = max(1, math.ceil(config.num_supernodes * config.fraction_train))
        min_eval_clients = max(1, math.ceil(config.num_supernodes * config.fraction_evaluate))
        initial_parameters = ndarrays_to_parameters(_model_parameters(initial_model))
        gamf_layer_plan(initial_model)

        return GAMFStrategy(
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
            sigma=config.gamf_sigma,
            initial_tau=config.gamf_initial_tau,
            descent_factor=config.gamf_descent_factor,
            min_tau=config.gamf_min_tau,
            max_iters=config.gamf_max_iters,
        )

    def _fit_config(self, config: ExperimentConfig):
        def fn(server_round: int) -> dict[str, bool | bytes | float | int | str]:
            return {
                "algorithm": self.name,
                "server_round": server_round,
                "local_epochs": config.local_epochs,
                "learning_rate": config.learning_rate,
            }

        return fn


def _model_parameters(model: torch.nn.Module) -> list[np.ndarray]:
    return [value.detach().cpu().numpy() for value in model.state_dict().values()]
