"""FedSiKD algorithm builder and clustered knowledge-distillation strategy."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import torch
from flwr.common import FitIns, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg

from fl_baselines.clients.torch_client import get_model_parameters
from fl_baselines.core.config import ExperimentConfig
from fl_baselines.core.types import ServerEvaluateFn
from fl_baselines.logging.checkpointing import save_round_checkpoints
from fl_baselines.logging.metrics import weighted_average
from fl_baselines.training.fedsikd import train_fedsikd_client


def cluster_fedsikd_clients(
    client_statistics: dict[str, tuple[float, float, float]],
    *,
    num_clusters: int,
    max_clusters: int,
) -> dict[str, int]:
    """Cluster clients from their (mean, std, skewness) statistics."""

    if not client_statistics:
        return {}

    client_ids = sorted(client_statistics)
    if len(client_ids) == 1:
        return {client_ids[0]: 0}

    feature_matrix = np.asarray(
        [client_statistics[client_id] for client_id in client_ids],
        dtype=np.float64,
    )
    standardized = _standardize_features(feature_matrix)

    if num_clusters > 0:
        cluster_count = min(num_clusters, len(client_ids))
        labels = _kmeans(standardized, cluster_count)
        return {client_id: int(label) for client_id, label in zip(client_ids, labels)}

    upper_bound = min(max_clusters, len(client_ids))
    if upper_bound < 2:
        return {client_id: 0 for client_id in client_ids}

    best_score: tuple[float, float, float] | None = None
    best_labels = np.zeros(len(client_ids), dtype=np.int64)
    for cluster_count in range(2, upper_bound + 1):
        labels = _kmeans(standardized, cluster_count)
        silhouette = _silhouette_score(standardized, labels)
        calinski = _calinski_harabasz_score(standardized, labels)
        davies = _davies_bouldin_score(standardized, labels)
        candidate = (silhouette, calinski, -davies)
        if best_score is None or candidate > best_score:
            best_score = candidate
            best_labels = labels

    return {client_id: int(label) for client_id, label in zip(client_ids, best_labels)}


def _standardize_features(feature_matrix: np.ndarray) -> np.ndarray:
    mean = feature_matrix.mean(axis=0, keepdims=True)
    std = feature_matrix.std(axis=0, keepdims=True)
    std = np.where(std > 1e-12, std, 1.0)
    return (feature_matrix - mean) / std


def _kmeans(feature_matrix: np.ndarray, cluster_count: int, max_iters: int = 50) -> np.ndarray:
    if cluster_count <= 1:
        return np.zeros(feature_matrix.shape[0], dtype=np.int64)

    rng = np.random.default_rng(0)
    initial_indices = rng.choice(feature_matrix.shape[0], size=cluster_count, replace=False)
    centroids = feature_matrix[initial_indices].copy()
    labels = np.full(feature_matrix.shape[0], -1, dtype=np.int64)

    for _ in range(max_iters):
        distances = np.linalg.norm(
            feature_matrix[:, None, :] - centroids[None, :, :],
            axis=2,
        )
        new_labels = np.argmin(distances, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        for centroid_index in range(cluster_count):
            members = feature_matrix[labels == centroid_index]
            if members.size == 0:
                farthest_point = int(np.argmax(np.min(distances, axis=1)))
                centroids[centroid_index] = feature_matrix[farthest_point]
            else:
                centroids[centroid_index] = members.mean(axis=0)

    distances = np.linalg.norm(
        feature_matrix[:, None, :] - centroids[None, :, :],
        axis=2,
    )
    return np.argmin(distances, axis=1).astype(np.int64)


def _silhouette_score(feature_matrix: np.ndarray, labels: np.ndarray) -> float:
    unique_labels = np.unique(labels)
    if unique_labels.size < 2:
        return 0.0

    distances = np.linalg.norm(
        feature_matrix[:, None, :] - feature_matrix[None, :, :],
        axis=2,
    )
    silhouettes: list[float] = []
    for index, label in enumerate(labels):
        same_cluster = labels == label
        same_cluster[index] = False
        if np.any(same_cluster):
            a = float(distances[index, same_cluster].mean())
        else:
            a = 0.0

        b = float("inf")
        for other_label in unique_labels:
            if other_label == label:
                continue
            other_cluster = labels == other_label
            if np.any(other_cluster):
                b = min(b, float(distances[index, other_cluster].mean()))
        if not np.isfinite(b):
            b = 0.0

        denominator = max(a, b)
        silhouettes.append(0.0 if denominator == 0.0 else (b - a) / denominator)
    return float(np.mean(silhouettes))


def _calinski_harabasz_score(feature_matrix: np.ndarray, labels: np.ndarray) -> float:
    unique_labels = np.unique(labels)
    cluster_count = unique_labels.size
    sample_count = feature_matrix.shape[0]
    if cluster_count < 2 or sample_count <= cluster_count:
        return 0.0

    overall_mean = feature_matrix.mean(axis=0)
    between_cluster = 0.0
    within_cluster = 0.0
    for label in unique_labels:
        cluster_points = feature_matrix[labels == label]
        if cluster_points.size == 0:
            continue
        centroid = cluster_points.mean(axis=0)
        between_cluster += cluster_points.shape[0] * float(
            np.sum((centroid - overall_mean) ** 2)
        )
        within_cluster += float(np.sum((cluster_points - centroid) ** 2))

    if within_cluster == 0.0:
        return float("inf")
    return float(
        (between_cluster / (cluster_count - 1))
        / (within_cluster / (sample_count - cluster_count))
    )


def _davies_bouldin_score(feature_matrix: np.ndarray, labels: np.ndarray) -> float:
    unique_labels = np.unique(labels)
    cluster_count = unique_labels.size
    if cluster_count < 2:
        return float("inf")

    centroids = []
    scatters = []
    for label in unique_labels:
        cluster_points = feature_matrix[labels == label]
        if cluster_points.size == 0:
            continue
        centroid = cluster_points.mean(axis=0)
        centroids.append(centroid)
        scatters.append(float(np.linalg.norm(cluster_points - centroid, axis=1).mean()))

    if len(centroids) < 2:
        return float("inf")

    centroids_array = np.asarray(centroids, dtype=np.float64)
    db_values: list[float] = []
    for index in range(len(centroids_array)):
        ratios = []
        for other_index in range(len(centroids_array)):
            if other_index == index:
                continue
            centroid_distance = float(
                np.linalg.norm(centroids_array[index] - centroids_array[other_index])
            )
            if centroid_distance == 0.0:
                continue
            ratios.append((scatters[index] + scatters[other_index]) / centroid_distance)
        if ratios:
            db_values.append(max(ratios))
    if not db_values:
        return float("inf")
    return float(np.mean(db_values))


def _extract_client_statistics(metrics: dict[str, Scalar]) -> tuple[float, float, float] | None:
    try:
        return (
            float(metrics["fedsikd_mean"]),
            float(metrics["fedsikd_std"]),
            float(metrics["fedsikd_skewness"]),
        )
    except KeyError:
        return None


class FedSiKDStrategy(FedAvg):
    """FedAvg-compatible strategy with clustering-aware teacher distillation."""

    def __init__(
        self,
        *,
        num_clusters: int,
        max_clusters: int,
        kd_alpha: float,
        kd_temperature: float,
        checkpoint_model: torch.nn.Module,
        output_dir: str,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        if self.initial_parameters is None:
            raise ValueError("FedSiKDStrategy requires initial_parameters")
        if num_clusters < 0:
            raise ValueError("FedSiKD num_clusters must be non-negative")
        if max_clusters <= 0:
            raise ValueError("FedSiKD max_clusters must be positive")
        if kd_alpha < 0:
            raise ValueError("FedSiKD kd_alpha must be non-negative")
        if kd_temperature <= 0:
            raise ValueError("FedSiKD kd_temperature must be positive")

        self.num_clusters = num_clusters
        self.max_clusters = max_clusters
        self.kd_alpha = kd_alpha
        self.kd_temperature = kd_temperature
        self._checkpoint_model = checkpoint_model
        self._output_dir = output_dir
        self._student_parameters = [
            array.copy() for array in parameters_to_ndarrays(self.initial_parameters)
        ]
        self._cluster_teacher_parameters: dict[int, list[np.ndarray]] = {
            0: [array.copy() for array in self._student_parameters]
        }
        self._client_clusters: dict[str, int] = {}

    def configure_fit(self, server_round, parameters, client_manager):
        client_fit_ins = super().configure_fit(server_round, parameters, client_manager)
        student_parameters = [array.copy() for array in parameters_to_ndarrays(parameters)]

        fit_instructions = []
        for client, fit_ins in client_fit_ins:
            cluster_id = self._client_clusters.get(client.cid)
            teacher_parameters = self._cluster_teacher_parameters.get(
                cluster_id if cluster_id is not None else 0,
                student_parameters,
            )
            combined_parameters = ndarrays_to_parameters(
                student_parameters + [array.copy() for array in teacher_parameters]
            )
            fit_config = dict(fit_ins.config)
            fit_config.update(
                {
                    "algorithm": self.name,
                    "server_round": server_round,
                    "local_epochs": self._resolve_local_epochs(),
                    "learning_rate": self._resolve_learning_rate(),
                    "fedsikd_kd_alpha": self.kd_alpha,
                    "fedsikd_kd_temperature": self.kd_temperature,
                    "fedsikd_num_clusters": self.num_clusters,
                    "fedsikd_cluster_id": -1 if cluster_id is None else cluster_id,
                }
            )
            fit_instructions.append((client, FitIns(combined_parameters, fit_config)))
        return fit_instructions

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        model_parameter_count = len(self._student_parameters)
        client_statistics: dict[str, tuple[float, float, float]] = {}
        client_payloads: list[tuple[str, list[np.ndarray], int, dict[str, Scalar]]] = []

        for client_proxy, fit_res in results:
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            if len(ndarrays) != model_parameter_count * 2:
                raise ValueError("FedSiKD client result must contain student and teacher parameters")
            client_payloads.append(
                (
                    client_proxy.cid,
                    [array.copy() for array in ndarrays[:model_parameter_count]],
                    fit_res.num_examples,
                    fit_res.metrics,
                )
            )
            stats = _extract_client_statistics(fit_res.metrics)
            if stats is not None:
                client_statistics[client_proxy.cid] = stats

        if client_statistics:
            self._client_clusters = cluster_fedsikd_clients(
                client_statistics,
                num_clusters=self.num_clusters,
                max_clusters=self.max_clusters,
            )
        else:
            self._client_clusters = {client_id: 0 for client_id, *_ in client_payloads}

        cluster_payloads: dict[int, list[tuple[list[np.ndarray], int]]] = defaultdict(list)
        for client_id, student_parameters, num_examples, _ in client_payloads:
            cluster_id = self._client_clusters.get(client_id, 0)
            cluster_payloads[cluster_id].append((student_parameters, num_examples))

        if not cluster_payloads:
            return None, {}

        cluster_teacher_parameters: dict[int, list[np.ndarray]] = {}
        occupied_clusters = sorted(cluster_payloads)
        for cluster_id in occupied_clusters:
            cluster_teacher_parameters[cluster_id] = _weighted_average_parameters(
                cluster_payloads[cluster_id]
            )

        if len(cluster_teacher_parameters) == 1:
            global_student_parameters = [array.copy() for array in next(iter(cluster_teacher_parameters.values()))]
        else:
            global_student_parameters = _mean_parameters(
                list(cluster_teacher_parameters.values())
            )

        self._cluster_teacher_parameters = {
            cluster_id: [array.copy() for array in parameters]
            for cluster_id, parameters in cluster_teacher_parameters.items()
        }
        self._student_parameters = [array.copy() for array in global_student_parameters]

        aggregated_parameters = ndarrays_to_parameters(global_student_parameters)
        metrics_aggregated: dict[str, Scalar] = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(fit_res.num_examples, fit_res.metrics) for _, fit_res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        metrics_aggregated = dict(metrics_aggregated)
        metrics_aggregated["fedsikd_cluster_count"] = len(cluster_teacher_parameters)
        metrics_aggregated["fedsikd_client_count"] = len(client_payloads)
        metrics_aggregated["fedsikd_distilled_client_count"] = len(client_statistics)
        metrics_aggregated["fedsikd_kd_alpha"] = self.kd_alpha
        metrics_aggregated["fedsikd_kd_temperature"] = self.kd_temperature

        save_round_checkpoints(
            self._checkpoint_model,
            aggregated_parameters,
            self._output_dir,
            server_round,
        )
        return aggregated_parameters, metrics_aggregated

    def _resolve_local_epochs(self) -> int:
        if self.on_fit_config_fn is None:
            return 1
        fit_config = self.on_fit_config_fn(1)
        return int(fit_config.get("local_epochs", 1))

    def _resolve_learning_rate(self) -> float:
        if self.on_fit_config_fn is None:
            return 0.01
        fit_config = self.on_fit_config_fn(1)
        return float(fit_config.get("learning_rate", 0.01))


def _weighted_average_parameters(
    payloads: list[tuple[list[np.ndarray], int]],
) -> list[np.ndarray]:
    total_examples = sum(num_examples for _, num_examples in payloads)
    if total_examples <= 0:
        return [array.copy() for array in payloads[0][0]]

    aggregated = []
    parameter_count = len(payloads[0][0])
    for parameter_index in range(parameter_count):
        accumulator = sum(
            parameters[parameter_index] * (num_examples / total_examples)
            for parameters, num_examples in payloads
        )
        aggregated.append(np.asarray(accumulator))
    return aggregated


def _mean_parameters(parameter_sets: list[list[np.ndarray]]) -> list[np.ndarray]:
    if not parameter_sets:
        return []
    parameter_count = len(parameter_sets[0])
    return [
        np.mean([parameter_set[index] for parameter_set in parameter_sets], axis=0)
        for index in range(parameter_count)
    ]


class FedSiKDBuilder:
    name = "fedsikd"

    def build_strategy(
        self,
        config: ExperimentConfig,
        initial_model: torch.nn.Module,
        evaluate_fn: ServerEvaluateFn | None,
    ) -> FedAvg:
        min_fit_clients = max(1, math.ceil(config.num_supernodes * config.fraction_train))
        min_eval_clients = max(1, math.ceil(config.num_supernodes * config.fraction_evaluate))
        initial_parameters = ndarrays_to_parameters(get_model_parameters(initial_model))

        return FedSiKDStrategy(
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
            num_clusters=config.fedsikd_num_clusters,
            max_clusters=config.fedsikd_max_clusters,
            kd_alpha=config.fedsikd_kd_alpha,
            kd_temperature=config.fedsikd_kd_temperature,
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
                "fedsikd_kd_alpha": config.fedsikd_kd_alpha,
                "fedsikd_kd_temperature": config.fedsikd_kd_temperature,
                "fedsikd_num_clusters": config.fedsikd_num_clusters,
            }

        return fn
