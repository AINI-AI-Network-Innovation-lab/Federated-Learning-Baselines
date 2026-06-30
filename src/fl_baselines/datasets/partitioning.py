"""Deterministic client partition helpers."""

from __future__ import annotations

import numpy as np


def create_iid_partitions(
    num_samples: int,
    num_partitions: int,
    seed: int,
) -> list[np.ndarray]:
    if num_partitions <= 0:
        raise ValueError("num_partitions must be positive")
    if num_samples < 0:
        raise ValueError("num_samples must be non-negative")

    rng = np.random.default_rng(seed)
    indices = np.arange(num_samples)
    rng.shuffle(indices)
    return [partition.astype(np.int64) for partition in np.array_split(indices, num_partitions)]


def create_dirichlet_partitions(
    targets: np.ndarray,
    num_partitions: int,
    alpha: float,
    seed: int,
) -> list[np.ndarray]:
    if num_partitions <= 0:
        raise ValueError("num_partitions must be positive")
    if alpha <= 0:
        raise ValueError("alpha must be positive")

    targets = np.asarray(targets)
    rng = np.random.default_rng(seed)
    partitions: list[list[int]] = [[] for _ in range(num_partitions)]

    for class_id in np.unique(targets):
        class_indices = np.flatnonzero(targets == class_id)
        rng.shuffle(class_indices)
        proportions = rng.dirichlet(np.full(num_partitions, alpha))
        split_points = (np.cumsum(proportions)[:-1] * len(class_indices)).astype(int)

        for partition_id, split in enumerate(np.split(class_indices, split_points)):
            partitions[partition_id].extend(int(index) for index in split)

    result = []
    for partition in partitions:
        partition_array = np.array(partition, dtype=np.int64)
        rng.shuffle(partition_array)
        result.append(partition_array)
    return _rebalance_empty_partitions(result)


def _rebalance_empty_partitions(partitions: list[np.ndarray]) -> list[np.ndarray]:
    total_samples = sum(len(partition) for partition in partitions)
    if total_samples < len(partitions):
        return partitions

    rebalanced = [partition.copy() for partition in partitions]
    empty_partition_ids = [
        partition_id for partition_id, partition in enumerate(rebalanced) if len(partition) == 0
    ]

    for empty_partition_id in empty_partition_ids:
        donor_partition_id = max(
            range(len(rebalanced)),
            key=lambda partition_id: len(rebalanced[partition_id]),
        )
        if len(rebalanced[donor_partition_id]) <= 1:
            break

        moved_sample = rebalanced[donor_partition_id][-1:]
        rebalanced[donor_partition_id] = rebalanced[donor_partition_id][:-1]
        rebalanced[empty_partition_id] = moved_sample

    return rebalanced


def create_partitions(
    targets: np.ndarray,
    num_partitions: int,
    partitioner: str,
    dirichlet_alpha: float,
    seed: int,
) -> list[np.ndarray]:
    if partitioner == "iid":
        return create_iid_partitions(len(targets), num_partitions, seed)
    if partitioner == "dirichlet":
        return create_dirichlet_partitions(targets, num_partitions, dirichlet_alpha, seed)
    raise ValueError("partitioner must be one of: iid, dirichlet")
