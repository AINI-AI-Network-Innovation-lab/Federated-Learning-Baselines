"""FedLAMA client-side helpers."""

from __future__ import annotations

import json
from typing import Any

import numpy as np


def parse_fedlama_sync_mask(raw_sync_mask: Any, num_layers: int) -> list[bool]:
    if raw_sync_mask is None:
        return [True] * num_layers
    if isinstance(raw_sync_mask, bytes):
        raw_sync_mask = raw_sync_mask.decode("utf-8")
    if isinstance(raw_sync_mask, str):
        parsed = json.loads(raw_sync_mask)
    elif isinstance(raw_sync_mask, (list, tuple)):
        parsed = list(raw_sync_mask)
    else:
        raise ValueError("FedLAMA sync mask must be a JSON list or sequence")

    if len(parsed) != num_layers:
        raise ValueError("FedLAMA sync mask length must match the model tensor count")
    return [bool(flag) for flag in parsed]


def parse_fedlama_layer_intervals(raw_intervals: Any, num_layers: int) -> list[int]:
    if raw_intervals is None:
        return [1] * num_layers
    if isinstance(raw_intervals, bytes):
        raw_intervals = raw_intervals.decode("utf-8")
    if isinstance(raw_intervals, str):
        parsed = json.loads(raw_intervals)
    elif isinstance(raw_intervals, (list, tuple)):
        parsed = list(raw_intervals)
    else:
        raise ValueError("FedLAMA layer intervals must be a JSON list or sequence")

    if len(parsed) != num_layers:
        raise ValueError("FedLAMA layer interval count must match the model tensor count")

    intervals = [int(interval) for interval in parsed]
    if any(interval <= 0 for interval in intervals):
        raise ValueError("FedLAMA layer intervals must be positive")
    return intervals


def fedlama_sync_mask_for_round(
    server_round: int,
    layer_intervals: list[int],
) -> list[bool]:
    if server_round <= 0:
        raise ValueError("FedLAMA server round must be positive")
    if not layer_intervals:
        return []

    sync_mask = [
        server_round == 1 or (server_round - 1) % interval == 0
        for interval in layer_intervals
    ]
    if any(sync_mask):
        return sync_mask

    fallback_index = min(range(len(layer_intervals)), key=layer_intervals.__getitem__)
    sync_mask[fallback_index] = True
    return sync_mask


def select_fedlama_parameters(
    parameters: list[np.ndarray],
    sync_mask: list[bool],
) -> list[np.ndarray]:
    if len(parameters) != len(sync_mask):
        raise ValueError("FedLAMA parameter count must match the sync mask")
    return [parameter.copy() for parameter, should_sync in zip(parameters, sync_mask) if should_sync]


def merge_fedlama_parameters(
    server_parameters: list[np.ndarray],
    local_parameters: list[np.ndarray],
    sync_mask: list[bool],
) -> list[np.ndarray]:
    if len(server_parameters) != len(sync_mask) or len(local_parameters) != len(sync_mask):
        raise ValueError("FedLAMA parameter count must match the sync mask")

    merged: list[np.ndarray] = []
    for server_parameter, local_parameter, should_sync in zip(
        server_parameters,
        local_parameters,
        sync_mask,
    ):
        merged.append(server_parameter.copy() if should_sync else local_parameter.copy())
    return merged


def compute_fedlama_layer_discrepancies(
    server_parameters: list[np.ndarray],
    local_parameters: list[np.ndarray],
    layer_intervals: list[int],
) -> list[float]:
    if not (
        len(server_parameters) == len(local_parameters) == len(layer_intervals)
    ):
        raise ValueError("FedLAMA discrepancy inputs must have matching lengths")

    discrepancies: list[float] = []
    for server_parameter, local_parameter, interval in zip(
        server_parameters,
        local_parameters,
        layer_intervals,
    ):
        local_array = local_parameter.astype(np.float64, copy=False)
        server_array = server_parameter.astype(np.float64, copy=False)
        diff = local_array - server_array
        denominator = max(interval * int(local_array.size), 1)
        discrepancies.append(float(np.sum(diff * diff) / denominator))
    return discrepancies
