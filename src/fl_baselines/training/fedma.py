"""FedMA local training helpers."""

from __future__ import annotations

import json
from collections import Counter

import torch
from torch import nn

from fl_baselines.training.train import train_one_client


def train_fedma_client(
    model: nn.Module,
    train_loader,
    *,
    epochs: int,
    learning_rate: float,
    device: str,
    frozen_layer_prefixes: list[str],
) -> dict[str, float | str]:
    original_requires_grad = {
        name: parameter.requires_grad for name, parameter in model.named_parameters()
    }
    for name, parameter in model.named_parameters():
        parameter.requires_grad = not any(
            name.startswith(prefix) for prefix in frozen_layer_prefixes
        )

    try:
        metrics = train_one_client(
            model,
            train_loader,
            epochs=epochs,
            learning_rate=learning_rate,
            device=device,
        )
    finally:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = original_requires_grad[name]

    label_counts = Counter()
    for _, labels in train_loader:
        label_counts.update(int(label) for label in labels.view(-1).tolist())
    num_classes = max(label_counts.keys(), default=-1) + 1
    counts = [label_counts.get(index, 0) for index in range(num_classes)]
    metrics["fedma_label_counts"] = json.dumps(counts)
    return metrics
