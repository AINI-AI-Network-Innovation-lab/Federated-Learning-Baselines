"""FedDRL local training helpers."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader

from fl_baselines.training.train import train_one_client


def _logits_from_output(output: torch.Tensor | tuple | list) -> torch.Tensor:
    if isinstance(output, (tuple, list)):
        return output[-1]
    return output


def compute_average_loss(
    model: nn.Module,
    data_loader: DataLoader,
    device: str,
) -> float:
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_examples = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device).long()
            logits = _logits_from_output(model(inputs))
            loss = criterion(logits, targets)
            batch_size = targets.size(0)
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size

    if total_examples == 0:
        return 0.0
    return total_loss / total_examples


def compute_feddrl_reward(pre_train_losses: Iterable[float]) -> float:
    losses = [float(loss) for loss in pre_train_losses]
    if not losses:
        return 0.0
    mean_loss = sum(losses) / len(losses)
    bias_gap = max(losses) - min(losses)
    return -(mean_loss + bias_gap)


def train_feddrl_client(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: str,
) -> dict[str, float]:
    pre_train_loss = compute_average_loss(model, train_loader, device)
    metrics = train_one_client(
        model,
        train_loader,
        epochs=epochs,
        learning_rate=learning_rate,
        device=device,
    )
    post_train_loss = compute_average_loss(model, train_loader, device)
    metrics["feddrl_pre_train_loss"] = pre_train_loss
    metrics["feddrl_post_train_loss"] = post_train_loss
    return metrics
