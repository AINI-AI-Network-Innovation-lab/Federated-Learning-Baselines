"""FedAAW local training helpers."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from fl_baselines.training.train import train_one_client


def _extract_logits(outputs: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor]) -> torch.Tensor:
    if isinstance(outputs, (tuple, list)):
        return outputs[-1]
    return outputs


def compute_full_batch_gradient_norm_sq(
    model: nn.Module,
    train_loader: DataLoader,
    device: str,
) -> float:
    model.to(device)
    model.train()
    model.zero_grad(set_to_none=True)
    criterion = nn.CrossEntropyLoss()

    total_loss: torch.Tensor | None = None
    total_examples = 0
    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits = _extract_logits(model(inputs))
        loss = criterion(logits, targets)
        batch_size = targets.size(0)
        weighted_loss = loss * batch_size
        total_loss = weighted_loss if total_loss is None else total_loss + weighted_loss
        total_examples += batch_size

    if total_examples == 0 or total_loss is None:
        return 0.0

    average_loss = total_loss / total_examples
    average_loss.backward()

    grad_norm_sq = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            grad_norm_sq += float(torch.sum(parameter.grad.detach() ** 2).item())
    model.zero_grad(set_to_none=True)
    return grad_norm_sq


def train_fedaaw_client(
    model: nn.Module,
    train_loader: DataLoader,
    *,
    epochs: int,
    learning_rate: float,
    device: str,
) -> dict[str, float]:
    grad_norm_sq = compute_full_batch_gradient_norm_sq(model, train_loader, device)
    metrics = train_one_client(
        model,
        train_loader,
        epochs=epochs,
        learning_rate=learning_rate,
        device=device,
    )
    metrics["fedaaw_grad_norm_sq"] = float(grad_norm_sq)
    return metrics
