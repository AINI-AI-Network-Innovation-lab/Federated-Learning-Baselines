"""FedRS client-side training loop."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader


def _logits_from_output(output: torch.Tensor | tuple | list) -> torch.Tensor:
    if isinstance(output, (tuple, list)):
        return output[-1]
    return output


def observed_class_mask(train_loader: DataLoader, num_classes: int) -> torch.Tensor:
    """Build a binary mask for classes observed on the local client."""

    mask = torch.zeros(num_classes, dtype=torch.float32)
    for _, targets in train_loader:
        labels = targets.detach().cpu().long().flatten()
        valid_labels = labels[(labels >= 0) & (labels < num_classes)]
        if valid_labels.numel() > 0:
            mask[valid_labels.unique()] = 1.0
    return mask


def fedrs_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_mask: torch.Tensor,
    *,
    alpha: float,
) -> torch.Tensor:
    """Compute restricted-softmax cross-entropy for FedRS."""

    if logits.ndim != 2:
        raise ValueError("FedRS expects 2D classification logits")
    if logits.shape[1] != class_mask.numel():
        raise ValueError("FedRS class_mask length must match logits classes")

    mask = class_mask.to(device=logits.device, dtype=logits.dtype)
    scale = torch.where(mask > 0, torch.ones_like(mask), torch.full_like(mask, alpha))
    scaled_exp = torch.exp(logits) * scale.unsqueeze(0)
    denominator = scaled_exp.sum(dim=1).clamp_min(torch.finfo(logits.dtype).tiny)
    target_logits = logits.gather(1, targets.long().unsqueeze(1)).squeeze(1)
    return (denominator.log() - target_logits).mean()


def train_fedrs_client(
    model: nn.Module,
    train_loader: DataLoader,
    *,
    epochs: int,
    learning_rate: float,
    device: str,
    num_classes: int,
    fedrs_alpha: float,
) -> dict[str, float]:
    """Train one client with the FedRS restricted-softmax loss."""

    model.to(device)
    model.train()

    class_mask = observed_class_mask(train_loader, num_classes).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    total_loss = 0.0
    total_examples = 0
    correct = 0

    for _ in range(epochs):
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device).long()

            optimizer.zero_grad()
            logits = _logits_from_output(model(inputs))
            loss = fedrs_loss(logits, targets, class_mask, alpha=fedrs_alpha)
            loss.backward()
            optimizer.step()

            batch_size = targets.size(0)
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size
            correct += int((logits.argmax(dim=1) == targets).sum().item())

    model.to("cpu")

    if total_examples == 0:
        return {"train_loss": 0.0, "train_accuracy": 0.0, "fedrs_loss": 0.0}
    average_loss = total_loss / total_examples
    return {
        "train_loss": average_loss,
        "train_accuracy": correct / total_examples,
        "fedrs_loss": average_loss,
    }
