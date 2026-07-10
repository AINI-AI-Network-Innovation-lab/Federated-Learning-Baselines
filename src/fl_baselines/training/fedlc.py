"""FedLC client-side training loop."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


def _logits_from_output(output: torch.Tensor | tuple | list) -> torch.Tensor:
    if isinstance(output, (tuple, list)):
        return output[-1]
    return output


def local_class_counts(train_loader: DataLoader, num_classes: int) -> torch.Tensor:
    """Count valid local labels for FedLC calibration."""

    counts = torch.zeros(num_classes, dtype=torch.float32)
    for _, targets in train_loader:
        labels = targets.detach().cpu().long().flatten()
        valid_labels = labels[(labels >= 0) & (labels < num_classes)]
        if valid_labels.numel() > 0:
            counts += torch.bincount(valid_labels, minlength=num_classes).to(torch.float32)
    return counts


def fedlc_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_counts: torch.Tensor,
    *,
    tau: float,
    epsilon: float,
) -> torch.Tensor:
    """Compute FedLC calibrated cross-entropy."""

    if logits.ndim != 2:
        raise ValueError("FedLC expects 2D classification logits")
    if logits.shape[1] != class_counts.numel():
        raise ValueError("FedLC class_counts length must match logits classes")

    counts = class_counts.to(device=logits.device, dtype=logits.dtype).clamp_min(epsilon)
    margins = tau * counts.pow(-0.25)
    adjusted_logits = logits - margins.unsqueeze(0)
    return F.cross_entropy(adjusted_logits, targets.long())


def train_fedlc_client(
    model: nn.Module,
    train_loader: DataLoader,
    *,
    epochs: int,
    learning_rate: float,
    device: str,
    num_classes: int,
    fedlc_tau: float,
    fedlc_epsilon: float,
) -> dict[str, float]:
    """Train one client with the FedLC local calibrated loss."""

    model.to(device)
    model.train()

    class_counts = local_class_counts(train_loader, num_classes).to(device)
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
            loss = fedlc_loss(
                logits,
                targets,
                class_counts,
                tau=fedlc_tau,
                epsilon=fedlc_epsilon,
            )
            loss.backward()
            optimizer.step()

            batch_size = targets.size(0)
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size
            correct += int((logits.argmax(dim=1) == targets).sum().item())

    model.to("cpu")

    if total_examples == 0:
        return {"train_loss": 0.0, "train_accuracy": 0.0, "fedlc_loss": 0.0}
    average_loss = total_loss / total_examples
    return {
        "train_loss": average_loss,
        "train_accuracy": correct / total_examples,
        "fedlc_loss": average_loss,
    }
