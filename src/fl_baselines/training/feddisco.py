"""FedDisco local training helpers."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from fl_baselines.training.train import train_one_client


def compute_label_distribution(
    train_loader: DataLoader,
    num_classes: int,
) -> torch.Tensor:
    """Compute a local empirical label distribution from a training loader."""

    counts = torch.zeros(num_classes, dtype=torch.float64)
    for _, targets in train_loader:
        labels = targets.detach().cpu().long().flatten()
        valid_labels = labels[(labels >= 0) & (labels < num_classes)]
        if valid_labels.numel() > 0:
            counts += torch.bincount(valid_labels, minlength=num_classes).to(
                torch.float64
            )

    total = torch.sum(counts)
    if total <= 0:
        return counts
    return counts / total


def compute_label_distribution_discrepancy(
    distribution: torch.Tensor,
    *,
    metric: str,
    epsilon: float,
) -> float:
    """Compare a local label distribution with FedDisco's uniform target."""

    distribution = distribution.to(dtype=torch.float64)
    if distribution.numel() == 0 or float(torch.sum(distribution)) <= 0:
        return 0.0

    target = torch.full_like(distribution, 1.0 / distribution.numel())
    if metric == "kl":
        value = torch.sum(
            distribution * torch.log((distribution + epsilon) / (target + epsilon))
        )
    elif metric == "l1":
        value = torch.sum(torch.abs(distribution - target))
    elif metric == "l2":
        value = torch.linalg.vector_norm(distribution - target, ord=2)
    elif metric == "cosine":
        numerator = torch.sum(distribution * target)
        denominator = torch.linalg.vector_norm(distribution) * torch.linalg.vector_norm(
            target
        )
        value = 1.0 - (numerator / torch.clamp(denominator, min=epsilon))
    else:
        raise ValueError("FedDisco metric must be one of: kl, l1, l2, cosine")

    return float(torch.clamp(value, min=0.0).item())


def train_feddisco_client(
    model: nn.Module,
    train_loader: DataLoader,
    *,
    epochs: int,
    learning_rate: float,
    device: str,
    num_classes: int,
    metric: str,
    epsilon: float,
) -> dict[str, float]:
    distribution = compute_label_distribution(train_loader, num_classes)
    discrepancy = compute_label_distribution_discrepancy(
        distribution,
        metric=metric,
        epsilon=epsilon,
    )
    metrics = train_one_client(
        model,
        train_loader,
        epochs=epochs,
        learning_rate=learning_rate,
        device=device,
    )
    metrics["feddisco_discrepancy"] = discrepancy
    return metrics
