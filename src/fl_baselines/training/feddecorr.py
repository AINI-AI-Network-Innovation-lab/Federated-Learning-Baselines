"""FedDecorr client-side training loop."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from fl_baselines.training.features import extract_features


def feddecorr_loss(features: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute the batch decorrelation penalty on representation features."""

    if features.ndim != 2:
        raise ValueError("FedDecorr expects a 2D feature tensor")
    if features.shape[0] <= 1:
        return torch.zeros((), dtype=features.dtype, device=features.device)

    centered = features - features.mean(dim=0, keepdim=True)
    scale = centered.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)
    normalized = centered / scale
    correlation = normalized.transpose(0, 1) @ normalized
    correlation = correlation / float(features.shape[0])
    return correlation.pow(2).mean()


def train_feddecorr_client(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: str,
    feddecorr_beta: float,
) -> dict[str, float]:
    """Train one client with the FedDecorr local objective."""

    model.to(device)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_examples = 0
    correct = 0
    total_decorrelation = 0.0

    for _ in range(epochs):
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device).long()

            optimizer.zero_grad()
            features = extract_features(model, inputs)
            logits = model(inputs)
            ce_loss = criterion(logits, targets)
            decor_loss = feddecorr_loss(features)
            loss = ce_loss + feddecorr_beta * decor_loss
            loss.backward()
            optimizer.step()

            batch_size = targets.size(0)
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size
            correct += int((logits.argmax(dim=1) == targets).sum().item())
            total_decorrelation += float(decor_loss.item()) * batch_size

    model.to("cpu")

    if total_examples == 0:
        return {
            "train_loss": 0.0,
            "train_accuracy": 0.0,
            "feddecorr_loss": 0.0,
        }
    return {
        "train_loss": total_loss / total_examples,
        "train_accuracy": correct / total_examples,
        "feddecorr_loss": total_decorrelation / total_examples,
    }
