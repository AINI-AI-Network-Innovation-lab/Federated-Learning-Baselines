"""FedSAM local training."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader


def train_fedsam_client(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: str,
    fedsam_rho: float,
) -> dict[str, float]:
    model.to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_examples = 0
    correct = 0

    for _ in range(epochs):
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            grad_norm = _gradient_norm(model)
            scale = fedsam_rho / (grad_norm + 1e-12) if grad_norm > 0 else 0.0
            perturbations: list[torch.Tensor | None] = []

            with torch.no_grad():
                for parameter in model.parameters():
                    if parameter.grad is None:
                        perturbations.append(None)
                        continue
                    epsilon = parameter.grad.detach().clone() * scale
                    parameter.add_(epsilon)
                    perturbations.append(epsilon)

            optimizer.zero_grad()
            perturbed_outputs = model(inputs)
            perturbed_loss = criterion(perturbed_outputs, targets)
            perturbed_loss.backward()

            with torch.no_grad():
                for parameter, epsilon in zip(model.parameters(), perturbations):
                    if epsilon is not None:
                        parameter.sub_(epsilon)

            optimizer.step()

            batch_size = targets.size(0)
            total_loss += float(perturbed_loss.item()) * batch_size
            total_examples += batch_size
            correct += int((perturbed_outputs.argmax(dim=1) == targets).sum().item())

    if total_examples == 0:
        return {"train_loss": 0.0, "train_accuracy": 0.0}
    return {
        "train_loss": total_loss / total_examples,
        "train_accuracy": correct / total_examples,
    }


def _gradient_norm(model: nn.Module) -> float:
    total = 0.0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        total += float(torch.sum(parameter.grad.detach() ** 2).item())
    return total ** 0.5
