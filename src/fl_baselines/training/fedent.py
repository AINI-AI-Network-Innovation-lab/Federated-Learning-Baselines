"""FedEnt local adaptive learning-rate helpers."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.utils.data import DataLoader


def compute_fedent_learning_rate(
    *,
    parameter_vector: torch.Tensor,
    gradient_vector: torch.Tensor,
    phi1_vector: torch.Tensor,
    phi2_scalar: float,
    aggregation_weight: float,
    epsilon: float,
    max_learning_rate: float,
) -> float:
    safe_phi2 = max(float(phi2_scalar), float(epsilon))
    weighted_norm = float(torch.dot(parameter_vector, parameter_vector).item())
    p_i = min(1.0, max(float(epsilon), weighted_norm / safe_phi2))
    numerator = aggregation_weight * float(torch.dot(phi1_vector, gradient_vector).item())
    denominator = max((1.0 - aggregation_weight) * safe_phi2, float(epsilon))
    eta = max(0.0, numerator / denominator * (1.0 + math.log(p_i)))
    return min(float(max_learning_rate), eta)


def apply_fedent_eta_decay(
    *,
    previous_eta: float | None,
    current_eta: float,
    gamma: float,
) -> float:
    if previous_eta is None:
        return current_eta
    return gamma * previous_eta + (1.0 - gamma) * current_eta


def flatten_model_parameters(model: nn.Module) -> torch.Tensor:
    return torch.cat([parameter.detach().flatten().cpu() for parameter in model.parameters()])


def estimate_full_gradient(
    model: nn.Module,
    train_loader: DataLoader,
    device: str,
) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    criterion = nn.CrossEntropyLoss()

    inputs, targets = next(iter(train_loader))
    inputs = inputs.to(device)
    targets = targets.to(device)
    outputs = model(inputs)
    if isinstance(outputs, (tuple, list)):
        outputs = outputs[-1]
    loss = criterion(outputs, targets)
    loss.backward()

    gradients = []
    for parameter in model.parameters():
        gradient = parameter.grad
        if gradient is None:
            gradient = torch.zeros_like(parameter)
        gradients.append(gradient.detach().flatten().cpu())
    model.zero_grad(set_to_none=True)
    return torch.cat(gradients)


def train_fedent_client(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: str,
    phi1_vector: torch.Tensor,
    phi2_scalar: float,
    fedent_beta: float,
    fedent_gamma: float,
    fedent_epsilon: float,
    fedent_max_learning_rate: float,
    previous_eta: float | None,
) -> dict[str, float]:
    model.to(device)
    model.train()

    parameter_vector = flatten_model_parameters(model)
    gradient_vector = estimate_full_gradient(model, train_loader, device)
    eta_raw = compute_fedent_learning_rate(
        parameter_vector=parameter_vector,
        gradient_vector=gradient_vector,
        phi1_vector=phi1_vector.detach().flatten().cpu(),
        phi2_scalar=phi2_scalar,
        aggregation_weight=fedent_beta,
        epsilon=fedent_epsilon,
        max_learning_rate=fedent_max_learning_rate,
    )
    eta = apply_fedent_eta_decay(
        previous_eta=previous_eta,
        current_eta=eta_raw,
        gamma=fedent_gamma,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=eta if eta > 0 else learning_rate)
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
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[-1]
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            batch_size = targets.size(0)
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size
            correct += int((outputs.argmax(dim=1) == targets).sum().item())

    if total_examples == 0:
        return {
            "train_loss": 0.0,
            "train_accuracy": 0.0,
            "fedent_learning_rate": float(eta),
        }
    return {
        "train_loss": total_loss / total_examples,
        "train_accuracy": correct / total_examples,
        "fedent_learning_rate": float(eta),
    }
