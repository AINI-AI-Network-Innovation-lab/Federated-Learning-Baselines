"""FedCurv client-side training loop."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def train_fedcurv_client(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: str,
    global_curvature: list[np.ndarray],
    global_weighted: list[np.ndarray],
    local_curvature: list[np.ndarray],
    local_weighted: list[np.ndarray],
    fedcurv_lambda: float,
    fisher_batches: int,
    fedcurv_stability_eps: float,
) -> tuple[dict[str, float], list[np.ndarray], list[np.ndarray]]:
    """Train one client with curvature regularization and return Fisher statistics."""

    model.to(device)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    state_keys = list(model.state_dict().keys())
    curvature_lookup = {
        key: (
            torch.as_tensor(global_value, dtype=torch.float32, device=device)
            - torch.as_tensor(local_value, dtype=torch.float32, device=device)
        ).clamp_min(0.0)
        for key, global_value, local_value in zip(state_keys, global_curvature, local_curvature)
    }
    weighted_lookup = {
        key: torch.as_tensor(global_value, dtype=torch.float32, device=device)
        - torch.as_tensor(local_value, dtype=torch.float32, device=device)
        for key, global_value, local_value in zip(state_keys, global_weighted, local_weighted)
    }

    total_loss = 0.0
    total_reg_loss = 0.0
    total_examples = 0
    correct = 0

    for _ in range(epochs):
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device).long()

            optimizer.zero_grad()
            logits = model(inputs)
            ce_loss = criterion(logits, targets)
            reg_loss = torch.tensor(0.0, device=device)
            for name, parameter in model.named_parameters():
                curvature = curvature_lookup[name]
                weighted = weighted_lookup[name]
                reg_loss = reg_loss + torch.sum(curvature * parameter.pow(2))
                reg_loss = reg_loss - (2.0 * torch.sum(weighted * parameter))

            loss = ce_loss + (fedcurv_lambda * reg_loss)
            loss.backward()
            optimizer.step()

            batch_size = targets.size(0)
            total_loss += float(loss.item()) * batch_size
            total_reg_loss += float(reg_loss.item()) * batch_size
            total_examples += batch_size
            correct += int((logits.argmax(dim=1) == targets).sum().item())

    fisher_diagonal = _estimate_fisher_diagonal(
        model,
        train_loader,
        device=device,
        fisher_batches=fisher_batches,
        stability_eps=fedcurv_stability_eps,
    )
    weighted_parameters = _compute_weighted_parameters(model, fisher_diagonal)
    model.to("cpu")

    if total_examples == 0:
        metrics = {"train_loss": 0.0, "train_accuracy": 0.0, "fedcurv_reg_loss": 0.0}
    else:
        metrics = {
            "train_loss": total_loss / total_examples,
            "train_accuracy": correct / total_examples,
            "fedcurv_reg_loss": total_reg_loss / total_examples,
        }
    return metrics, fisher_diagonal, weighted_parameters


def _estimate_fisher_diagonal(
    model: nn.Module,
    train_loader: DataLoader,
    *,
    device: str,
    fisher_batches: int,
    stability_eps: float,
) -> list[np.ndarray]:
    criterion = nn.CrossEntropyLoss()
    parameter_items = list(model.named_parameters())
    parameter_names = [name for name, _ in parameter_items]
    fisher_accumulator = {
        name: torch.zeros_like(parameter, dtype=torch.float32, device=device)
        for name, parameter in parameter_items
    }
    observed_batches = 0
    was_training = model.training
    model.train()

    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device).long()
        model.zero_grad()
        loss = criterion(model(inputs), targets)
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                fisher_accumulator[name].add_(parameter.grad.detach().pow(2))
        observed_batches += 1
        if observed_batches >= fisher_batches:
            break

    if not was_training:
        model.eval()

    if observed_batches == 0:
        return [
            np.full_like(value.detach().cpu().numpy(), stability_eps, dtype=np.float32)
            for value in model.state_dict().values()
        ]

    fisher_by_name = {
        name: (fisher_accumulator[name] / float(observed_batches))
        .clamp_min(stability_eps)
        .cpu()
        .numpy()
        .astype(np.float32)
        for name in parameter_names
    }
    fisher_diagonal = []
    for key, value in model.state_dict().items():
        fisher_diagonal.append(
            fisher_by_name.get(
                key,
                np.zeros_like(value.detach().cpu().numpy(), dtype=np.float32),
            )
        )
    return fisher_diagonal


def _compute_weighted_parameters(
    model: nn.Module,
    fisher_diagonal: list[np.ndarray],
) -> list[np.ndarray]:
    state_items = list(model.state_dict().items())
    return [
        fisher * value.detach().cpu().numpy().astype(np.float32)
        for (_, value), fisher in zip(state_items, fisher_diagonal)
    ]
