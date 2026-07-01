"""Ditto client-side personalized training loop."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader


def train_ditto_personalized(
    model: nn.Module,
    global_model: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: str,
    ditto_lambda: float,
) -> dict[str, float]:
    """Train a personalized Ditto model regularized toward the global model."""

    model.to(device)
    global_model.to(device)
    model.train()
    global_model.eval()

    global_parameters = [
        parameter.detach().clone().to(device)
        for parameter in global_model.parameters()
    ]

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_examples = 0
    correct = 0

    for _ in range(epochs):
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device).long()

            optimizer.zero_grad()
            outputs = model(inputs)
            ce_loss = criterion(outputs, targets)

            reg_term = torch.tensor(0.0, device=device)
            for parameter, global_parameter in zip(model.parameters(), global_parameters):
                reg_term += torch.sum((parameter - global_parameter) ** 2)

            loss = ce_loss + (ditto_lambda / 2.0) * reg_term
            loss.backward()
            optimizer.step()

            batch_size = targets.size(0)
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size
            correct += int((outputs.argmax(dim=1) == targets).sum().item())

    model.to("cpu")
    global_model.to("cpu")

    if total_examples == 0:
        return {"train_loss": 0.0, "train_accuracy": 0.0}
    return {
        "train_loss": total_loss / total_examples,
        "train_accuracy": correct / total_examples,
    }
