"""PyTorch training loop."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader


def train_one_client(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: str,
    proximal_mu: float = 0.0,
) -> dict[str, float]:
    model.to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    global_parameters = [
        parameter.detach().clone() for parameter in model.parameters()
    ] if proximal_mu > 0 else []

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
            if proximal_mu > 0:
                proximal_term = torch.tensor(0.0, device=inputs.device)
                for local_parameter, global_parameter in zip(
                    model.parameters(), global_parameters
                ):
                    proximal_term += torch.sum((local_parameter - global_parameter) ** 2)
                loss = loss + (proximal_mu / 2.0) * proximal_term
            loss.backward()
            optimizer.step()

            batch_size = targets.size(0)
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size
            correct += int((outputs.argmax(dim=1) == targets).sum().item())

    if total_examples == 0:
        return {"train_loss": 0.0, "train_accuracy": 0.0}
    return {
        "train_loss": total_loss / total_examples,
        "train_accuracy": correct / total_examples,
    }
