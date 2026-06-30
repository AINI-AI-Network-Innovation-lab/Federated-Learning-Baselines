"""PyTorch evaluation loop."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader


def _logits_from_output(output: torch.Tensor | tuple | list) -> torch.Tensor:
    if isinstance(output, (tuple, list)):
        return output[-1]
    return output


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: str,
) -> tuple[float, dict[str, float]]:
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_examples = 0
    correct = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = _logits_from_output(model(inputs))
            loss = criterion(outputs, targets)

            batch_size = targets.size(0)
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size
            correct += int((outputs.argmax(dim=1) == targets).sum().item())

    if total_examples == 0:
        return 0.0, {"accuracy": 0.0}
    average_loss = total_loss / total_examples
    return average_loss, {"accuracy": correct / total_examples}
