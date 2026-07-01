"""FedDyn client-side training loop."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader


def train_feddyn_client(
    model: nn.Module,
    global_model: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: str,
    alpha: float,
    state: list[torch.Tensor],
) -> dict[str, float]:
    """Train one client with the FedDyn dynamic regularization objective."""

    model.to(device)
    global_model.to(device)
    model.train()
    global_model.eval()

    global_parameters = [
        parameter.detach().clone().to(device)
        for parameter in global_model.parameters()
    ]
    state_on_device = [tensor.detach().clone().to(device) for tensor in state]

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

            linear_term = torch.tensor(0.0, device=device)
            quadratic_term = torch.tensor(0.0, device=device)
            for parameter, global_parameter, state_tensor in zip(
                model.parameters(),
                global_parameters,
                state_on_device,
            ):
                linear_term += torch.sum(state_tensor * parameter)
                quadratic_term += torch.sum((parameter - global_parameter) ** 2)

            loss = ce_loss - linear_term + (alpha / 2.0) * quadratic_term
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


def update_feddyn_state(
    previous_state: list[torch.Tensor],
    local_model: nn.Module,
    global_model: nn.Module,
    alpha: float,
) -> list[torch.Tensor]:
    """Update the persisted FedDyn client state after local optimization."""

    local_parameters = list(local_model.state_dict().values())
    global_parameters = list(global_model.state_dict().values())
    return [
        state_tensor.detach().cpu().clone()
        - alpha
        * (
            local_tensor.detach().cpu()
            - global_tensor.detach().cpu()
        )
        for state_tensor, local_tensor, global_tensor in zip(
            previous_state,
            local_parameters,
            global_parameters,
        )
    ]
