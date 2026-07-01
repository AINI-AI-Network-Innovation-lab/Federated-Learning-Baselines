"""FedDC client-side training loop."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader


def train_feddc_client(
    model: nn.Module,
    global_model: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: str,
    alpha: float,
    drift_state: list[torch.Tensor],
    local_update_state: list[torch.Tensor],
    server_update_state: list[torch.Tensor],
) -> tuple[dict[str, float], list[torch.Tensor], list[torch.Tensor]]:
    """Train one client with the FedDC objective and updated local states."""

    model.to(device)
    global_model.to(device)
    model.train()
    global_model.eval()

    parameters = list(model.parameters())
    global_parameters = [
        parameter.detach().clone().to(device)
        for parameter in global_model.parameters()
    ]
    initial_parameters = [
        parameter.detach().clone().to(device)
        for parameter in model.parameters()
    ]
    drift_on_device = [tensor.detach().clone().to(device) for tensor in drift_state]
    local_update_on_device = [
        tensor.detach().clone().to(device) for tensor in local_update_state
    ]
    server_update_on_device = [
        tensor.detach().clone().to(device) for tensor in server_update_state
    ]

    optimizer = torch.optim.SGD(parameters, lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    total_local_steps = max(1, epochs * len(train_loader))

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

            penalty_term = torch.tensor(0.0, device=device)
            correction_term = torch.tensor(0.0, device=device)
            for parameter, global_parameter, drift_tensor, local_update_tensor, server_update_tensor in zip(
                parameters,
                global_parameters,
                drift_on_device,
                local_update_on_device,
                server_update_on_device,
            ):
                penalty_term += torch.sum((drift_tensor + parameter - global_parameter) ** 2)
                correction_term += torch.sum(
                    parameter * (local_update_tensor - server_update_tensor)
                )

            loss = ce_loss + (alpha / 2.0) * penalty_term + (
                correction_term / (learning_rate * total_local_steps)
            )
            loss.backward()
            optimizer.step()

            batch_size = targets.size(0)
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size
            correct += int((outputs.argmax(dim=1) == targets).sum().item())

    updated_local_state = [
        local_parameter.detach().clone()
        - initial_parameter.detach().clone()
        for local_parameter, initial_parameter in zip(parameters, initial_parameters)
    ]
    updated_drift_state = [
        drift_tensor + local_delta
        for drift_tensor, local_delta in zip(drift_on_device, updated_local_state)
    ]

    model.to("cpu")
    global_model.to("cpu")

    if total_examples == 0:
        metrics = {"train_loss": 0.0, "train_accuracy": 0.0}
    else:
        metrics = {
            "train_loss": total_loss / total_examples,
            "train_accuracy": correct / total_examples,
        }
    return metrics, updated_drift_state, updated_local_state
