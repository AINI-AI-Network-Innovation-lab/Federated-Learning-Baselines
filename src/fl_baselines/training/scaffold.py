"""SCAFFOLD client-side training loop."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def _controls_to_tensors(
    controls: list[np.ndarray],
    parameters: list[nn.Parameter],
    device: str,
) -> list[torch.Tensor]:
    return [
        torch.as_tensor(control, dtype=parameter.dtype, device=device)
        for control, parameter in zip(controls, parameters)
    ]


def train_scaffold_client(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: str,
    server_control: list[np.ndarray],
    client_control: list[np.ndarray],
) -> tuple[dict[str, float], list[np.ndarray], list[np.ndarray]]:
    """Train one client with SCAFFOLD gradient correction.

    Returns regular training metrics, the updated client control variates, and
    the control deltas to be aggregated on the server.
    """

    model.to(device)
    model.train()
    parameters = list(model.parameters())
    if len(server_control) != len(parameters):
        raise ValueError("server_control must match the number of trainable parameters")
    if len(client_control) != len(parameters):
        raise ValueError("client_control must match the number of trainable parameters")

    optimizer = torch.optim.SGD(parameters, lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    global_parameters = [parameter.detach().clone() for parameter in parameters]
    server_control_tensors = _controls_to_tensors(server_control, parameters, device)
    client_control_tensors = _controls_to_tensors(client_control, parameters, device)

    total_loss = 0.0
    total_examples = 0
    correct = 0
    local_steps = 0

    for _ in range(epochs):
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            for parameter, server_c, client_c in zip(
                parameters, server_control_tensors, client_control_tensors
            ):
                if parameter.grad is not None:
                    parameter.grad.add_(server_c - client_c)
            optimizer.step()

            batch_size = targets.size(0)
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size
            correct += int((outputs.argmax(dim=1) == targets).sum().item())
            local_steps += 1

    if local_steps == 0:
        new_control_tensors = [control.detach().clone() for control in client_control_tensors]
    else:
        scale = 1.0 / (local_steps * learning_rate)
        new_control_tensors = [
            client_c - server_c + (global_parameter - local_parameter.detach()) * scale
            for client_c, server_c, global_parameter, local_parameter in zip(
                client_control_tensors,
                server_control_tensors,
                global_parameters,
                parameters,
            )
        ]
    control_delta_tensors = [
        new_control - old_control
        for new_control, old_control in zip(new_control_tensors, client_control_tensors)
    ]

    if total_examples == 0:
        metrics = {"train_loss": 0.0, "train_accuracy": 0.0}
    else:
        metrics = {
            "train_loss": total_loss / total_examples,
            "train_accuracy": correct / total_examples,
        }

    new_control = [
        control.detach().cpu().numpy() for control in new_control_tensors
    ]
    control_delta = [
        delta.detach().cpu().numpy() for delta in control_delta_tensors
    ]
    return metrics, new_control, control_delta
