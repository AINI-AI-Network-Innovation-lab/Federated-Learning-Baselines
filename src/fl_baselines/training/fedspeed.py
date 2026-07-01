"""FedSpeed client-side training loop."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader


def train_fedspeed_client(
    model: nn.Module,
    global_model: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: str,
    fedspeed_lambda: float,
    fedspeed_alpha: float,
    fedspeed_rho: float,
    state: list[torch.Tensor],
) -> tuple[dict[str, float], list[torch.Tensor], list[torch.Tensor]]:
    """Train one client with the FedSpeed local update and return payload state."""

    model.to(device)
    global_model.to(device)
    model.train()
    global_model.eval()

    parameters = list(model.parameters())
    global_parameters = [
        parameter.detach().clone().to(device)
        for parameter in global_model.parameters()
    ]
    state_on_device = [tensor.detach().clone().to(device) for tensor in state]
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_examples = 0
    correct = 0

    for _ in range(epochs):
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device).long()

            outputs = model(inputs)
            ce_loss = criterion(outputs, targets)
            grad_first = torch.autograd.grad(ce_loss, parameters)

            with torch.no_grad():
                for parameter, grad in zip(parameters, grad_first):
                    parameter.add_(fedspeed_rho * grad)

            perturbed_outputs = model(inputs)
            perturbed_loss = criterion(perturbed_outputs, targets)
            grad_second = torch.autograd.grad(perturbed_loss, parameters)

            with torch.no_grad():
                for parameter, grad in zip(parameters, grad_first):
                    parameter.sub_(fedspeed_rho * grad)

                for parameter, global_parameter, state_tensor, g1, g2 in zip(
                    parameters,
                    global_parameters,
                    state_on_device,
                    grad_first,
                    grad_second,
                ):
                    quasi_gradient = ((1.0 - fedspeed_alpha) * g1) + (fedspeed_alpha * g2)
                    correction = (parameter - global_parameter) / fedspeed_lambda
                    update_direction = quasi_gradient - state_tensor + correction
                    parameter.sub_(learning_rate * update_direction)

            batch_size = targets.size(0)
            total_loss += float(ce_loss.item()) * batch_size
            total_examples += batch_size
            correct += int((outputs.argmax(dim=1) == targets).sum().item())

    updated_state = [
        state_tensor.detach().cpu().clone()
        - (
            local_parameter.detach().cpu()
            - global_parameter.detach().cpu()
        )
        / fedspeed_lambda
        for state_tensor, local_parameter, global_parameter in zip(
            state_on_device,
            parameters,
            global_parameters,
        )
    ]
    payload = [
        local_parameter.detach().cpu().clone()
        - (fedspeed_lambda * new_state)
        for local_parameter, new_state in zip(parameters, updated_state)
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
    return metrics, updated_state, payload
