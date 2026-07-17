"""FedADMM client-side training helpers."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader


def train_fedadmm_client(
    model: nn.Module,
    global_model: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: str,
    alpha: float,
    state: list[torch.Tensor],
) -> tuple[dict[str, float], list[torch.Tensor]]:
    """Train one client with an ADMM-style local objective.

    The local loss is:

        CE(model(x), y) + <z, w - w_global> + (alpha / 2) ||w - w_global||^2

    where ``z`` is the persisted dual state for this client.
    """

    model.to(device)
    global_model.to(device)
    model.train()
    global_model.eval()

    local_parameters = list(model.parameters())
    global_parameters = [
        parameter.detach().clone().to(device)
        for parameter in global_model.parameters()
    ]
    dual_state = [tensor.detach().clone().to(device) for tensor in state]

    if len(dual_state) != len(local_parameters):
        raise ValueError("FedADMM state must match the number of trainable parameters")

    optimizer = torch.optim.SGD(local_parameters, lr=learning_rate)
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
                local_parameters,
                global_parameters,
                dual_state,
            ):
                linear_term += torch.sum(state_tensor * (parameter - global_parameter))
                quadratic_term += torch.sum((parameter - global_parameter) ** 2)

            loss = ce_loss + linear_term + (alpha / 2.0) * quadratic_term
            loss.backward()
            optimizer.step()

            batch_size = targets.size(0)
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size
            correct += int((outputs.argmax(dim=1) == targets).sum().item())

    updated_state = update_fedadmm_state(
        dual_state,
        model,
        global_model,
        alpha,
    )

    model.to("cpu")
    global_model.to("cpu")

    if total_examples == 0:
        metrics = {"train_loss": 0.0, "train_accuracy": 0.0}
    else:
        metrics = {
            "train_loss": total_loss / total_examples,
            "train_accuracy": correct / total_examples,
        }
    return metrics, updated_state


def update_fedadmm_state(
    previous_state: list[torch.Tensor],
    local_model: nn.Module,
    global_model: nn.Module,
    alpha: float,
) -> list[torch.Tensor]:
    """Update the persisted dual state after local optimization."""

    local_parameters = list(local_model.parameters())
    global_parameters = list(global_model.parameters())

    if len(previous_state) != len(local_parameters):
        raise ValueError("FedADMM state must match the number of trainable parameters")

    return [
        state_tensor.detach().cpu().clone()
        + alpha * (
            local_parameter.detach().cpu() - global_parameter.detach().cpu()
        )
        for state_tensor, local_parameter, global_parameter in zip(
            previous_state,
            local_parameters,
            global_parameters,
        )
    ]
