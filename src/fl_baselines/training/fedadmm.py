"""Client-side FedADMM local solves and primal/dual state updates."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.utils.data import DataLoader


def _logits(output: torch.Tensor | tuple | list) -> torch.Tensor:
    return output[-1] if isinstance(output, (tuple, list)) else output


def _state_values(model: nn.Module) -> list[torch.Tensor]:
    return [value.detach().clone() for value in model.state_dict().values()]


def _trainable_state_indices(model: nn.Module) -> list[int]:
    names = list(model.state_dict().keys())
    parameter_names = set(dict(model.named_parameters()))
    return [index for index, name in enumerate(names) if name in parameter_names]


def train_fedadmm_client(
    model: nn.Module,
    global_model: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: str,
    alpha: float,
    state: list[torch.Tensor],
    *,
    local_steps: int = 0,
    tolerance: float = 0.0,
) -> tuple[
    dict[str, float],
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
]:
    """Run an inexact FedADMM local minimization.

    ``alpha`` is the paper's penalty ``eta``. ``state`` contains the previous
    dual tensors in model ``state_dict`` order. The returned values are the
    metrics, current primal state, updated dual state, and transformed
    ``hat_x = x + z / eta`` state, respectively.
    """

    if alpha <= 0:
        raise ValueError("FedADMM penalty must be positive")
    if local_steps < 0:
        raise ValueError("FedADMM local_steps must be non-negative")
    if tolerance < 0:
        raise ValueError("FedADMM tolerance must be non-negative")

    model.to(device)
    global_model.to(device)
    model.train()
    global_model.eval()

    local_parameters = list(model.parameters())
    global_parameters = [
        parameter.detach().clone().to(device) for parameter in global_model.parameters()
    ]
    previous_state = [tensor.detach().clone() for tensor in state]
    global_state = [value.detach().clone().to(device) for value in global_model.state_dict().values()]
    if len(previous_state) != len(global_state):
        raise ValueError("FedADMM state must match model state_dict")

    trainable_indices = _trainable_state_indices(model)
    dual_trainable = {
        parameter_index: previous_state[state_index].to(device)
        for parameter_index, state_index in enumerate(trainable_indices)
    }

    optimizer = torch.optim.SGD(local_parameters, lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    requested_steps = local_steps or (epochs * len(train_loader))
    if requested_steps < 0:
        raise ValueError("FedADMM local step count must be non-negative")

    total_loss = 0.0
    total_examples = 0
    correct = 0
    completed_steps = 0
    last_step_norm = math.inf
    loader_iterator = iter(train_loader)

    for _ in range(requested_steps):
        try:
            inputs, targets = next(loader_iterator)
        except StopIteration:
            loader_iterator = iter(train_loader)
            try:
                inputs, targets = next(loader_iterator)
            except StopIteration:
                break

        inputs = inputs.to(device)
        targets = targets.to(device).long()
        before = [parameter.detach().clone() for parameter in local_parameters]
        optimizer.zero_grad()
        outputs = _logits(model(inputs))
        ce_loss = criterion(outputs, targets)

        linear_term = torch.tensor(0.0, device=device)
        quadratic_term = torch.tensor(0.0, device=device)
        for parameter, global_parameter, parameter_index in zip(
            local_parameters,
            global_parameters,
            range(len(local_parameters)),
        ):
            dual_tensor = dual_trainable[parameter_index]
            difference = parameter - global_parameter
            linear_term = linear_term + torch.sum(dual_tensor * difference)
            quadratic_term = quadratic_term + torch.sum(difference**2)

        loss = ce_loss + linear_term + (alpha / 2.0) * quadratic_term
        loss.backward()
        optimizer.step()
        completed_steps += 1

        last_step_norm = math.sqrt(
            sum(float(torch.sum((after.detach() - before_value) ** 2).item())
                for after, before_value in zip(local_parameters, before))
        )
        batch_size = targets.size(0)
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size
        correct += int((outputs.argmax(dim=1) == targets).sum().item())
        if tolerance > 0 and last_step_norm <= tolerance:
            break

    local_state = _state_values(model)
    updated_state = update_fedadmm_state(
        previous_state,
        local_state,
        global_state,
        alpha,
    )
    hat_state = [
        local_value.clone()
        if not torch.is_floating_point(local_value)
        else local_value + dual_value.to(local_value.device) / alpha
        for local_value, dual_value in zip(local_state, updated_state)
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
    metrics["fedadmm_local_steps"] = float(completed_steps)
    metrics["fedadmm_step_norm"] = 0.0 if not math.isfinite(last_step_norm) else last_step_norm
    return metrics, local_state, updated_state, hat_state


def update_fedadmm_state(
    previous_state: list[torch.Tensor],
    local_state: list[torch.Tensor],
    global_state: list[torch.Tensor],
    alpha: float,
) -> list[torch.Tensor]:
    """Apply ``z_new = z_old + eta * (x_new - bar_x)`` to all state tensors."""

    if alpha <= 0:
        raise ValueError("FedADMM penalty must be positive")
    if not len(previous_state) == len(local_state) == len(global_state):
        raise ValueError("FedADMM state must match model state_dict")
    updated: list[torch.Tensor] = []
    for previous, local, global_value in zip(previous_state, local_state, global_state):
        if not torch.is_floating_point(local):
            updated.append(torch.zeros_like(previous.detach().cpu()))
            continue
        updated.append(
            previous.detach().cpu().clone()
            + alpha * (local.detach().cpu() - global_value.detach().cpu())
        )
    return updated
