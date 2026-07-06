"""FedALA adaptive local aggregation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset


@dataclass
class FedALAState:
    weights: list[torch.Tensor]
    start_phase: bool


def _logits(outputs: torch.Tensor | tuple[object, ...]) -> torch.Tensor:
    if isinstance(outputs, tuple):
        return outputs[-1]
    return outputs


def _selected_parameter_names(model: nn.Module, layer_count: int) -> set[str]:
    parameterized_modules = [
        name
        for name, module in model.named_modules()
        if name and any(True for _ in module.parameters(recurse=False))
    ]
    if not parameterized_modules:
        return {name for name, _ in model.named_parameters()}
    selected_modules = tuple(parameterized_modules[-layer_count:])
    return {
        name
        for name, _ in model.named_parameters()
        if name.startswith(tuple(f"{module_name}." for module_name in selected_modules))
    }


def _sample_loader(train_loader: DataLoader, rand_percent: int) -> DataLoader:
    dataset = train_loader.dataset
    if rand_percent >= 100:
        return train_loader
    sample_count = max(1, int(len(dataset) * rand_percent / 100))
    subset = Subset(dataset, range(sample_count))
    return DataLoader(
        subset,
        batch_size=train_loader.batch_size,
        shuffle=False,
        drop_last=False,
    )


def adaptive_local_aggregation(
    local_model: nn.Module,
    global_model: nn.Module,
    train_loader: DataLoader,
    weights: list[torch.Tensor] | None,
    layer_count: int,
    eta: float,
    rand_percent: int,
    threshold: float,
    num_pre_loss: int,
    start_max_steps: int,
    device: str,
    start_phase: bool,
) -> FedALAState:
    """Initialize a local model by learning FedALA element-wise weights."""

    local_model.to(device)
    global_model.to(device)
    local_model.train()
    global_model.eval()

    selected_names = _selected_parameter_names(local_model, layer_count)
    local_parameters = dict(local_model.named_parameters())
    global_parameters = dict(global_model.named_parameters())
    selected_local = [local_parameters[name] for name in local_parameters if name in selected_names]
    selected_global = [global_parameters[name] for name in local_parameters if name in selected_names]

    with torch.no_grad():
        for name, parameter in local_parameters.items():
            if name not in selected_names:
                parameter.copy_(global_parameters[name])

    if not selected_local:
        return FedALAState(weights=[], start_phase=False)

    if all(
        torch.equal(local.detach(), global_parameter.detach())
        for local, global_parameter in zip(selected_local, selected_global)
    ):
        return FedALAState(
            weights=[
                torch.ones_like(parameter.detach(), device="cpu")
                for parameter in selected_local
            ],
            start_phase=start_phase,
        )

    if weights is None:
        active_weights = [
            torch.ones_like(parameter.detach(), device=device)
            for parameter in selected_local
        ]
    else:
        active_weights = [weight.detach().clone().to(device) for weight in weights]
    base_selected = [parameter.detach().clone() for parameter in selected_local]

    for parameter in global_model.parameters():
        parameter.requires_grad = False

    for name, parameter in local_model.named_parameters():
        parameter.requires_grad = name in selected_names

    criterion = nn.CrossEntropyLoss()
    sampled_loader = _sample_loader(train_loader, rand_percent)
    losses: list[float] = []
    steps = 0

    while True:
        for inputs, targets in sampled_loader:
            inputs = inputs.to(device)
            targets = targets.to(device).long()
            local_model.zero_grad(set_to_none=True)

            with torch.no_grad():
                for local, base_parameter, global_parameter, weight in zip(
                    selected_local,
                    base_selected,
                    selected_global,
                    active_weights,
                ):
                    local.copy_(base_parameter + (global_parameter - base_parameter) * weight)

            loss = criterion(_logits(local_model(inputs)), targets)
            loss.backward()

            with torch.no_grad():
                for local, base_parameter, global_parameter, weight in zip(
                    selected_local,
                    base_selected,
                    selected_global,
                    active_weights,
                ):
                    if local.grad is None:
                        continue
                    update = global_parameter - base_parameter
                    weight.copy_(
                        torch.clamp(
                            weight - eta * local.grad * update,
                            0.0,
                            1.0,
                        )
                    )
                    local.copy_(base_parameter + update * weight)

            losses.append(float(loss.item()))
            steps += 1

            if not start_phase:
                break
            if steps >= start_max_steps:
                break
            if len(losses) >= num_pre_loss:
                recent = losses[-num_pre_loss:]
                if float(np.std(recent)) < threshold:
                    break

        if not start_phase:
            break
        if steps >= start_max_steps:
            break
        if len(losses) >= num_pre_loss and float(np.std(losses[-num_pre_loss:])) < threshold:
            break

    for parameter in local_model.parameters():
        parameter.requires_grad = True
    for parameter in global_model.parameters():
        parameter.requires_grad = True

    local_model.to("cpu")
    global_model.to("cpu")
    return FedALAState(
        weights=[weight.detach().cpu().clone() for weight in active_weights],
        start_phase=False,
    )
