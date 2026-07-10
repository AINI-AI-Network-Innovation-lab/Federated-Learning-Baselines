"""FedMeta client-side meta-gradient computation."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from torch.func import functional_call
from torch.utils.data import DataLoader


def train_fedmeta_client(
    model: nn.Module,
    train_loader: DataLoader,
    device: str,
    inner_learning_rate: float,
    support_fraction: float,
    inner_steps: int,
    first_order: bool,
    alpha_parameters: list[np.ndarray] | None = None,
) -> tuple[dict[str, float], list[np.ndarray]]:
    """Compute FedMeta MAML or Meta-SGD gradients on local support/query splits."""

    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()

    param_items = list(model.named_parameters())
    buffer_items = list(model.named_buffers())
    parameter_names = [name for name, _ in param_items]
    base_parameters = OrderedDict(
        (name, parameter) for name, parameter in param_items
    )
    buffers = OrderedDict((name, buffer) for name, buffer in buffer_items)
    meta_sgd = alpha_parameters is not None
    alpha_tensors = _alpha_tensors(
        alpha_parameters,
        [parameter for _, parameter in param_items],
        device,
    )

    gradient_sums = [
        torch.zeros_like(parameter, device=device)
        for _, parameter in param_items
    ]
    alpha_gradient_sums = [
        torch.zeros_like(alpha, device=device)
        for alpha in alpha_tensors
    ]
    total_query_loss = 0.0
    total_query_examples = 0
    correct = 0
    episodes = 0

    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device).long()
        if targets.size(0) < 2:
            continue

        support_inputs, support_targets, query_inputs, query_targets = _support_query_split(
            inputs,
            targets,
            support_fraction,
        )
        adapted_parameters = base_parameters
        for _ in range(inner_steps):
            support_outputs = functional_call(
                model,
                (adapted_parameters, buffers),
                (support_inputs,),
            )
            support_loss = criterion(_logits_from_output(support_outputs), support_targets)
            support_gradients = torch.autograd.grad(
                support_loss,
                tuple(adapted_parameters.values()),
                create_graph=not first_order,
            )
            adapted_parameters = OrderedDict(
                (
                    name,
                    parameter - _inner_step_size(
                        index,
                        inner_learning_rate,
                        alpha_tensors,
                    )
                    * support_gradients[index],
                )
                for index, (name, parameter) in enumerate(adapted_parameters.items())
            )

        query_outputs = functional_call(
            model,
            (adapted_parameters, buffers),
            (query_inputs,),
        )
        query_logits = _logits_from_output(query_outputs)
        query_loss = criterion(query_logits, query_targets)

        gradient_targets: tuple[torch.Tensor, ...]
        if meta_sgd:
            gradient_targets = tuple(base_parameters.values()) + tuple(alpha_tensors)
            gradients = torch.autograd.grad(query_loss, gradient_targets)
            model_gradients = gradients[: len(parameter_names)]
            alpha_gradients = gradients[len(parameter_names) :]
            for index, gradient in enumerate(alpha_gradients):
                alpha_gradient_sums[index] += gradient.detach()
        else:
            model_gradients = torch.autograd.grad(
                query_loss,
                tuple(base_parameters.values()),
            )

        for index, gradient in enumerate(model_gradients):
            gradient_sums[index] += gradient.detach()

        batch_size = query_targets.size(0)
        total_query_loss += float(query_loss.item()) * batch_size
        total_query_examples += batch_size
        correct += int((query_logits.argmax(dim=1) == query_targets).sum().item())
        episodes += 1

    model.to("cpu")

    if episodes == 0 or total_query_examples == 0:
        metrics = {"meta_query_loss": 0.0, "meta_query_accuracy": 0.0}
        empty_gradients = [
            torch.zeros_like(parameter).detach().cpu().numpy()
            for _, parameter in param_items
        ]
        if meta_sgd:
            empty_gradients += [
                torch.zeros_like(alpha).detach().cpu().numpy()
                for alpha in alpha_tensors
            ]
        return metrics, empty_gradients

    scale = 1.0 / float(episodes)
    gradients_np = [
        (gradient * scale).detach().cpu().numpy()
        for gradient in gradient_sums
    ]
    if meta_sgd:
        gradients_np += [
            (gradient * scale).detach().cpu().numpy()
            for gradient in alpha_gradient_sums
        ]
    metrics = {
        "meta_query_loss": total_query_loss / total_query_examples,
        "meta_query_accuracy": correct / total_query_examples,
    }
    return metrics, gradients_np


def _alpha_tensors(
    alpha_parameters: list[np.ndarray] | None,
    model_parameters: list[torch.nn.Parameter],
    device: str,
) -> list[torch.Tensor]:
    if alpha_parameters is None:
        return []
    return [
        torch.as_tensor(alpha, dtype=parameter.dtype, device=device).requires_grad_(True)
        for alpha, parameter in zip(alpha_parameters, model_parameters)
    ]


def _inner_step_size(
    parameter_index: int,
    inner_learning_rate: float,
    alpha_tensors: list[torch.Tensor],
) -> float | torch.Tensor:
    if alpha_tensors:
        return alpha_tensors[parameter_index]
    return inner_learning_rate


def _support_query_split(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    support_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    support_size = int(round(float(targets.size(0)) * support_fraction))
    support_size = min(max(1, support_size), targets.size(0) - 1)
    return (
        inputs[:support_size],
        targets[:support_size],
        inputs[support_size:],
        targets[support_size:],
    )


def _logits_from_output(output: torch.Tensor | tuple | list) -> torch.Tensor:
    if isinstance(output, (tuple, list)):
        return output[-1]
    return output
