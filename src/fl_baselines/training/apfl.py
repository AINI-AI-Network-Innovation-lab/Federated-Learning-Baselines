"""APFL client-side training loop."""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn
from torch.func import functional_call
from torch.utils.data import DataLoader


def train_apfl_client(
    global_model: nn.Module,
    personalized_model: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    personal_learning_rate: float,
    device: str,
    alpha: float,
    adaptive_alpha: bool,
    alpha_learning_rate: float,
) -> tuple[dict[str, float], float]:
    """Train APFL global and personalized branches in place."""

    global_model.to(device)
    personalized_model.to(device)
    global_model.train()
    personalized_model.train()

    global_optimizer = torch.optim.SGD(global_model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    alpha_value = float(alpha)
    total_loss = 0.0
    total_examples = 0
    correct = 0

    for _ in range(epochs):
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device).long()

            global_optimizer.zero_grad()
            global_outputs = global_model(inputs)
            global_loss = criterion(global_outputs, targets)
            global_loss.backward()
            global_optimizer.step()

            local_parameters = OrderedDict(personalized_model.named_parameters())
            global_parameters = OrderedDict(
                (name, parameter.detach())
                for name, parameter in global_model.named_parameters()
            )
            alpha_tensor = torch.tensor(
                alpha_value,
                dtype=inputs.dtype,
                device=device,
                requires_grad=adaptive_alpha,
            )
            mixed_parameters = OrderedDict(
                (
                    name,
                    alpha_tensor * local_parameters[name]
                    + (1.0 - alpha_tensor) * global_parameters[name],
                )
                for name in local_parameters
            )
            mixed_state = OrderedDict(mixed_parameters)
            mixed_state.update(
                (name, buffer)
                for name, buffer in personalized_model.named_buffers()
            )
            personalized_outputs = functional_call(personalized_model, mixed_state, (inputs,))
            personalized_loss = criterion(personalized_outputs, targets)

            grad_targets = list(local_parameters.values())
            if adaptive_alpha:
                grad_targets.append(alpha_tensor)
            gradients = torch.autograd.grad(personalized_loss, tuple(grad_targets))
            parameter_gradients = gradients[: len(local_parameters)]
            with torch.no_grad():
                for parameter, gradient in zip(personalized_model.parameters(), parameter_gradients):
                    parameter.add_(-personal_learning_rate * gradient)

            if adaptive_alpha:
                alpha_gradient = gradients[-1]
                alpha_value = float(
                    min(
                        1.0,
                        max(0.0, alpha_value - alpha_learning_rate * float(alpha_gradient.item())),
                    )
                )

            batch_size = targets.size(0)
            total_loss += float(personalized_loss.item()) * batch_size
            total_examples += batch_size
            correct += int((personalized_outputs.argmax(dim=1) == targets).sum().item())

    global_model.to("cpu")
    personalized_model.to("cpu")

    if total_examples == 0:
        return {"train_loss": 0.0, "train_accuracy": 0.0, "apfl_alpha": alpha_value}, alpha_value
    return {
        "train_loss": total_loss / total_examples,
        "train_accuracy": correct / total_examples,
        "apfl_alpha": alpha_value,
    }, alpha_value
