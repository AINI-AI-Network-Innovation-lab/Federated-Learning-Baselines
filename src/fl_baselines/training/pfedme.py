"""pFedMe client-side training loop."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader


def train_pfedme_client(
    reference_model: nn.Module,
    personalized_model: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    personal_learning_rate: float,
    personal_steps: int,
    device: str,
    pfedme_lambda: float,
) -> dict[str, float]:
    """Train pFedMe local reference and personalized models in place."""

    reference_model.to(device)
    personalized_model.to(device)
    reference_model.train()
    personalized_model.train()

    reference_optimizer = torch.optim.SGD(reference_model.parameters(), lr=learning_rate)
    personalized_optimizer = torch.optim.SGD(
        personalized_model.parameters(),
        lr=personal_learning_rate,
    )
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_examples = 0
    correct = 0

    for _ in range(epochs):
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device).long()

            reference_parameters = [
                parameter.detach().clone()
                for parameter in reference_model.parameters()
            ]

            final_personalized_loss = None
            for _ in range(personal_steps):
                personalized_optimizer.zero_grad()
                outputs = personalized_model(inputs)
                ce_loss = criterion(outputs, targets)

                reg_term = torch.tensor(0.0, device=device)
                for parameter, reference_parameter in zip(
                    personalized_model.parameters(),
                    reference_parameters,
                ):
                    reg_term += torch.sum((parameter - reference_parameter) ** 2)

                loss = ce_loss + (pfedme_lambda / 2.0) * reg_term
                loss.backward()
                personalized_optimizer.step()
                final_personalized_loss = loss

            reference_optimizer.zero_grad()
            with torch.no_grad():
                for reference_parameter, personalized_parameter in zip(
                    reference_model.parameters(),
                    personalized_model.parameters(),
                ):
                    reference_parameter.grad = pfedme_lambda * (
                        reference_parameter - personalized_parameter.detach()
                    )
            reference_optimizer.step()

            with torch.no_grad():
                reference_outputs = reference_model(inputs)
                reference_loss = criterion(reference_outputs, targets)

            batch_size = targets.size(0)
            total_loss += float(reference_loss.item()) * batch_size
            total_examples += batch_size
            correct += int((reference_outputs.argmax(dim=1) == targets).sum().item())

    reference_model.to("cpu")
    personalized_model.to("cpu")

    if total_examples == 0:
        return {"train_loss": 0.0, "train_accuracy": 0.0}
    return {
        "train_loss": total_loss / total_examples,
        "train_accuracy": correct / total_examples,
    }
