"""MOON client-side training loop."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


def _representation_and_logits(
    output: torch.Tensor | tuple | list,
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(output, (tuple, list)):
        logits = output[-1]
        representation = output[-2] if len(output) >= 2 else logits
    else:
        logits = output
        representation = output

    if representation.dim() > 2:
        representation = torch.flatten(representation, start_dim=1)
    return representation, logits


def train_moon_client(
    model: nn.Module,
    global_model: nn.Module,
    previous_model: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: str,
    moon_mu: float,
    temperature: float,
) -> dict[str, float]:
    """Train one client with MOON model-contrastive loss."""

    model.to(device)
    global_model.to(device)
    previous_model.to(device)
    model.train()
    global_model.eval()
    previous_model.eval()

    for parameter in global_model.parameters():
        parameter.requires_grad = False
    for parameter in previous_model.parameters():
        parameter.requires_grad = False

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_ce_loss = 0.0
    total_contrastive_loss = 0.0
    total_examples = 0
    correct = 0

    for _ in range(epochs):
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device).long()

            optimizer.zero_grad()
            current_representation, logits = _representation_and_logits(model(inputs))
            with torch.no_grad():
                global_representation, _ = _representation_and_logits(
                    global_model(inputs)
                )
                previous_representation, _ = _representation_and_logits(
                    previous_model(inputs)
                )

            positive = F.cosine_similarity(
                current_representation,
                global_representation,
                dim=-1,
            )
            negative = F.cosine_similarity(
                current_representation,
                previous_representation,
                dim=-1,
            )
            contrastive_logits = torch.stack((positive, negative), dim=1) / temperature
            contrastive_targets = torch.zeros(
                inputs.size(0),
                dtype=torch.long,
                device=device,
            )

            ce_loss = criterion(logits, targets)
            contrastive_loss = criterion(contrastive_logits, contrastive_targets)
            loss = ce_loss + moon_mu * contrastive_loss
            loss.backward()
            optimizer.step()

            batch_size = targets.size(0)
            total_loss += float(loss.item()) * batch_size
            total_ce_loss += float(ce_loss.item()) * batch_size
            total_contrastive_loss += float(contrastive_loss.item()) * batch_size
            total_examples += batch_size
            correct += int((logits.argmax(dim=1) == targets).sum().item())

    model.to("cpu")
    global_model.to("cpu")
    previous_model.to("cpu")

    if total_examples == 0:
        return {
            "train_loss": 0.0,
            "train_accuracy": 0.0,
            "moon_ce_loss": 0.0,
            "moon_contrastive_loss": 0.0,
        }

    return {
        "train_loss": total_loss / total_examples,
        "train_accuracy": correct / total_examples,
        "moon_ce_loss": total_ce_loss / total_examples,
        "moon_contrastive_loss": total_contrastive_loss / total_examples,
    }
