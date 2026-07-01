"""FedProto client-side training loop."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from fl_baselines.training.features import extract_features


def train_fedproto_client(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: str,
    fedproto_lambda: float,
    global_prototypes: np.ndarray,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Train one client with prototype regularization and return prototype sums/counts."""

    model.to(device)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    global_proto_tensor = torch.as_tensor(global_prototypes, dtype=torch.float32, device=device)
    num_classes = global_prototypes.shape[0]
    feature_dim = global_prototypes.shape[1]

    total_loss = 0.0
    total_examples = 0
    correct = 0
    prototype_sums = torch.zeros((num_classes, feature_dim), dtype=torch.float32, device=device)
    prototype_counts = torch.zeros(num_classes, dtype=torch.float32, device=device)

    for _ in range(epochs):
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device).long()

            optimizer.zero_grad()
            features = extract_features(model, inputs)
            logits = model(inputs)
            ce_loss = criterion(logits, targets)

            reg_loss = torch.tensor(0.0, device=device)
            unique_targets = targets.unique(sorted=True)
            for class_index in unique_targets.tolist():
                mask = targets == class_index
                class_features = features[mask]
                class_mean = class_features.mean(dim=0)
                reg_loss += torch.sum((class_mean - global_proto_tensor[class_index]) ** 2)
                prototype_sums[class_index] += class_features.detach().sum(dim=0)
                prototype_counts[class_index] += float(mask.sum().item())

            loss = ce_loss + fedproto_lambda * reg_loss
            loss.backward()
            optimizer.step()

            batch_size = targets.size(0)
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size
            correct += int((logits.argmax(dim=1) == targets).sum().item())

    model.to("cpu")

    if total_examples == 0:
        metrics = {"train_loss": 0.0, "train_accuracy": 0.0}
    else:
        metrics = {
            "train_loss": total_loss / total_examples,
            "train_accuracy": correct / total_examples,
        }
    return (
        metrics,
        prototype_sums.detach().cpu().numpy(),
        prototype_counts.detach().cpu().numpy(),
    )
