"""FedGen client-side training and evaluation helpers."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from fl_baselines.training.evaluate import _compute_macro_classification_metrics
from fl_baselines.training.features import (
    classifier_weight_matrix,
    extract_features,
    logits_from_features,
)


def train_fedgen_client(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: str,
    global_mask: np.ndarray,
    fedgen_alpha: float,
    fedgen_lambda: float,
    fedgen_beta: float,
    fedgen_delta: float,
    fedgen_warmup_epochs: int,
    fedgen_l1_weight: float,
) -> tuple[dict[str, float], np.ndarray]:
    """Train one client with FedGen feature masking and return the updated mask."""

    model.to(device)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    mask_tensor = torch.as_tensor(global_mask, dtype=torch.float32, device=device).clone()
    running_mean = torch.zeros_like(mask_tensor)
    running_var = torch.zeros_like(mask_tensor)

    total_loss = 0.0
    total_examples = 0
    correct = 0

    for epoch_index in range(epochs):
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device).long()

            optimizer.zero_grad()
            features = extract_features(model, inputs)
            if features.shape[1] != mask_tensor.shape[0]:
                raise ValueError("FedGen global mask shape does not match extracted feature dimension")
            masked_features = features * torch.sigmoid(mask_tensor).unsqueeze(0)
            logits = logits_from_features(model, masked_features)
            ce_loss = criterion(logits, targets)
            classifier_weight = classifier_weight_matrix(model)
            penalty_grad = torch.autograd.grad(
                ce_loss,
                classifier_weight,
                create_graph=True,
                retain_graph=True,
            )[0]
            penalty = torch.sum((classifier_weight * penalty_grad) ** 2)
            l1_penalty = torch.tensor(0.0, device=device)
            for parameter in model.parameters():
                l1_penalty = l1_penalty + parameter.abs().sum()

            loss = ce_loss + (fedgen_l1_weight * l1_penalty) + (fedgen_lambda * penalty)
            loss.backward()
            optimizer.step()

            batch_size = targets.size(0)
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size
            correct += int((logits.argmax(dim=1) == targets).sum().item())

            current_feature_weights = classifier_weight_matrix(model).detach().pow(2).mean(dim=0).sqrt()
            previous_mean = running_mean.clone()
            running_mean = (fedgen_beta * current_feature_weights) + ((1.0 - fedgen_beta) * previous_mean)
            running_var = (fedgen_delta * running_var) + (
                (1.0 - fedgen_delta) * (current_feature_weights - previous_mean) ** 2
            )
            if epoch_index >= fedgen_warmup_epochs:
                average_variance = running_var.mean()
                mask_tensor = mask_tensor + (average_variance - (fedgen_alpha * running_var))

    model.to("cpu")

    if total_examples == 0:
        metrics = {"train_loss": 0.0, "train_accuracy": 0.0}
    else:
        metrics = {
            "train_loss": total_loss / total_examples,
            "train_accuracy": correct / total_examples,
        }
    metrics["fedgen_mask_mean"] = float(torch.sigmoid(mask_tensor).mean().item())
    return metrics, mask_tensor.detach().cpu().numpy().astype(np.float32)


def evaluate_fedgen_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: str,
    global_mask: np.ndarray,
) -> tuple[float, dict[str, float]]:
    """Evaluate a model using the FedGen global mask in feature space."""

    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    mask_tensor = torch.as_tensor(global_mask, dtype=torch.float32, device=device)

    total_loss = 0.0
    total_examples = 0
    correct = 0
    true_positives: torch.Tensor | None = None
    predicted_positives: torch.Tensor | None = None
    actual_positives: torch.Tensor | None = None

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device).long()
            features = extract_features(model, inputs)
            if features.shape[1] != mask_tensor.shape[0]:
                raise ValueError("FedGen global mask shape does not match extracted feature dimension")
            masked_features = features * torch.sigmoid(mask_tensor).unsqueeze(0)
            logits = logits_from_features(model, masked_features)
            loss = criterion(logits, targets)
            predictions = logits.argmax(dim=1)

            batch_size = targets.size(0)
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size
            correct += int((predictions == targets).sum().item())

            num_classes = logits.shape[1]
            if true_positives is None:
                true_positives = torch.zeros(num_classes, dtype=torch.float32)
                predicted_positives = torch.zeros(num_classes, dtype=torch.float32)
                actual_positives = torch.zeros(num_classes, dtype=torch.float32)

            predicted_cpu = predictions.detach().cpu()
            targets_cpu = targets.detach().cpu()
            predicted_positives += torch.bincount(predicted_cpu, minlength=num_classes).to(torch.float32)
            actual_positives += torch.bincount(targets_cpu, minlength=num_classes).to(torch.float32)
            true_positives += torch.bincount(
                targets_cpu[predicted_cpu == targets_cpu],
                minlength=num_classes,
            ).to(torch.float32)

    model.to("cpu")

    if total_examples == 0:
        return 0.0, {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    metrics = {"accuracy": correct / total_examples}
    metrics.update(
        _compute_macro_classification_metrics(
            true_positives=true_positives,
            predicted_positives=predicted_positives,
            actual_positives=actual_positives,
        )
    )
    return total_loss / total_examples, metrics
