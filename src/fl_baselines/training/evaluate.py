"""PyTorch evaluation loop."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader


def _logits_from_output(output: torch.Tensor | tuple | list) -> torch.Tensor:
    if isinstance(output, (tuple, list)):
        return output[-1]
    return output


def _compute_macro_classification_metrics(
    true_positives: torch.Tensor,
    predicted_positives: torch.Tensor,
    actual_positives: torch.Tensor,
) -> dict[str, float]:
    precision_per_class = torch.zeros_like(true_positives, dtype=torch.float32)
    recall_per_class = torch.zeros_like(true_positives, dtype=torch.float32)

    precision_mask = predicted_positives > 0
    recall_mask = actual_positives > 0

    precision_per_class[precision_mask] = (
        true_positives[precision_mask] / predicted_positives[precision_mask]
    )
    recall_per_class[recall_mask] = (
        true_positives[recall_mask] / actual_positives[recall_mask]
    )

    f1_per_class = torch.zeros_like(true_positives, dtype=torch.float32)
    f1_denominator = precision_per_class + recall_per_class
    f1_mask = f1_denominator > 0
    f1_per_class[f1_mask] = (
        2.0
        * precision_per_class[f1_mask]
        * recall_per_class[f1_mask]
        / f1_denominator[f1_mask]
    )

    return {
        "precision": float(precision_per_class.mean().item()),
        "recall": float(recall_per_class.mean().item()),
        "f1": float(f1_per_class.mean().item()),
    }


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: str,
) -> tuple[float, dict[str, float]]:
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_examples = 0
    correct = 0
    true_positives: torch.Tensor | None = None
    predicted_positives: torch.Tensor | None = None
    actual_positives: torch.Tensor | None = None

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = _logits_from_output(model(inputs))
            loss = criterion(outputs, targets)
            predictions = outputs.argmax(dim=1)

            batch_size = targets.size(0)
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size
            correct += int((predictions == targets).sum().item())

            num_classes = outputs.shape[1]
            if true_positives is None:
                true_positives = torch.zeros(num_classes, dtype=torch.float32)
                predicted_positives = torch.zeros(num_classes, dtype=torch.float32)
                actual_positives = torch.zeros(num_classes, dtype=torch.float32)

            predicted_cpu = predictions.detach().cpu()
            targets_cpu = targets.detach().cpu()

            predicted_positives += torch.bincount(
                predicted_cpu, minlength=num_classes
            ).to(torch.float32)
            actual_positives += torch.bincount(
                targets_cpu, minlength=num_classes
            ).to(torch.float32)
            true_positives += torch.bincount(
                targets_cpu[predicted_cpu == targets_cpu],
                minlength=num_classes,
            ).to(torch.float32)

    if total_examples == 0:
        return 0.0, {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    average_loss = total_loss / total_examples
    metrics = {"accuracy": correct / total_examples}
    metrics.update(
        _compute_macro_classification_metrics(
            true_positives=true_positives,
            predicted_positives=predicted_positives,
            actual_positives=actual_positives,
        )
    )
    return average_loss, metrics
