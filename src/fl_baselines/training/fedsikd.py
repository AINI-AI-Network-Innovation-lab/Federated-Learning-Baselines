"""FedSiKD client-side statistics and distillation helpers."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


def _logits_from_output(output: torch.Tensor | tuple | list) -> torch.Tensor:
    if isinstance(output, (tuple, list)):
        return output[-1]
    return output


def compute_fedsikd_statistics(train_loader: DataLoader) -> tuple[float, float, float]:
    """Return mean, standard deviation, and skewness for the local inputs."""

    sample_count = 0
    sum_1 = 0.0
    sum_2 = 0.0
    sum_3 = 0.0

    for inputs, _ in train_loader:
        batch = inputs.detach().cpu().to(torch.float64).reshape(-1)
        if batch.numel() == 0:
            continue
        sample_count += int(batch.numel())
        sum_1 += float(batch.sum().item())
        sum_2 += float((batch * batch).sum().item())
        sum_3 += float((batch * batch * batch).sum().item())

    if sample_count == 0:
        return 0.0, 0.0, 0.0

    mean = sum_1 / sample_count
    second_moment = sum_2 / sample_count
    variance = max(0.0, second_moment - mean * mean)
    std = math.sqrt(variance)
    if std == 0.0:
        return mean, 0.0, 0.0

    third_moment = sum_3 / sample_count
    central_third = third_moment - 3.0 * mean * second_moment + 2.0 * (mean**3)
    skewness = central_third / (std**3)
    return mean, std, skewness


def train_fedsikd_client(
    model: nn.Module,
    teacher_model: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: str,
    fedsikd_kd_alpha: float,
    fedsikd_kd_temperature: float,
) -> dict[str, float]:
    """Train a local FedSiKD client with cross-entropy and distillation loss."""

    model.to(device)
    teacher_model.to(device)
    model.train()
    teacher_model.eval()

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    temperature = max(float(fedsikd_kd_temperature), 1e-8)

    total_loss = 0.0
    total_examples = 0
    total_ce_loss = 0.0
    total_kd_loss = 0.0
    correct = 0

    for _ in range(epochs):
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device).long()

            optimizer.zero_grad()
            student_logits = _logits_from_output(model(inputs))
            with torch.no_grad():
                teacher_logits = _logits_from_output(teacher_model(inputs))

            ce_loss = criterion(student_logits, targets)
            student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
            teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
            kd_loss = F.kl_div(
                student_log_probs,
                teacher_probs,
                reduction="batchmean",
            ) * (temperature**2)
            loss = ce_loss + float(fedsikd_kd_alpha) * kd_loss
            loss.backward()
            optimizer.step()

            batch_size = targets.size(0)
            total_loss += float(loss.item()) * batch_size
            total_ce_loss += float(ce_loss.item()) * batch_size
            total_kd_loss += float(kd_loss.item()) * batch_size
            total_examples += batch_size
            correct += int((student_logits.argmax(dim=1) == targets).sum().item())

    model.to("cpu")
    teacher_model.to("cpu")

    mean, std, skewness = compute_fedsikd_statistics(train_loader)

    if total_examples == 0:
        return {
            "train_loss": 0.0,
            "train_accuracy": 0.0,
            "fedsikd_ce_loss": 0.0,
            "fedsikd_kd_loss": 0.0,
            "fedsikd_mean": mean,
            "fedsikd_std": std,
            "fedsikd_skewness": skewness,
        }

    return {
        "train_loss": total_loss / total_examples,
        "train_accuracy": correct / total_examples,
        "fedsikd_ce_loss": total_ce_loss / total_examples,
        "fedsikd_kd_loss": total_kd_loss / total_examples,
        "fedsikd_mean": mean,
        "fedsikd_std": std,
        "fedsikd_skewness": skewness,
    }
