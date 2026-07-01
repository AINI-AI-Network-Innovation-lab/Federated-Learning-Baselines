"""FedNTD client-side training loop."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


def _logits_from_output(output: torch.Tensor | tuple | list) -> torch.Tensor:
    if isinstance(output, (tuple, list)):
        return output[-1]
    return output


def _not_true_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if student_logits.shape[1] <= 1:
        return torch.tensor(0.0, device=student_logits.device)

    scaled_student = student_logits / temperature
    scaled_teacher = teacher_logits / temperature
    mask = F.one_hot(targets, num_classes=student_logits.shape[1]).to(torch.bool)

    min_value = torch.finfo(scaled_student.dtype).min
    masked_student = scaled_student.masked_fill(mask, min_value)
    masked_teacher = scaled_teacher.masked_fill(mask, min_value)

    student_log_probs = F.log_softmax(masked_student, dim=1)
    teacher_probs = F.softmax(masked_teacher, dim=1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")


def train_fedntd_client(
    model: nn.Module,
    teacher_model: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: str,
    fedntd_beta: float,
    fedntd_temperature: float,
) -> dict[str, float]:
    """Train a local FedNTD client against a fixed global teacher snapshot."""

    model.to(device)
    teacher_model.to(device)
    model.train()
    teacher_model.eval()

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_examples = 0
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
            ntd_loss = _not_true_distillation_loss(
                student_logits,
                teacher_logits,
                targets,
                fedntd_temperature,
            )
            loss = ce_loss + fedntd_beta * ntd_loss
            loss.backward()
            optimizer.step()

            batch_size = targets.size(0)
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size
            correct += int((student_logits.argmax(dim=1) == targets).sum().item())

    model.to("cpu")
    teacher_model.to("cpu")

    if total_examples == 0:
        return {"train_loss": 0.0, "train_accuracy": 0.0}
    return {
        "train_loss": total_loss / total_examples,
        "train_accuracy": correct / total_examples,
    }
