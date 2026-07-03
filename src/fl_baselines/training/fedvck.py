"""FedVCK local condensation and server replay helpers."""

from __future__ import annotations

import copy
from collections.abc import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from fl_baselines.training.features import extract_features


def train_fedvck_client(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: str,
    condensed_ratio: float,
    condensed_steps: int,
    condensed_learning_rate: float,
    importance_alpha: float,
    enable_latent_constraints: bool,
    previous_model_state: dict[str, torch.Tensor] | None,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Train a client and return condensed knowledge plus class prototype stats."""

    del condensed_steps
    del condensed_learning_rate
    del enable_latent_constraints

    model.to(device)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    previous_model = _load_previous_model(model, previous_model_state, device)

    total_loss = 0.0
    total_examples = 0
    correct = 0
    cached_inputs: list[torch.Tensor] = []
    cached_targets: list[torch.Tensor] = []
    cached_scores: list[torch.Tensor] = []
    prototype_sums: torch.Tensor | None = None
    prototype_counts: torch.Tensor | None = None

    for _ in range(epochs):
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device).long()

            optimizer.zero_grad()
            outputs = model(inputs)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[-1]
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            batch_size = targets.size(0)
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size
            correct += int((outputs.argmax(dim=1) == targets).sum().item())

            with torch.no_grad():
                if prototype_sums is None:
                    num_classes = outputs.shape[1]
                    prototype_sums = torch.zeros(
                        (num_classes, num_classes),
                        dtype=torch.float32,
                        device=device,
                    )
                    prototype_counts = torch.zeros(num_classes, dtype=torch.float32, device=device)

                previous_outputs = None
                if previous_model is not None:
                    previous_outputs = previous_model(inputs)
                    if isinstance(previous_outputs, (tuple, list)):
                        previous_outputs = previous_outputs[-1]
                importance_scores = _compute_importance_scores(
                    outputs=outputs.detach(),
                    targets=targets,
                    previous_outputs=previous_outputs,
                    importance_alpha=importance_alpha,
                )
                cached_inputs.append(inputs.detach().cpu())
                cached_targets.append(targets.detach().cpu())
                cached_scores.append(importance_scores.detach().cpu())
                _accumulate_logit_prototypes(
                    outputs.detach(),
                    targets,
                    prototype_sums,
                    prototype_counts,
                )

    model.to("cpu")
    if prototype_sums is None or prototype_counts is None:
        num_classes = _infer_num_classes(model)
        prototype_sums_np = np.zeros((num_classes, num_classes), dtype=np.float32)
        prototype_counts_np = np.zeros(num_classes, dtype=np.float32)
        condensed_inputs, condensed_labels = _empty_condensed_payload(train_loader)
        metrics = {
            "train_loss": 0.0,
            "train_accuracy": 0.0,
            "fedvck_condensed_size": 0,
        }
        return metrics, condensed_inputs, condensed_labels, prototype_sums_np, prototype_counts_np

    condensed_inputs, condensed_labels = build_condensed_dataset(
        cached_inputs=cached_inputs,
        cached_targets=cached_targets,
        cached_scores=cached_scores,
        condensed_ratio=condensed_ratio,
    )
    metrics = {
        "train_loss": total_loss / max(total_examples, 1),
        "train_accuracy": correct / max(total_examples, 1),
        "fedvck_condensed_size": int(condensed_labels.shape[0]),
    }
    return (
        metrics,
        condensed_inputs,
        condensed_labels,
        prototype_sums.detach().cpu().numpy(),
        prototype_counts.detach().cpu().numpy(),
    )


def build_condensed_dataset(
    *,
    cached_inputs: Iterable[torch.Tensor],
    cached_targets: Iterable[torch.Tensor],
    cached_scores: Iterable[torch.Tensor],
    condensed_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Select a small informative synthetic proxy directly from cached examples."""

    inputs = torch.cat(list(cached_inputs), dim=0)
    targets = torch.cat(list(cached_targets), dim=0)
    scores = torch.cat(list(cached_scores), dim=0)
    condensed_size = max(1, int(round(inputs.shape[0] * condensed_ratio)))
    top_indices = torch.topk(scores, k=min(condensed_size, scores.numel()), largest=True).indices
    return (
        inputs[top_indices].detach().cpu().numpy().astype(np.float32, copy=False),
        targets[top_indices].detach().cpu().numpy().astype(np.int64, copy=False),
    )


def replay_fedvck_server(
    model: nn.Module,
    condensed_memory: list[tuple[np.ndarray, np.ndarray]],
    global_logit_prototypes: np.ndarray,
    *,
    epochs: int,
    learning_rate: float,
    temperature: float,
    hard_negative_k: int,
    device: str,
) -> dict[str, float]:
    """Replay accumulated condensed knowledge on the server."""

    del temperature

    if not condensed_memory:
        return {"fedvck_replay_loss": 0.0}

    model.to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    feature_prototypes = build_feature_prototypes(
        model,
        condensed_memory,
        device=device,
    )
    hard_negatives = compute_hard_negative_indices(global_logit_prototypes, hard_negative_k)
    total_loss = 0.0
    total_examples = 0

    for _ in range(epochs):
        for condensed_inputs, condensed_labels in condensed_memory:
            inputs = torch.tensor(condensed_inputs, dtype=torch.float32, device=device)
            labels = torch.tensor(condensed_labels, dtype=torch.long, device=device)

            optimizer.zero_grad()
            features = extract_features(model, inputs)
            logits = model(inputs)
            if isinstance(logits, (tuple, list)):
                logits = logits[-1]

            ce_loss = criterion(logits, labels)
            contrastive_loss = relational_contrastive_loss(
                features,
                labels,
                feature_prototypes,
                hard_negatives,
            )
            loss = ce_loss + contrastive_loss
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size

    model.to("cpu")
    return {"fedvck_replay_loss": total_loss / max(total_examples, 1)}


def build_feature_prototypes(
    model: nn.Module,
    condensed_memory: list[tuple[np.ndarray, np.ndarray]],
    *,
    device: str,
) -> dict[int, torch.Tensor]:
    """Build class-wise feature prototypes from condensed memory."""

    prototype_sums: dict[int, torch.Tensor] = {}
    prototype_counts: dict[int, int] = {}
    model.to(device)
    model.eval()
    with torch.no_grad():
        for condensed_inputs, condensed_labels in condensed_memory:
            inputs = torch.tensor(condensed_inputs, dtype=torch.float32, device=device)
            labels = torch.tensor(condensed_labels, dtype=torch.long, device=device)
            features = extract_features(model, inputs)
            for class_index in labels.unique(sorted=True).tolist():
                mask = labels == class_index
                feature_sum = features[mask].sum(dim=0)
                if class_index not in prototype_sums:
                    prototype_sums[class_index] = feature_sum
                    prototype_counts[class_index] = int(mask.sum().item())
                else:
                    prototype_sums[class_index] += feature_sum
                    prototype_counts[class_index] += int(mask.sum().item())
    model.train()
    return {
        class_index: prototype_sums[class_index] / max(prototype_counts[class_index], 1)
        for class_index in prototype_sums
    }


def compute_hard_negative_indices(
    global_logit_prototypes: np.ndarray,
    hard_negative_k: int,
) -> dict[int, list[int]]:
    """Return top-k hard negative class indices per class."""

    hard_negatives: dict[int, list[int]] = {}
    num_classes = global_logit_prototypes.shape[0]
    for class_index in range(num_classes):
        row = global_logit_prototypes[class_index].copy()
        if class_index < row.shape[0]:
            row[class_index] = -np.inf
        negative_indices = np.argsort(row)[::-1][: min(hard_negative_k, max(num_classes - 1, 0))]
        hard_negatives[class_index] = [
            int(index) for index in negative_indices if np.isfinite(row[index])
        ]
    return hard_negatives


def relational_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    feature_prototypes: dict[int, torch.Tensor],
    hard_negatives: dict[int, list[int]],
) -> torch.Tensor:
    """A lightweight relational loss aligned with class prototypes and hard negatives."""

    if not feature_prototypes:
        return torch.tensor(0.0, device=features.device)

    loss = torch.tensor(0.0, device=features.device)
    used = 0
    for feature, label_tensor in zip(features, labels):
        label = int(label_tensor.item())
        positive = feature_prototypes.get(label)
        if positive is None:
            continue
        loss = loss + torch.sum((feature - positive.detach()) ** 2)
        used += 1
        for negative_index in hard_negatives.get(label, []):
            negative = feature_prototypes.get(negative_index)
            if negative is None:
                continue
            loss = loss + F.relu(1.0 - torch.sum((feature - negative.detach()) ** 2))
    if used == 0:
        return torch.tensor(0.0, device=features.device)
    return loss / used


def _compute_importance_scores(
    *,
    outputs: torch.Tensor,
    targets: torch.Tensor,
    previous_outputs: torch.Tensor | None,
    importance_alpha: float,
) -> torch.Tensor:
    criterion = nn.CrossEntropyLoss(reduction="none")
    current_scores = criterion(outputs, targets)
    if previous_outputs is None:
        return current_scores
    smoothed_logits = importance_alpha * outputs + (1.0 - importance_alpha) * previous_outputs
    return criterion(smoothed_logits, targets)


def _accumulate_logit_prototypes(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    prototype_sums: torch.Tensor,
    prototype_counts: torch.Tensor,
) -> None:
    for class_index in targets.unique(sorted=True).tolist():
        mask = targets == class_index
        prototype_sums[class_index] += outputs[mask].sum(dim=0)
        prototype_counts[class_index] += float(mask.sum().item())


def _load_previous_model(
    model: nn.Module,
    previous_model_state: dict[str, torch.Tensor] | None,
    device: str,
) -> nn.Module | None:
    if previous_model_state is None:
        return None
    previous_model = copy.deepcopy(model)
    previous_model.load_state_dict(previous_model_state, strict=True)
    previous_model.to(device)
    previous_model.eval()
    return previous_model


def _infer_num_classes(model: nn.Module) -> int:
    last_parameter = next(reversed(list(model.state_dict().values())))
    if last_parameter.ndim == 1:
        return int(last_parameter.shape[0])
    return int(last_parameter.shape[0]) if last_parameter.ndim > 0 else 1


def _empty_condensed_payload(train_loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    dataset = getattr(train_loader, "dataset", None)
    if dataset is not None and len(dataset) > 0:
        first_input, _ = dataset[0]
        input_shape = tuple(first_input.shape)
    else:
        input_shape = (1,)
    return (
        np.zeros((0, *input_shape), dtype=np.float32),
        np.zeros((0,), dtype=np.int64),
    )
