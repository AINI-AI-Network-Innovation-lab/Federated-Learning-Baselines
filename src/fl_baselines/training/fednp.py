"""FedNP client-side training loop."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from fl_baselines.training.features import extract_features


def train_fednp_client(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: str,
    latent_mean: np.ndarray,
    latent_var: np.ndarray,
    fednp_lambda: float,
    fednp_stability_eps: float,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """Train one client with latent Gaussian regularization and return sufficient statistics."""

    model.to(device)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    global_mean = torch.as_tensor(latent_mean, dtype=torch.float32, device=device)
    global_var = torch.as_tensor(latent_var, dtype=torch.float32, device=device).clamp_min(
        fednp_stability_eps
    )

    latent_sum = torch.zeros_like(global_mean)
    latent_square_sum = torch.zeros_like(global_mean)
    latent_count = 0.0
    total_loss = 0.0
    total_reg_loss = 0.0
    total_examples = 0
    correct = 0

    for _ in range(epochs):
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device).long()

            optimizer.zero_grad()
            features = extract_features(model, inputs)
            logits = model(inputs)
            ce_loss = criterion(logits, targets)

            local_mean = features.mean(dim=0)
            centered = features - local_mean.unsqueeze(0)
            local_var = centered.pow(2).mean(dim=0).clamp_min(fednp_stability_eps)
            reg_loss = _gaussian_kl(local_mean, local_var, global_mean, global_var)
            loss = ce_loss + (fednp_lambda * reg_loss)
            loss.backward()
            optimizer.step()

            batch_size = targets.size(0)
            latent_sum += features.detach().sum(dim=0)
            latent_square_sum += features.detach().pow(2).sum(dim=0)
            latent_count += float(batch_size)
            total_loss += float(loss.item()) * batch_size
            total_reg_loss += float(reg_loss.item()) * batch_size
            total_examples += batch_size
            correct += int((logits.argmax(dim=1) == targets).sum().item())

    model.to("cpu")

    if total_examples == 0:
        metrics = {"train_loss": 0.0, "train_accuracy": 0.0, "fednp_reg_loss": 0.0}
    else:
        metrics = {
            "train_loss": total_loss / total_examples,
            "train_accuracy": correct / total_examples,
            "fednp_reg_loss": total_reg_loss / total_examples,
        }
    return (
        metrics,
        latent_sum.detach().cpu().numpy().astype(np.float32),
        latent_square_sum.detach().cpu().numpy().astype(np.float32),
        np.asarray([latent_count], dtype=np.float32),
    )


def _gaussian_kl(
    local_mean: torch.Tensor,
    local_var: torch.Tensor,
    global_mean: torch.Tensor,
    global_var: torch.Tensor,
) -> torch.Tensor:
    log_ratio = torch.log(global_var) - torch.log(local_var)
    mean_diff = (local_mean - global_mean).pow(2)
    kl_per_dim = log_ratio + ((local_var + mean_diff) / global_var) - 1.0
    return 0.5 * kl_per_dim.sum()
