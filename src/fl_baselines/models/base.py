"""Model extension interface."""

from __future__ import annotations

from typing import Protocol

import torch

from fl_baselines.core.config import ExperimentConfig


class ModelBuilder(Protocol):
    name: str

    def build_model(self, config: ExperimentConfig) -> torch.nn.Module:
        """Build a fresh model instance."""
