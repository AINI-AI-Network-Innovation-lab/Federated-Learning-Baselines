"""Algorithm extension interface."""

from __future__ import annotations

from typing import Protocol

import torch
from flwr.server.strategy import Strategy

from fl_baselines.core.config import ExperimentConfig
from fl_baselines.core.types import ServerEvaluateFn


class AlgorithmBuilder(Protocol):
    name: str

    def build_strategy(
        self,
        config: ExperimentConfig,
        initial_model: torch.nn.Module,
        evaluate_fn: ServerEvaluateFn | None,
    ) -> Strategy:
        """Build a Flower server strategy for this algorithm."""
