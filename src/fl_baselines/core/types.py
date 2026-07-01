"""Shared framework types."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


@dataclass(frozen=True)
class ClientDataLoaders:
    train: "DataLoader"
    test: "DataLoader"


MetricDict = dict[str, bool | bytes | float | int | str]
ServerEvaluateFn = Callable[[int, list[np.ndarray], MetricDict], tuple[float, MetricDict] | None]
