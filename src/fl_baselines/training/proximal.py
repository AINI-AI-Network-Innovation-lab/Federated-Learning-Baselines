"""Proximal operators used by composite federated objectives."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

import numpy as np


class ProxOperator(Protocol):
    """Apply ``prox_{g/eta}`` to an ordered model-parameter payload."""

    def __call__(
        self,
        parameters: Sequence[np.ndarray],
        penalty: float,
    ) -> list[np.ndarray]:
        ...


class IdentityProx:
    """The identity operator, corresponding to ``g = 0``."""

    def __call__(self, parameters: Sequence[np.ndarray], penalty: float) -> list[np.ndarray]:
        del penalty
        return [np.array(parameter, copy=True) for parameter in parameters]


class L1Prox:
    """Element-wise soft-thresholding for ``g(x) = weight * ||x||_1``."""

    def __init__(self, weight: float = 1.0) -> None:
        if weight < 0:
            raise ValueError("fedadmm-l1-weight must be non-negative")
        self.weight = float(weight)

    def __call__(self, parameters: Sequence[np.ndarray], penalty: float) -> list[np.ndarray]:
        if penalty <= 0:
            raise ValueError("FedADMM penalty must be positive")
        threshold = self.weight / penalty
        return [
            np.sign(parameter) * np.maximum(np.abs(parameter) - threshold, 0).astype(
                parameter.dtype,
                copy=False,
            )
            for parameter in parameters
        ]


class BoxProx:
    """Projection onto an element-wise closed box."""

    def __init__(self, lower: float, upper: float) -> None:
        if lower > upper:
            raise ValueError("fedadmm-box-min must not exceed fedadmm-box-max")
        self.lower = float(lower)
        self.upper = float(upper)

    def __call__(self, parameters: Sequence[np.ndarray], penalty: float) -> list[np.ndarray]:
        del penalty
        return [
            np.clip(parameter, self.lower, self.upper).astype(parameter.dtype, copy=False)
            for parameter in parameters
        ]


def build_prox_operator(
    name: str,
    *,
    l1_weight: float,
    box_min: float,
    box_max: float,
) -> ProxOperator:
    """Build a configured proximal operator by stable runtime name."""

    factories: dict[str, Callable[[], ProxOperator]] = {
        "identity": IdentityProx,
        "l1": lambda: L1Prox(l1_weight),
        "box": lambda: BoxProx(box_min, box_max),
    }
    try:
        return factories[name]()
    except KeyError as exc:
        available = ", ".join(sorted(factories))
        raise KeyError(
            f"Unknown FedADMM prox '{name}'. Available: {available}"
        ) from exc
