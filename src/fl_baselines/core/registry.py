"""Small registry abstraction for extensible framework components."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Generic, TypeVar


T = TypeVar("T")


class Registry(Generic[T]):
    """Map stable component names to builder instances."""

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._items: dict[str, T] = {}

    def __contains__(self, name: str) -> bool:
        return name in self._items

    def register(self, name: str, component: T) -> None:
        if name in self._items:
            raise ValueError(f"{self._kind.title()} '{name}' is already registered")
        self._items[name] = component

    def get(self, name: str) -> T:
        try:
            return self._items[name]
        except KeyError as exc:
            available = ", ".join(self.names()) or "<none>"
            raise KeyError(
                f"Unknown {self._kind} '{name}'. Available: {available}"
            ) from exc

    def names(self) -> list[str]:
        return sorted(self._items)

    def values(self) -> Iterable[T]:
        return self._items.values()


DATASETS: Registry[object] = Registry("dataset")
MODELS: Registry[object] = Registry("model")
ALGORITHMS: Registry[object] = Registry("algorithm")
