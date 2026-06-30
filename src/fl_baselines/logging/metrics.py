"""Metric aggregation helpers for Flower strategies."""

from __future__ import annotations


Scalar = bool | bytes | float | int | str


def weighted_average(metrics: list[tuple[int, dict[str, Scalar]]]) -> dict[str, Scalar]:
    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0:
        return {}

    aggregated: dict[str, float] = {}
    for num_examples, metric_dict in metrics:
        for key, value in metric_dict.items():
            if isinstance(value, bool | bytes | str):
                continue
            aggregated[key] = aggregated.get(key, 0.0) + float(value) * num_examples

    return {key: value / total_examples for key, value in aggregated.items()}
