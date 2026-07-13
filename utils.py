"""Small utility helpers shared by benchmark modules."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, Sequence, Tuple


def clamp(value: float, low: float, high: float) -> float:
    """Clamp value into the inclusive [low, high] range."""
    return max(low, min(high, value))


def weighted_average(items: Sequence[Tuple[float, float]]) -> float:
    """Compute weighted mean from (value, weight) pairs.

    Returns 0.0 when there is no usable weight.
    """
    total_weight = sum(weight for _, weight in items if weight > 0)
    if total_weight <= 0:
        return 0.0
    weighted_sum = sum(value * weight for value, weight in items if weight > 0)
    return weighted_sum / total_weight


def mean(values: Iterable[float]) -> float:
    """Compute the arithmetic mean and default to 0.0 for empty iterables."""
    values_list = list(values)
    if not values_list:
        return 0.0
    return sum(values_list) / len(values_list)


def utc_timestamp() -> str:
    """Generate an ISO-8601 UTC timestamp for reports."""
    return datetime.now(timezone.utc).isoformat()
