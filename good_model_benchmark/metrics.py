"""Metric scoring primitives for benchmark pillars."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from .utils import clamp, weighted_average


@dataclass(frozen=True, slots=True)
class MetricSpec:
    """Configuration for one metric in a pillar evaluator."""

    key: str
    label: str
    weight: float
    min_value: float
    max_value: float
    higher_is_better: bool = True
    required: bool = False


@dataclass(slots=True)
class PillarEvaluation:
    """Result of evaluating one pillar."""

    pillar_name: str
    score: float
    metric_scores: Dict[str, float] = field(default_factory=dict)
    missing_required_metrics: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True when all required metrics were present."""
        return not self.missing_required_metrics


def normalize_metric(value: float, spec: MetricSpec) -> float:
    """Normalize raw metric values into a 0..100 score.

    Uses linear normalization over the metric range and inverts when lower values
    are better.
    """
    if spec.max_value <= spec.min_value:
        raise ValueError(f"Invalid metric range for {spec.key}")

    ratio = (value - spec.min_value) / (spec.max_value - spec.min_value)
    ratio = clamp(ratio, 0.0, 1.0)
    if not spec.higher_is_better:
        ratio = 1.0 - ratio
    return ratio * 100.0


def evaluate_metrics(metric_obj: Any, specs: List[MetricSpec]) -> Tuple[float, Dict[str, float], List[str]]:
    """Score a dataclass-like metric object using metric specs."""
    weighted_scores: List[Tuple[float, float]] = []
    metric_scores: Dict[str, float] = {}
    missing_required: List[str] = []

    for spec in specs:
        raw = getattr(metric_obj, spec.key, None)
        if raw is None:
            if spec.required:
                missing_required.append(spec.label)
            continue

        score = normalize_metric(float(raw), spec)
        metric_scores[spec.label] = round(score, 2)
        weighted_scores.append((score, spec.weight))

    return round(weighted_average(weighted_scores), 2), metric_scores, missing_required
