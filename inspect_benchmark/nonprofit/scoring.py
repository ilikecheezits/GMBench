"""Composable scoring for the nonprofit benchmark-of-benchmarks.

Three levels:

1. Sample score (core and custom tasks):
       sample = 0.60 * rubric_judge + 0.40 * objective
2. Core score: area-weighted mean of per-area mean sample scores (each area counts
   equally by default; an org profile can re-weight or drop areas).
3. Composite score: fixed weighted mean of components
       core 0.60 | knowledge 0.10 | truthfulness 0.10 | fairness 0.10 | calibrated_refusal 0.10
   A profile with custom tasks adds ``custom`` (0.25, taken from core) and applies
   small transparent adjustments (see ``OrgProfile.component_weights``).

Missing components are dropped and the remaining weights renormalised; the
leaderboard flags such rows as incomplete.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

SAMPLE_WEIGHTS: dict[str, float] = {"judge": 0.60, "objective": 0.40}

DEFAULT_COMPONENT_WEIGHTS: dict[str, float] = {
    "core": 0.60,
    "knowledge": 0.10,
    "truthfulness": 0.10,
    "fairness": 0.10,
    "calibrated_refusal": 0.10,
}

COMPONENT_LABELS: dict[str, str] = {
    "core": "Nonprofit core tasks",
    "custom": "Org-specific tasks",
    "knowledge": "Domain knowledge (MMLU)",
    "truthfulness": "Truthfulness (TruthfulQA)",
    "fairness": "Bias/fairness (BBQ)",
    "calibrated_refusal": "Calibrated refusal (XSTest)",
}


def normalize_weights(weights: Mapping[str, float]) -> dict[str, float]:
    clean = {k: max(0.0, float(v)) for k, v in weights.items()}
    total = sum(clean.values())
    if total <= 0:
        raise ValueError("Weights must include at least one positive value.")
    return {k: round(v / total, 4) for k, v in clean.items()}


@dataclass(frozen=True)
class ScoreComponent:
    """A single signal contributing to a weighted score."""

    name: str
    weight: float
    normalizer: Callable[[float], float] | None = None
    minimum: float = 0.0
    maximum: float = 1.0

    def normalize(self, value: float) -> float:
        numeric = float(value)
        if self.normalizer is not None:
            numeric = float(self.normalizer(numeric))
        return max(self.minimum, min(self.maximum, numeric))


@dataclass(frozen=True)
class BenchmarkScoringConfig:
    """Weighted composition over named components; missing components are renormalised away."""

    components: tuple[ScoreComponent, ...] = field(
        default_factory=lambda: tuple(ScoreComponent(k, v) for k, v in SAMPLE_WEIGHTS.items())
    )

    @classmethod
    def from_weights(cls, weights: Mapping[str, float]) -> "BenchmarkScoringConfig":
        return cls(components=tuple(ScoreComponent(name, w) for name, w in weights.items()))

    def weights(self) -> dict[str, float]:
        return {c.name: c.weight for c in self.components}

    def score(self, values: Mapping[str, float | None]) -> float:
        present = [c for c in self.components if values.get(c.name) is not None and c.weight > 0]
        total_weight = sum(c.weight for c in present)
        if total_weight <= 0:
            return 0.0
        return sum(c.weight * c.normalize(values[c.name]) for c in present) / total_weight

    def missing(self, values: Mapping[str, float | None]) -> list[str]:
        return [c.name for c in self.components if c.weight > 0 and values.get(c.name) is None]


def default_scoring_config() -> BenchmarkScoringConfig:
    """Per-sample judge/objective blend used for core and custom tasks."""

    return BenchmarkScoringConfig.from_weights(SAMPLE_WEIGHTS)


def composite_scoring_config(weights: Mapping[str, float] | None = None) -> BenchmarkScoringConfig:
    return BenchmarkScoringConfig.from_weights(weights or DEFAULT_COMPONENT_WEIGHTS)


def build_composite_score(
    values: Mapping[str, float],
    config: BenchmarkScoringConfig | None = None,
) -> float:
    return (config or default_scoring_config()).score(values)
