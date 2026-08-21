"""Composable scoring config for nonprofit benchmark evaluation.

This keeps the current objective + judge hybrid model in place while making it
simple to add more benchmark components later (for example external evaluator
signals, domain rollups, or organization-specific weighting).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping


@dataclass(frozen=True)
class ScoreComponent:
    """A single signal contributing to the final benchmark score."""

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
    """A weighted score composition that can grow over time.

    The default configuration models the current nonprofit benchmark as a hybrid of
    objective rubric performance and judge-based rubric quality, but the same
    structure can be extended with future external benchmarks or specialized
    signals without reworking leaderboard logic.
    """

    components: tuple[ScoreComponent, ...] = field(
        default=(
            ScoreComponent(name="objective", weight=0.65),
            ScoreComponent(name="judge", weight=0.35),
        )
    )

    def score(self, values: Mapping[str, float]) -> float:
        total_weight = sum(component.weight for component in self.components)
        if total_weight <= 0:
            raise ValueError("Scoring config must include positive component weights.")

        weighted = 0.0
        for component in self.components:
            if component.name not in values:
                continue
            weighted += component.weight * component.normalize(values[component.name])

        if not weighted:
            return 0.0

        return weighted / total_weight


def default_scoring_config() -> BenchmarkScoringConfig:
    """Default nonprofit benchmark weighting used across the repo."""

    return BenchmarkScoringConfig(
        components=(
            ScoreComponent(name="objective", weight=0.65),
            ScoreComponent(name="judge", weight=0.35),
        )
    )


def build_composite_score(
    values: Mapping[str, float],
    config: BenchmarkScoringConfig | None = None,
) -> float:
    """Compute a weighted composite score for a set of component values."""

    return (config or default_scoring_config()).score(values)
