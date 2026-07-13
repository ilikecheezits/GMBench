"""Metric protocol and evaluation context."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List

from dataset import BenchmarkDataset
from runner import ExampleRun
from workflow import SystemUnderTest


@dataclass(slots=True)
class EvaluationContext:
    """Shared context passed to each metric."""

    system: SystemUnderTest
    dataset: BenchmarkDataset
    runs: List[ExampleRun]


@dataclass(slots=True)
class MetricResult:
    """Metric output value plus optional debug details."""

    name: str
    value: float
    metric_type: str = "rule_based"
    details: Dict[str, Any] = field(default_factory=dict)


class MetricType(str, Enum):
    RULE_BASED = "rule_based"
    REFERENCE_BASED = "reference_based"
    LLM_JUDGE = "llm_judge"


class Metric(ABC):
    """Interface for pluggable benchmark metrics."""

    name: str
    metric_type: MetricType = MetricType.RULE_BASED

    @abstractmethod
    async def compute(self, context: EvaluationContext) -> MetricResult:
        """Compute metric for a completed benchmark run."""


class RuleBasedMetric(Metric):
    metric_type = MetricType.RULE_BASED


class ReferenceBasedMetric(Metric):
    metric_type = MetricType.REFERENCE_BASED


class LLMJudgeMetric(Metric):
    metric_type = MetricType.LLM_JUDGE


class MetricEvaluator:
    """Applies metrics to a benchmark context."""

    async def evaluate(self, metrics: List[Metric], context: EvaluationContext) -> List[MetricResult]:
        return [await metric.compute(context) for metric in metrics]
