"""Core system-under-test abstraction for workflow benchmarking."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from dataset import BenchmarkExample


@dataclass(slots=True)
class TraceEvent:
    """One serialized execution step for debugging and observability."""

    step: str
    status: str
    started_at: str
    ended_at: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CallTelemetry:
    """Telemetry for a single provider call inside a system pipeline."""

    provider: str
    model: str
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    retry_count: int = 0
    temperature: float = 0.0
    context_length: int = 0
    failure: bool = False
    rate_limited: bool = False
    exception: Optional[str] = None


@dataclass(slots=True)
class SystemTelemetry:
    """Aggregated telemetry for one example run of a system under test."""

    calls: List[CallTelemetry] = field(default_factory=list)

    @property
    def total_api_calls(self) -> int:
        return len(self.calls)

    @property
    def prompt_tokens(self) -> int:
        return sum(call.prompt_tokens for call in self.calls)

    @property
    def completion_tokens(self) -> int:
        return sum(call.completion_tokens for call in self.calls)

    @property
    def total_cost_usd(self) -> float:
        return sum(call.cost_usd for call in self.calls)

    @property
    def total_latency_ms(self) -> float:
        return sum(call.latency_ms for call in self.calls)

    @property
    def peak_latency_ms(self) -> float:
        if not self.calls:
            return 0.0
        return max(call.latency_ms for call in self.calls)

    @property
    def average_latency_ms(self) -> float:
        if not self.calls:
            return 0.0
        return self.total_latency_ms / len(self.calls)

    @property
    def retry_count(self) -> int:
        return sum(call.retry_count for call in self.calls)

    @property
    def failures(self) -> int:
        return sum(1 for call in self.calls if call.failure)

    @property
    def rate_limits(self) -> int:
        return sum(1 for call in self.calls if call.rate_limited)


@dataclass(slots=True)
class SystemOutput:
    """Output returned by a system under test for one example."""

    structured_output: Optional[Dict[str, Any]] = None
    text_output: Optional[str] = None
    raw_output: Any = None
    telemetry: SystemTelemetry = field(default_factory=SystemTelemetry)
    traces: List[TraceEvent] = field(default_factory=list)
    prompts: List[Dict[str, Any]] = field(default_factory=list)
    raw_model_responses: List[Dict[str, Any]] = field(default_factory=list)
    intermediate_outputs: List[Dict[str, Any]] = field(default_factory=list)
    exceptions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SystemUnderTest(ABC):
    """Black-box interface benchmarked by the framework."""

    name: str
    description: str
    system_type: str

    @abstractmethod
    async def run(self, example: BenchmarkExample) -> SystemOutput:
        """Run one benchmark example and return a normalized system output."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
