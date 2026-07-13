"""Execution runner for systems under test."""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass
from typing import List, Optional

from dataset import BenchmarkDataset, BenchmarkExample
from workflow import SystemOutput, SystemTelemetry, SystemUnderTest


@dataclass(slots=True)
class ExampleRun:
    """Runtime output and metadata for one evaluated example."""

    example: BenchmarkExample
    output: SystemOutput
    latency_ms: float
    error: Optional[str] = None
    exception_stack: Optional[str] = None


class Runner:
    """Runs a system over all examples in a dataset."""

    async def evaluate(self, system: SystemUnderTest, dataset: BenchmarkDataset) -> List[ExampleRun]:
        runs: List[ExampleRun] = []
        for example in dataset:
            start = time.perf_counter()
            error = None
            stack = None
            try:
                output = await system.run(example)
            except Exception as exc:  # pragma: no cover - error path still captured in result
                output = SystemOutput(structured_output=None, text_output=None, raw_output=None, telemetry=SystemTelemetry())
                error = str(exc)
                stack = traceback.format_exc()
                output.exceptions.append(error)
            latency_ms = (time.perf_counter() - start) * 1000.0
            runs.append(ExampleRun(example=example, output=output, latency_ms=latency_ms, error=error, exception_stack=stack))
        return runs
