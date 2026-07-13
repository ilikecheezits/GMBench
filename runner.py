"""Execution runner for systems under test."""

from __future__ import annotations

import asyncio
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

    def __init__(self, max_concurrency: int = 4) -> None:
        self.max_concurrency = max(1, max_concurrency)

    async def _evaluate_example(self, system: SystemUnderTest, example: BenchmarkExample) -> ExampleRun:
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
        return ExampleRun(example=example, output=output, latency_ms=latency_ms, error=error, exception_stack=stack)

    async def evaluate(self, system: SystemUnderTest, dataset: BenchmarkDataset) -> List[ExampleRun]:
        semaphore = asyncio.Semaphore(self.max_concurrency)
        examples = list(dataset)

        async def _run(index: int, example: BenchmarkExample) -> tuple[int, ExampleRun]:
            async with semaphore:
                return index, await self._evaluate_example(system, example)

        ordered_runs = await asyncio.gather(*(_run(index, example) for index, example in enumerate(examples)))
        ordered_runs.sort(key=lambda item: item[0])
        return [run for _, run in ordered_runs]
