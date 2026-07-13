"""Top-level benchmark orchestration for a dataset and metric suite."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import metrics  # noqa: F401
from dataset import BenchmarkDataset
from evaluators import EvaluationContext, Metric, MetricEvaluator
from registry import METRIC_REGISTRY
from runner import ExampleRun, Runner
from workflow import SystemUnderTest


@dataclass(slots=True)
class BenchmarkResult:
    """Result payload for one system evaluated on one dataset."""

    workflow_name: str
    workflow_type: str
    dataset_name: str
    task_name: str
    metrics: Dict[str, float]
    metric_details: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    failure_cases: List[Dict[str, Any]] = field(default_factory=list)
    telemetry_summary: Dict[str, Any] = field(default_factory=dict)
    trace_path: str = ""
    failure_log_dir: str = ""


class Benchmark:
    """Runs a black-box system against a dataset using selected metrics."""

    def __init__(self, dataset: BenchmarkDataset, metrics: List[Metric], artifacts_dir: str = "artifacts") -> None:
        self.dataset = dataset
        self.metrics = list(metrics)
        self.runner = Runner()
        self.metric_evaluator = MetricEvaluator()
        self.artifacts_dir = Path(artifacts_dir)

        auto_metric_names = [
            "latency_ms",
            "cost_usd",
            "prompt_tokens",
            "completion_tokens",
            "total_api_calls",
            "failure_rate",
            "retry_count",
            "peak_latency_ms",
            "average_call_latency_ms",
            "cost_per_success",
            "token_efficiency",
        ]
        seen = {metric.name for metric in self.metrics}
        for metric_name in auto_metric_names:
            if metric_name in seen:
                continue
            metric_cls = METRIC_REGISTRY.get(metric_name)
            if metric_cls is not None:
                self.metrics.append(metric_cls())

    @staticmethod
    def _aggregate_telemetry(runs: List[ExampleRun]) -> Dict[str, Any]:
        call_count = 0
        prompt_tokens = 0
        completion_tokens = 0
        total_cost = 0.0
        total_latency = 0.0
        peak_latency = 0.0
        retry_count = 0
        failures = 0
        rate_limits = 0
        providers = set()
        models = set()
        temperatures = []
        context_lengths = []

        for run in runs:
            telem = run.output.telemetry
            call_count += telem.total_api_calls
            prompt_tokens += telem.prompt_tokens
            completion_tokens += telem.completion_tokens
            total_cost += telem.total_cost_usd
            total_latency += telem.total_latency_ms
            peak_latency = max(peak_latency, telem.peak_latency_ms)
            retry_count += telem.retry_count
            failures += telem.failures
            rate_limits += telem.rate_limits
            for call in telem.calls:
                providers.add(call.provider)
                models.add(call.model)
                temperatures.append(call.temperature)
                context_lengths.append(call.context_length)

        avg_latency = (total_latency / call_count) if call_count else 0.0
        avg_temp = (sum(temperatures) / len(temperatures)) if temperatures else 0.0
        avg_context_length = (sum(context_lengths) / len(context_lengths)) if context_lengths else 0.0

        return {
            "total_api_calls": call_count,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "total_cost_usd": round(total_cost, 6),
            "total_latency_ms": round(total_latency, 4),
            "peak_latency_ms": round(peak_latency, 4),
            "average_latency_ms": round(avg_latency, 4),
            "retry_count": retry_count,
            "provider": ",".join(sorted(providers)) if providers else "unknown",
            "model": ",".join(sorted(models)) if models else "unknown",
            "temperature": round(avg_temp, 4),
            "context_length": round(avg_context_length, 2),
            "failures": failures,
            "rate_limits": rate_limits,
        }

    def _write_failure_logs(self, system: SystemUnderTest, runs: List[ExampleRun]) -> tuple[str, List[Dict[str, Any]]]:
        log_dir = self.artifacts_dir / "failure_logs" / system.name.replace(" ", "_").lower()
        log_dir.mkdir(parents=True, exist_ok=True)

        failures: List[Dict[str, Any]] = []
        for run in runs:
            if not run.error:
                continue
            payload = {
                "example_id": run.example.id,
                "input": run.example.input_text,
                "expected_output": run.example.ground_truth,
                "actual_output": run.output.structured_output,
                "raw_model_responses": run.output.raw_model_responses,
                "prompts": run.output.prompts,
                "intermediate_outputs": run.output.intermediate_outputs,
                "exceptions": run.output.exceptions,
                "exception_stack": run.exception_stack,
                "timing": {
                    "example_latency_ms": run.latency_ms,
                    "total_call_latency_ms": run.output.telemetry.total_latency_ms,
                },
                "telemetry": self._aggregate_telemetry([run]),
            }
            out_path = log_dir / f"{run.example.id}.json"
            out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            failures.append({"example_id": run.example.id, "error": run.error, "log_path": str(out_path)})

        return str(log_dir), failures

    def _write_traces(self, system: SystemUnderTest, runs: List[ExampleRun]) -> str:
        trace_dir = self.artifacts_dir / "traces"
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_payload = {
            "system": system.name,
            "dataset": self.dataset.name,
            "examples": [
                {
                    "example_id": run.example.id,
                    "trace": [asdict(event) for event in run.output.traces],
                }
                for run in runs
            ],
        }
        trace_path = trace_dir / f"{system.name.replace(' ', '_').lower()}__{self.dataset.name}.json"
        trace_path.write_text(json.dumps(trace_payload, indent=2), encoding="utf-8")
        return str(trace_path)

    async def arun(self, system: SystemUnderTest) -> BenchmarkResult:
        runs: List[ExampleRun] = await self.runner.evaluate(system, self.dataset)
        context = EvaluationContext(system=system, dataset=self.dataset, runs=runs)
        metric_results = await self.metric_evaluator.evaluate(self.metrics, context)

        failure_log_dir, failures = self._write_failure_logs(system, runs)
        trace_path = self._write_traces(system, runs)
        telemetry_summary = self._aggregate_telemetry(runs)

        return BenchmarkResult(
            workflow_name=system.name,
            workflow_type=system.system_type,
            dataset_name=self.dataset.name,
            task_name=self.dataset.task_name,
            metrics={result.name: result.value for result in metric_results},
            metric_details={result.name: {"metric_type": result.metric_type, **result.details} for result in metric_results},
            failure_cases=failures,
            telemetry_summary=telemetry_summary,
            trace_path=trace_path,
            failure_log_dir=failure_log_dir,
        )

    def run(self, system: SystemUnderTest) -> BenchmarkResult:
        return asyncio.run(self.arun(system))
