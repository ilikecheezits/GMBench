"""Report generation helpers for benchmark outputs."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import List

from benchmark import BenchmarkResult
from leaderboard import RankedResult


def result_to_json(result: BenchmarkResult, indent: int = 2) -> str:
    return json.dumps(asdict(result), indent=indent)


def leaderboard_to_json(rankings: List[RankedResult], indent: int = 2) -> str:
    payload = [
        {
            "rank": row.rank,
            "score": row.score,
            "workflow_name": row.result.workflow_name,
            "workflow_type": row.result.workflow_type,
            "dataset_name": row.result.dataset_name,
            "task_name": row.result.task_name,
            "metrics": row.result.metrics,
            "telemetry_summary": row.result.telemetry_summary,
            "trace_path": row.result.trace_path,
            "failure_log_dir": row.result.failure_log_dir,
        }
        for row in rankings
    ]
    return json.dumps(payload, indent=indent)


def leaderboard_to_markdown(rankings: List[RankedResult]) -> str:
    lines = [
        "# Benchmark Leaderboard",
        "",
        "| Rank | System | Score | Accuracy | Judge | Hallucination | Schema Validity | API Calls | Prompt Tok | Completion Tok | Failure Rate | Latency (ms) | Cost (USD) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rankings:
        metrics = row.result.metrics
        lines.append(
            "| "
            f"{row.rank} | "
            f"{row.result.workflow_name} | "
            f"{row.score:.3f} | "
            f"{metrics.get('accuracy', 0.0):.3f} | "
            f"{metrics.get('llm_judge_quality', 0.0):.3f} | "
            f"{metrics.get('hallucination_rate', 0.0):.3f} | "
            f"{metrics.get('json_validity', 0.0):.3f} | "
            f"{metrics.get('total_api_calls', 0.0):.2f} | "
            f"{metrics.get('prompt_tokens', 0.0):.1f} | "
            f"{metrics.get('completion_tokens', 0.0):.1f} | "
            f"{metrics.get('failure_rate', 0.0):.3f} | "
            f"{metrics.get('latency_ms', 0.0):.2f} | "
            f"{metrics.get('cost_usd', 0.0):.5f} |"
        )
    return "\n".join(lines)
