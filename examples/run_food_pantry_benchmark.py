"""Example script for running the Food Pantry Intake benchmark."""

from __future__ import annotations

import metrics  # noqa: F401
from benchmark import Benchmark
from leaderboard import Leaderboard
from registry import (
    DATASET_REGISTRY,
    SYSTEM_REGISTRY,
    build_metric_suite,
    discover_package_modules,
)
from reports import leaderboard_to_markdown


def run() -> None:
    discover_package_modules("datasets")
    discover_package_modules("workflows")

    dataset = DATASET_REGISTRY["food_pantry_intake_v1"]()
    systems = [
        SYSTEM_REGISTRY["food_pantry_intake_a"](),
        SYSTEM_REGISTRY["food_pantry_intake_b"](),
        SYSTEM_REGISTRY["food_pantry_intake_c"](),
    ]

    metric_names = [
        "accuracy",
        "precision",
        "recall",
        "json_validity",
        "hallucination_rate",
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
        "robustness",
        "safety",
        "pii_leakage",
        "prompt_injection",
        "cross_org",
        "cross_language",
        "llm_judge_quality",
    ]
    metrics = build_metric_suite(metric_names)

    benchmark = Benchmark(dataset=dataset, metrics=metrics)
    leaderboard = Leaderboard(
        benchmark=benchmark,
        ranking_weights={
            "accuracy": 0.35,
            "json_validity": 0.2,
            "robustness": 0.2,
            "prompt_injection": 0.1,
            "cross_org": 0.1,
            "llm_judge_quality": 0.1,
            "cost_usd": -0.05,
        },
    )

    ranking = leaderboard.rank(systems)
    print(leaderboard_to_markdown(ranking))


if __name__ == "__main__":
    run()
