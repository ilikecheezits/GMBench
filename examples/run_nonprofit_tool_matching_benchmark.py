"""Example script for running the Nonprofit Tool Matching benchmark."""

from __future__ import annotations

import metrics  # noqa: F401
from benchmark import Benchmark
from leaderboard import Leaderboard
from registry import DATASET_REGISTRY, SYSTEM_REGISTRY, build_metric_suite, discover_package_modules
from reports import leaderboard_to_markdown


def run() -> None:
    discover_package_modules("datasets")
    discover_package_modules("workflows")

    dataset = DATASET_REGISTRY["nonprofit_tool_matching_v1"]()
    systems = [
        SYSTEM_REGISTRY["nonprofit_tool_matcher_balanced"](),
        SYSTEM_REGISTRY["nonprofit_tool_matcher_budget"](),
        SYSTEM_REGISTRY["nonprofit_tool_matcher_expansive"](),
    ]

    metric_names = [
        "accuracy",
        "precision",
        "recall",
        "json_validity",
        "hallucination_rate",
        "latency_ms",
        "cost_usd",
        "failure_rate",
        "cross_org",
        "cross_language",
        "robustness",
        "prompt_injection",
        "pii_leakage",
        "llm_judge_quality",
    ]
    metric_suite = build_metric_suite(metric_names)

    benchmark = Benchmark(dataset=dataset, metrics=metric_suite)
    leaderboard = Leaderboard(
        benchmark=benchmark,
        ranking_weights={
            "accuracy": 0.15,
            "json_validity": 0.15,
            "hallucination_rate": -0.25,
            "cross_org": 0.1,
            "cross_language": 0.25,
            "robustness": 0.2,
            "prompt_injection": 0.2,
            "pii_leakage": -0.2,
            "llm_judge_quality": 0.1,
            "cost_usd": -0.05,
            "failure_rate": -0.1,
        },
    )

    ranking = leaderboard.rank(systems)
    print(leaderboard_to_markdown(ranking))


if __name__ == "__main__":
    run()
