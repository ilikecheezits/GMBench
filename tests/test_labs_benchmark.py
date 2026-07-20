from __future__ import annotations

from pathlib import Path
import unittest

import metrics  # noqa: F401
from benchmark import Benchmark
from leaderboard import Leaderboard
from registry import (
    DATASET_REGISTRY,
    SYSTEM_REGISTRY,
    build_metric_suite,
    discover_package_modules,
)


class LabsBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        discover_package_modules("datasets")
        discover_package_modules("workflows")

    def test_food_pantry_benchmark_runs(self) -> None:
        dataset = DATASET_REGISTRY["food_pantry_intake_v1"]()
        metrics = build_metric_suite(
            [
                "accuracy",
                "json_validity",
                "hallucination_rate",
                "latency_ms",
                "cost_usd",
                "prompt_tokens",
                "completion_tokens",
                "total_api_calls",
                "failure_rate",
            ]
        )
        benchmark = Benchmark(dataset=dataset, metrics=metrics)

        result = benchmark.run(SYSTEM_REGISTRY["food_pantry_intake_b"]())

        self.assertEqual(result.dataset_name, "food_pantry_intake_v1")
        self.assertIn("accuracy", result.metrics)
        self.assertIn("json_validity", result.metrics)
        self.assertIn("prompt_tokens", result.metrics)
        self.assertIn("total_api_calls", result.metrics)
        self.assertTrue(Path(result.trace_path).exists())

    def test_leaderboard_ranks_three_systems(self) -> None:
        dataset = DATASET_REGISTRY["food_pantry_intake_v1"]()
        metrics = build_metric_suite(["accuracy", "json_validity", "hallucination_rate", "cost_usd"])
        benchmark = Benchmark(dataset=dataset, metrics=metrics)
        leaderboard = Leaderboard(
            benchmark=benchmark,
            ranking_weights={"accuracy": 0.7, "json_validity": 0.2, "cost_usd": -0.1},
        )

        rankings = leaderboard.rank(
            [
                SYSTEM_REGISTRY["food_pantry_intake_a"](),
                SYSTEM_REGISTRY["food_pantry_intake_b"](),
                SYSTEM_REGISTRY["food_pantry_intake_c"](),
            ]
        )

        self.assertEqual(len(rankings), 3)
        self.assertEqual(rankings[0].rank, 1)
        self.assertGreaterEqual(rankings[0].score, rankings[-1].score)

    def test_nonprofit_tool_matching_benchmark_runs(self) -> None:
        dataset = DATASET_REGISTRY["nonprofit_tool_matching_v1"]()
        metrics = build_metric_suite(["accuracy", "json_validity", "cost_usd", "failure_rate"])
        benchmark = Benchmark(dataset=dataset, metrics=metrics)

        result = benchmark.run(SYSTEM_REGISTRY["nonprofit_tool_matcher_balanced"]())

        self.assertEqual(result.dataset_name, "nonprofit_tool_matching_v1")
        self.assertIn("accuracy", result.metrics)
        self.assertIn("json_validity", result.metrics)
        self.assertIn("failure_rate", result.metrics)
        self.assertTrue(Path(result.trace_path).exists())


if __name__ == "__main__":
    unittest.main()
