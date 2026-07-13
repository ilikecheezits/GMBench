"""Leaderboard ranking over multiple benchmark results."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Dict, List, Sequence

from benchmark import Benchmark, BenchmarkResult
from workflow import SystemUnderTest


@dataclass(slots=True)
class RankedResult:
    rank: int
    result: BenchmarkResult
    score: float


class Leaderboard:
    """Ranks systems based on caller-provided metric weighting."""

    def __init__(self, benchmark: Benchmark, ranking_weights: Dict[str, float], max_concurrency: int = 3) -> None:
        if not ranking_weights:
            raise ValueError("ranking_weights cannot be empty")
        self.benchmark = benchmark
        self.ranking_weights = ranking_weights
        self.max_concurrency = max(1, max_concurrency)

    def _score(self, result: BenchmarkResult) -> float:
        weighted_sum = 0.0
        total_weight = 0.0
        for metric_name, weight in self.ranking_weights.items():
            if metric_name not in result.metrics:
                continue
            weighted_sum += result.metrics[metric_name] * weight
            total_weight += weight
        if total_weight == 0:
            return 0.0
        return weighted_sum / total_weight

    async def arank(self, systems: Sequence[SystemUnderTest]) -> List[RankedResult]:
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def _score_system(system: SystemUnderTest) -> tuple[float, BenchmarkResult]:
            async with semaphore:
                result = await self.benchmark.arun(system)
                return self._score(result), result

        scored = list(await asyncio.gather(*(_score_system(system) for system in systems)))
        scored.sort(key=lambda item: item[0], reverse=True)

        ranked: List[RankedResult] = []
        for idx, (score, result) in enumerate(scored, start=1):
            ranked.append(RankedResult(rank=idx, result=result, score=round(score, 4)))
        return ranked

    def rank(self, systems: Sequence[SystemUnderTest]) -> List[RankedResult]:
        return asyncio.run(self.arank(systems))
