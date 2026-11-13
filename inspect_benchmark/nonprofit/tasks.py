"""Inspect task definitions for the nonprofit core suite and org-specific custom tasks."""

from __future__ import annotations

from dataclasses import asdict
from typing import Iterable

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate, system_message

from .judge import rubric_judge_scorer
from .objective_metrics import objective_rubric_scorer
from .task_bank import AREA_ALIASES, NonprofitSampleSpec, all_samples, filter_samples

SYSTEM_PROMPT = """
You are an expert nonprofit strategy advisor.
Give practical, concrete recommendations suitable for real nonprofit teams.
Prioritize safety, dignity, and realistic execution over generic advice.
""".strip()

# Generous cap: reasoning models spend most tokens thinking; non-reasoning models stop far below it.
DEFAULT_MAX_TOKENS = 16000


def to_sample(spec: NonprofitSampleSpec) -> Sample:
    return Sample(
        id=spec.sample_id,
        input=spec.prompt,
        target=spec.criterion,
        metadata={
            "task_area": spec.task_area,
            "difficulty": spec.difficulty,
            "language": spec.language,
            "risk_level": spec.risk_level,
            "reference_answer": spec.reference_answer,
            "tags": list(spec.tags),
            "source": spec.source,
            "rubric": [asdict(item) for item in spec.rubric],
            "checks": [{**asdict(c), "patterns": list(c.patterns)} for c in spec.checks],
        },
    )


def build_task(
    samples: Iterable[NonprofitSampleSpec],
    *,
    name: str,
    suite: str,
    limit: int | None = None,
    judge_model: str | None = None,
) -> Task:
    mapped = [to_sample(s) for s in samples]
    if limit and limit > 0:
        mapped = mapped[:limit]
    if not mapped:
        raise ValueError(f"Task {name} has no samples after filtering.")
    return Task(
        dataset=MemoryDataset(mapped, name=name),
        solver=[system_message(SYSTEM_PROMPT), generate()],
        scorer=[rubric_judge_scorer(judge_model=judge_model), objective_rubric_scorer()],
        config=GenerateConfig(max_tokens=DEFAULT_MAX_TOKENS),
        metadata={"suite": suite},
        name=name,
    )


def _split(value: str | list[str] | None) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return list(value)


@task
def nonprofit_core(
    areas: str | list[str] | None = None,
    languages: str | list[str] | None = None,
    tags: str | list[str] | None = None,
    limit: int | None = None,
    judge_model: str | None = None,
) -> Task:
    """Full nonprofit core suite (optionally filtered by area, language or tag)."""

    areas_list = _split(areas)
    if areas_list and "all" in areas_list:
        areas_list = None
    samples = filter_samples(all_samples(), areas=areas_list, languages=_split(languages), tags=_split(tags))
    return build_task(samples, name="nonprofit_core", suite="core", limit=limit, judge_model=judge_model)


def custom_task(samples: list[NonprofitSampleSpec], judge_model: str | None = None) -> Task:
    return build_task(samples, name="nonprofit_custom", suite="custom", judge_model=judge_model)


# Backwards-compatible per-area tasks (e.g. `inspect eval inspect_benchmark/nonprofit/tasks.py@nonprofit_grants`).
def _area_task(alias: str):
    def _fn(limit: int | None = None) -> Task:
        return nonprofit_core(areas=[AREA_ALIASES[alias]], limit=limit)

    _fn.__name__ = f"nonprofit_{alias}"
    return task(_fn)


nonprofit_fundraising = _area_task("fundraising")
nonprofit_grants = _area_task("grants")
nonprofit_operations = _area_task("operations")
nonprofit_communications = _area_task("communications")
nonprofit_impact = _area_task("impact")
nonprofit_governance = _area_task("governance")
nonprofit_compliance = _area_task("compliance")
nonprofit_finance = _area_task("finance")


@task
def nonprofit_all(limit: int | None = None) -> Task:
    return nonprofit_core(limit=limit)
