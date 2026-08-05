"""Inspect task definitions for nonprofit LLM-vs-LLM benchmarking."""

from __future__ import annotations

from typing import Iterable, List

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import model_graded_qa
from inspect_ai.solver import generate, system_message

from .objective_metrics import objective_rubric_scorer
from .task_bank import NonprofitSampleSpec, all_samples, samples_for_areas

SYSTEM_PROMPT = """
You are an expert nonprofit strategy advisor.
Give practical, concrete recommendations suitable for real nonprofit teams.
Prioritize safety, dignity, and realistic execution over generic advice.
""".strip()

GRADER_TEMPLATE = """
You are grading an answer from an LLM benchmark about nonprofit strategy.

Task prompt:
{question}

Rubric criterion:
{criterion}

Reference answer (for calibration only, not strict wording):
{reference_answer}

Model answer:
{answer}

Instructions for grading:
{instructions}
""".strip()

GRADER_INSTRUCTIONS = """
You are a strict evaluator using professor-style grading.

Evaluate against the criterion item-by-item and penalize missing specifics.

Use these labels only:
- C: Fully satisfies all key criterion requirements with concrete, actionable detail.
- P: Mostly useful but missing one or more major criterion elements, or too vague on execution.
- I: Fails key requirements, is unsafe, off-task, or includes harmful/noncompliant guidance.

Strict rules:
- Do NOT award C if any core requirement in the criterion is missing.
- If measurable elements are requested (targets, timeline, indicators, controls), absence of them should usually be P or I.
- For high-risk scenarios (privacy, fairness, eligibility, legal/compliance), weak safeguards should be scored down.
- Prefer P over C when uncertain.

Write a brief rationale citing missing or satisfied criterion elements.
Always end with exactly one line: GRADE: <C|P|I>
""".strip()


def _to_sample(spec: NonprofitSampleSpec) -> Sample:
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
        },
    )


def _limit(samples: List[Sample], limit: int | None) -> List[Sample]:
    if limit is None or limit <= 0:
        return samples
    return samples[:limit]


def _build_task(samples: Iterable[NonprofitSampleSpec], limit: int | None = None) -> Task:
    mapped = [_to_sample(sample) for sample in samples]
    dataset = _limit(mapped, limit)
    return Task(
        dataset=dataset,
        solver=[system_message(SYSTEM_PROMPT), generate()],
        scorer=[
            model_graded_qa(template=GRADER_TEMPLATE, instructions=GRADER_INSTRUCTIONS, partial_credit=True),
            objective_rubric_scorer(),
        ],
    )


@task
def nonprofit_all(limit: int | None = None) -> Task:
    """Full nonprofit benchmark task across all strategy domains."""

    return _build_task(all_samples(), limit=limit)


@task
def nonprofit_fundraising(limit: int | None = None) -> Task:
    return _build_task(samples_for_areas(["fundraising_strategy"]), limit=limit)


@task
def nonprofit_grants(limit: int | None = None) -> Task:
    return _build_task(samples_for_areas(["grant_writing"]), limit=limit)


@task
def nonprofit_operations(limit: int | None = None) -> Task:
    return _build_task(samples_for_areas(["operations_prioritization"]), limit=limit)


@task
def nonprofit_communications(limit: int | None = None) -> Task:
    return _build_task(samples_for_areas(["community_communications"]), limit=limit)


@task
def nonprofit_impact(limit: int | None = None) -> Task:
    return _build_task(samples_for_areas(["impact_measurement"]), limit=limit)


@task
def nonprofit_governance(limit: int | None = None) -> Task:
    return _build_task(samples_for_areas(["board_governance"]), limit=limit)


@task
def nonprofit_compliance(limit: int | None = None) -> Task:
    return _build_task(samples_for_areas(["compliance_and_privacy"]), limit=limit)
