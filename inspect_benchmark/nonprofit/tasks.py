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
You are a strict evaluator using professor-style grading for nonprofit strategy tasks.

Evaluate the answer against the actual task criterion, not against general quality. A strong answer must address the specific nonprofit problem, required actions, and risk considerations in the prompt.

Use these labels only:
- C: Fully satisfies all core criterion requirements with concrete, realistic, and actionable details. The answer directly addresses the task, includes required execution details, and includes the right safeguards or implementation specifics when the task calls for them.
- P: Mostly relevant and useful, but misses one or more important criterion elements, is too generic, or lacks concrete implementation details needed for real nonprofit execution.
- I: Fails the task, omits critical requirements, is generic filler, unsafe, off-task, or includes harmful/noncompliant guidance.

Strict rules:
- Do NOT award C if any core requirement in the criterion is missing.
- Do NOT award C unless the answer includes concrete execution detail (timeline, ownership, measurable indicators, and practical constraints) when those are relevant to the prompt.
- Default to P unless there is clear evidence the answer is fully task-complete.
- To award C, require at least 3 concrete implementation details (for example: timeline, numeric target/KPI, named owner/role, decision trigger, review cadence, or explicit safeguard).
- A generic answer like 'do outreach, improve messaging, train staff, build partnerships' is not C unless it is tied to the task, concrete actions, responsibilities, timelines, indicators, or safeguards required by the prompt.
- If the task asks for measurable elements (targets, timeline, KPIs, controls, stakeholder roles, triggers, review cadence), absence of these should usually cap the score at P or I.
- For privacy, fairness, eligibility, legal/compliance, or donor-facing risk scenarios, weak or missing safeguards should substantially reduce the score.
- If the answer is fluent but lacks enough specific nonprofit execution details to be realistically actionable, score P rather than C.
- Prefer P over C when the answer is directionally good but not fully task-complete.
- Prefer I over P when the answer is largely generic, misses required deliverables, or overlooks critical risk requirements.
- For high-risk tasks, if any critical safeguard is missing (for example consent/privacy controls, human review, escalation path, or compliance control), do not award C.
- If an answer is off-task or mostly generic prose without operational decisions, score I.

Write a brief rationale citing which task requirements were met versus missed.
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
