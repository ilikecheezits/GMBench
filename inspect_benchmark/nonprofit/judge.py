"""Rubric-item judge scorer (HealthBench-style).

Instead of asking a judge for one holistic letter grade (which saturated: every
GPT-5-family model scored ~0.98 in earlier runs), the judge marks each
sample-specific rubric item as met / not met in a single call. Items carry
points; negative items penalise harmful or incorrect content.

    judge_score = clip( sum(points of met items) / sum(positive points), 0, 1 )

Only one judge call is made per sample, so cost scales with sample count, not
rubric length.
"""

from __future__ import annotations

import json
import re
from typing import Any

from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.scorer import Score, Target, mean as mean_metric, scorer, stderr
from inspect_ai.solver import TaskState

DEFAULT_JUDGE_MODEL = "openai/gpt-4.1-mini"

JUDGE_TEMPLATE = """You are an expert evaluator of AI assistance for nonprofit organisations (fundraising, grants, operations, \
communications, impact measurement, governance, compliance, finance).

You will see a task given to an AI assistant, a reference answer for calibration, and the assistant's response. \
Decide for EACH rubric item whether the response meets it.

Rules:
- Judge only what is actually in the response. Do not give credit for things merely implied or promised.
- A positive item is met only if the response clearly and correctly satisfies it. Partial or vague coverage = not met.
- A negative item (negative points) describes an undesirable behaviour. Mark it met=true if the response EXHIBITS that behaviour.
- Factual or arithmetic errors mean accuracy items are not met, even if the response is confident and well formatted.
- Length and polish are not evidence of quality. Do not reward verbosity.
- The reference answer is one acceptable answer, not the only one.

TASK:
<<<
{prompt}
>>>

REFERENCE ANSWER (calibration only):
<<<
{reference}
>>>

RESPONSE TO EVALUATE:
<<<
{response}
>>>

RUBRIC ITEMS:
{items}

Return ONLY a JSON object, no prose before or after, in this format:
{{"items": [{{"id": 1, "met": true, "reason": "<=20 words"}}, ...]}}
Include every item id exactly once."""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _is_reasoning_model(name: str) -> bool:
    short = name.split("/")[-1].lower()
    return short.startswith(("o1", "o3", "o4", "gpt-5"))


def judge_config(model_name: str) -> GenerateConfig:
    if _is_reasoning_model(model_name):
        return GenerateConfig(max_tokens=6000, reasoning_effort="low")
    return GenerateConfig(temperature=0.0, max_tokens=1500)


def format_items(rubric: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"{i}. [{item['points']:+d} pts, {item.get('tag', 'completeness')}] {item['criterion']}"
        for i, item in enumerate(rubric, start=1)
    )


def parse_judgement(text: str, n_items: int) -> dict[int, dict[str, Any]] | None:
    """Parse judge JSON; returns {item_id: {"met": bool, "reason": str}} or None."""

    match = _JSON_RE.search(text or "")
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return None
    result: dict[int, dict[str, Any]] = {}
    for entry in items:
        try:
            idx = int(entry["id"])
        except (KeyError, TypeError, ValueError):
            continue
        met = entry.get("met")
        if isinstance(met, str):
            met = met.strip().lower() in {"true", "yes", "1"}
        if 1 <= idx <= n_items:
            result[idx] = {"met": bool(met), "reason": str(entry.get("reason", ""))[:300]}
    if len(result) != n_items:
        return None
    return result


def rubric_score(rubric: list[dict[str, Any]], judgement: dict[int, dict[str, Any]]) -> float:
    positive = sum(item["points"] for item in rubric if item["points"] > 0)
    if positive <= 0:
        return 0.0
    earned = sum(item["points"] for i, item in enumerate(rubric, start=1) if judgement[i]["met"])
    return max(0.0, min(1.0, earned / positive))


def tag_breakdown(rubric: list[dict[str, Any]], judgement: dict[int, dict[str, Any]]) -> dict[str, float]:
    """Per-tag earned/possible ratios (e.g. accuracy, safety) for diagnostics."""

    possible: dict[str, float] = {}
    earned: dict[str, float] = {}
    for i, item in enumerate(rubric, start=1):
        tag = item.get("tag", "completeness")
        if item["points"] > 0:
            possible[tag] = possible.get(tag, 0) + item["points"]
        if judgement[i]["met"]:
            earned[tag] = earned.get(tag, 0) + item["points"]
    return {tag: round(max(0.0, earned.get(tag, 0) / total), 4) for tag, total in possible.items() if total > 0}


@scorer(metrics=[mean_metric(), stderr()], name="rubric_judge")
def rubric_judge_scorer(judge_model: str | None = None, max_attempts: int = 2):
    """Judge each sample-specific rubric item with a single grader call."""

    async def score(state: TaskState, target: Target) -> Score:
        metadata = state.metadata or {}
        rubric: list[dict[str, Any]] = metadata.get("rubric", [])
        if not rubric:
            return Score(value=0.0, explanation="no rubric items", metadata={"error": "no_rubric"})

        grader = get_model(judge_model) if judge_model else get_model(role="grader", default=DEFAULT_JUDGE_MODEL)
        prompt = JUDGE_TEMPLATE.format(
            prompt=state.input_text,
            reference=metadata.get("reference_answer", ""),
            response=state.output.completion or "(empty response)",
            items=format_items(rubric),
        )
        config = judge_config(str(grader))

        judgement = None
        raw = ""
        for _ in range(max_attempts):
            result = await grader.generate(prompt, config=config)
            raw = result.completion
            judgement = parse_judgement(raw, len(rubric))
            if judgement is not None:
                break

        if judgement is None:
            return Score(
                value=0.0,
                explanation=f"judge output unparseable: {raw[:300]}",
                metadata={"error": "judge_parse_failure", "judge_model": str(grader)},
            )

        value = rubric_score(rubric, judgement)
        explanation = "\n".join(
            f"[{'x' if judgement[i]['met'] else ' '}] ({item['points']:+d}) {item['criterion']} -- {judgement[i]['reason']}"
            for i, item in enumerate(rubric, start=1)
        )
        return Score(
            value=round(value, 4),
            explanation=explanation,
            metadata={
                "items_met": [judgement[i]["met"] for i in range(1, len(rubric) + 1)],
                "tag_scores": tag_breakdown(rubric, judgement),
                "judge_model": str(grader),
            },
        )

    return score
