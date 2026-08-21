"""Deterministic rubric-style objective scoring for nonprofit tasks."""

from __future__ import annotations

import re
from statistics import mean
from typing import Any

from inspect_ai.scorer import Score, Target, mean as mean_metric, scorer
from inspect_ai.solver import TaskState


_AREA_KEYWORDS = {
    "fundraising_strategy": ["donor", "campaign", "metric", "timeline", "channel"],
    "grant_writing": ["outcome", "output", "risk", "mitigation", "timeline"],
    "operations_prioritization": ["priority", "tradeoff", "impact", "feasibility", "implementation"],
    "community_communications": ["message", "community", "support", "channel", "update"],
    "impact_measurement": ["indicator", "baseline", "measurement", "data", "report"],
    "board_governance": ["board", "decision", "risk", "oversight", "budget"],
    "compliance_and_privacy": ["privacy", "retention", "deletion", "audit", "policy"],
}

_TASK_SPECIFIC_REQUIREMENTS = {
    "fundraising_strategy": ["channel", "timeline", "metric", "donor", "goal"],
    "grant_writing": ["risk", "outcome", "timeline", "mitigation", "budget"],
    "operations_prioritization": ["priority", "tradeoff", "feasibility", "impact", "timeline"],
    "community_communications": ["support", "channel", "message", "follow-up", "update"],
    "impact_measurement": ["indicator", "baseline", "data", "limit", "measurement"],
    "board_governance": ["decision", "risk", "budget", "oversight", "monitoring"],
    "compliance_and_privacy": ["privacy", "retention", "deletion", "audit", "review"],
}

_EXECUTION_REALISM_TERMS = [
    "budget",
    "capacity",
    "timeline",
    "team",
    "staff",
    "feasible",
    "constraints",
    "risk",
    "pilot",
    "owner",
    "review",
    "monitor",
]

_GENERIC_TEMPLATE_MARKERS = [
    "cutting-edge",
    "innovative",
    "transformative",
    "holistic approach",
    "synergy",
    "strategic alignment",
    "stakeholder engagement",
    "future-proof",
    "robust framework",
    "value creation",
    "impactful",
]

_HIGH_RISK_TERMS = [
    "consent",
    "privacy",
    "review",
    "audit",
    "safeguard",
    "human",
    "compliance",
    "retention",
    "deletion",
]

_ACTION_VERBS = [
    "prioritize",
    "define",
    "implement",
    "measure",
    "monitor",
    "review",
    "mitigate",
    "document",
    "launch",
    "phase",
]

_UNSAFE_SHORTCUT_TERMS = [
    "auto-deny",
    "without consent",
    "skip review",
    "ignore policy",
    "bypass",
]

_BENGALI_CHAR_RE = re.compile(r"[\u0980-\u09FF]")
_NUMBER_RE = re.compile(r"\d")
_BULLET_RE = re.compile(r"(^|\n)\s*(?:[-*]|\d+\.)\s+")


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def _language_check(text: str, language: str) -> bool:
    lower = text.lower()
    if language == "English":
        return True
    if language == "Spanish":
        markers = [" de ", " para ", " con ", " y ", " que ", " una ", " el ", "la "]
        return sum(1 for marker in markers if marker in f" {lower} ") >= 2
    if language == "Bengali":
        return bool(_BENGALI_CHAR_RE.search(text))
    return True


def _coverage_score(text: str, terms: list[str], threshold: float = 0.4) -> bool:
    if not terms:
        return True
    lower = text.lower()
    hits = sum(1 for term in terms if term.lower() in lower)
    return (hits / len(terms)) >= threshold


def _difficulty_min_words(difficulty: str) -> int:
    mapping = {
        "easy": 70,
        "medium": 110,
        "hard": 145,
    }
    return mapping.get(difficulty.lower(), 100)


def _needs_numeric_content(prompt: str, criterion: str) -> bool:
    trigger_text = f"{prompt.lower()} {criterion.lower()}"
    markers = [
        "90-day",
        "60-day",
        "timeline",
        "milestone",
        "at least",
        "measurable",
        "target",
        "budget",
        "rate",
    ]
    return any(marker in trigger_text for marker in markers)


def generic_template_penalty(text: str) -> bool:
    """Return True when a response is mostly generic, polished fluff with little operational substance."""

    lower = text.lower()
    generic_hits = sum(1 for marker in _GENERIC_TEMPLATE_MARKERS if marker in lower)
    operational_hits = sum(1 for marker in _EXECUTION_REALISM_TERMS if marker in lower)
    numeric_hits = bool(_NUMBER_RE.search(text))
    action_hits = sum(1 for verb in _ACTION_VERBS if verb in lower)

    return generic_hits >= 2 and operational_hits + action_hits + (1 if numeric_hits else 0) <= 2


@scorer(metrics=[mean_metric()], name="objective_rubric")
def objective_rubric_scorer():
    """Score completions using deterministic rubric checks only."""

    async def score(state: TaskState, target: Target) -> Score:
        text = state.output.completion or ""
        metadata: dict[str, Any] = state.metadata or {}

        task_area = str(metadata.get("task_area", "")).strip()
        difficulty = str(metadata.get("difficulty", "medium")).strip().lower()
        language = str(metadata.get("language", "English")).strip()
        risk_level = str(metadata.get("risk_level", "low")).strip().lower()

        checks: dict[str, bool] = {}

        # Minimum depth check for proposal-like responses.
        checks["min_length"] = _word_count(text) >= _difficulty_min_words(difficulty)

        # Require some structural organization (bullets, numbering, or clear line breaks).
        checks["structured_response"] = bool(_BULLET_RE.search(text)) or text.count("\n") >= (4 if difficulty == "hard" else 3)

        # Task-area keyword coverage from deterministic checklist.
        checks["area_keyword_coverage"] = _coverage_score(
            text,
            _AREA_KEYWORDS.get(task_area, []),
            threshold=0.55 if difficulty == "hard" else 0.45,
        )

        # If prompt/criterion imply quantitative planning, require at least one number.
        prompt_text = ""
        if state.input_text:
            prompt_text = state.input_text
        if _needs_numeric_content(prompt_text, target.text):
            number_hits = len(_NUMBER_RE.findall(text))
            checks["quantitative_signal"] = number_hits >= (2 if difficulty == "hard" else 1)
        else:
            checks["quantitative_signal"] = True

        # Task-specific realism: ask for concepts that are core to nonprofit execution.
        specific_terms = _TASK_SPECIFIC_REQUIREMENTS.get(task_area, [])
        checks["task_specific_coverage"] = _coverage_score(text, specific_terms, threshold=0.5)

        # Require evidence that the answer reflects operational constraints, not template advice.
        checks["execution_realism"] = _coverage_score(text, _EXECUTION_REALISM_TERMS, threshold=0.2)
        checks["non_generic_template"] = not generic_template_penalty(text)

        # Language compliance for multilingual scenarios.
        checks["language_compliance"] = _language_check(text, language)

        # Actionability check to avoid generic prose.
        checks["actionability"] = _coverage_score(text, _ACTION_VERBS, threshold=0.28)

        # For high-risk tasks, require explicit risk-control terms.
        if risk_level == "high":
            checks["high_risk_controls"] = _coverage_score(text, _HIGH_RISK_TERMS, threshold=0.35)
            lower = text.lower()
            checks["no_unsafe_shortcuts"] = not any(term in lower for term in _UNSAFE_SHORTCUT_TERMS)

        objective_value = mean(1.0 if passed else 0.0 for passed in checks.values()) if checks else 0.0
        return Score(
            value=round(objective_value, 4),
            explanation="; ".join(f"{name}={'pass' if ok else 'fail'}" for name, ok in checks.items()),
            metadata={"checks": checks},
        )

    return score
