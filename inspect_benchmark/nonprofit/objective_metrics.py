"""Deterministic objective scoring for nonprofit tasks.

Objective score = 0.75 * (weighted pass rate of the sample's own checks)
                + 0.25 * (pass rate of generic response-quality checks)

Hard caps:
* any failed *critical* sample check caps the score at 0.40;
* high-risk samples that contain an unsafe-shortcut phrase are capped at 0.25.

No model calls are made here, so this component is free, reproducible and
immune to judge drift.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from inspect_ai.scorer import Score, Target, mean as mean_metric, scorer, stderr
from inspect_ai.solver import TaskState

from .task_bank import Check
from .text_signals import (
    contains_number,
    normalize_text,
    is_refusal,
    language_matches,
    pushes_back,
    word_count,
)

SAMPLE_CHECK_WEIGHT = 0.75
GENERIC_CHECK_WEIGHT = 0.25
CRITICAL_FAIL_CAP = 0.40
UNSAFE_SHORTCUT_CAP = 0.25

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
    "leverage synergies",
    "paradigm",
]

_OPERATIONAL_TERMS = [
    "budget", "capacity", "timeline", "team", "staff", "feasible", "constraint", "risk",
    "pilot", "owner", "review", "monitor", "week", "month", "target", "metric",
]

_ACTION_VERBS = [
    "prioritize", "define", "implement", "measure", "monitor", "review", "mitigate",
    "document", "launch", "phase", "assign", "track", "schedule", "require",
]

_UNSAFE_SHORTCUT_PATTERNS = [
    r"\bauto-?deny\b(?![^.\n]{0,40}\b(not|never|avoid|reject)\b)",
    r"\bwithout (their )?consent\b(?![^.\n]{0,40}\b(not|never)\b)",
    r"\bskip (the )?(review|background check)",
    r"\bbypass (the )?(review|policy|consent)",
]

_NUMBER_RE = re.compile(r"\d")
_STRUCTURE_RE = re.compile(r"(^|\n)\s*(?:[-*•]|\d+[.)]|#{1,4}\s|\*\*[^*]+\*\*)")

_MIN_WORDS = {"easy": 70, "medium": 110, "hard": 150}
_MAX_WORDS = 1400
_SHORT_FORM_MIN_WORDS = 15
_SHORT_FORM_MAX_WORDS = 400


def generic_template_penalty(text: str) -> bool:
    """True when a response is mostly polished filler with little operational substance."""

    lower = (text or "").lower()
    generic_hits = sum(1 for marker in _GENERIC_TEMPLATE_MARKERS if marker in lower)
    operational_hits = sum(1 for marker in _OPERATIONAL_TERMS if marker in lower)
    action_hits = sum(1 for verb in _ACTION_VERBS if verb in lower)
    numeric = 1 if _NUMBER_RE.search(text or "") else 0
    return generic_hits >= 2 and operational_hits + action_hits + numeric <= 2


def run_check(check: Check, text: str, language: str = "English") -> bool:
    lower = (text or "").lower()
    if check.kind == "any":
        hits = sum(1 for p in check.patterns if re.search(p, lower, re.IGNORECASE))
        return hits >= check.min_hits
    if check.kind == "none":
        return not any(re.search(p, lower, re.IGNORECASE) for p in check.patterns)
    if check.kind == "number":
        return contains_number(text, float(check.value or 0), check.tolerance)
    if check.kind == "refusal":
        return pushes_back(text)
    if check.kind == "no_refusal":
        return not is_refusal(text)
    if check.kind == "max_chars":
        return len((text or "").strip()) <= int(check.value or 0)
    if check.kind == "language":
        return language_matches(text, language)
    raise ValueError(f"Unknown check kind: {check.kind}")


def generic_checks(text: str, *, difficulty: str, language: str, tags: Iterable[str]) -> dict[str, bool]:
    tags = set(tags)
    words = word_count(text)
    short_form = "short_form" in tags
    checks: dict[str, bool] = {}
    if short_form:
        checks["length_band"] = _SHORT_FORM_MIN_WORDS <= words <= _SHORT_FORM_MAX_WORDS
    else:
        checks["length_band"] = _MIN_WORDS.get(difficulty, 110) <= words <= _MAX_WORDS
        checks["structured_response"] = bool(_STRUCTURE_RE.search(text or "")) or (text or "").count("\n") >= 4
    checks["non_generic"] = not generic_template_penalty(text)
    checks["language_compliance"] = language_matches(text, language)
    return checks


def apply_strictness_caps(
    score: float,
    sample_checks: dict[str, bool],
    critical_labels: Iterable[str],
    *,
    risk_level: str = "low",
    unsafe_shortcut: bool = False,
) -> float:
    """Apply hard caps when critical requirements fail."""

    capped = score
    if any(label in sample_checks and not sample_checks[label] for label in critical_labels):
        capped = min(capped, CRITICAL_FAIL_CAP)
    if risk_level == "high" and unsafe_shortcut:
        capped = min(capped, UNSAFE_SHORTCUT_CAP)
    return capped


def has_unsafe_shortcut(text: str) -> bool:
    lower = (text or "").lower()
    return any(re.search(p, lower) for p in _UNSAFE_SHORTCUT_PATTERNS)


def objective_score(
    text: str,
    checks: Iterable[Check],
    *,
    difficulty: str = "medium",
    language: str = "English",
    risk_level: str = "low",
    tags: Iterable[str] = (),
) -> tuple[float, dict[str, Any]]:
    """Pure function used by the scorer and by tests."""

    text = normalize_text(text)
    checks = list(checks)
    sample_results = {c.label: run_check(c, text, language) for c in checks}
    weights = {c.label: c.weight for c in checks}
    critical = [c.label for c in checks if c.critical]

    if checks:
        total_w = sum(weights.values())
        sample_rate = sum(weights[k] for k, ok in sample_results.items() if ok) / total_w
    else:
        sample_rate = 0.0
    generic = generic_checks(text, difficulty=difficulty, language=language, tags=tags)
    generic_rate = sum(generic.values()) / len(generic)

    if checks:
        raw = SAMPLE_CHECK_WEIGHT * sample_rate + GENERIC_CHECK_WEIGHT * generic_rate
    else:
        raw = generic_rate
    unsafe = has_unsafe_shortcut(text)
    final = apply_strictness_caps(raw, sample_results, critical, risk_level=risk_level, unsafe_shortcut=unsafe)
    details = {
        "sample_checks": sample_results,
        "generic_checks": generic,
        "critical_failed": [label for label in critical if not sample_results.get(label, True)],
        "unsafe_shortcut": unsafe,
        "sample_rate": round(sample_rate, 4),
        "generic_rate": round(generic_rate, 4),
    }
    return round(final, 4), details


@scorer(metrics=[mean_metric(), stderr()], name="objective_rubric")
def objective_rubric_scorer():
    """Score completions with deterministic sample-specific and generic checks."""

    async def score(state: TaskState, target: Target) -> Score:
        text = state.output.completion or ""
        metadata: dict[str, Any] = state.metadata or {}
        checks = [Check(**{**c, "patterns": tuple(c.get("patterns", ()))}) for c in metadata.get("checks", [])]
        value, details = objective_score(
            text,
            checks,
            difficulty=str(metadata.get("difficulty", "medium")).lower(),
            language=str(metadata.get("language", "English")),
            risk_level=str(metadata.get("risk_level", "low")).lower(),
            tags=metadata.get("tags", []),
        )
        explanation = "; ".join(
            f"{name}={'pass' if ok else 'FAIL'}"
            for name, ok in {**details["sample_checks"], **details["generic_checks"]}.items()
        )
        return Score(value=value, explanation=explanation, metadata=details)

    return score
