"""Programmatic runner for nonprofit inspect_ai eval sets."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List


def _resolve_tasks(task_names: Iterable[str], limit_per_task: int | None):
    try:
        from .tasks import (
            nonprofit_all,
            nonprofit_compliance,
            nonprofit_communications,
            nonprofit_fundraising,
            nonprofit_governance,
            nonprofit_grants,
            nonprofit_impact,
            nonprofit_operations,
        )
    except ImportError as exc:
        raise RuntimeError(
            "inspect_ai is not installed. Install it with: pip install inspect-ai pandas"
        ) from exc

    task_map = {
        "all": nonprofit_all,
        "fundraising": nonprofit_fundraising,
        "grants": nonprofit_grants,
        "operations": nonprofit_operations,
        "communications": nonprofit_communications,
        "impact": nonprofit_impact,
        "governance": nonprofit_governance,
        "compliance": nonprofit_compliance,
    }

    selected = []
    for name in task_names:
        key = name.strip().lower()
        if key not in task_map:
            valid = ", ".join(sorted(task_map.keys()))
            raise ValueError(f"Unknown task '{name}'. Valid tasks: {valid}")
        selected.append(task_map[key](limit=limit_per_task))
    return selected


def run_nonprofit_eval_set(
    models: List[str],
    task_names: Iterable[str],
    log_dir: str = "logs/nonprofit_llm_benchmark",
    judge_model: str | None = None,
    limit_per_task: int | None = None,
) -> Dict[str, object]:
    """Run all requested nonprofit tasks across all requested models."""

    try:
        from inspect_ai import eval_set
    except ImportError as exc:
        raise RuntimeError(
            "inspect_ai is not installed. Install it with: pip install inspect-ai pandas"
        ) from exc

    if not models:
        raise ValueError("At least one model is required.")

    tasks = _resolve_tasks(task_names=task_names, limit_per_task=limit_per_task)
    output_dir = Path(log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    kwargs = {
        "tasks": tasks,
        "model": models,
        "log_dir": str(output_dir),
        "display": "none",
    }
    if judge_model:
        kwargs["model_roles"] = {"grader": judge_model}

    success, logs = eval_set(**kwargs)
    return {
        "success": success,
        "logs": [str(log) for log in logs],
        "log_dir": str(output_dir),
        "models": models,
        "tasks": list(task_names),
        "judge_model": judge_model,
    }
