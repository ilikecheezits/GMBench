"""Programmatic runner for the nonprofit benchmark-of-benchmarks."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .judge import DEFAULT_JUDGE_MODEL

ALL_SUITES = ("core", "external", "custom")


@dataclass
class RunPlan:
    """Everything needed to build the Inspect tasks for a run (also used for cost estimates)."""

    models: list[str]
    suites: list[str] = field(default_factory=lambda: ["core", "external"])
    areas: list[str] | None = None
    languages: list[str] | None = None
    profile_path: str | None = None
    judge_model: str = DEFAULT_JUDGE_MODEL
    limit_per_task: int | None = None
    external_limit: int | None = None
    external_components: list[str] | None = None
    epochs: int = 1

    def profile(self):
        if not self.profile_path:
            return None
        from .profile import OrgProfile

        return OrgProfile.load(self.profile_path)

    def core_samples(self):
        from .task_bank import all_samples, filter_samples

        profile = self.profile()
        areas = self.areas
        if areas and "all" in areas:
            areas = None
        samples = filter_samples(all_samples(), areas=areas, languages=self.languages)
        if profile:
            selected = set(profile.selected_areas())
            samples = [s for s in samples if s.task_area in selected and profile.include_language(s.language)]
        if self.limit_per_task:
            samples = samples[: self.limit_per_task]
        return samples

    def custom_samples(self):
        profile = self.profile()
        samples = profile.custom_samples() if profile else []
        if self.limit_per_task:
            samples = samples[: self.limit_per_task]
        return samples

    def external_sizes(self) -> dict[str, int]:
        from .external import COMPONENTS, external_counts

        counts = external_counts()
        names = self.external_components or list(COMPONENTS)
        sizes = {}
        for name in names:
            n = counts.get(name, 0)
            if self.external_limit:
                n = min(n, self.external_limit)
            sizes[name] = n
        return sizes

    def build_tasks(self) -> list[Any]:
        from .external import EXTERNAL_TASKS
        from .tasks import build_task, custom_task

        tasks = []
        if "core" in self.suites:
            samples = self.core_samples()
            if samples:
                tasks.append(build_task(samples, name="nonprofit_core", suite="core", judge_model=None))
        if "custom" in self.suites:
            samples = self.custom_samples()
            if samples:
                tasks.append(custom_task(samples))
        if "external" in self.suites:
            for name in self.external_sizes():
                if name not in EXTERNAL_TASKS:
                    raise ValueError(f"Unknown external component {name}. Valid: {', '.join(EXTERNAL_TASKS)}")
                tasks.append(EXTERNAL_TASKS[name](limit=self.external_limit))
        if not tasks:
            raise ValueError("Nothing to run: check --suites, --areas and the profile.")
        return tasks


def run_benchmark(
    plan: RunPlan,
    log_dir: str = "logs/nonprofit_benchmark",
    *,
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
    max_connections: int | None = None,
    fresh: bool = False,
) -> dict[str, Any]:
    """Run the plan with inspect_ai.eval_set.

    By default an existing log dir is *resumed* (completed work is not re-billed).
    Pass ``fresh=True`` to delete it first.
    """

    from inspect_ai import eval_set

    if not plan.models:
        raise ValueError("At least one model is required.")

    output_dir = Path(log_dir)
    if fresh and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    kwargs: dict[str, Any] = {
        "tasks": plan.build_tasks(),
        "model": plan.models,
        "log_dir": str(output_dir),
        "model_roles": {"grader": plan.judge_model},
        "display": "plain",
        "epochs": plan.epochs,
        "fail_on_error": 0.1,
        "retry_attempts": 3,
    }
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    if max_connections:
        kwargs["max_connections"] = max_connections

    # Record run configuration next to the logs so reports can reproduce weights.
    run_meta = {
        "models": plan.models,
        "suites": plan.suites,
        "judge_model": plan.judge_model,
        "profile_path": plan.profile_path,
        "areas": plan.areas,
        "languages": plan.languages,
        "limit_per_task": plan.limit_per_task,
        "external_limit": plan.external_limit,
        "epochs": plan.epochs,
    }
    profile = plan.profile()
    if profile:
        run_meta["component_weights"] = profile.component_weights()
        run_meta["area_weights"] = profile.area_weights()
        run_meta["organization"] = profile.name
    (output_dir / "run_config.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    success, logs = eval_set(**kwargs)
    return {
        "success": success,
        "logs": [str(getattr(log, "location", "")) for log in logs],
        "log_dir": str(output_dir),
        **run_meta,
    }


def run_nonprofit_eval_set(
    models: list[str],
    task_names: Iterable[str] = ("all",),
    log_dir: str = "logs/nonprofit_benchmark",
    judge_model: str | None = None,
    limit_per_task: int | None = None,
) -> dict[str, Any]:
    """Backwards-compatible wrapper: core suite only, filtered by area aliases."""

    plan = RunPlan(
        models=models,
        suites=["core"],
        areas=list(task_names),
        judge_model=judge_model or DEFAULT_JUDGE_MODEL,
        limit_per_task=limit_per_task,
    )
    return run_benchmark(plan, log_dir=log_dir, fresh=True)
