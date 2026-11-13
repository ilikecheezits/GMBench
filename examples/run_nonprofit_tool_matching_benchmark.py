"""Programmatic example: estimate, run, and report the nonprofit benchmark."""

from __future__ import annotations

from inspect_benchmark.nonprofit import RunPlan, export_report, run_benchmark
from inspect_benchmark.nonprofit.cost import estimate_plan


def run() -> None:
    plan = RunPlan(
        models=["openai/gpt-4o-mini", "openai/gpt-5-nano"],
        suites=["core", "external"],
        judge_model="openai/gpt-4.1-mini",
        # profile_path="profiles/my-org.json",  # tailor tasks and weights to an organisation
    )
    print(estimate_plan(plan).render())
    result = run_benchmark(plan, log_dir="logs/example_run")
    paths = export_report(result["log_dir"], "artifacts/example_run")
    print(paths)


if __name__ == "__main__":
    run()
