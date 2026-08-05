"""Example script for running the Inspect-based nonprofit LLM benchmark."""

from __future__ import annotations

from inspect_benchmark.nonprofit import (
    export_markdown_leaderboard,
    export_markdown_task_scores,
    run_nonprofit_eval_set,
)


def run() -> None:
    models = [
        "openai/gpt-5",
        "anthropic/claude-sonnet-4-6",
        "google/gemini-2.5-pro",
    ]

    result = run_nonprofit_eval_set(
        models=models,
        task_names=[
            "all",
            "fundraising",
            "grants",
            "operations",
            "communications",
            "impact",
            "governance",
            "compliance",
        ],
        log_dir="logs/nonprofit_llm_benchmark",
        judge_model="openai/gpt-5-mini",
    )
    print(f"Run success: {result['success']}")
    print(f"Logs in: {result['log_dir']}")

    output_path, model_count = export_markdown_leaderboard(
        log_dir=result["log_dir"],
        output_path="artifacts/nonprofit_llm_leaderboard.md",
    )
    print(f"Ranked models: {model_count}")
    print(f"Leaderboard: {output_path}")

    task_scores_path, row_count = export_markdown_task_scores(
        log_dir=result["log_dir"],
        output_path="artifacts/nonprofit_task_scores.md",
    )
    print(f"Task score rows: {row_count}")
    print(f"Task scores: {task_scores_path}")


if __name__ == "__main__":
    run()
