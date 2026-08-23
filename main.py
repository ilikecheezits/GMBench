"""Entry point for the Inspect-based nonprofit LLM benchmark suite."""

from __future__ import annotations

import argparse

from inspect_benchmark.nonprofit import (
    export_markdown_leaderboard,
    export_markdown_task_scores,
    run_nonprofit_eval_set,
    validate_task_bank,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run nonprofit LLM-vs-LLM benchmarks on inspect_ai.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run benchmark tasks across one or more models.")
    run_parser.add_argument(
        "--models",
        required=True,
        help=(
            "Comma-separated model list (for example: "
            "openai/gpt-5,anthropic/claude-sonnet-4-6,google/gemini-2.5-pro)."
        ),
    )
    run_parser.add_argument(
        "--tasks",
        default="all,fundraising,grants,operations,communications,impact,governance,compliance",
        help=(
            "Comma-separated task groups: "
            "all,fundraising,grants,operations,communications,impact,governance,compliance"
        ),
    )
    run_parser.add_argument(
        "--judge-model",
        default=None,
        help="Optional dedicated grading model (bound to inspect role 'grader').",
    )
    run_parser.add_argument(
        "--limit-per-task",
        type=int,
        default=None,
        help="Optional sample limit per selected task, useful for quick smoke tests.",
    )
    run_parser.add_argument(
        "--log-dir",
        default="logs/nonprofit_llm_benchmark",
        help="Directory to store inspect eval logs. If it exists, it is replaced for a clean run.",
    )

    leaderboard_parser = subparsers.add_parser("leaderboard", help="Build leaderboard markdown from inspect logs.")
    leaderboard_parser.add_argument(
        "--log-dir",
        default="logs/nonprofit_llm_benchmark",
        help="Directory containing inspect eval logs.",
    )
    leaderboard_parser.add_argument(
        "--output",
        default="artifacts/nonprofit_llm_leaderboard.md",
        help="Output markdown file path.",
    )

    task_scores_parser = subparsers.add_parser(
        "task-scores", help="Build task-level score breakdown from inspect logs."
    )
    task_scores_parser.add_argument(
        "--log-dir",
        default="logs/nonprofit_llm_benchmark",
        help="Directory containing inspect eval logs.",
    )
    task_scores_parser.add_argument(
        "--output",
        default="artifacts/nonprofit_task_scores.md",
        help="Output markdown file path.",
    )

    validate_parser = subparsers.add_parser(
        "validate", help="Check benchmark task-bank coverage and schema validity."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.command == "run":
        models = [model.strip() for model in args.models.split(",") if model.strip()]
        task_names = [task.strip() for task in args.tasks.split(",") if task.strip()]

        result = run_nonprofit_eval_set(
            models=models,
            task_names=task_names,
            log_dir=args.log_dir,
            judge_model=args.judge_model,
            limit_per_task=args.limit_per_task,
        )

        print("Run complete")
        print(f"Success: {result['success']}")
        print(f"Log directory: {result['log_dir']}")
        print(f"Models: {', '.join(result['models'])}")
        print(f"Tasks: {', '.join(result['tasks'])}")
        if result["judge_model"]:
            print(f"Judge model: {result['judge_model']}")
    elif args.command == "leaderboard":
        output_path, model_count = export_markdown_leaderboard(args.log_dir, args.output)
        print("Leaderboard generated")
        print(f"Models ranked: {model_count}")
        print(f"Output: {output_path}")
    elif args.command == "validate":
        report = validate_task_bank()
        print("Benchmark validation report")
        print(f"Valid: {report['valid']}")
        print(f"Total samples: {report['total_samples']}")
        print(f"Task areas: {', '.join(report['task_areas'])}")
        if report["errors"]:
            print("Errors:")
            for error in report["errors"]:
                print(f"- {error}")
    else:
        output_path, row_count = export_markdown_task_scores(args.log_dir, args.output)
        print("Task score breakdown generated")
        print(f"Rows: {row_count}")
        print(f"Output: {output_path}")
