"""CLI for the nonprofit AI benchmark-of-benchmarks (`npbench` or `python main.py`).

Commands
  validate        check task bank schema/coverage (no API)
  estimate        print expected API calls and cost for a run (no API)
  run             run the benchmark (asks for confirmation after showing the estimate)
  report          write leaderboard.md / leaderboard.html / results.json / samples.csv
  profile         build an organisation profile (intake agent, or --no-agent questionnaire)
  fetch-external  re-download fixed-seed external benchmark slices (no API key)
  explorer        build an interactive HTML page to browse a run (tasks, answers, judge decisions, weights)
  export-dataset  write the core task bank as JSONL for sharing
  leaderboard / task-scores   legacy single-file reports
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from inspect_benchmark.nonprofit.env import load_env

DEFAULT_MODELS = "openai/gpt-4o-mini,openai/gpt-5-nano"


def _csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


def _add_plan_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--models", default=DEFAULT_MODELS, help=f"Comma-separated Inspect model ids (default: {DEFAULT_MODELS}).")
    p.add_argument("--suites", default="core,external", help="Comma-separated: core, external, custom (custom needs --profile).")
    p.add_argument("--areas", default=None, help="Core task areas to include (aliases: fundraising,grants,operations,communications,impact,governance,compliance,finance).")
    p.add_argument("--languages", default=None, help="Restrict core tasks to these languages, e.g. English,Spanish.")
    p.add_argument("--profile", default=None, help="Org profile JSON from `main.py profile`. Adds the custom suite automatically.")
    p.add_argument("--judge-model", default=None, help="Grader model (default: openai/gpt-4.1-mini).")
    p.add_argument("--limit-per-task", type=int, default=None, help="Cap core/custom samples (smoke tests).")
    p.add_argument("--external-limit", type=int, default=None, help="Cap items per external benchmark.")
    p.add_argument("--external", default=None, help="External components: knowledge,truthfulness,fairness,calibrated_refusal.")
    p.add_argument("--epochs", type=int, default=1, help="Repeat each sample N times (variance estimation; multiplies cost).")


def _plan_from_args(args):
    from inspect_benchmark.nonprofit.judge import DEFAULT_JUDGE_MODEL
    from inspect_benchmark.nonprofit.run_eval_set import RunPlan

    suites = _csv(args.suites) or ["core", "external"]
    if args.profile and "custom" not in suites:
        suites.append("custom")
    return RunPlan(
        models=_csv(args.models) or [],
        suites=suites,
        areas=_csv(args.areas),
        languages=_csv(args.languages),
        profile_path=args.profile,
        judge_model=args.judge_model or DEFAULT_JUDGE_MODEL,
        limit_per_task=args.limit_per_task,
        external_limit=args.external_limit,
        external_components=_csv(args.external),
        epochs=args.epochs,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nonprofit AI benchmark-of-benchmarks (inspect_ai).")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("validate", help="Check task bank schema and coverage.")

    est = sub.add_parser("estimate", help="Estimate API calls and cost (no API calls).")
    _add_plan_args(est)

    run = sub.add_parser("run", help="Run the benchmark.")
    _add_plan_args(run)
    run.add_argument("--log-dir", default="logs/nonprofit_benchmark")
    run.add_argument("--fresh", action="store_true", help="Delete the log dir first (otherwise completed work is resumed, not re-billed).")
    run.add_argument("--max-tokens", type=int, default=None, help="Global output-token cap per generation.")
    run.add_argument("--reasoning-effort", default=None, choices=["minimal", "low", "medium", "high"], help="For reasoning models; 'low' or 'minimal' cuts cost.")
    run.add_argument("--max-connections", type=int, default=None)
    run.add_argument("--yes", "-y", action="store_true", help="Skip the cost confirmation prompt.")
    run.add_argument("--report-dir", default=None, help="Write the report here after the run (default artifacts/<log-dir name>).")

    rep = sub.add_parser("report", help="Build leaderboard report files from logs.")
    rep.add_argument("--log-dir", default="logs/nonprofit_benchmark", help="One dir, or several comma-separated dirs to merge into one leaderboard.")
    rep.add_argument("--out-dir", default=None, help="Default: artifacts/<log-dir name>")
    rep.add_argument("--profile", default=None, help="Re-weight an existing run with a profile's weights.")

    prof = sub.add_parser("profile", help="Create an organisation profile for tailored benchmarking.")
    prof.add_argument("--output", default=None, help="Where to save (default profiles/<org-name>.json).")
    prof.add_argument("--no-agent", action="store_true", help="Use the fixed questionnaire instead of the AI intake agent (no API calls).")
    prof.add_argument("--model", default=None, help="Intake agent model (default openai/gpt-4.1-mini).")
    prof.add_argument("--description-file", default=None, help="Read the org description from a file instead of prompting.")
    prof.add_argument("--non-interactive", action="store_true", help="Skip follow-up questions (agent records assumptions).")
    prof.add_argument("--custom-tasks", type=int, default=4, help="Number of org-specific tasks to generate (0-8).")

    fetch = sub.add_parser("fetch-external", help="Download fixed-seed slices of external benchmarks.")
    fetch.add_argument("--mmlu-per-subject", type=int, default=8)
    fetch.add_argument("--truthfulqa", type=int, default=50)
    fetch.add_argument("--bbq-per-category", type=int, default=6)
    fetch.add_argument("--xstest-per-label", type=int, default=30)

    ex = sub.add_parser("explorer", help="Build the interactive explorer page (one HTML file) from a run's logs.")
    ex.add_argument("--log-dir", default="logs/baseline")
    ex.add_argument("--output", default=None, help="Default: artifacts/<log-dir name>/explorer.html")
    ex.add_argument("--profile", default=None, help="Include an org profile in the Tailoring view.")

    exp = sub.add_parser("export-dataset", help="Export the core task bank to JSONL.")
    exp.add_argument("--output", default="artifacts/nonprofit_core_tasks.jsonl")

    lb = sub.add_parser("leaderboard", help="(legacy) Leaderboard markdown only.")
    lb.add_argument("--log-dir", default="logs/nonprofit_benchmark")
    lb.add_argument("--output", default="artifacts/nonprofit_llm_leaderboard.md")

    ts = sub.add_parser("task-scores", help="(legacy) Per-area/difficulty/language breakdown.")
    ts.add_argument("--log-dir", default="logs/nonprofit_benchmark")
    ts.add_argument("--output", default="artifacts/nonprofit_task_scores.md")

    return parser.parse_args()


def _default_report_dir(log_dir: str) -> str:
    names = [Path(d.strip()).name for d in log_dir.split(",") if d.strip()]
    return str(Path("artifacts") / "+".join(names))


def cmd_validate() -> int:
    from inspect_benchmark.nonprofit.external import external_counts
    from inspect_benchmark.nonprofit.task_bank import validate_task_bank

    report = validate_task_bank()
    print("Benchmark validation report")
    print(f"Valid: {report['valid']}")
    print(f"Core samples: {report['total_samples']}")
    print(f"Task areas: {json.dumps(report['tasks_by_area'])}")
    print(f"Tags: {json.dumps(report['tags'])}")
    print(f"Languages: {', '.join(report['languages'])}")
    counts = external_counts()
    print(f"External items: {json.dumps(counts)}")
    missing = [k for k, v in counts.items() if v == 0]
    if missing:
        print(f"Missing external data for {', '.join(missing)} - run: python main.py fetch-external")
    for error in report["errors"]:
        print(f"- {error}")
    return 0 if report["valid"] and not missing else 1


def cmd_estimate(args) -> int:
    from inspect_benchmark.nonprofit.cost import estimate_plan

    plan = _plan_from_args(args)
    print(estimate_plan(plan).render())
    return 0


def cmd_run(args) -> int:
    from inspect_benchmark.nonprofit.cost import estimate_plan
    from inspect_benchmark.nonprofit.leaderboard import export_report
    from inspect_benchmark.nonprofit.run_eval_set import run_benchmark

    plan = _plan_from_args(args)
    if not plan.models:
        print("No models given.")
        return 2
    print(estimate_plan(plan).render())
    if not args.yes:
        if not sys.stdin.isatty():
            print("\nRefusing to start a paid run non-interactively without --yes.")
            return 2
        if input("\nProceed? [y/N] ").strip().lower() not in {"y", "yes"}:
            print("Cancelled.")
            return 1

    result = run_benchmark(
        plan,
        log_dir=args.log_dir,
        max_tokens=args.max_tokens,
        reasoning_effort=args.reasoning_effort,
        max_connections=args.max_connections,
        fresh=args.fresh,
    )
    print("\nRun complete" if result["success"] else "\nRun finished with incomplete tasks (re-run the same command to resume)")
    print(f"Log directory: {result['log_dir']}")
    out_dir = args.report_dir or _default_report_dir(args.log_dir)
    try:
        paths = export_report(args.log_dir, out_dir)
        print("Report:")
        for name, path in paths.items():
            print(f"  {name}: {path}")
    except ValueError as exc:
        print(f"Could not build report: {exc}")
    return 0 if result["success"] else 1


def cmd_report(args) -> int:
    from inspect_benchmark.nonprofit.leaderboard import build_report, export_report, report_to_markdown

    kwargs = {}
    if args.profile:
        from inspect_benchmark.nonprofit.profile import OrgProfile

        profile = OrgProfile.load(args.profile)
        kwargs = {"component_weights": profile.component_weights(), "area_weights": profile.area_weights()}
    paths = export_report(args.log_dir, args.out_dir or _default_report_dir(args.log_dir), **kwargs)
    print(report_to_markdown(build_report(args.log_dir, **kwargs)))
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


def cmd_profile(args) -> int:
    import re

    from inspect_benchmark.nonprofit.profile import DEFAULT_AGENT_MODEL, run_intake, run_questionnaire

    def ask(q: str) -> str:
        try:
            return input(f"{q}\n> ")
        except EOFError:
            return ""

    warnings: list[str] = []
    if args.no_agent:
        profile = run_questionnaire(ask)
    else:
        if args.description_file:
            description = Path(args.description_file).read_text(encoding="utf-8")
        else:
            print("Describe your organisation: mission, size, who you serve, languages, and what you'd want AI to help with.")
            print("(Finish with an empty line.)")
            lines = []
            while True:
                try:
                    line = input()
                except EOFError:
                    break
                if not line.strip():
                    break
                lines.append(line)
            description = "\n".join(lines)
        if not description.strip():
            print("No description given.")
            return 2
        profile, warnings = run_intake(
            description,
            ask,
            model=args.model or DEFAULT_AGENT_MODEL,
            n_custom_tasks=max(0, min(8, args.custom_tasks)),
            interactive=not args.non_interactive,
        )

    slug = re.sub(r"[^a-z0-9]+", "-", (profile.name or "organization").lower()).strip("-") or "organization"
    path = profile.save(args.output or f"profiles/{slug}.json")
    print(f"\nSaved profile: {path}")
    print(f"Organisation: {profile.name}  | staff {profile.staff_size}  | budget {profile.annual_budget}")
    print(f"Languages: {', '.join(profile.languages)}  | data sensitivity {profile.data_sensitivity}  | risk tolerance {profile.risk_tolerance}")
    print("Area priorities (0-3): " + ", ".join(f"{a}={p}" for a, p in profile.area_priorities.items()))
    print("Composite weights: " + ", ".join(f"{k}={v:.2f}" for k, v in profile.component_weights().items()))
    for t in profile.custom_tasks:
        print(f"  custom task {t['sample_id']} [{t['task_area']}, {t['risk_level']} risk]: {t['prompt'].splitlines()[-1][:110]}")
    for w in warnings:
        print(f"  warning: {w}")
    print(f"\nReview/edit the JSON, then: python main.py estimate --profile {path}")
    return 0


def cmd_fetch(args) -> int:
    from inspect_benchmark.nonprofit.external import fetch_external_data

    counts = fetch_external_data(
        mmlu_per_subject=args.mmlu_per_subject,
        truthfulqa_n=args.truthfulqa,
        bbq_per_category=args.bbq_per_category,
        xstest_per_label=args.xstest_per_label,
    )
    print(json.dumps(counts, indent=2))
    return 0


def cmd_export(args) -> int:
    from inspect_benchmark.nonprofit.task_bank import all_samples, to_record

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for sample in all_samples():
            fh.write(json.dumps(to_record(sample), ensure_ascii=False) + "\n")
    print(f"Wrote {len(all_samples())} tasks to {path}")
    return 0


def main() -> int:
    load_env()
    args = _parse_args()
    if args.command == "validate":
        return cmd_validate()
    if args.command == "estimate":
        return cmd_estimate(args)
    if args.command == "run":
        return cmd_run(args)
    if args.command == "report":
        return cmd_report(args)
    if args.command == "profile":
        return cmd_profile(args)
    if args.command == "fetch-external":
        return cmd_fetch(args)
    if args.command == "explorer":
        from inspect_benchmark.nonprofit.explorer import export_explorer

        path = export_explorer(args.log_dir, args.output or str(Path(_default_report_dir(args.log_dir)) / "explorer.html"), args.profile)
        print(f"Explorer page: {path}  (open it in a browser)")
        return 0
    if args.command == "export-dataset":
        return cmd_export(args)
    if args.command == "leaderboard":
        from inspect_benchmark.nonprofit import export_markdown_leaderboard

        path, n = export_markdown_leaderboard(args.log_dir, args.output)
        print(f"Leaderboard ({n} models): {path}")
        return 0
    if args.command == "task-scores":
        from inspect_benchmark.nonprofit import export_markdown_task_scores

        path, n = export_markdown_task_scores(args.log_dir, args.output)
        print(f"Task scores ({n} rows): {path}")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
