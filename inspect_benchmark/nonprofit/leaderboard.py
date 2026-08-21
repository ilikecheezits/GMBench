"""Leaderboard generation from inspect_ai nonprofit benchmark logs."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

from .scoring import default_scoring_config

OBJECTIVE_WEIGHT = 0.65
JUDGE_WEIGHT = 0.35


def _score_to_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip().upper()
    mapping = {"C": 1.0, "P": 0.5, "I": 0.0}
    if text in mapping:
        return mapping[text]

    try:
        return float(text)
    except ValueError:
        return None


def _pick_score_column(columns: list[str]) -> str:
    if "score_headline_value" in columns:
        return "score_headline_value"

    accuracy_cols = [name for name in columns if name.startswith("score_") and name.endswith("_accuracy")]
    if accuracy_cols:
        return sorted(accuracy_cols)[0]

    raise ValueError(
        "Could not find a usable score column in eval logs. "
        "Expected score_headline_value or score_*_accuracy."
    )


def _pick_sample_score_columns(columns: list[str]) -> tuple[str, str | None]:
    judge_col = "score_model_graded_qa" if "score_model_graded_qa" in columns else ""

    objective_candidates = [name for name in columns if name.startswith("score_objective_rubric")]
    objective_col = objective_candidates[0] if objective_candidates else None

    if not judge_col:
        generic = [name for name in columns if name.startswith("score_")]
        if not generic:
            raise ValueError("No sample score columns found in inspect logs.")
        judge_col = generic[0]

    return judge_col, objective_col


def build_leaderboard_dataframe(log_dir: str):
    """Read Inspect logs and produce model-level aggregated leaderboard dataframe."""

    try:
        from inspect_ai.analysis import evals_df, samples_df
    except ImportError as exc:
        raise RuntimeError(
            "inspect_ai/pandas not installed. Install with: pip install inspect-ai pandas pyarrow"
        ) from exc

    evals = evals_df(log_dir)
    samples = samples_df(log_dir)
    if evals.empty or samples.empty:
        raise ValueError(f"No eval logs found in {log_dir}")

    if "model" not in evals.columns or "eval_id" not in evals.columns:
        raise ValueError("Inspect eval dataframe is missing required model/eval_id columns.")

    judge_col, objective_col = _pick_sample_score_columns(list(samples.columns))

    merged = samples.merge(evals[["eval_id", "model", "task_name"]], on="eval_id", how="left")
    merged["judge_score"] = merged[judge_col].map(_score_to_float)

    if objective_col:
        merged["objective_score"] = merged[objective_col].map(_score_to_float)
    else:
        merged["objective_score"] = merged["judge_score"]

    merged = merged[merged["judge_score"].notna() & merged["objective_score"].notna()].copy()
    if merged.empty:
        raise ValueError("No parseable sample scores were found in logs.")

    scoring_cfg = default_scoring_config()
    merged["hybrid_score"] = merged.apply(
        lambda row: scoring_cfg.score({"objective": row["objective_score"], "judge": row["judge_score"]}),
        axis=1,
    )

    grouped = (
        merged.groupby("model", as_index=False)
        .agg(
            mean_hybrid_score=("hybrid_score", "mean"),
            mean_objective_score=("objective_score", "mean"),
            mean_judge_score=("judge_score", "mean"),
            median_hybrid_score=("hybrid_score", "median"),
            samples=("hybrid_score", "count"),
            tasks=("task_name", "nunique"),
        )
        .sort_values(by="mean_hybrid_score", ascending=False)
        .reset_index(drop=True)
    )

    grouped.insert(0, "rank", grouped.index + 1)
    for col in [
        "mean_hybrid_score",
        "mean_objective_score",
        "mean_judge_score",
        "median_hybrid_score",
    ]:
        grouped[col] = grouped[col].round(4)
    return grouped


def dataframe_to_markdown(df) -> str:
    lines = [
        "# Nonprofit LLM Benchmark Leaderboard",
        "",
        "| Rank | Model | Hybrid Mean | Objective Mean | Judge Mean | Hybrid Median | Samples | Task Coverage |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in df.itertuples(index=False):
        lines.append(
            f"| {row.rank} | {row.model} | {row.mean_hybrid_score:.4f} | {row.mean_objective_score:.4f} | "
            f"{row.mean_judge_score:.4f} | {row.median_hybrid_score:.4f} | {row.samples} | {row.tasks} |"
        )
    return "\n".join(lines) + "\n"


def export_markdown_leaderboard(log_dir: str, output_path: str) -> Tuple[str, int]:
    """Generate leaderboard markdown from logs and write it to disk."""

    leaderboard = build_leaderboard_dataframe(log_dir=log_dir)
    markdown = dataframe_to_markdown(leaderboard)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")

    return str(path), len(leaderboard)


def build_task_scores_dataframe(log_dir: str):
    """Build task-area score breakdown from inspect sample logs."""

    try:
        from inspect_ai.analysis import evals_df, samples_df
    except ImportError as exc:
        raise RuntimeError(
            "inspect_ai/pandas not installed. Install with: pip install inspect-ai pandas pyarrow"
        ) from exc

    evals = evals_df(log_dir)
    samples = samples_df(log_dir)

    if evals.empty or samples.empty:
        raise ValueError(f"No eval/sample logs found in {log_dir}")

    judge_col, objective_col = _pick_sample_score_columns(list(samples.columns))

    if "eval_id" not in samples.columns or "eval_id" not in evals.columns:
        raise ValueError("Inspect logs missing eval_id needed for joining samples to models.")

    join_cols = ["eval_id", "model", "task_name"]
    for col in join_cols:
        if col not in evals.columns:
            raise ValueError(f"Inspect eval logs missing required column: {col}")

    merged = samples.merge(evals[join_cols], on="eval_id", how="left")
    merged["judge_score"] = merged[judge_col].map(_score_to_float)
    if objective_col:
        merged["objective_score"] = merged[objective_col].map(_score_to_float)
    else:
        merged["objective_score"] = merged["judge_score"]

    scoring_cfg = default_scoring_config()
    merged["hybrid_score"] = merged.apply(
        lambda row: scoring_cfg.score({"objective": row["objective_score"], "judge": row["judge_score"]}),
        axis=1,
    )

    merged = merged[
        merged["judge_score"].notna() & merged["objective_score"].notna() & merged["hybrid_score"].notna()
    ].copy()
    if merged.empty:
        raise ValueError("No numeric sample scores could be parsed from logs.")

    def _grade_rate(df, grade: str) -> float:
        return (df[judge_col].astype(str).str.upper() == grade).mean()

    grouped = (
        merged.groupby(
            [
                "model",
                "task_name",
                "metadata_task_area",
                "metadata_difficulty",
                "metadata_language",
                "metadata_risk_level",
            ],
            as_index=False,
        )
        .agg(
            mean_hybrid_score=("hybrid_score", "mean"),
            mean_objective_score=("objective_score", "mean"),
            mean_judge_score=("judge_score", "mean"),
            n=("hybrid_score", "count"),
        )
        .reset_index(drop=True)
        .sort_values(by=["model", "mean_hybrid_score", "n"], ascending=[True, False, False])
    )

    # Judge grade frequencies are still informative for rubric interpretation.
    keys = [
        "model",
        "task_name",
        "metadata_task_area",
        "metadata_difficulty",
        "metadata_language",
        "metadata_risk_level",
    ]
    rates = (
        merged.assign(_judge_letter=merged[judge_col].astype(str).str.upper())
        .groupby(keys, as_index=False)
        .agg(
            rate_C=("_judge_letter", lambda s: (s == "C").mean()),
            rate_P=("_judge_letter", lambda s: (s == "P").mean()),
            rate_I=("_judge_letter", lambda s: (s == "I").mean()),
        )
    )
    grouped = grouped.merge(
        rates,
        on=keys,
        how="left",
    )

    for col in ["mean_hybrid_score", "mean_objective_score", "mean_judge_score", "rate_C", "rate_P", "rate_I"]:
        grouped[col] = grouped[col].round(4)
    return grouped


def task_scores_to_markdown(df) -> str:
    lines = [
        "# Nonprofit LLM Benchmark Task Scores",
        "",
        "| Model | Task | Area | Difficulty | Language | Risk | Hybrid Mean | Objective Mean | Judge Mean | C Rate | P Rate | I Rate | Samples |",
        "|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in df.itertuples(index=False):
        lines.append(
            "| "
            f"{row.model} | {row.task_name} | {row.metadata_task_area} | {row.metadata_difficulty} | "
            f"{row.metadata_language} | {row.metadata_risk_level} | {row.mean_hybrid_score:.4f} | "
            f"{row.mean_objective_score:.4f} | {row.mean_judge_score:.4f} | "
            f"{row.rate_C:.4f} | {row.rate_P:.4f} | {row.rate_I:.4f} | {int(row.n)} |"
        )

    return "\n".join(lines) + "\n"


def export_markdown_task_scores(log_dir: str, output_path: str) -> Tuple[str, int]:
    """Generate task-level markdown score breakdown and write it to disk."""

    breakdown = build_task_scores_dataframe(log_dir=log_dir)
    markdown = task_scores_to_markdown(breakdown)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")

    return str(path), len(breakdown)
