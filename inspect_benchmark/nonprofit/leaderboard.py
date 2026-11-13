"""Leaderboard and report generation from inspect_ai logs.

Produces, per log directory:
* composite leaderboard with bootstrap 95% confidence intervals,
* component table (core, custom, external benchmarks),
* core per-area table, special-subset table (traps, over-refusal, exact-answer),
* scorer diagnostics (judge vs objective rank agreement, judge parse failures),
* outputs as Markdown, JSON, a self-contained HTML page, and a per-sample CSV.
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any, Tuple

import numpy as np

from .external import COMPONENTS, component_for_task
from .scoring import (
    COMPONENT_LABELS,
    DEFAULT_COMPONENT_WEIGHTS,
    SAMPLE_WEIGHTS,
    composite_scoring_config,
    normalize_weights,
)
from .task_bank import TASK_AREAS

BOOTSTRAP_ITERS = 1000
BOOTSTRAP_SEED = 7


def _score_to_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return None if (isinstance(value, float) and math.isnan(value)) else float(value)
    text = str(value).strip().upper()
    mapping = {"C": 1.0, "P": 0.5, "I": 0.0}
    if text in mapping:
        return mapping[text]
    try:
        return float(text)
    except ValueError:
        return None


def _suite_for_task(task_name: str) -> str | None:
    short = str(task_name).split("/")[-1]
    if short == "nonprofit_custom":
        return "custom"
    if short.startswith("nonprofit_"):
        return "core"
    return component_for_task(short)


def _json_list(value: Any) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.startswith("["):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return []
    return []


def load_sample_scores(log_dir: str):
    """Return one row per (model, task, sample, epoch) with a normalised ``score``."""

    from inspect_ai.analysis import evals_df, samples_df

    dirs = split_log_dirs(log_dir)
    evals = evals_df(dirs, quiet=True)
    samples = samples_df(dirs, quiet=True)
    if evals.empty or samples.empty:
        raise ValueError(f"No eval logs found in {log_dir}")

    keep = [c for c in ["eval_id", "model", "task_name", "created", "status"] if c in evals.columns]
    merged = samples.merge(evals[keep], on="eval_id", how="left")
    merged["component"] = merged["task_name"].map(_suite_for_task)
    merged = merged[merged["component"].notna()].copy()

    def col(name):
        return merged[name].map(_score_to_float) if name in merged.columns else None

    judge = col("score_rubric_judge")
    legacy_judge = col("score_model_graded_qa")
    if judge is None:
        judge = legacy_judge
    elif legacy_judge is not None:
        judge = judge.fillna(legacy_judge)
    objective = col("score_objective_rubric")
    choice_s = col("score_choice")
    refusal_s = col("score_calibrated_refusal")

    merged["judge"] = judge if judge is not None else np.nan
    merged["objective"] = objective if objective is not None else np.nan
    is_graded = merged["component"].isin(["core", "custom"])
    graded_score = SAMPLE_WEIGHTS["judge"] * merged["judge"] + SAMPLE_WEIGHTS["objective"] * merged["objective"]
    ext_score = np.nan
    if choice_s is not None:
        ext_score = choice_s
    if refusal_s is not None:
        ext_score = refusal_s if choice_s is None else choice_s.fillna(refusal_s)
    merged["score"] = np.where(is_graded, graded_score, ext_score)
    merged = merged[merged["score"].notna()].copy()
    merged["score"] = merged["score"].astype(float)

    if "metadata_tags" in merged.columns:
        merged["tags"] = merged["metadata_tags"].map(_json_list)
    else:
        merged["tags"] = [[] for _ in range(len(merged))]

    # Re-runs into the same directory: keep the most recent result per sample.
    sort_col = "created" if "created" in merged.columns else "eval_id"
    # Inspect's `sample_id` column is a per-run UUID; `id` is the dataset's sample id.
    if "id" in merged.columns:
        merged["sample_uuid"] = merged["sample_id"]
        merged["sample_id"] = merged["id"].astype(str)
    merged = merged.sort_values(sort_col).drop_duplicates(
        subset=["model", "task_name", "sample_id", "epoch"], keep="last"
    )
    if merged.empty:
        raise ValueError("No parseable sample scores were found in logs.")
    return merged.reset_index(drop=True)


def split_log_dirs(log_dir: str) -> list[str]:
    """`log_dir` may be several directories separated by commas; their runs are merged."""

    return [d.strip() for d in str(log_dir).split(",") if d.strip()]


def _read_run_config(log_dir: str) -> dict[str, Any]:
    """Merge run configs; weights come from the first directory, judges are listed together."""

    merged: dict[str, Any] = {}
    judges: list[str] = []
    for d in split_log_dirs(log_dir):
        path = Path(d) / "run_config.json"
        if not path.exists():
            continue
        cfg = json.loads(path.read_text(encoding="utf-8"))
        for k, v in cfg.items():
            merged.setdefault(k, v)
        if cfg.get("judge_model") and cfg["judge_model"] not in judges:
            judges.append(cfg["judge_model"])
    if len(judges) > 1:
        merged["judge_model"] = ", ".join(judges)
        merged["mixed_judges"] = True
    return merged


def _core_score(df, area_weights: dict[str, float]) -> float | None:
    if df.empty:
        return None
    means = df.groupby("metadata_task_area")["score"].mean()
    num = sum(area_weights.get(a, 0.0) * m for a, m in means.items())
    den = sum(area_weights.get(a, 0.0) for a in means.index)
    return float(num / den) if den > 0 else float(df["score"].mean())


def _component_scores(model_df, area_weights) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for comp, sub in model_df.groupby("component"):
        out[comp] = _core_score(sub, area_weights) if comp == "core" else float(sub["score"].mean())
    return out


def _bootstrap_ci(model_df, area_weights, weights, rng) -> tuple[float, float]:
    """Stratified bootstrap over samples (within component, and within area for core)."""

    groups = []
    for comp, sub in model_df.groupby("component"):
        if comp == "core":
            for area, asub in sub.groupby("metadata_task_area"):
                groups.append((comp, area, asub["score"].to_numpy()))
        else:
            groups.append((comp, None, sub["score"].to_numpy()))

    cfg = composite_scoring_config(weights)
    draws = []
    for _ in range(BOOTSTRAP_ITERS):
        core_num = core_den = 0.0
        comp_vals: dict[str, float] = {}
        for comp, area, arr in groups:
            m = float(arr[rng.integers(0, len(arr), len(arr))].mean())
            if comp == "core":
                w = area_weights.get(area, 0.0)
                core_num += w * m
                core_den += w
            else:
                comp_vals[comp] = m
        if core_den > 0:
            comp_vals["core"] = core_num / core_den
        draws.append(cfg.score(comp_vals))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def _spearman(a, b) -> float | None:
    if len(a) < 4:
        return None
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    if np.std(ra) == 0 or np.std(rb) == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def build_report(
    log_dir: str,
    *,
    component_weights: dict[str, float] | None = None,
    area_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compute every leaderboard table as plain data (also written to results.json)."""

    df = load_sample_scores(log_dir)
    run_cfg = _read_run_config(log_dir)

    area_w = area_weights or run_cfg.get("area_weights") or {a: 1.0 for a in TASK_AREAS}
    base_w = component_weights or run_cfg.get("component_weights") or dict(DEFAULT_COMPONENT_WEIGHTS)
    present_components = set(df["component"].unique())
    weights = {k: v for k, v in base_w.items() if k in present_components or k in DEFAULT_COMPONENT_WEIGHTS}
    weights = normalize_weights(weights)
    cfg = composite_scoring_config(weights)
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    rows = []
    for model, mdf in df.groupby("model"):
        comps = _component_scores(mdf, area_w)
        missing = cfg.missing(comps)
        composite = cfg.score(comps)
        lo, hi = _bootstrap_ci(mdf, area_w, weights, rng)
        rows.append({
            "model": model,
            "composite": round(composite, 4),
            "ci_low": round(lo, 4),
            "ci_high": round(hi, 4),
            "components": {k: (round(v, 4) if v is not None else None) for k, v in comps.items()},
            "missing_components": missing,
            "n_samples": int(len(mdf)),
        })
    rows.sort(key=lambda r: r["composite"], reverse=True)
    for i, r in enumerate(rows, start=1):
        r["rank"] = i

    graded = df[df["component"].isin(["core", "custom"])]
    core = df[df["component"] == "core"]

    area_table = []
    for model, mdf in core.groupby("model"):
        area_table.append({
            "model": model,
            **{a: round(float(mdf[mdf["metadata_task_area"] == a]["score"].mean()), 4)
               for a in TASK_AREAS if (mdf["metadata_task_area"] == a).any()},
        })

    subset_table = []
    for model, mdf in graded.groupby("model"):
        entry = {"model": model}
        for tag in ("trap", "over_refusal", "exact"):
            sub = mdf[mdf["tags"].map(lambda t: tag in t)]
            entry[tag] = round(float(sub["score"].mean()), 4) if len(sub) else None
        hr = mdf[mdf["metadata_risk_level"] == "high"] if "metadata_risk_level" in mdf else mdf.iloc[0:0]
        entry["high_risk"] = round(float(hr["score"].mean()), 4) if len(hr) else None
        entry["judge_mean"] = round(float(mdf["judge"].mean()), 4)
        entry["objective_mean"] = round(float(mdf["objective"].mean()), 4)
        subset_table.append(entry)

    rho = _spearman(graded["judge"].to_numpy(), graded["objective"].to_numpy()) if len(graded) else None
    judge_model = run_cfg.get("judge_model")
    warnings = []
    if run_cfg.get("mixed_judges"):
        warnings.append(f"Runs were graded by different judges ({judge_model}); judge scores are not directly comparable.")
    if judge_model and judge_model in set(df["model"]):
        warnings.append(f"Judge model {judge_model} is also a candidate; its core/custom scores may carry self-preference bias.")
    for r in rows:
        if r["missing_components"]:
            warnings.append(f"{r['model']} is missing components {', '.join(r['missing_components'])}; composite renormalised over the rest.")
    if "error" in df.columns:
        n_err = int(df["error"].notna().sum() - (df["error"] == "").sum()) if df["error"].dtype == object else 0
        if n_err > 0:
            warnings.append(f"{n_err} samples recorded errors; see logs.")

    return {
        "log_dir": log_dir,
        "organization": run_cfg.get("organization"),
        "judge_model": judge_model,
        "component_weights": weights,
        "area_weights": area_w,
        "sample_weights": SAMPLE_WEIGHTS,
        "leaderboard": rows,
        "areas": area_table,
        "subsets": subset_table,
        "diagnostics": {
            "judge_objective_spearman": None if rho is None else round(rho, 3),
            "graded_samples": int(len(graded)),
            "total_samples": int(len(df)),
        },
        "warnings": warnings,
        "external_sources": {k: {"source": c.source, "license": c.license, "citation": c.citation} for k, c in COMPONENTS.items()},
    }


# ------------------------------------------------------------------------------ rendering

def _fmt(v) -> str:
    return "—" if v is None else f"{v:.3f}"


def _component_order(report) -> list[str]:
    order = ["core", "custom", "knowledge", "truthfulness", "fairness", "calibrated_refusal"]
    return [c for c in order if c in report["component_weights"]]


def report_to_markdown(report: dict[str, Any]) -> str:
    comps = _component_order(report)
    title = "Nonprofit AI Benchmark Leaderboard"
    if report.get("organization"):
        title += f" — tailored for {report['organization']}"
    lines = [f"# {title}", ""]
    weights_txt = ", ".join(f"{c} {report['component_weights'][c]:.2f}" for c in comps)
    lines += [f"Composite weights: {weights_txt}. Judge: `{report.get('judge_model') or 'n/a'}`.", ""]
    lines += ["| Rank | Model | Composite | 95% CI | " + " | ".join(COMPONENT_LABELS.get(c, c) for c in comps) + " |",
              "|---:|---|---:|---|" + "---:|" * len(comps)]
    for r in report["leaderboard"]:
        cells = " | ".join(_fmt(r["components"].get(c)) for c in comps)
        lines.append(f"| {r['rank']} | {r['model']} | **{r['composite']:.3f}** | {r['ci_low']:.3f}–{r['ci_high']:.3f} | {cells} |")

    if report["areas"]:
        areas = [a for a in TASK_AREAS if any(a in row for row in report["areas"])]
        lines += ["", "## Core suite by task area", "",
                  "| Model | " + " | ".join(a.replace("_", " ") for a in areas) + " |",
                  "|---|" + "---:|" * len(areas)]
        for row in report["areas"]:
            lines.append(f"| {row['model']} | " + " | ".join(_fmt(row.get(a)) for a in areas) + " |")

    if report["subsets"]:
        lines += ["", "## Stress subsets and scorer split", "",
                  "| Model | Traps | Over-refusal | Exact-answer | High-risk | Judge mean | Objective mean |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for row in report["subsets"]:
            lines.append(
                f"| {row['model']} | {_fmt(row['trap'])} | {_fmt(row['over_refusal'])} | {_fmt(row['exact'])} | "
                f"{_fmt(row['high_risk'])} | {_fmt(row['judge_mean'])} | {_fmt(row['objective_mean'])} |"
            )

    d = report["diagnostics"]
    lines += ["", "## Diagnostics", "",
              f"- Graded samples: {d['graded_samples']} of {d['total_samples']} total.",
              f"- Judge vs objective Spearman rank correlation (graded samples): {d['judge_objective_spearman']}"]
    for w in report["warnings"]:
        lines.append(f"- ⚠️ {w}")
    lines += ["", "Sample score = 0.60 × rubric judge + 0.40 × deterministic checks. Core score weights each task area "
              "equally unless a profile says otherwise. CIs are stratified bootstrap over samples (1,000 draws).", ""]
    return "\n".join(lines)


def report_to_html(report: dict[str, Any]) -> str:
    comps = _component_order(report)
    title = "Nonprofit AI Benchmark"
    org = report.get("organization")

    def bar(v):
        if v is None:
            return '<td class="num muted">—</td>'
        pct = max(0, min(100, v * 100))
        return f'<td class="num" style="background:linear-gradient(90deg,var(--bar) {pct:.1f}%,transparent {pct:.1f}%)">{v:.3f}</td>'

    head = "".join(f"<th class='num'>{html.escape(COMPONENT_LABELS.get(c, c))}<br><span class='muted'>w={report['component_weights'][c]:.2f}</span></th>" for c in comps)
    body = ""
    for r in report["leaderboard"]:
        cells = "".join(bar(r["components"].get(c)) for c in comps)
        body += (f"<tr><td class='num'>{r['rank']}</td><td>{html.escape(r['model'])}</td>"
                 f"<td class='num strong'>{r['composite']:.3f}</td><td class='num muted'>{r['ci_low']:.3f}–{r['ci_high']:.3f}</td>{cells}</tr>")

    areas = [a for a in TASK_AREAS if any(a in row for row in report["areas"])]
    area_head = "".join(f"<th class='num'>{html.escape(a.replace('_', ' '))}</th>" for a in areas)
    area_body = "".join(
        f"<tr><td>{html.escape(row['model'])}</td>" + "".join(bar(row.get(a)) for a in areas) + "</tr>" for row in report["areas"]
    )
    sub_body = "".join(
        f"<tr><td>{html.escape(row['model'])}</td>{bar(row['trap'])}{bar(row['over_refusal'])}{bar(row['exact'])}{bar(row['high_risk'])}{bar(row['judge_mean'])}{bar(row['objective_mean'])}</tr>"
        for row in report["subsets"]
    )
    warn = "".join(f"<li>{html.escape(w)}</li>" for w in report["warnings"])
    d = report["diagnostics"]
    sources = "".join(
        f"<li><b>{html.escape(COMPONENT_LABELS.get(k, k))}</b>: {html.escape(v['citation'])} ({html.escape(v['license'])})</li>"
        for k, v in report["external_sources"].items() if k in comps
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root{{--bg:#fbfaf7;--fg:#1d1d1b;--muted:#6b6a65;--line:#e4e1d8;--accent:#2f6f5e;--bar:#cfe3dc;--card:#fff}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#141513;--fg:#ecebe6;--muted:#9b9a94;--line:#2c2d29;--accent:#7cc4ad;--bar:#24443b;--card:#1b1c1a}}}}
:root[data-theme="dark"]{{--bg:#141513;--fg:#ecebe6;--muted:#9b9a94;--line:#2c2d29;--accent:#7cc4ad;--bar:#24443b;--card:#1b1c1a}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}}
main{{max-width:1100px;margin:0 auto;padding:32px 16px 64px}}
h1{{font-size:26px;margin:0 0 4px}} h2{{font-size:18px;margin:36px 0 10px}}
.muted{{color:var(--muted)}} .strong{{font-weight:650;color:var(--accent)}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;overflow-x:auto}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}
th,td{{padding:9px 12px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}}
th{{font-size:12px;font-weight:600;color:var(--muted);vertical-align:bottom}}
tr:last-child td{{border-bottom:0}} .num{{text-align:right}}
ul{{padding-left:20px}} code{{font-size:13px}}
</style></head><body><main>
<h1>{html.escape(title)}</h1>
<p class="muted">{('Tailored for <b>' + html.escape(org) + '</b> · ') if org else ''}Judge <code>{html.escape(str(report.get('judge_model')))}</code> · sample score = 0.60 rubric judge + 0.40 deterministic checks · 95% CIs by stratified bootstrap</p>
<h2>Leaderboard</h2>
<div class="card"><table><thead><tr><th class="num">#</th><th>Model</th><th class="num">Composite</th><th class="num">95% CI</th>{head}</tr></thead><tbody>{body}</tbody></table></div>
<h2>Core suite by task area</h2>
<div class="card"><table><thead><tr><th>Model</th>{area_head}</tr></thead><tbody>{area_body}</tbody></table></div>
<h2>Stress subsets</h2>
<p class="muted">Traps: false premises / unsafe shortcuts the model should push back on. Over-refusal: sensitive-sounding but legitimate requests. Exact: numerically checkable answers.</p>
<div class="card"><table><thead><tr><th>Model</th><th class="num">Traps</th><th class="num">Over-refusal</th><th class="num">Exact</th><th class="num">High-risk</th><th class="num">Judge mean</th><th class="num">Objective mean</th></tr></thead><tbody>{sub_body}</tbody></table></div>
<h2>Diagnostics</h2>
<ul><li>Judge vs objective Spearman ρ: {d['judge_objective_spearman']} over {d['graded_samples']} graded samples</li>{warn}</ul>
<h2>External sources</h2><ul>{sources}</ul>
</main></body></html>
"""


def export_report(log_dir: str, out_dir: str, **kwargs) -> dict[str, str]:
    report = build_report(log_dir, **kwargs)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "markdown": out / "leaderboard.md",
        "html": out / "leaderboard.html",
        "json": out / "results.json",
        "samples_csv": out / "samples.csv",
    }
    paths["markdown"].write_text(report_to_markdown(report), encoding="utf-8")
    paths["html"].write_text(report_to_html(report), encoding="utf-8")
    paths["json"].write_text(json.dumps(report, indent=2), encoding="utf-8")
    df = load_sample_scores(log_dir)
    cols = [c for c in ["model", "component", "sample_id", "epoch", "metadata_task_area", "metadata_language",
                        "metadata_risk_level", "tags", "judge", "objective", "score"] if c in df.columns]
    df[cols].to_csv(paths["samples_csv"], index=False)
    return {k: str(v) for k, v in paths.items()}


# --------------------------------------------------------------------- compatibility API

def build_leaderboard_dataframe(log_dir: str):
    import pandas as pd

    report = build_report(log_dir)
    return pd.DataFrame(
        [{"rank": r["rank"], "model": r["model"], "composite": r["composite"], "ci_low": r["ci_low"],
          "ci_high": r["ci_high"], **{f"{k}": v for k, v in r["components"].items()}, "samples": r["n_samples"]}
         for r in report["leaderboard"]]
    )


def export_markdown_leaderboard(log_dir: str, output_path: str) -> Tuple[str, int]:
    report = build_report(log_dir)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report_to_markdown(report), encoding="utf-8")
    return str(path), len(report["leaderboard"])


def build_task_scores_dataframe(log_dir: str):
    df = load_sample_scores(log_dir)
    graded = df[df["component"].isin(["core", "custom"])]
    keys = ["model", "component", "metadata_task_area", "metadata_difficulty", "metadata_language", "metadata_risk_level"]
    keys = [k for k in keys if k in graded.columns]
    return (
        graded.groupby(keys, as_index=False)
        .agg(mean_score=("score", "mean"), mean_judge=("judge", "mean"), mean_objective=("objective", "mean"), n=("score", "count"))
        .sort_values(["model", "mean_score"], ascending=[True, False])
        .round(4)
    )


def export_markdown_task_scores(log_dir: str, output_path: str) -> Tuple[str, int]:
    df = build_task_scores_dataframe(log_dir)
    lines = ["# Nonprofit Benchmark Task Scores", "",
             "| Model | Suite | Area | Difficulty | Language | Risk | Score | Judge | Objective | n |",
             "|---|---|---|---|---|---|---:|---:|---:|---:|"]
    for row in df.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.component} | {row.metadata_task_area} | {row.metadata_difficulty} | {row.metadata_language} | "
            f"{row.metadata_risk_level} | {row.mean_score:.3f} | {row.mean_judge:.3f} | {row.mean_objective:.3f} | {int(row.n)} |"
        )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path), len(df)
