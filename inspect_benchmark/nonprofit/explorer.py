"""Build the interactive explorer page (single self-contained HTML file) from a run's logs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .external import COMPONENTS
from .leaderboard import build_report, split_log_dirs
from .task_bank import all_samples

TEMPLATE = Path(__file__).parent / "templates" / "explorer.html"
MAX_EXTERNAL_ANSWER_CHARS = 600


def _short(model: str) -> str:
    return model.split("/")[-1]


def build_explorer_data(log_dir: str, profile_path: str | None = None, profile_description: str | None = None) -> dict[str, Any]:
    from inspect_ai.log import list_eval_logs, read_eval_log

    tasks = {
        s.sample_id: {
            "id": s.sample_id, "area": s.task_area, "difficulty": s.difficulty, "language": s.language,
            "risk": s.risk_level, "tags": list(s.tags), "prompt": s.prompt,
            "rubric": [{"c": r.criterion, "p": r.points, "t": r.tag} for r in s.rubric],
            "checks": [{"l": c.label, "k": c.kind, "crit": c.critical} for c in s.checks],
            "answers": {},
        }
        for s in all_samples()
    }
    ext: dict[str, dict[str, list]] = {}
    infos = [info for d in split_log_dirs(log_dir) for info in list_eval_logs(d)]
    for info in infos:
        log = read_eval_log(info)
        model = _short(log.eval.model)
        task = log.eval.task.split("/")[-1]
        for s in log.samples or []:
            if task == "nonprofit_core" and s.id in tasks and s.scores:
                j, o = s.scores.get("rubric_judge"), s.scores.get("objective_rubric")
                if not j or not o:
                    continue
                reasons = [ln.split(" -- ", 1)[1] if " -- " in ln else "" for ln in (j.explanation or "").split("\n")]
                tasks[s.id]["answers"][model] = {
                    "text": s.output.completion, "judge": j.value, "obj": o.value,
                    "met": (j.metadata or {}).get("items_met"), "why": reasons,
                    "checks": (o.metadata or {}).get("sample_checks"), "generic": (o.metadata or {}).get("generic_checks"),
                    "critfail": (o.metadata or {}).get("critical_failed"),
                }
            elif task.startswith("ext_") and s.scores:
                sc = next(iter(s.scores.values()))
                ext.setdefault(task.removeprefix("ext_"), {}).setdefault(model, []).append({
                    "id": s.id, "q": s.input if isinstance(s.input, str) else str(s.input), "choices": s.choices,
                    "target": s.target, "ans": (s.output.completion or "")[:MAX_EXTERNAL_ANSWER_CHARS], "v": sc.value, "a": sc.answer,
                })

    profile = None
    if profile_path:
        profile = json.loads(Path(profile_path).read_text(encoding="utf-8"))
    report = build_report(log_dir)
    return {
        "run_label": "Runs: " + ", ".join(Path(d).name for d in split_log_dirs(log_dir)),
        "tasks": [t for t in tasks.values() if t["answers"]],
        "ext": ext,
        "results": report,
        "profile": profile,
        "profile_desc": profile_description or (profile or {}).get("description", ""),
        "components": {k: {"source": c.source, "license": c.license, "citation": c.citation, "description": c.description}
                       for k, c in COMPONENTS.items()},
    }


def export_explorer(log_dir: str, output: str, profile_path: str | None = None) -> str:
    data = build_explorer_data(log_dir, profile_path)
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    page = TEMPLATE.read_text(encoding="utf-8").replace("__DATA__", payload)
    if not page.lstrip().lower().startswith("<!doctype"):
        page = "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1,viewport-fit=cover\">\n" + page.replace("</style>", "</style></head><body>", 1) + "\n</body></html>\n"
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    return str(out)
