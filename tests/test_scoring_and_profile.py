import json
import tempfile
import unittest
from pathlib import Path

from inspect_benchmark.nonprofit.judge import parse_judgement, rubric_score, tag_breakdown
from inspect_benchmark.nonprofit.profile import OrgProfile, sanitize_custom_tasks
from inspect_benchmark.nonprofit.run_eval_set import RunPlan
from inspect_benchmark.nonprofit.task_bank import TASK_AREAS
from inspect_benchmark.nonprofit.scoring import (
    DEFAULT_COMPONENT_WEIGHTS,
    composite_scoring_config,
    default_scoring_config,
    normalize_weights,
)

RUBRIC = [
    {"criterion": "a", "points": 3, "tag": "accuracy"},
    {"criterion": "b", "points": 2, "tag": "safety"},
    {"criterion": "harm", "points": -4, "tag": "safety"},
]


class JudgeTest(unittest.TestCase):
    def test_parse_and_score(self):
        raw = 'Sure. {"items": [{"id": 1, "met": true, "reason": "x"}, {"id": 2, "met": false}, {"id": 3, "met": "false"}]}'
        j = parse_judgement(raw, 3)
        self.assertIsNotNone(j)
        self.assertAlmostEqual(rubric_score(RUBRIC, j), 3 / 5)
        self.assertEqual(tag_breakdown(RUBRIC, j), {"accuracy": 1.0, "safety": 0.0})

    def test_negative_items_penalise_and_clip(self):
        j = parse_judgement('{"items": [{"id": 1, "met": false}, {"id": 2, "met": true}, {"id": 3, "met": true}]}', 3)
        self.assertEqual(rubric_score(RUBRIC, j), 0.0)

    def test_incomplete_judgement_rejected(self):
        self.assertIsNone(parse_judgement('{"items": [{"id": 1, "met": true}]}', 3))
        self.assertIsNone(parse_judgement("GRADE: C", 3))


class ChoiceParserTest(unittest.TestCase):
    def test_parse_choice_variants(self):
        from inspect_benchmark.nonprofit.external import parse_choice

        self.assertEqual(parse_choice("ANSWER: C"), "C")
        self.assertEqual(parse_choice("ANSWER: $C"), "C")  # gpt-5-mini copies the template's $
        self.assertEqual(parse_choice("reasoning... ANSWER: (b)"), "B")
        self.assertEqual(parse_choice("ANSWER: A\nwait. ANSWER: D"), "D")
        self.assertIsNone(parse_choice("I think it's C"))


class ScoringTest(unittest.TestCase):
    def test_sample_blend(self):
        self.assertAlmostEqual(default_scoring_config().score({"judge": 1.0, "objective": 0.5}), 0.8)

    def test_composite_renormalises_missing(self):
        cfg = composite_scoring_config(DEFAULT_COMPONENT_WEIGHTS)
        self.assertAlmostEqual(cfg.score({"core": 0.5}), 0.5)
        self.assertIn("fairness", cfg.missing({"core": 0.5}))

    def test_normalize(self):
        self.assertAlmostEqual(sum(normalize_weights({"a": 2, "b": 2}).values()), 1.0)


class ProfileTest(unittest.TestCase):
    def test_weights_respond_to_profile(self):
        base = OrgProfile()
        self.assertEqual(base.component_weights(), normalize_weights(DEFAULT_COMPONENT_WEIGHTS))
        risky = OrgProfile(data_sensitivity="high", languages=["English", "Spanish"])
        w = risky.component_weights()
        self.assertGreater(w["calibrated_refusal"], base.component_weights()["calibrated_refusal"])
        self.assertGreater(w["fairness"], base.component_weights()["fairness"])
        self.assertAlmostEqual(sum(w.values()), 1.0, places=3)

    def test_area_priorities_filter_core_tasks(self):
        profile = OrgProfile(
            languages=["English", "Spanish"],
            area_priorities={**{a: 0 for a in TASK_AREAS}, "grant_writing": 3, "finance_and_accounting": 1},
        )
        with tempfile.TemporaryDirectory() as d:
            path = profile.save(Path(d) / "p.json")
            samples = RunPlan(models=["x"], profile_path=str(path)).core_samples()
        areas = {s.task_area for s in samples}
        self.assertEqual(areas, {"grant_writing", "finance_and_accounting"})
        self.assertTrue(all(s.language in {"English", "Spanish"} for s in samples))

    def test_sanitize_custom_tasks(self):
        tasks = [
            {
                "task_area": "grant_writing", "difficulty": "medium", "risk_level": "high",
                "prompt": "Draft the report. (Trap: the model should refuse.)", "criterion": "c", "reference_answer": "r",
                "rubric": [{"criterion": "good", "points": 3}, {"criterion": "bad", "points": -3}],
                "checks": [{"kind": "any", "label": "money", "patterns": ["$400,000"]}],
                "tags": ["trap", "made_up_tag"],
            },
            {"task_area": "not_an_area", "prompt": "x", "rubric": [{"criterion": "a", "points": 1}]},
        ]
        kept, warnings = sanitize_custom_tasks(tasks, "Organization: Test Org.")
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(warnings), 1)
        self.assertNotIn("Trap", kept[0]["prompt"])
        self.assertTrue(kept[0]["prompt"].startswith("Organization: Test Org."))
        self.assertEqual(kept[0]["tags"], ["trap"])
        from inspect_benchmark.nonprofit.objective_metrics import run_check
        from inspect_benchmark.nonprofit.task_bank import from_record

        check = from_record(kept[0]).checks[0]
        self.assertTrue(run_check(check, "We received $400,000 in funding."))


class PipelineTest(unittest.TestCase):
    """Runs the whole pipeline offline with Inspect's mock model (no API calls)."""

    def test_end_to_end_with_mock_model(self):
        from inspect_benchmark.nonprofit.leaderboard import build_report, export_report
        from inspect_benchmark.nonprofit.run_eval_set import run_benchmark

        with tempfile.TemporaryDirectory() as d:
            plan = RunPlan(models=["mockllm/model"], judge_model="mockllm/model", limit_per_task=2, external_limit=2)
            result = run_benchmark(plan, log_dir=str(Path(d) / "logs"), fresh=True)
            self.assertTrue(result["success"])
            report = build_report(str(Path(d) / "logs"))
            self.assertEqual(len(report["leaderboard"]), 1)
            row = report["leaderboard"][0]
            self.assertEqual(set(row["components"]), set(DEFAULT_COMPONENT_WEIGHTS))
            self.assertLessEqual(row["ci_low"], row["composite"])
            self.assertLessEqual(row["composite"], row["ci_high"])
            paths = export_report(str(Path(d) / "logs"), str(Path(d) / "out"))
            self.assertIn("Leaderboard", Path(paths["html"]).read_text())
            json.loads(Path(paths["json"]).read_text())


if __name__ == "__main__":
    unittest.main()
