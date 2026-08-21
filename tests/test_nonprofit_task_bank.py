import unittest

from inspect_benchmark.nonprofit.objective_metrics import generic_template_penalty
from inspect_benchmark.nonprofit.task_bank import (
    build_context_variant,
    validate_task_bank,
)


class NonprofitTaskBankValidationTest(unittest.TestCase):
    def test_validate_task_bank_reports_expected_coverage(self):
        report = validate_task_bank()

        self.assertGreater(report["total_samples"], 0)
        self.assertIn("fundraising_strategy", report["task_areas"])
        self.assertIn("grant_writing", report["task_areas"])
        self.assertIn("operations_prioritization", report["task_areas"])
        self.assertIn("community_communications", report["task_areas"])
        self.assertIn("impact_measurement", report["task_areas"])
        self.assertIn("board_governance", report["task_areas"])
        self.assertIn("compliance_and_privacy", report["task_areas"])
        self.assertTrue(all(count > 0 for count in report["tasks_by_area"].values()))
        self.assertTrue(report["valid"], report["errors"])

    def test_build_context_variant_includes_org_constraints(self):
        sample = build_context_variant(
            sample_id="fund-001",
            task_area="fundraising_strategy",
            difficulty="medium",
            language="English",
            risk_level="low",
            prompt="Design a donor plan.",
            criterion="Include a plan.",
            reference_answer="Use a plan.",
            organization_name="City Food Relief",
            staff_size=12,
            budget=25000,
            major_constraints=["two staff on donor outreach", "no new CRM tools"],
        )

        self.assertIn("City Food Relief", sample.prompt)
        self.assertIn("12", sample.prompt)
        self.assertIn("no new CRM tools", sample.prompt)
        self.assertIn("budget", sample.prompt.lower())

    def test_generic_template_penalty_rejects_fluffy_nonprofit_advice(self):
        generic = (
            "This innovative and transformative approach creates strategic alignment and value creation "
            "through holistic stakeholder engagement with a robust framework."
        )
        realistic = (
            "We will prioritize a 90-day donor reactivation plan with two staff, a $25,000 budget, "
            "and a weekly review of conversion and retention metrics."
        )

        self.assertTrue(generic_template_penalty(generic))
        self.assertFalse(generic_template_penalty(realistic))


if __name__ == "__main__":
    unittest.main()
