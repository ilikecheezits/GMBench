import unittest

from inspect_benchmark.nonprofit.task_bank import validate_task_bank


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


if __name__ == "__main__":
    unittest.main()
