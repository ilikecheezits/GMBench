from __future__ import annotations

import json
import os
import tempfile
import unittest

from benchmark import GoodModelBenchmark
from deployment import DataSensitivityTier
from deployment import DeploymentStage
from test_benchmark_phases import make_deployment
from reports import PhaseGateResult


class PolicyConfigTests(unittest.TestCase):
    def test_policy_file_override_changes_phase_threshold(self) -> None:
        deployment = make_deployment(stage=DeploymentStage.CUSTOM_BUILD)

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump({"phase_gate": {"alpha_min_impact": 95}}, handle)
            path = handle.name

        try:
            benchmark = GoodModelBenchmark.from_policy_file(path)
            report = benchmark.run(deployment)
            self.assertEqual(report.phase_result, PhaseGateResult.NOT_READY)
        finally:
            os.remove(path)

    def test_env_override_changes_high_sensitivity_governance_gate(self) -> None:
        deployment = make_deployment(stage=DeploymentStage.CUSTOM_BUILD)
        deployment.governance_profile.data_sensitivity = DataSensitivityTier.HIGH

        previous = os.environ.get("GMBENCH_GOV_HIGH_SENS_MIN_SCORE")
        os.environ["GMBENCH_GOV_HIGH_SENS_MIN_SCORE"] = "99"

        try:
            benchmark = GoodModelBenchmark()
            report = benchmark.run(deployment)
            self.assertTrue(report.blocked_by_governance)
        finally:
            if previous is None:
                del os.environ["GMBENCH_GOV_HIGH_SENS_MIN_SCORE"]
            else:
                os.environ["GMBENCH_GOV_HIGH_SENS_MIN_SCORE"] = previous


if __name__ == "__main__":
    unittest.main()
