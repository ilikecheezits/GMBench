from __future__ import annotations

import unittest

from benchmark import GoodModelBenchmark
from deployment import (
    CapacityEvidence,
    ContinuityMetrics,
    DataSensitivityTier,
    Deployment,
    DeploymentFlags,
    DeploymentStage,
    EvidenceSource,
    GeneralizationSignals,
    GovernanceMetrics,
    GovernanceProfile,
    OperationalImpactMetrics,
    TechnicalMetrics,
)
from reports import PhaseGateResult


def make_deployment(
    *,
    stage: DeploymentStage = DeploymentStage.CUSTOM_BUILD,
    evidence_source: EvidenceSource = EvidenceSource.SYSTEM_LOGS,
    data_sensitivity: DataSensitivityTier = DataSensitivityTier.MODERATE,
    has_human_in_the_loop: bool = True,
    is_excluded_use_case: bool = False,
) -> Deployment:
    return Deployment(
        organization="Test Org",
        workflow="AI Intake Summarization",
        deployment_type="Food Pantry",
        technical=TechnicalMetrics(
            schema_validity=98,
            structured_output_success=96,
            hallucination_rate=3,
            missing_facts_rate=4,
            factual_correctness=95,
            latency_ms=750,
            api_failure_rate=1,
            retry_rate=3,
            robustness_input_style=91,
            security_checks_pass_rate=99,
            pii_leakage_rate=1,
        ),
        impact=OperationalImpactMetrics(
            hours_saved_per_week=24,
            cost_savings_usd_per_month=4100,
            roi=180,
            staff_workload_reduction=42,
            volunteer_workload_reduction=34,
            error_reduction=28,
            response_time_improvement=45,
            client_throughput_improvement=22,
            adoption_rate=84,
            satisfaction_score=90,
        ),
        continuity=ContinuityMetrics(
            volunteer_onboarding_time_hours=8,
            documentation_quality=93,
            maintenance_complexity=30,
            dependency_count=10,
            workflow_simplicity=85,
            manual_intervention_frequency=18,
            retention_rate=87,
            ease_of_updating=82,
        ),
        governance=GovernanceMetrics(
            privacy_risk=8,
            pii_handling_score=95,
            human_oversight=94,
            explainability=87,
            auditability=96,
            legal_compliance=98,
            failure_severity=9,
            security_posture=95,
        ),
        generalization=GeneralizationSignals(
            cross_organization_success=84,
            cross_dataset_robustness=80,
            cross_region_robustness=78,
            cross_language_robustness=72,
            cross_model_robustness=86,
            cross_deployment_consistency=85,
        ),
        flags=DeploymentFlags(
            critical_technical_failure=False,
            critical_governance_failure=False,
            case_study_exists=True,
            documentation_exists=True,
            volunteer_onboarding_succeeded=True,
            validated_organizations=3,
            repeated_success_count=4,
        ),
        stage=stage,
        capacity_evidence=CapacityEvidence(
            baseline_task_minutes=64,
            followup_task_minutes=30,
            baseline_error_rate=14,
            followup_error_rate=5,
            staff_confidence_delta=20,
            continuity_improvement=80,
            evidence_source=evidence_source,
            sample_size=20,
            measured_weeks=10,
        ),
        governance_profile=GovernanceProfile(
            data_sensitivity=data_sensitivity,
            serves_vulnerable_population=True,
            has_human_in_the_loop=has_human_in_the_loop,
            pii_stored=True,
            external_model_data_retention_days=14,
            has_incident_response_plan=True,
            has_dpa_or_contract_controls=True,
            is_excluded_use_case=is_excluded_use_case,
            exclusion_reason="Excluded for legal constraints" if is_excluded_use_case else None,
        ),
    )


class BenchmarkPhaseTests(unittest.TestCase):
    def test_stage_caps_result_to_alpha_for_ai_setup(self) -> None:
        benchmark = GoodModelBenchmark()
        deployment = make_deployment(stage=DeploymentStage.AI_SETUP)

        report = benchmark.run(deployment)

        self.assertEqual(report.phase_result, PhaseGateResult.READY_FOR_ALPHA)

    def test_governance_gate_blocks_high_sensitivity_without_human_loop(self) -> None:
        benchmark = GoodModelBenchmark()
        deployment = make_deployment(
            data_sensitivity=DataSensitivityTier.HIGH,
            has_human_in_the_loop=False,
        )

        report = benchmark.run(deployment)

        self.assertTrue(report.blocked_by_governance)
        self.assertEqual(report.phase_result, PhaseGateResult.NOT_READY)
        self.assertIn("human-in-the-loop", " ".join(report.block_reasons).lower())

    def test_excluded_use_case_blocks_even_with_strong_scores(self) -> None:
        benchmark = GoodModelBenchmark()
        deployment = make_deployment(is_excluded_use_case=True)

        report = benchmark.run(deployment)

        self.assertTrue(report.blocked_by_governance)
        self.assertEqual(report.phase_result, PhaseGateResult.NOT_READY)
        self.assertEqual(report.claim_confidence, "VERY LOW")

    def test_estimated_evidence_reduces_evidence_score(self) -> None:
        benchmark = GoodModelBenchmark()
        strong = make_deployment(evidence_source=EvidenceSource.SYSTEM_LOGS)
        weak = make_deployment(evidence_source=EvidenceSource.ESTIMATE)

        strong_report = benchmark.run(strong)
        weak_report = benchmark.run(weak)

        self.assertGreater(strong_report.evidence_score, weak_report.evidence_score)


if __name__ == "__main__":
    unittest.main()
