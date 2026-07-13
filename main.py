"""Example runner for the Good Model Benchmark framework."""

from __future__ import annotations

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
    GovernanceProfile,
    GovernanceMetrics,
    OperationalImpactMetrics,
    TechnicalMetrics,
)


def build_sample_deployments() -> list[Deployment]:
    """Create sample deployments for demonstration and quick local testing."""
    common_workflow = "AI Intake Summarization"

    dep1 = Deployment(
        organization="Hope Food Pantry",
        workflow=common_workflow,
        deployment_type="Food Pantry",
        technical=TechnicalMetrics(
            schema_validity=98,
            structured_output_success=95,
            hallucination_rate=4,
            missing_facts_rate=6,
            factual_correctness=94,
            latency_ms=850,
            api_failure_rate=2,
            retry_rate=5,
            robustness_input_style=90,
            security_checks_pass_rate=99,
            pii_leakage_rate=1,
        ),
        impact=OperationalImpactMetrics(
            hours_saved_per_week=22,
            cost_savings_usd_per_month=4200,
            roi=180,
            staff_workload_reduction=40,
            volunteer_workload_reduction=35,
            error_reduction=25,
            response_time_improvement=45,
            client_throughput_improvement=20,
            adoption_rate=82,
            satisfaction_score=88,
        ),
        continuity=ContinuityMetrics(
            volunteer_onboarding_time_hours=8,
            documentation_quality=92,
            maintenance_complexity=28,
            dependency_count=12,
            workflow_simplicity=84,
            manual_intervention_frequency=20,
            retention_rate=85,
            ease_of_updating=80,
        ),
        governance=GovernanceMetrics(
            privacy_risk=8,
            pii_handling_score=96,
            human_oversight=93,
            explainability=86,
            auditability=95,
            legal_compliance=97,
            failure_severity=10,
            security_posture=94,
        ),
        generalization=GeneralizationSignals(
            cross_organization_success=82,
            cross_dataset_robustness=78,
            cross_region_robustness=75,
            cross_language_robustness=70,
            cross_model_robustness=85,
            cross_deployment_consistency=84,
        ),
        flags=DeploymentFlags(
            critical_technical_failure=False,
            critical_governance_failure=False,
            case_study_exists=True,
            documentation_exists=True,
            volunteer_onboarding_succeeded=True,
            validated_organizations=2,
            repeated_success_count=3,
        ),
        stage=DeploymentStage.WORKFLOW_AUTOMATION,
        capacity_evidence=CapacityEvidence(
            baseline_task_minutes=60,
            followup_task_minutes=32,
            baseline_error_rate=12,
            followup_error_rate=6,
            staff_confidence_delta=18,
            continuity_improvement=78,
            evidence_source=EvidenceSource.SYSTEM_LOGS,
            sample_size=24,
            measured_weeks=10,
        ),
        governance_profile=GovernanceProfile(
            data_sensitivity=DataSensitivityTier.MODERATE,
            serves_vulnerable_population=True,
            has_human_in_the_loop=True,
            pii_stored=True,
            external_model_data_retention_days=14,
            has_incident_response_plan=True,
            has_dpa_or_contract_controls=True,
        ),
        region="US-West",
        language="English",
        model_name="GPT-5",
        dataset_name="intake-v1",
    )

    dep2 = Deployment(
        organization="Northside Pantry Network",
        workflow=common_workflow,
        deployment_type="Food Pantry",
        technical=TechnicalMetrics(
            schema_validity=95,
            structured_output_success=92,
            hallucination_rate=7,
            missing_facts_rate=8,
            factual_correctness=90,
            latency_ms=980,
            api_failure_rate=3,
            retry_rate=7,
            robustness_input_style=87,
            security_checks_pass_rate=97,
            pii_leakage_rate=2,
        ),
        impact=OperationalImpactMetrics(
            hours_saved_per_week=18,
            cost_savings_usd_per_month=3300,
            roi=140,
            staff_workload_reduction=35,
            volunteer_workload_reduction=28,
            error_reduction=22,
            response_time_improvement=38,
            client_throughput_improvement=16,
            adoption_rate=74,
            satisfaction_score=85,
        ),
        continuity=ContinuityMetrics(
            volunteer_onboarding_time_hours=10,
            documentation_quality=88,
            maintenance_complexity=35,
            dependency_count=14,
            workflow_simplicity=80,
            manual_intervention_frequency=26,
            retention_rate=80,
            ease_of_updating=76,
        ),
        governance=GovernanceMetrics(
            privacy_risk=12,
            pii_handling_score=92,
            human_oversight=90,
            explainability=80,
            auditability=92,
            legal_compliance=94,
            failure_severity=14,
            security_posture=90,
        ),
        generalization=GeneralizationSignals(
            cross_organization_success=80,
            cross_dataset_robustness=76,
            cross_region_robustness=72,
            cross_language_robustness=65,
            cross_model_robustness=82,
            cross_deployment_consistency=79,
        ),
        flags=DeploymentFlags(
            critical_technical_failure=False,
            critical_governance_failure=False,
            case_study_exists=True,
            documentation_exists=True,
            volunteer_onboarding_succeeded=True,
            validated_organizations=2,
            repeated_success_count=3,
        ),
        stage=DeploymentStage.WORKFLOW_AUTOMATION,
        capacity_evidence=CapacityEvidence(
            baseline_task_minutes=58,
            followup_task_minutes=36,
            baseline_error_rate=10,
            followup_error_rate=7,
            staff_confidence_delta=14,
            continuity_improvement=70,
            evidence_source=EvidenceSource.MANUAL_LOGS,
            sample_size=12,
            measured_weeks=8,
        ),
        governance_profile=GovernanceProfile(
            data_sensitivity=DataSensitivityTier.HIGH,
            serves_vulnerable_population=True,
            has_human_in_the_loop=True,
            pii_stored=True,
            external_model_data_retention_days=21,
            has_incident_response_plan=True,
            has_dpa_or_contract_controls=True,
        ),
        region="US-South",
        language="English",
        model_name="Claude",
        dataset_name="intake-v2",
    )

    dep3 = Deployment(
        organization="City Relief Pantry",
        workflow=common_workflow,
        deployment_type="Food Pantry",
        technical=TechnicalMetrics(
            schema_validity=93,
            structured_output_success=90,
            hallucination_rate=9,
            missing_facts_rate=10,
            factual_correctness=88,
            latency_ms=1200,
            api_failure_rate=4,
            retry_rate=8,
            robustness_input_style=84,
            security_checks_pass_rate=96,
            pii_leakage_rate=2,
        ),
        impact=OperationalImpactMetrics(
            hours_saved_per_week=16,
            cost_savings_usd_per_month=2800,
            roi=115,
            staff_workload_reduction=30,
            volunteer_workload_reduction=24,
            error_reduction=20,
            response_time_improvement=33,
            client_throughput_improvement=12,
            adoption_rate=71,
            satisfaction_score=82,
        ),
        continuity=ContinuityMetrics(
            volunteer_onboarding_time_hours=12,
            documentation_quality=84,
            maintenance_complexity=38,
            dependency_count=16,
            workflow_simplicity=77,
            manual_intervention_frequency=30,
            retention_rate=78,
            ease_of_updating=73,
        ),
        governance=GovernanceMetrics(
            privacy_risk=14,
            pii_handling_score=90,
            human_oversight=88,
            explainability=78,
            auditability=90,
            legal_compliance=92,
            failure_severity=15,
            security_posture=89,
        ),
        generalization=GeneralizationSignals(
            cross_organization_success=78,
            cross_dataset_robustness=74,
            cross_region_robustness=70,
            cross_language_robustness=62,
            cross_model_robustness=78,
            cross_deployment_consistency=76,
        ),
        flags=DeploymentFlags(
            critical_technical_failure=False,
            critical_governance_failure=False,
            case_study_exists=True,
            documentation_exists=True,
            volunteer_onboarding_succeeded=True,
            validated_organizations=2,
            repeated_success_count=3,
        ),
        stage=DeploymentStage.CUSTOM_BUILD,
        capacity_evidence=CapacityEvidence(
            baseline_task_minutes=65,
            followup_task_minutes=44,
            baseline_error_rate=14,
            followup_error_rate=11,
            staff_confidence_delta=10,
            continuity_improvement=65,
            evidence_source=EvidenceSource.SURVEY,
            sample_size=8,
            measured_weeks=5,
        ),
        governance_profile=GovernanceProfile(
            data_sensitivity=DataSensitivityTier.HIGH,
            serves_vulnerable_population=True,
            has_human_in_the_loop=True,
            pii_stored=True,
            external_model_data_retention_days=30,
            has_incident_response_plan=True,
            has_dpa_or_contract_controls=True,
        ),
        region="LATAM",
        language="Spanish",
        model_name="Gemini",
        dataset_name="intake-v2-es",
    )

    return [dep1, dep2, dep3]


def run_demo() -> None:
    benchmark = GoodModelBenchmark()
    deployments = build_sample_deployments()

    single_report = benchmark.run(deployments[0])
    print(single_report.render_text())
    print("\n")

    comparison = benchmark.compare(deployments[1:])
    print("COMPARISON SUMMARY")
    print("------------------")
    for idx, report in enumerate(comparison.reports, start=1):
        print(f"{idx}. {report.deployment.organization}: {report.overall_score:.1f} ({report.phase_result.value})")

    print("\n")
    generalized = benchmark.generalize(workflow_name="AI Intake Summarization")
    print("GENERALIZATION SUMMARY")
    print("----------------------")
    print(f"Workflow: {generalized.workflow_name}")
    print(f"Deployments: {generalized.deployment_count}")
    print(f"Organizations: {', '.join(generalized.represented_organizations)}")
    print(f"Generalization Score: {generalized.generalization_score:.1f}")
    print(f"Candidate For Labs: {'YES' if generalized.candidate_for_labs else 'NO'}")


if __name__ == "__main__":
    run_demo()
