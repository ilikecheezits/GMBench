"""Top-level benchmark orchestrator with phase-gate logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from deployment import DataSensitivityTier, Deployment, DeploymentStage
from evaluators import (
    BasePillarEvaluator,
    GeneralizationEvaluator,
    default_evaluators,
    default_generalization_specs,
)
from metrics import PillarEvaluation
from reports import BenchmarkReport, ComparisonReport, GeneralizationReport, PhaseGateResult
from utils import weighted_average


@dataclass(slots=True)
class PhaseGateEvaluator:
    """Implements GMP Alpha/Beta/Labs graduation requirements."""

    @staticmethod
    def _stage_rank(stage: DeploymentStage) -> int:
        rank_map = {
            DeploymentStage.NEEDS_ASSESSMENT: 0,
            DeploymentStage.AI_SETUP: 1,
            DeploymentStage.WORKFLOW_AUTOMATION: 2,
            DeploymentStage.CUSTOM_BUILD: 3,
        }
        return rank_map[stage]

    def _cap_result_by_stage(self, stage: DeploymentStage, result: PhaseGateResult) -> PhaseGateResult:
        # Stage-aware caps prevent overclaiming maturity before deployment readiness.
        if self._stage_rank(stage) == 0:
            return PhaseGateResult.NOT_READY
        if self._stage_rank(stage) == 1 and result in {
            PhaseGateResult.READY_FOR_BETA,
            PhaseGateResult.READY_FOR_LABS,
        }:
            return PhaseGateResult.READY_FOR_ALPHA
        if self._stage_rank(stage) == 2 and result == PhaseGateResult.READY_FOR_LABS:
            return PhaseGateResult.READY_FOR_BETA
        return result

    def evaluate(self, report: BenchmarkReport, cohort_size: int, org_count: int) -> PhaseGateResult:
        deployment = report.deployment
        evidence_org_count = max(org_count, deployment.flags.validated_organizations)

        alpha_pass = (
            report.impact_score >= 55
            and not deployment.flags.critical_technical_failure
            and not deployment.flags.critical_governance_failure
            and deployment.flags.case_study_exists
        )

        beta_pass = (
            alpha_pass
            and evidence_org_count >= 2
            and deployment.flags.volunteer_onboarding_succeeded
            and deployment.flags.documentation_exists
            and report.technical_score >= 75
            and report.governance_score >= 80
        )

        maintenance_low = (
            deployment.continuity.maintenance_complexity is not None
            and deployment.continuity.maintenance_complexity <= 40
        )

        adoption_high = (
            deployment.impact.adoption_rate is not None
            and deployment.impact.adoption_rate >= 70
        )

        labs_pass = (
            beta_pass
            and report.generalization_score >= 80
            and adoption_high
            and maintenance_low
            and report.continuity_score >= 75
            and report.governance_score >= 85
            and deployment.flags.repeated_success_count >= 3
        )

        if labs_pass:
            return self._cap_result_by_stage(deployment.stage, PhaseGateResult.READY_FOR_LABS)
        if beta_pass:
            return self._cap_result_by_stage(deployment.stage, PhaseGateResult.READY_FOR_BETA)
        if alpha_pass:
            return self._cap_result_by_stage(deployment.stage, PhaseGateResult.READY_FOR_ALPHA)
        return self._cap_result_by_stage(deployment.stage, PhaseGateResult.NOT_READY)


@dataclass(slots=True)
class GoodModelBenchmark:
    """Core framework API for GMP deployment benchmarking."""

    evaluators: List[BasePillarEvaluator] = field(default_factory=default_evaluators)
    generalization_evaluator: GeneralizationEvaluator = field(
        default_factory=lambda: GeneralizationEvaluator(default_generalization_specs())
    )
    phase_gate_evaluator: PhaseGateEvaluator = field(default_factory=PhaseGateEvaluator)

    # Internal history enables cohort analysis and workflow generalization.
    _history: List[BenchmarkReport] = field(default_factory=list, init=False, repr=False)

    def _governance_gate_reasons(self, deployment: Deployment, governance_score: float) -> List[str]:
        profile = deployment.governance_profile
        reasons: List[str] = []

        if profile.is_excluded_use_case:
            reason = profile.exclusion_reason or "Deployment marked as excluded use case."
            reasons.append(reason)

        if (
            profile.data_sensitivity in {DataSensitivityTier.HIGH, DataSensitivityTier.CRITICAL}
            and not profile.has_human_in_the_loop
        ):
            reasons.append("High-sensitivity deployment requires human-in-the-loop review.")

        if profile.pii_stored and not profile.has_dpa_or_contract_controls:
            reasons.append("PII storage requires DPA or equivalent contract controls.")

        if profile.serves_vulnerable_population and not profile.has_incident_response_plan:
            reasons.append("Vulnerable-population workflows require an incident response plan.")

        if (
            profile.data_sensitivity == DataSensitivityTier.CRITICAL
            and profile.external_model_data_retention_days is not None
            and profile.external_model_data_retention_days > 30
        ):
            reasons.append("Critical data tier requires short external model retention windows.")

        if profile.data_sensitivity in {DataSensitivityTier.HIGH, DataSensitivityTier.CRITICAL} and governance_score < 75:
            reasons.append("Governance score below minimum threshold for high-sensitivity deployment.")

        return reasons

    @staticmethod
    def _claim_confidence(evidence_score: float, governance_score: float, blocked: bool) -> str:
        if blocked:
            return "VERY LOW"

        confidence = weighted_average(
            [
                (evidence_score, 0.7),
                (governance_score, 0.3),
            ]
        )
        if confidence >= 80:
            return "HIGH"
        if confidence >= 60:
            return "MODERATE"
        if confidence >= 40:
            return "LOW"
        return "VERY LOW"

    def run(self, deployment: Deployment) -> BenchmarkReport:
        """Run benchmark for a single deployment."""
        pillar_map: Dict[str, PillarEvaluation] = {}

        for evaluator in self.evaluators:
            result = evaluator.evaluate(deployment)
            pillar_map[result.pillar_name] = result

        generalization_eval = self.generalization_evaluator.evaluate(deployment)
        pillar_map["Generalization"] = PillarEvaluation(
            pillar_name="Generalization",
            score=generalization_eval.score,
            metric_scores=generalization_eval.metric_scores,
            missing_required_metrics=generalization_eval.missing_required_metrics,
            notes=generalization_eval.notes,
        )

        technical_score = pillar_map["Technical Reliability"].score
        impact_score = pillar_map["Operational Impact"].score
        evidence_score = pillar_map["Evidence and Capacity Delta"].score
        continuity_score = pillar_map["Continuity"].score
        governance_score = pillar_map["Risk and Governance"].score
        generalization_score = pillar_map["Generalization"].score
        capacity_delta_score = evidence_score
        governance_gate_reasons = self._governance_gate_reasons(deployment, governance_score)
        blocked_by_governance = bool(governance_gate_reasons)

        overall_score = round(
            weighted_average(
                [
                    (technical_score, 0.22),
                    (impact_score, 0.18),
                    (evidence_score, 0.12),
                    (continuity_score, 0.17),
                    (generalization_score, 0.13),
                    (governance_score, 0.18),
                ]
            ),
            2,
        )

        workflow_reports = [
            rep for rep in self._history if rep.deployment.workflow == deployment.workflow
        ]
        cohort_size = len(workflow_reports) + 1
        org_count = len({rep.deployment.organization for rep in workflow_reports} | {deployment.organization})

        preliminary = BenchmarkReport(
            deployment=deployment,
            technical_score=technical_score,
            impact_score=impact_score,
            evidence_score=evidence_score,
            capacity_delta_score=capacity_delta_score,
            continuity_score=continuity_score,
            generalization_score=generalization_score,
            governance_score=governance_score,
            overall_score=overall_score,
            phase_result=PhaseGateResult.NOT_READY,
            pillar_evaluations=pillar_map,
            claim_confidence=self._claim_confidence(
                evidence_score=evidence_score,
                governance_score=governance_score,
                blocked=blocked_by_governance,
            ),
            blocked_by_governance=blocked_by_governance,
            block_reasons=governance_gate_reasons,
        )

        if blocked_by_governance:
            preliminary.phase_result = PhaseGateResult.NOT_READY
        else:
            preliminary.phase_result = self.phase_gate_evaluator.evaluate(
                preliminary, cohort_size=cohort_size, org_count=org_count
            )

        self._history.append(preliminary)
        return preliminary

    def compare(self, deployments: Sequence[Deployment]) -> ComparisonReport:
        """Benchmark and compare multiple deployments."""
        reports = [self.run(dep) for dep in deployments]
        reports.sort(key=lambda rep: rep.overall_score, reverse=True)
        return ComparisonReport(reports=reports)

    def generalize(
        self,
        workflow_name: str,
        deployments: Optional[Sequence[Deployment]] = None,
    ) -> GeneralizationReport:
        """Aggregate workflow-level generalization for one pattern."""
        if deployments:
            new_reports = [self.run(dep) for dep in deployments if dep.workflow == workflow_name]
            _ = new_reports

        matching_reports = [
            rep for rep in self._history if rep.deployment.workflow == workflow_name
        ]
        matching_deployments = [rep.deployment for rep in matching_reports]
        pillar_eval = self.generalization_evaluator.evaluate_group(matching_deployments)

        organizations = sorted({dep.organization for dep in matching_deployments})
        candidate_for_labs = (
            pillar_eval.score >= 80
            and len(matching_deployments) >= 3
            and len(organizations) >= 2
        )

        return GeneralizationReport(
            workflow_name=workflow_name,
            deployment_count=len(matching_deployments),
            represented_organizations=organizations,
            generalization_score=pillar_eval.score,
            candidate_for_labs=candidate_for_labs,
            pillar_evaluation=pillar_eval,
        )

    @property
    def history(self) -> List[BenchmarkReport]:
        """Read-only snapshot of reports generated by this benchmark instance."""
        return list(self._history)
