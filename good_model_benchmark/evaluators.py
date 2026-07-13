"""Pillar evaluators and configurable scoring specs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from .deployment import Deployment
from .metrics import MetricSpec, PillarEvaluation, evaluate_metrics
from .utils import mean


class BasePillarEvaluator(ABC):
    """Abstract pillar evaluator contract for extensible benchmark logic."""

    pillar_name: str

    @abstractmethod
    def evaluate(self, deployment: Deployment) -> PillarEvaluation:
        """Evaluate a deployment and return a pillar-level result."""


@dataclass(slots=True)
class SpecBasedEvaluator(BasePillarEvaluator):
    """Evaluator that scores one deployment attribute using metric specs."""

    pillar_name: str
    deployment_attr: str
    specs: List[MetricSpec]

    def evaluate(self, deployment: Deployment) -> PillarEvaluation:
        metric_obj = getattr(deployment, self.deployment_attr)
        score, metric_scores, missing_required = evaluate_metrics(metric_obj, self.specs)
        notes: List[str] = []
        if missing_required:
            notes.append(
                "Missing required metrics: " + ", ".join(missing_required)
            )
        return PillarEvaluation(
            pillar_name=self.pillar_name,
            score=score,
            metric_scores=metric_scores,
            missing_required_metrics=missing_required,
            notes=notes,
        )


class GeneralizationEvaluator(BasePillarEvaluator):
    """Evaluates Pillar 4 for single deployments and multi-deployment cohorts."""

    pillar_name = "Generalizability"

    def __init__(self, specs: List[MetricSpec]) -> None:
        self.specs = specs

    def evaluate(self, deployment: Deployment) -> PillarEvaluation:
        score, metric_scores, missing_required = evaluate_metrics(
            deployment.generalization, self.specs
        )
        notes: List[str] = []
        if not metric_scores:
            notes.append("No deployment-level generalization signals were provided.")
        return PillarEvaluation(
            pillar_name=self.pillar_name,
            score=score,
            metric_scores=metric_scores,
            missing_required_metrics=missing_required,
            notes=notes,
        )

    def evaluate_group(self, deployments: Sequence[Deployment]) -> PillarEvaluation:
        """Compute one generalized workflow score across multiple deployments."""
        if not deployments:
            return PillarEvaluation(
                pillar_name=self.pillar_name,
                score=0.0,
                notes=["No deployments available for group generalization."],
            )

        per_deployment_scores = [self.evaluate(dep).score for dep in deployments]

        organizations = {dep.organization for dep in deployments}
        regions = {dep.region for dep in deployments if dep.region}
        languages = {dep.language for dep in deployments if dep.language}
        models = {dep.model_name for dep in deployments if dep.model_name}

        # Diversity heuristics increase confidence that the workflow generalizes.
        diversity_signals = {
            "Cross-Organization": min(100.0, len(organizations) * 25.0),
            "Cross-Region": min(100.0, len(regions) * 33.33) if regions else 0.0,
            "Cross-Language": min(100.0, len(languages) * 33.33) if languages else 0.0,
            "Cross-Model": min(100.0, len(models) * 33.33) if models else 0.0,
        }

        aggregate_score = mean(per_deployment_scores + list(diversity_signals.values()))
        details = {f"Deployment {idx + 1}": round(score, 2) for idx, score in enumerate(per_deployment_scores)}
        details.update({key: round(val, 2) for key, val in diversity_signals.items()})

        return PillarEvaluation(
            pillar_name=self.pillar_name,
            score=round(aggregate_score, 2),
            metric_scores=details,
            notes=[
                f"Generalization computed from {len(deployments)} deployment(s).",
                f"Organizations represented: {len(organizations)}",
            ],
        )


def default_technical_specs() -> List[MetricSpec]:
    return [
        MetricSpec("schema_validity", "Schema Validity", 1.0, 0, 100, True, True),
        MetricSpec("structured_output_success", "Structured Output Success", 1.0, 0, 100, True, True),
        MetricSpec("hallucination_rate", "Hallucination Rate", 1.0, 0, 100, False, False),
        MetricSpec("missing_facts_rate", "Missing Facts", 0.8, 0, 100, False, False),
        MetricSpec("factual_correctness", "Factual Correctness", 1.2, 0, 100, True, True),
        MetricSpec("latency_ms", "Latency", 0.8, 100, 5000, False, False),
        MetricSpec("api_failure_rate", "API Failures", 1.0, 0, 100, False, False),
        MetricSpec("retry_rate", "Retry Rate", 0.8, 0, 100, False, False),
        MetricSpec("robustness_input_style", "Input Robustness", 1.0, 0, 100, True, False),
        MetricSpec("security_checks_pass_rate", "Security Checks", 1.2, 0, 100, True, True),
        MetricSpec("pii_leakage_rate", "PII Leakage", 1.2, 0, 100, False, True),
    ]


def default_impact_specs() -> List[MetricSpec]:
    return [
        MetricSpec("hours_saved_per_week", "Hours Saved", 1.0, 0, 25, True, False),
        MetricSpec("cost_savings_usd_per_month", "Cost Savings", 0.8, 0, 5000, True, False),
        MetricSpec("roi", "ROI", 1.0, -50, 250, True, False),
        MetricSpec("staff_workload_reduction", "Staff Workload Reduction", 1.0, 0, 50, True, False),
        MetricSpec("volunteer_workload_reduction", "Volunteer Workload Reduction", 1.0, 0, 50, True, False),
        MetricSpec("error_reduction", "Error Reduction", 0.8, 0, 50, True, False),
        MetricSpec("response_time_improvement", "Response Time Improvement", 0.8, 0, 60, True, False),
        MetricSpec("client_throughput_improvement", "Client Throughput", 0.8, 0, 40, True, False),
        MetricSpec("adoption_rate", "Adoption Rate", 1.2, 0, 100, True, False),
        MetricSpec("satisfaction_score", "Satisfaction", 1.0, 0, 100, True, False),
    ]


def default_continuity_specs() -> List[MetricSpec]:
    return [
        MetricSpec("volunteer_onboarding_time_hours", "Volunteer Onboarding", 1.2, 1, 80, False, False),
        MetricSpec("documentation_quality", "Documentation", 1.2, 0, 100, True, True),
        MetricSpec("maintenance_complexity", "Maintenance Complexity", 1.0, 0, 100, False, False),
        MetricSpec("dependency_count", "Dependency Count", 0.6, 1, 60, False, False),
        MetricSpec("workflow_simplicity", "Workflow Simplicity", 0.8, 0, 100, True, False),
        MetricSpec("manual_intervention_frequency", "Manual Intervention", 1.0, 0, 100, False, False),
        MetricSpec("retention_rate", "Retention", 1.0, 0, 100, True, False),
        MetricSpec("ease_of_updating", "Ease of Updating", 1.0, 0, 100, True, False),
    ]


def default_generalization_specs() -> List[MetricSpec]:
    return [
        MetricSpec("cross_organization_success", "Cross-Organization", 1.2, 0, 100, True, False),
        MetricSpec("cross_dataset_robustness", "Cross-Dataset", 1.0, 0, 100, True, False),
        MetricSpec("cross_region_robustness", "Cross-Region", 1.0, 0, 100, True, False),
        MetricSpec("cross_language_robustness", "Cross-Language", 1.0, 0, 100, True, False),
        MetricSpec("cross_model_robustness", "Cross-Model", 1.0, 0, 100, True, False),
        MetricSpec("cross_deployment_consistency", "Cross-Deployment Consistency", 1.2, 0, 100, True, False),
    ]


def default_governance_specs() -> List[MetricSpec]:
    return [
        MetricSpec("privacy_risk", "Privacy Risk", 1.2, 0, 100, False, True),
        MetricSpec("pii_handling_score", "PII Handling", 1.2, 0, 100, True, True),
        MetricSpec("human_oversight", "Human Oversight", 1.0, 0, 100, True, True),
        MetricSpec("explainability", "Explainability", 1.0, 0, 100, True, False),
        MetricSpec("auditability", "Auditability", 1.0, 0, 100, True, True),
        MetricSpec("legal_compliance", "Legal Compliance", 1.2, 0, 100, True, True),
        MetricSpec("failure_severity", "Failure Severity", 1.0, 0, 100, False, True),
        MetricSpec("security_posture", "Security", 1.2, 0, 100, True, True),
    ]


def default_evaluators() -> List[BasePillarEvaluator]:
    """Create default pillar evaluators used by GoodModelBenchmark."""
    return [
        SpecBasedEvaluator("Technical Reliability", "technical", default_technical_specs()),
        SpecBasedEvaluator("Operational Impact", "impact", default_impact_specs()),
        SpecBasedEvaluator("Continuity", "continuity", default_continuity_specs()),
        SpecBasedEvaluator("Risk and Governance", "governance", default_governance_specs()),
    ]
