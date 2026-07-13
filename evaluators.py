"""Pillar evaluators and configurable scoring specs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from deployment import Deployment
from metrics import MetricSpec, PillarEvaluation, evaluate_metrics
from policy import BenchmarkPolicy, EvidencePolicy, GeneralizationPolicy, default_policy
from utils import clamp, mean, weighted_average


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

    def __init__(
        self,
        specs: List[MetricSpec],
        policy: GeneralizationPolicy | None = None,
    ) -> None:
        self.specs = specs
        self.policy = policy or default_policy().generalization

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
            "Cross-Organization": min(100.0, len(organizations) * self.policy.cross_organization_step),
            "Cross-Region": min(100.0, len(regions) * self.policy.cross_region_step) if regions else 0.0,
            "Cross-Language": min(100.0, len(languages) * self.policy.cross_language_step) if languages else 0.0,
            "Cross-Model": min(100.0, len(models) * self.policy.cross_model_step) if models else 0.0,
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


class EvidenceEvaluator(BasePillarEvaluator):
    """Scores evidence quality and observed operational capacity improvements."""

    pillar_name = "Evidence and Capacity Delta"

    def __init__(self, policy: EvidencePolicy | None = None) -> None:
        self.policy = policy or default_policy().evidence

    def evaluate(self, deployment: Deployment) -> PillarEvaluation:
        evidence = deployment.capacity_evidence
        metric_scores = {}
        notes: List[str] = []

        source_reliability = self.policy.source_reliability
        source_score = source_reliability[evidence.evidence_source.value]
        metric_scores["Evidence Source Reliability"] = source_score

        weighted_scores = [(source_score, self.policy.source_weight)]

        if (
            evidence.baseline_task_minutes is not None
            and evidence.followup_task_minutes is not None
            and evidence.baseline_task_minutes > 0
        ):
            time_saved_pct = (
                (evidence.baseline_task_minutes - evidence.followup_task_minutes)
                / evidence.baseline_task_minutes
            ) * 100.0
            time_saved_score = clamp(
                self.policy.baseline_score + (time_saved_pct * self.policy.time_delta_multiplier),
                0.0,
                100.0,
            )
            metric_scores["Task Time Delta"] = round(time_saved_score, 2)
            weighted_scores.append((time_saved_score, self.policy.time_delta_weight))
        else:
            notes.append("Missing baseline/follow-up task time for capacity delta.")

        if (
            evidence.baseline_error_rate is not None
            and evidence.followup_error_rate is not None
            and evidence.baseline_error_rate >= 0
            and evidence.followup_error_rate >= 0
        ):
            baseline = max(evidence.baseline_error_rate, 1e-9)
            error_delta_pct = ((baseline - evidence.followup_error_rate) / baseline) * 100.0
            error_score = clamp(
                self.policy.baseline_score + (error_delta_pct * self.policy.error_delta_multiplier),
                0.0,
                100.0,
            )
            metric_scores["Error Rate Delta"] = round(error_score, 2)
            weighted_scores.append((error_score, self.policy.error_delta_weight))

        if evidence.staff_confidence_delta is not None:
            confidence_score = clamp(self.policy.baseline_score + evidence.staff_confidence_delta, 0.0, 100.0)
            metric_scores["Staff Confidence Delta"] = round(confidence_score, 2)
            weighted_scores.append((confidence_score, self.policy.confidence_delta_weight))

        if evidence.continuity_improvement is not None:
            continuity_score = clamp(float(evidence.continuity_improvement), 0.0, 100.0)
            metric_scores["Continuity Improvement"] = round(continuity_score, 2)
            weighted_scores.append((continuity_score, self.policy.continuity_delta_weight))

        if evidence.sample_size is not None:
            sample_score = clamp(evidence.sample_size * self.policy.sample_size_factor, 0.0, 100.0)
            metric_scores["Sample Size Confidence"] = round(sample_score, 2)
            weighted_scores.append((sample_score, self.policy.sample_size_weight))
            if evidence.sample_size < self.policy.sample_size_low_threshold:
                notes.append("Small sample size lowers confidence in impact claims.")
        else:
            notes.append("Sample size not provided for impact evidence.")

        if evidence.measured_weeks is not None:
            duration_score = clamp(
                (evidence.measured_weeks / self.policy.duration_target_weeks) * 100.0,
                0.0,
                100.0,
            )
            metric_scores["Measurement Duration"] = round(duration_score, 2)
            weighted_scores.append((duration_score, self.policy.duration_weight))
            if evidence.measured_weeks < self.policy.duration_short_threshold_weeks:
                notes.append("Measurement period is short; verify durability over time.")
        else:
            notes.append("Measurement duration not provided.")

        if evidence.evidence_source.value in set(self.policy.low_trust_sources):
            notes.append("Evidence quality is low-trust without logs or structured records.")

        overall = round(weighted_average(weighted_scores), 2)
        return PillarEvaluation(
            pillar_name=self.pillar_name,
            score=overall,
            metric_scores=metric_scores,
            notes=notes,
        )


def default_technical_specs() -> List[MetricSpec]:
    return default_policy().metric_specs.technical


def default_impact_specs() -> List[MetricSpec]:
    return default_policy().metric_specs.impact


def default_continuity_specs() -> List[MetricSpec]:
    return default_policy().metric_specs.continuity


def default_generalization_specs() -> List[MetricSpec]:
    return default_policy().metric_specs.generalization


def default_governance_specs() -> List[MetricSpec]:
    return default_policy().metric_specs.governance


def default_evaluators(policy: BenchmarkPolicy | None = None) -> List[BasePillarEvaluator]:
    """Create default pillar evaluators used by GoodModelBenchmark."""
    active = policy or default_policy()
    return [
        SpecBasedEvaluator("Technical Reliability", "technical", list(active.metric_specs.technical)),
        SpecBasedEvaluator("Operational Impact", "impact", list(active.metric_specs.impact)),
        EvidenceEvaluator(active.evidence),
        SpecBasedEvaluator("Continuity", "continuity", list(active.metric_specs.continuity)),
        SpecBasedEvaluator("Risk and Governance", "governance", list(active.metric_specs.governance)),
    ]
