"""Runtime policy configuration for benchmark thresholds, weights, and scoring specs."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from metrics import MetricSpec


@dataclass(slots=True)
class MetricCatalogPolicy:
    technical: List[MetricSpec] = field(default_factory=list)
    impact: List[MetricSpec] = field(default_factory=list)
    continuity: List[MetricSpec] = field(default_factory=list)
    generalization: List[MetricSpec] = field(default_factory=list)
    governance: List[MetricSpec] = field(default_factory=list)


@dataclass(slots=True)
class GeneralizationPolicy:
    cross_organization_step: float = 25.0
    cross_region_step: float = 33.33
    cross_language_step: float = 33.33
    cross_model_step: float = 33.33


@dataclass(slots=True)
class EvidencePolicy:
    source_reliability: Dict[str, float] = field(
        default_factory=lambda: {
            "SYSTEM_LOGS": 100.0,
            "MANUAL_LOGS": 80.0,
            "SURVEY": 70.0,
            "INTERVIEW": 60.0,
            "ESTIMATE": 45.0,
        }
    )
    source_weight: float = 0.25
    time_delta_weight: float = 0.25
    error_delta_weight: float = 0.20
    confidence_delta_weight: float = 0.10
    continuity_delta_weight: float = 0.10
    sample_size_weight: float = 0.05
    duration_weight: float = 0.05
    baseline_score: float = 50.0
    time_delta_multiplier: float = 0.5
    error_delta_multiplier: float = 0.5
    sample_size_factor: float = 8.0
    sample_size_low_threshold: int = 5
    duration_target_weeks: float = 12.0
    duration_short_threshold_weeks: float = 4.0
    low_trust_sources: List[str] = field(default_factory=lambda: ["INTERVIEW", "ESTIMATE"])


@dataclass(slots=True)
class PhaseGatePolicy:
    alpha_min_impact: float = 55.0
    beta_min_org_count: int = 2
    beta_min_technical: float = 75.0
    beta_min_governance: float = 80.0
    labs_max_maintenance_complexity: float = 40.0
    labs_min_adoption: float = 70.0
    labs_min_generalization: float = 80.0
    labs_min_continuity: float = 75.0
    labs_min_governance: float = 85.0
    labs_min_repeated_success_count: int = 3


@dataclass(slots=True)
class GovernanceGatePolicy:
    critical_max_retention_days: int = 30
    high_sensitivity_min_governance_score: float = 75.0


@dataclass(slots=True)
class ClaimConfidencePolicy:
    evidence_weight: float = 0.7
    governance_weight: float = 0.3
    high_min: float = 80.0
    moderate_min: float = 60.0
    low_min: float = 40.0
    blocked_label: str = "VERY LOW"
    high_label: str = "HIGH"
    moderate_label: str = "MODERATE"
    low_label: str = "LOW"
    very_low_label: str = "VERY LOW"


@dataclass(slots=True)
class OverallScoreWeights:
    technical: float = 0.22
    impact: float = 0.18
    evidence: float = 0.12
    continuity: float = 0.17
    generalization: float = 0.13
    governance: float = 0.18


@dataclass(slots=True)
class LabsCandidatePolicy:
    min_generalization_score: float = 80.0
    min_deployments: int = 3
    min_organizations: int = 2


@dataclass(slots=True)
class StagePolicy:
    rank: Dict[str, int] = field(
        default_factory=lambda: {
            "NEEDS ASSESSMENT": 0,
            "AI SETUP": 1,
            "WORKFLOW AUTOMATION": 2,
            "CUSTOM BUILD": 3,
        }
    )
    max_phase: Dict[str, str] = field(
        default_factory=lambda: {
            "NEEDS ASSESSMENT": "NOT READY",
            "AI SETUP": "READY FOR ALPHA",
            "WORKFLOW AUTOMATION": "READY FOR BETA",
            "CUSTOM BUILD": "READY FOR LABS",
        }
    )
    phase_order: Dict[str, int] = field(
        default_factory=lambda: {
            "NOT READY": 0,
            "READY FOR ALPHA": 1,
            "READY FOR BETA": 2,
            "READY FOR LABS": 3,
        }
    )


@dataclass(slots=True)
class BenchmarkPolicy:
    metric_specs: MetricCatalogPolicy
    generalization: GeneralizationPolicy = field(default_factory=GeneralizationPolicy)
    evidence: EvidencePolicy = field(default_factory=EvidencePolicy)
    phase_gate: PhaseGatePolicy = field(default_factory=PhaseGatePolicy)
    governance_gate: GovernanceGatePolicy = field(default_factory=GovernanceGatePolicy)
    claim_confidence: ClaimConfidencePolicy = field(default_factory=ClaimConfidencePolicy)
    overall_weights: OverallScoreWeights = field(default_factory=OverallScoreWeights)
    labs_candidate: LabsCandidatePolicy = field(default_factory=LabsCandidatePolicy)
    stage: StagePolicy = field(default_factory=StagePolicy)


def _default_metric_specs() -> MetricCatalogPolicy:
    technical = [
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

    impact = [
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

    continuity = [
        MetricSpec("volunteer_onboarding_time_hours", "Volunteer Onboarding", 1.2, 1, 80, False, False),
        MetricSpec("documentation_quality", "Documentation", 1.2, 0, 100, True, True),
        MetricSpec("maintenance_complexity", "Maintenance Complexity", 1.0, 0, 100, False, False),
        MetricSpec("dependency_count", "Dependency Count", 0.6, 1, 60, False, False),
        MetricSpec("workflow_simplicity", "Workflow Simplicity", 0.8, 0, 100, True, False),
        MetricSpec("manual_intervention_frequency", "Manual Intervention", 1.0, 0, 100, False, False),
        MetricSpec("retention_rate", "Retention", 1.0, 0, 100, True, False),
        MetricSpec("ease_of_updating", "Ease of Updating", 1.0, 0, 100, True, False),
    ]

    generalization = [
        MetricSpec("cross_organization_success", "Cross-Organization", 1.2, 0, 100, True, False),
        MetricSpec("cross_dataset_robustness", "Cross-Dataset", 1.0, 0, 100, True, False),
        MetricSpec("cross_region_robustness", "Cross-Region", 1.0, 0, 100, True, False),
        MetricSpec("cross_language_robustness", "Cross-Language", 1.0, 0, 100, True, False),
        MetricSpec("cross_model_robustness", "Cross-Model", 1.0, 0, 100, True, False),
        MetricSpec("cross_deployment_consistency", "Cross-Deployment Consistency", 1.2, 0, 100, True, False),
    ]

    governance = [
        MetricSpec("privacy_risk", "Privacy Risk", 1.2, 0, 100, False, True),
        MetricSpec("pii_handling_score", "PII Handling", 1.2, 0, 100, True, True),
        MetricSpec("human_oversight", "Human Oversight", 1.0, 0, 100, True, True),
        MetricSpec("explainability", "Explainability", 1.0, 0, 100, True, False),
        MetricSpec("auditability", "Auditability", 1.0, 0, 100, True, True),
        MetricSpec("legal_compliance", "Legal Compliance", 1.2, 0, 100, True, True),
        MetricSpec("failure_severity", "Failure Severity", 1.0, 0, 100, False, True),
        MetricSpec("security_posture", "Security", 1.2, 0, 100, True, True),
    ]

    return MetricCatalogPolicy(
        technical=technical,
        impact=impact,
        continuity=continuity,
        generalization=generalization,
        governance=governance,
    )


def default_policy() -> BenchmarkPolicy:
    return BenchmarkPolicy(metric_specs=_default_metric_specs())


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _set_nested(raw: Dict[str, Any], path: List[str], value: Any) -> None:
    cursor = raw
    for key in path[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[path[-1]] = value


def _apply_env_overrides(raw: Dict[str, Any]) -> Dict[str, Any]:
    env_map = {
        "GMBENCH_PHASE_ALPHA_MIN_IMPACT": (["phase_gate", "alpha_min_impact"], float),
        "GMBENCH_PHASE_BETA_MIN_TECHNICAL": (["phase_gate", "beta_min_technical"], float),
        "GMBENCH_PHASE_BETA_MIN_GOVERNANCE": (["phase_gate", "beta_min_governance"], float),
        "GMBENCH_PHASE_LABS_MIN_GOVERNANCE": (["phase_gate", "labs_min_governance"], float),
        "GMBENCH_GOV_HIGH_SENS_MIN_SCORE": (["governance_gate", "high_sensitivity_min_governance_score"], float),
        "GMBENCH_GOV_CRITICAL_MAX_RETENTION_DAYS": (["governance_gate", "critical_max_retention_days"], int),
        "GMBENCH_LABS_MIN_GENERALIZATION": (["labs_candidate", "min_generalization_score"], float),
    }

    for var_name, (path, caster) in env_map.items():
        value = os.getenv(var_name)
        if value is None:
            continue
        _set_nested(raw, path, caster(value))

    return raw


def _metric_specs_from_dict(items: List[Dict[str, Any]]) -> List[MetricSpec]:
    return [MetricSpec(**item) for item in items]


def _policy_from_dict(raw: Dict[str, Any]) -> BenchmarkPolicy:
    metric_specs_raw = raw["metric_specs"]
    metric_specs = MetricCatalogPolicy(
        technical=_metric_specs_from_dict(metric_specs_raw["technical"]),
        impact=_metric_specs_from_dict(metric_specs_raw["impact"]),
        continuity=_metric_specs_from_dict(metric_specs_raw["continuity"]),
        generalization=_metric_specs_from_dict(metric_specs_raw["generalization"]),
        governance=_metric_specs_from_dict(metric_specs_raw["governance"]),
    )

    return BenchmarkPolicy(
        metric_specs=metric_specs,
        generalization=GeneralizationPolicy(**raw["generalization"]),
        evidence=EvidencePolicy(**raw["evidence"]),
        phase_gate=PhaseGatePolicy(**raw["phase_gate"]),
        governance_gate=GovernanceGatePolicy(**raw["governance_gate"]),
        claim_confidence=ClaimConfidencePolicy(**raw["claim_confidence"]),
        overall_weights=OverallScoreWeights(**raw["overall_weights"]),
        labs_candidate=LabsCandidatePolicy(**raw["labs_candidate"]),
        stage=StagePolicy(**raw["stage"]),
    )


def load_policy(config_path: str | None = None) -> BenchmarkPolicy:
    base = default_policy()
    raw = asdict(base)

    policy_path = config_path or os.getenv("GMBENCH_POLICY_FILE")
    if policy_path:
        with open(policy_path, "r", encoding="utf-8") as handle:
            overrides = json.load(handle)
        raw = _deep_merge(raw, overrides)

    raw = _apply_env_overrides(raw)
    return _policy_from_dict(raw)
