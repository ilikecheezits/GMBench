"""Core deployment domain model and pillar metric dataclasses."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class DeploymentStage(str, Enum):
    """Lifecycle stage for a GMP deployment."""

    NEEDS_ASSESSMENT = "NEEDS ASSESSMENT"
    AI_SETUP = "AI SETUP"
    WORKFLOW_AUTOMATION = "WORKFLOW AUTOMATION"
    CUSTOM_BUILD = "CUSTOM BUILD"


class DataSensitivityTier(str, Enum):
    """Data sensitivity used for governance gating."""

    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class EvidenceSource(str, Enum):
    """How operational impact evidence was collected."""

    SYSTEM_LOGS = "SYSTEM_LOGS"
    MANUAL_LOGS = "MANUAL_LOGS"
    SURVEY = "SURVEY"
    INTERVIEW = "INTERVIEW"
    ESTIMATE = "ESTIMATE"


@dataclass(slots=True)
class TechnicalMetrics:
    """Pillar 1: Technical Reliability metrics."""

    schema_validity: Optional[float] = None
    structured_output_success: Optional[float] = None
    hallucination_rate: Optional[float] = None
    missing_facts_rate: Optional[float] = None
    factual_correctness: Optional[float] = None
    latency_ms: Optional[float] = None
    api_failure_rate: Optional[float] = None
    retry_rate: Optional[float] = None
    robustness_input_style: Optional[float] = None
    security_checks_pass_rate: Optional[float] = None
    pii_leakage_rate: Optional[float] = None


@dataclass(slots=True)
class OperationalImpactMetrics:
    """Pillar 2: Operational Impact metrics.

    Metrics are optional because nonprofit deployments vary by context and data maturity.
    """

    hours_saved_per_week: Optional[float] = None
    cost_savings_usd_per_month: Optional[float] = None
    roi: Optional[float] = None
    staff_workload_reduction: Optional[float] = None
    volunteer_workload_reduction: Optional[float] = None
    error_reduction: Optional[float] = None
    response_time_improvement: Optional[float] = None
    client_throughput_improvement: Optional[float] = None
    adoption_rate: Optional[float] = None
    satisfaction_score: Optional[float] = None


@dataclass(slots=True)
class ContinuityMetrics:
    """Pillar 3: Continuity metrics for sustainability after handoff."""

    volunteer_onboarding_time_hours: Optional[float] = None
    documentation_quality: Optional[float] = None
    maintenance_complexity: Optional[float] = None
    dependency_count: Optional[float] = None
    workflow_simplicity: Optional[float] = None
    manual_intervention_frequency: Optional[float] = None
    retention_rate: Optional[float] = None
    ease_of_updating: Optional[float] = None


@dataclass(slots=True)
class GeneralizationSignals:
    """Pillar 4 signal set used for per-deployment and cohort generalization."""

    cross_organization_success: Optional[float] = None
    cross_dataset_robustness: Optional[float] = None
    cross_region_robustness: Optional[float] = None
    cross_language_robustness: Optional[float] = None
    cross_model_robustness: Optional[float] = None
    cross_deployment_consistency: Optional[float] = None


@dataclass(slots=True)
class GovernanceMetrics:
    """Pillar 5: Risk and Governance metrics."""

    privacy_risk: Optional[float] = None
    pii_handling_score: Optional[float] = None
    human_oversight: Optional[float] = None
    explainability: Optional[float] = None
    auditability: Optional[float] = None
    legal_compliance: Optional[float] = None
    failure_severity: Optional[float] = None
    security_posture: Optional[float] = None


@dataclass(slots=True)
class DeploymentFlags:
    """Binary evidence used by GMP phase gates."""

    critical_technical_failure: bool = False
    critical_governance_failure: bool = False
    case_study_exists: bool = False
    documentation_exists: bool = False
    volunteer_onboarding_succeeded: bool = False
    validated_organizations: int = 1
    repeated_success_count: int = 1


@dataclass(slots=True)
class CapacityEvidence:
    """Evidence used to validate real operational capacity improvements."""

    baseline_task_minutes: Optional[float] = None
    followup_task_minutes: Optional[float] = None
    baseline_error_rate: Optional[float] = None
    followup_error_rate: Optional[float] = None
    staff_confidence_delta: Optional[float] = None
    continuity_improvement: Optional[float] = None
    evidence_source: EvidenceSource = EvidenceSource.ESTIMATE
    sample_size: Optional[int] = None
    measured_weeks: Optional[float] = None


@dataclass(slots=True)
class GovernanceProfile:
    """Operational governance controls beyond metric scores."""

    data_sensitivity: DataSensitivityTier = DataSensitivityTier.MODERATE
    serves_vulnerable_population: bool = False
    has_human_in_the_loop: bool = True
    pii_stored: bool = False
    external_model_data_retention_days: Optional[int] = None
    has_incident_response_plan: bool = False
    has_dpa_or_contract_controls: bool = False
    is_excluded_use_case: bool = False
    exclusion_reason: Optional[str] = None


@dataclass(slots=True)
class Deployment:
    """Represents one nonprofit AI deployment to benchmark."""

    organization: str
    workflow: str
    deployment_type: str
    technical: TechnicalMetrics
    impact: OperationalImpactMetrics
    continuity: ContinuityMetrics
    governance: GovernanceMetrics
    generalization: GeneralizationSignals = field(default_factory=GeneralizationSignals)
    flags: DeploymentFlags = field(default_factory=DeploymentFlags)
    stage: DeploymentStage = DeploymentStage.AI_SETUP
    capacity_evidence: CapacityEvidence = field(default_factory=CapacityEvidence)
    governance_profile: GovernanceProfile = field(default_factory=GovernanceProfile)

    # Optional metadata used in cross-context generalization analysis.
    region: Optional[str] = None
    language: Optional[str] = None
    model_name: Optional[str] = None
    dataset_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert deployment to plain dictionary for serialization."""
        return asdict(self)
