"""Benchmark report objects and text/JSON serialization."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List

from deployment import Deployment
from metrics import PillarEvaluation
from utils import mean, utc_timestamp


class PhaseGateResult(str, Enum):
    """Lifecycle stage reached by a deployment after evaluation."""

    NOT_READY = "NOT READY"
    READY_FOR_ALPHA = "READY FOR ALPHA"
    READY_FOR_BETA = "READY FOR BETA"
    READY_FOR_LABS = "READY FOR LABS"


@dataclass(slots=True)
class BenchmarkReport:
    """Full benchmark output for a single deployment."""

    deployment: Deployment
    technical_score: float
    impact_score: float
    continuity_score: float
    generalization_score: float
    governance_score: float
    overall_score: float
    phase_result: PhaseGateResult
    pillar_evaluations: Dict[str, PillarEvaluation] = field(default_factory=dict)
    generated_at: str = field(default_factory=utc_timestamp)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["phase_result"] = self.phase_result.value
        return data

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def render_text(self) -> str:
        lines: List[str] = []
        lines.append("====================================")
        lines.append("GOOD MODEL BENCHMARK REPORT")
        lines.append("====================================")
        lines.append("")
        lines.append("Organization:")
        lines.append(self.deployment.organization)
        lines.append("")
        lines.append("Workflow:")
        lines.append(self.deployment.workflow)
        lines.append("")

        section_order = [
            "Technical Reliability",
            "Operational Impact",
            "Continuity",
            "Generalization",
            "Risk and Governance",
        ]

        score_map = {
            "Technical Reliability": self.technical_score,
            "Operational Impact": self.impact_score,
            "Continuity": self.continuity_score,
            "Generalization": self.generalization_score,
            "Risk and Governance": self.governance_score,
        }

        for section in section_order:
            lines.append(section)
            lines.append("----------------------")
            lines.append(f"Score: {score_map[section]:.1f}")
            lines.append("")
            details = self.pillar_evaluations.get(section)
            if details:
                for metric_label in details.metric_scores.keys():
                    lines.append(metric_label)
                if details.notes:
                    lines.extend(details.notes)
            lines.append("")

        lines.append("Overall Score")
        lines.append("")
        lines.append(f"{self.overall_score:.1f}")
        lines.append("")
        lines.append("PHASE RESULT")
        lines.append("")
        lines.append(self.phase_result.value)

        return "\n".join(lines)


@dataclass(slots=True)
class ComparisonReport:
    """Result of benchmarking multiple deployments."""

    reports: List[BenchmarkReport]

    @property
    def average_score(self) -> float:
        return round(mean(report.overall_score for report in self.reports), 2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "average_score": self.average_score,
            "reports": [report.to_dict() for report in self.reports],
        }


@dataclass(slots=True)
class GeneralizationReport:
    """Aggregated generalization analysis for one workflow pattern."""

    workflow_name: str
    deployment_count: int
    represented_organizations: List[str]
    generalization_score: float
    candidate_for_labs: bool
    pillar_evaluation: PillarEvaluation
    generated_at: str = field(default_factory=utc_timestamp)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_name": self.workflow_name,
            "deployment_count": self.deployment_count,
            "represented_organizations": self.represented_organizations,
            "generalization_score": self.generalization_score,
            "candidate_for_labs": self.candidate_for_labs,
            "pillar_evaluation": asdict(self.pillar_evaluation),
            "generated_at": self.generated_at,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
