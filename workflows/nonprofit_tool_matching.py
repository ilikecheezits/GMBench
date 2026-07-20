"""Systems under test for nonprofit AI tool matching."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict

from dataset import BenchmarkExample
from registry import register_system
from workflow import SystemOutput, SystemUnderTest


@dataclass(frozen=True)
class PackageProfile:
    package_name: str
    category: str
    activation_days: int
    monthly_cost: int
    risk_level: str
    success_metric: str


PACKAGE_CATALOG: Dict[str, PackageProfile] = {
    "meeting": PackageProfile(
        package_name="meeting_intelligence_basic",
        category="meeting_assistant",
        activation_days=14,
        monthly_cost=64,
        risk_level="low",
        success_metric="follow_up_completion_rate",
    ),
    "intake": PackageProfile(
        package_name="intake_automation_lite",
        category="intake_automation",
        activation_days=21,
        monthly_cost=72,
        risk_level="medium",
        success_metric="intake_turnaround_time",
    ),
    "volunteer": PackageProfile(
        package_name="volunteer_coordination_basic",
        category="volunteer_coordination",
        activation_days=14,
        monthly_cost=48,
        risk_level="low",
        success_metric="volunteer_shift_fill_rate",
    ),
    "grant": PackageProfile(
        package_name="grant_support_standard",
        category="grant_support",
        activation_days=30,
        monthly_cost=180,
        risk_level="medium",
        success_metric="grant_draft_cycle_time",
    ),
    "identity": PackageProfile(
        package_name="identity_access_starter",
        category="identity_access",
        activation_days=28,
        monthly_cost=95,
        risk_level="high",
        success_metric="access_audit_completion",
    ),
}


def _extract_budget(text: str) -> int:
    patterns = [
        r"budget:\s*\$(\d+)",
        r"presupuesto:\s*\$(\d+)",
        r"monthly budget:\s*\$(\d+)",
    ]
    match = None
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            break
    if not match:
        return 100
    return int(match.group(1))


def _extract_sensitivity(text: str) -> str:
    lower = text.lower()
    if "data sensitivity: high" in lower or "sensibilidad de datos: alta" in lower:
        return "high"
    if "data sensitivity: low" in lower or "sensibilidad de datos: baja" in lower:
        return "low"
    return "medium"


def _choose_painpoint_bucket(text: str) -> str:
    lower = text.lower()
    if any(token in lower for token in ["identity", "access", "permissions", "onboarding", "consent", "identidad"]):
        return "identity"
    if "grant" in lower:
        return "grant"
    if any(token in lower for token in ["intake", "forms", "form", "formularios", "admisiones"]):
        return "intake"
    if any(token in lower for token in ["volunteer", "no-shows", "voluntarios", "ausencias"]):
        return "volunteer"
    return "meeting"


class _BaseToolMatcher(SystemUnderTest):
    system_type = "workflow_orchestration"
    description = "Matches nonprofit constraints to a practical AI tool package."

    cost_multiplier = 1.0

    def _match(self, example: BenchmarkExample) -> Dict[str, object]:
        budget = _extract_budget(example.input_text)
        sensitivity = _extract_sensitivity(example.input_text)
        bucket = _choose_painpoint_bucket(example.input_text)
        profile = PACKAGE_CATALOG[bucket]

        adjusted_cost = int(round(profile.monthly_cost * self.cost_multiplier))
        if adjusted_cost > budget:
            adjusted_cost = budget

        risk_level = profile.risk_level
        if sensitivity == "high":
            risk_level = "high"

        return {
            "recommended_package": profile.package_name,
            "primary_tool_category": profile.category,
            "activation_horizon_days": profile.activation_days,
            "estimated_monthly_cost_usd": adjusted_cost,
            "risk_level": risk_level,
            "success_metric": profile.success_metric,
        }

    async def run(self, example: BenchmarkExample) -> SystemOutput:
        return SystemOutput(structured_output=self._match(example), text_output="local heuristic match")


@register_system("nonprofit_tool_matcher_balanced")
class NonprofitToolMatcherBalanced(_BaseToolMatcher):
    name = "Nonprofit Tool Matcher Balanced"
    cost_multiplier = 1.0


@register_system("nonprofit_tool_matcher_budget")
class NonprofitToolMatcherBudget(_BaseToolMatcher):
    name = "Nonprofit Tool Matcher Budget"
    cost_multiplier = 0.85


@register_system("nonprofit_tool_matcher_expansive")
class NonprofitToolMatcherExpansive(_BaseToolMatcher):
    name = "Nonprofit Tool Matcher Expansive"
    cost_multiplier = 1.2
