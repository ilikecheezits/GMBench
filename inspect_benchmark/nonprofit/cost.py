"""Rough pre-run cost estimates so runs never surprise the API bill.

Token counts are averages measured on this benchmark; prices are list prices per
1M tokens (input, output) and should be checked against the provider's pricing
page. Unknown models are reported with token counts only.
"""

from __future__ import annotations

from dataclasses import dataclass

from .run_eval_set import RunPlan

# USD per 1M tokens (input, output). Edit if prices change.
PRICES: dict[str, tuple[float, float]] = {
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-5-nano": (0.05, 0.40),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5": (1.25, 10.00),
    "o4-mini": (1.10, 4.40),
}

# Average tokens per call: (input, output). Reasoning models multiply output.
CALL_PROFILE = {
    "core_generate": (250, 900),
    "core_judge": (2200, 450),
    "mc_generate": (220, 15),
    "refusal_generate": (60, 350),
    "refusal_classify": (600, 8),
}
REASONING_OUTPUT_MULTIPLIER = 4.0


def _price(model: str) -> tuple[float, float] | None:
    short = model.split("/")[-1].lower()
    for key in sorted(PRICES, key=len, reverse=True):
        if short == key or short.startswith(key + "-20"):
            return PRICES[key]
    return None


def _is_reasoning(model: str) -> bool:
    short = model.split("/")[-1].lower()
    return short.startswith(("o1", "o3", "o4", "gpt-5"))


@dataclass
class Estimate:
    lines: list[dict]
    total_usd: float | None
    total_calls: int

    def render(self) -> str:
        out = ["Estimated API usage (rough; actual varies with answer length):", ""]
        out.append(f"{'model':<28}{'role':<10}{'calls':>7}{'in tok':>11}{'out tok':>11}{'est $':>10}")
        for line in self.lines:
            usd = f"{line['usd']:.3f}" if line["usd"] is not None else "n/a"
            out.append(f"{line['model']:<28}{line['role']:<10}{line['calls']:>7}{line['in']:>11,}{line['out']:>11,}{usd:>10}")
        total = f"${self.total_usd:.2f}" if self.total_usd is not None else "unknown (some models lack prices)"
        out += ["", f"Total calls: {self.total_calls}   Estimated total: {total}"]
        return "\n".join(out)


def estimate_plan(plan: RunPlan) -> Estimate:
    n_core = len(plan.core_samples()) if "core" in plan.suites else 0
    n_custom = len(plan.custom_samples()) if "custom" in plan.suites else 0
    ext = plan.external_sizes() if "external" in plan.suites else {}
    n_mc = sum(v for k, v in ext.items() if k != "calibrated_refusal")
    n_refusal = ext.get("calibrated_refusal", 0)
    epochs = max(1, plan.epochs)

    lines = []
    total_usd = 0.0
    unknown = False
    total_calls = 0

    def add(model: str, role: str, calls: int, tin: int, tout: int):
        nonlocal total_usd, unknown, total_calls
        if calls <= 0:
            return
        price = _price(model)
        usd = None
        if price:
            usd = (tin * price[0] + tout * price[1]) / 1_000_000
            total_usd += usd
        else:
            unknown = True
        total_calls += calls
        lines.append({"model": model, "role": role, "calls": calls, "in": tin, "out": tout, "usd": usd})

    graded = (n_core + n_custom) * epochs
    for model in plan.models:
        mult = REASONING_OUTPUT_MULTIPLIER if _is_reasoning(model) else 1.0
        ci, co = CALL_PROFILE["core_generate"]
        mi, mo = CALL_PROFILE["mc_generate"]
        ri, ro = CALL_PROFILE["refusal_generate"]
        calls = graded + (n_mc + n_refusal) * epochs
        tin = graded * ci + n_mc * epochs * mi + n_refusal * epochs * ri
        tout = int(graded * co * mult + n_mc * epochs * (mo + (300 if mult > 1 else 0)) + n_refusal * epochs * ro * mult)
        add(model, "candidate", calls, tin, tout)

    ji, jo = CALL_PROFILE["core_judge"]
    jmult = REASONING_OUTPUT_MULTIPLIER if _is_reasoning(plan.judge_model) else 1.0
    judge_calls = graded * len(plan.models)
    refusal_calls = n_refusal * epochs * len(plan.models)
    ki, ko = CALL_PROFILE["refusal_classify"]
    add(
        plan.judge_model,
        "judge",
        judge_calls + refusal_calls,
        judge_calls * ji + refusal_calls * ki,
        int((judge_calls * jo + refusal_calls * ko) * jmult),
    )

    return Estimate(lines=lines, total_usd=None if unknown else round(total_usd, 4), total_calls=total_calls)
