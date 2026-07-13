# Good Model Benchmark

Good Model Benchmark is a modular evaluation framework for nonprofit AI deployments.
It is designed for Good Model Project (GMP) use cases where the benchmark target is a
real deployment inside an organization, not a generic model leaderboard.

## What This Framework Evaluates

Each deployment is scored across six pillars:

1. Technical Reliability
2. Operational Impact
3. Evidence and Capacity Delta
4. Continuity
5. Generalizability
6. Risk and Governance

Each run produces a `BenchmarkReport` with a phase-gate decision:

- Ready for Alpha
- Ready for Beta
- Ready for Labs
- Not Ready

Additional GMP-aligned controls are enforced:

- Stage-aware phase caps (`Needs Assessment`, `AI Setup`, `Workflow Automation`, `Custom Build`)
- Governance hard gates for sensitive use cases
- Evidence quality weighting so low-trust claims cannot dominate outcomes

## Package Layout

```text
.
├── benchmark.py
├── deployment.py
├── metrics.py
├── evaluators.py
├── reports.py
├── datasets/
├── examples/
├── utils.py
└── main.py
```

## Quick Start

Run the demo:

```bash
python main.py
```

Run tests:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

Programmatic usage:

```python
from benchmark import GoodModelBenchmark
from deployment import Deployment

benchmark = GoodModelBenchmark()
report = benchmark.run(deployment)
comparison = benchmark.compare([deployment1, deployment2, deployment3])
generalization = benchmark.generalize(workflow_name="Intake Summarization")

print(report.to_json())
```

## Extending The Framework

### Add new metric fields

1. Add fields to the relevant dataclass in `deployment.py`.
2. Add corresponding `MetricSpec` entries in `evaluators.py`.
3. Optionally tune weighting and required flags.

### Add a new deployment domain

No code changes are required for domain type itself. Use `deployment_type` and `metadata`
fields to capture domain-specific context for food pantries, shelters, clinics, schools,
legal aid organizations, animal welfare groups, or climate organizations.

### Add a custom evaluator

1. Implement `BasePillarEvaluator`.
2. Pass custom evaluators to `GoodModelBenchmark(evaluators=[...])`.
3. Reweight overall score logic in `benchmark.py` if needed.

## Design Notes

- Dataclasses and strict typing are used throughout.
- Pillar evaluators are decoupled from report rendering.
- Phase-gate logic is centralized in `PhaseGateEvaluator`.
- Governance hard-gating is centralized in `GoodModelBenchmark._governance_gate_reasons`.
- Reports serialize cleanly to JSON and can later be exported to markdown or PDF.
