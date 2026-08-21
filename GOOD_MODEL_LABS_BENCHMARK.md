# Nonprofit LLM Benchmark: Implementation and Runbook

This document is the technical companion to [README.md](README.md). It is meant for people who want to understand the benchmark implementation, how scoring works, and how to reproduce the workflow.

## Purpose

The benchmark compares models on nonprofit-relevant decision-making tasks under a shared prompt set and shared scoring rules. It is designed to measure practical nonprofit reasoning rather than generic conversational fluency.

## Benchmark structure

Core implementation files:

- [main.py](main.py): CLI entrypoint
- [inspect_benchmark/nonprofit/task_bank.py](inspect_benchmark/nonprofit/task_bank.py): task definitions and metadata
- [inspect_benchmark/nonprofit/tasks.py](inspect_benchmark/nonprofit/tasks.py): task generation and judge scoring setup
- [inspect_benchmark/nonprofit/objective_metrics.py](inspect_benchmark/nonprofit/objective_metrics.py): objective checks
- [inspect_benchmark/nonprofit/leaderboard.py](inspect_benchmark/nonprofit/leaderboard.py): aggregation and reporting
- [inspect_benchmark/nonprofit/scoring.py](inspect_benchmark/nonprofit/scoring.py): flexible weighted scoring configuration

## Task domains

The benchmark currently includes these task areas:

- fundraising_strategy
- grant_writing
- operations_prioritization
- community_communications
- impact_measurement
- board_governance
- compliance_and_privacy

## Scoring model

Each model response is scored in two ways:

1. Objective score
   - structured response checks
   - task-area coverage checks
   - numeric or milestone requirements when relevant
   - language compliance checks
   - high-risk safeguard checks
   - realism checks to reject generic template answers

2. Judge score
   - model-graded rubric labels: C, P, or I
   - converted numerically as: C = 1.0, P = 0.5, I = 0.0

The hybrid score is:

$$
\text{Hybrid} = 0.65 \times \text{Objective} + 0.35 \times \text{Judge}
$$

## Workflow

### 1. Validate benchmark health

```bash
cd /Users/Samuel/GMBench
source .venv/bin/activate
python main.py validate
```

This checks that the task bank is structurally valid and that all required task areas are present.

### 2. Run the benchmark

```bash
python main.py run \
  --models openai/gpt-4.1-nano,openai/gpt-4.1-mini \
  --tasks fundraising,grants,operations,communications,impact,governance,compliance \
  --judge-model openai/gpt-4.1-mini \
  --log-dir logs/nonprofit_llm_benchmark_budget_compare
```

What this does:

- resolves the selected task groups
- creates the sample sets
- runs each model on each sample
- evaluates with both objective and judge scoring
- writes Inspect logs to the provided log directory

### 3. Produce reports

```bash
python main.py leaderboard \
  --log-dir logs/nonprofit_llm_benchmark_budget_compare \
  --output artifacts/nonprofit_llm_leaderboard_budget_compare.md

python main.py task-scores \
  --log-dir logs/nonprofit_llm_benchmark_budget_compare \
  --output artifacts/nonprofit_task_scores_budget_compare.md
```

Outputs:

- [artifacts/nonprofit_llm_leaderboard_budget_compare.md](artifacts/nonprofit_llm_leaderboard_budget_compare.md)
- [artifacts/nonprofit_task_scores_budget_compare.md](artifacts/nonprofit_task_scores_budget_compare.md)

### 4. Inspect logs

Raw logs are stored in:

- [logs/nonprofit_llm_benchmark_budget_compare](logs/nonprofit_llm_benchmark_budget_compare)

These logs include prompts, outputs, score metadata, and task slices.

## Benchmark rigor and evaluation choices

The benchmark is intentionally not judge-only. The deterministic objective scorer anchors evaluation and reduces grader variance. The judge score still matters, but it does not dominate the final ranking.

The current implementation also includes:

- context-aware task variants
- safeguard checks for privacy and high-risk decisions
- generic-fluff detection to reduce template-style answers
- a modular weighted-scoring layer for future benchmark-of-benchmarks composition

## Limitations

This remains a benchmark of capability, not direct proof of real nonprofit impact. It is best used for comparative model evaluation under a defined task set, not as a substitute for field trials or practitioner validation.

## Recommended next steps

- add expert calibration data
- add adversarial realism cases
- add external benchmark components into the weighted scoring layer
- make org-context tailoring configurable in benchmark setup

## References

- [README.md](README.md)
- [inspect_ai](https://inspect.aisi.org.uk/)
- [HealthBench](https://openai.com/index/healthbench/)
- [BetterBench](https://betterbench.stanford.edu/index.html)
- [LegalBench](https://github.com/HazyResearch/legalbench)
