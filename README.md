# Nonprofit LLM Benchmark

This project evaluates how large language models perform on realistic nonprofit decision-making tasks. It compares models on the same task bank, uses the same evaluation rubric, and produces a shared leaderboard.

## Why this exists

Most public LLM benchmarks focus on general knowledge or coding. Very few measure how well a model handles actual nonprofit work: grant strategy, compliance, board governance, impact measurement, fundraising, and operational prioritization.

This benchmark is designed to fill that gap.

## What it measures

- nonprofit strategy quality
- operational realism
- risk and safety awareness
- governance and compliance judgment
- language and context sensitivity

## High-level design

The benchmark combines two signals:

- objective scoring: deterministic checks for structure, realism, safeguards, and task-specific requirements
- judge scoring: an LLM rubric grader that assigns C, P, or I based on quality and completeness

The final score blends those signals into a single leaderboard result.

## Quick start

```bash
cd /Users/Samuel/GMBench
source .venv/bin/activate
python main.py validate
python main.py run \
  --models openai/gpt-4.1-nano,openai/gpt-4.1-mini \
  --tasks fundraising,grants,operations,communications,impact,governance,compliance \
  --judge-model openai/gpt-4.1-mini \
  --log-dir logs/nonprofit_llm_benchmark_budget_compare
python main.py leaderboard \
  --log-dir logs/nonprofit_llm_benchmark_budget_compare \
  --output artifacts/nonprofit_llm_leaderboard_budget_compare.md
python main.py task-scores \
  --log-dir logs/nonprofit_llm_benchmark_budget_compare \
  --output artifacts/nonprofit_task_scores_budget_compare.md
```

## Repository at a glance

- [main.py](main.py): CLI entrypoint
- [inspect_benchmark/nonprofit/task_bank.py](inspect_benchmark/nonprofit/task_bank.py): benchmark tasks and task metadata
- [inspect_benchmark/nonprofit/tasks.py](inspect_benchmark/nonprofit/tasks.py): task wiring and judge grading
- [inspect_benchmark/nonprofit/objective_metrics.py](inspect_benchmark/nonprofit/objective_metrics.py): deterministic quality checks
- [inspect_benchmark/nonprofit/scoring.py](inspect_benchmark/nonprofit/scoring.py): weighted score composition
- [inspect_benchmark/nonprofit/leaderboard.py](inspect_benchmark/nonprofit/leaderboard.py): leaderboard and per-task reporting

## Current benchmark direction

This project is intentionally designed to be more realistic than a generic benchmark:

- tasks are grounded in nonprofit operations
- objective scoring checks for realistic execution constraints
- high-risk cases require explicit safeguards
- scoring is structured so future benchmark-of-benchmarks components can be added without rewriting the system

## What external readers need to know

This is a capability benchmark, not a field-impact study. It is useful for comparing model behavior on realistic nonprofit tasks, but it should not be interpreted as direct proof of real-world organizational outcomes.

## External resources

- [GOOD_MODEL_LABS_BENCHMARK.md](GOOD_MODEL_LABS_BENCHMARK.md): detailed workflow, scoring logic, and replication instructions
- [inspect_ai](https://inspect.aisi.org.uk/)
- [HealthBench](https://openai.com/index/healthbench/)
- [BetterBench](https://betterbench.stanford.edu/index.html)
