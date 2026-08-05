# Nonprofit LLM Benchmark Notes

This document explains how the benchmark works from start to finish, with one concrete run example.

## Purpose

The benchmark compares LLMs on nonprofit strategy tasks under the same prompts and scoring rules.

- Model-vs-model only
- Shared task bank
- Shared rubric criteria
- Shared deterministic objective checks

## Core Components

1. Task bank: [inspect_benchmark/nonprofit/task_bank.py](inspect_benchmark/nonprofit/task_bank.py)
2. Task execution and judge prompting: [inspect_benchmark/nonprofit/tasks.py](inspect_benchmark/nonprofit/tasks.py)
3. Deterministic objective checks: [inspect_benchmark/nonprofit/objective_metrics.py](inspect_benchmark/nonprofit/objective_metrics.py)
4. Aggregation and reporting: [inspect_benchmark/nonprofit/leaderboard.py](inspect_benchmark/nonprofit/leaderboard.py)
5. CLI entrypoint: [main.py](main.py)

## Start-To-Finish Workflow (Example)

### Step 1: Run the benchmark

```bash
python main.py run \
	--models openai/gpt-4.1-nano,openai/gpt-4.1-mini \
	--tasks fundraising,grants,operations,communications,impact,governance,compliance \
	--judge-model openai/gpt-4.1-mini \
	--log-dir logs/nonprofit_llm_benchmark_budget_compare
```

What happens:

1. Task groups are resolved into sample sets.
2. Each model answers each sample.
3. Two scorers evaluate each answer.

### Step 2: Per-sample scoring

For each model response, there are two independent scores.

1. Judge score (LLM rubric grader): C, P, or I

- C = 1.0
- P = 0.5
- I = 0.0

2. Objective score (deterministic checks): value in [0, 1]

- Length/depth checks
- Structure checks
- Task-keyword coverage
- Numeric-signal checks when requested
- Language compliance checks
- High-risk safeguard checks

Hybrid score combines both:

$$
	ext{Hybrid} = 0.65 \times \text{Objective} + 0.35 \times \text{Judge}
$$

### Step 3: Logs are written

Inspect writes raw eval logs to:

- [logs/nonprofit_llm_benchmark_budget_compare](logs/nonprofit_llm_benchmark_budget_compare)

These logs include prompts, model outputs, metadata slices, scores, and runtime usage.

### Step 4: Build reports

```bash
python main.py leaderboard \
	--log-dir logs/nonprofit_llm_benchmark_budget_compare \
	--output artifacts/nonprofit_llm_leaderboard_budget_compare.md

python main.py task-scores \
	--log-dir logs/nonprofit_llm_benchmark_budget_compare \
	--output artifacts/nonprofit_task_scores_budget_compare.md
```

Outputs:

1. Overall model ranking: [artifacts/nonprofit_llm_leaderboard_budget_compare.md](artifacts/nonprofit_llm_leaderboard_budget_compare.md)
2. Task-slice breakdown: [artifacts/nonprofit_task_scores_budget_compare.md](artifacts/nonprofit_task_scores_budget_compare.md)

## How To Interpret Reports

Leaderboard columns:

- Hybrid Mean: final weighted score used for ranking
- Objective Mean: deterministic score only
- Judge Mean: LLM rubric score only
- Samples: total sample evaluations for that model
- Task Coverage: number of task groups covered

Task-slice columns:

- Task/Area/Difficulty/Language/Risk identify the slice
- Hybrid Mean/Objectve Mean/Judge Mean show quality by slice
- C/P/I rates show how judge labels are distributed

## Why This Is More Reliable Than Judge-Only

The deterministic objective rubric anchors scoring to fixed checks, reducing variance from LLM grader behavior.

The judge rubric still contributes, but no longer dominates ranking.

## Limitation

This benchmark is a controlled capability comparison and should not be treated as direct proof of real-world nonprofit impact without field validation.
