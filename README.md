# Nonprofit LLM Benchmark (inspect_ai)

This repository runs a nonprofit-focused LLM-vs-LLM benchmark using inspect_ai.

## What It Measures

- Practical strategy quality on nonprofit tasks
- Deterministic objective rubric checks
- Strict LLM-judge rubric checks
- Hybrid score that combines objective and judge signals

Current task domains in [inspect_benchmark/nonprofit/task_bank.py](inspect_benchmark/nonprofit/task_bank.py):

- fundraising_strategy
- grant_writing
- operations_prioritization
- community_communications
- impact_measurement
- board_governance
- compliance_and_privacy

## Scoring Model

Task definitions are in [inspect_benchmark/nonprofit/tasks.py](inspect_benchmark/nonprofit/tasks.py).

- Judge scorer: model_graded_qa (strict C/P/I instructions)
- Objective scorer: deterministic rubric checks in [inspect_benchmark/nonprofit/objective_metrics.py](inspect_benchmark/nonprofit/objective_metrics.py)
- Hybrid score in [inspect_benchmark/nonprofit/leaderboard.py](inspect_benchmark/nonprofit/leaderboard.py):
	- objective weight: 0.65
	- judge weight: 0.35

## Install

```bash
pip install -r requirements.txt
```

Set credentials as needed:

```bash
export OPENAI_API_KEY=...
```

## Run Benchmark

OpenAI budget comparison example:

```bash
python main.py run \
	--models openai/gpt-4.1-nano,openai/gpt-4.1-mini \
	--tasks fundraising,grants,operations,communications,impact,governance,compliance \
	--judge-model openai/gpt-4.1-mini \
	--log-dir logs/nonprofit_llm_benchmark_budget_compare
```

## Start-To-Finish Example Workflow

This is the exact lifecycle for one benchmark run.

1. You run the CLI command.

```bash
python main.py run \
	--models openai/gpt-4.1-nano,openai/gpt-4.1-mini \
	--tasks fundraising,grants,operations,communications,impact,governance,compliance \
	--judge-model openai/gpt-4.1-mini \
	--log-dir logs/nonprofit_llm_benchmark_budget_compare
```

2. Task loading happens from [inspect_benchmark/nonprofit/task_bank.py](inspect_benchmark/nonprofit/task_bank.py).

- Each sample includes prompt, criterion, reference_answer, difficulty, language, and risk level metadata.

3. For each sample and each model, two scorers run.

- Judge scorer in [inspect_benchmark/nonprofit/tasks.py](inspect_benchmark/nonprofit/tasks.py): returns C/P/I.
- Objective scorer in [inspect_benchmark/nonprofit/objective_metrics.py](inspect_benchmark/nonprofit/objective_metrics.py): deterministic checklist score from 0 to 1.

4. Hybrid score is computed per sample.

- Hybrid = 0.65 * objective + 0.35 * judge_numeric
- Judge numeric mapping: C=1.0, P=0.5, I=0.0
- Aggregation is implemented in [inspect_benchmark/nonprofit/leaderboard.py](inspect_benchmark/nonprofit/leaderboard.py).

5. Raw logs are written by inspect_ai.

- Directory: [logs/nonprofit_llm_benchmark_budget_compare](logs/nonprofit_llm_benchmark_budget_compare)
- These logs contain model outputs, scorer values, metadata slices, and usage/timing data.

6. You generate reports from those logs.

```bash
python main.py leaderboard \
	--log-dir logs/nonprofit_llm_benchmark_budget_compare \
	--output artifacts/nonprofit_llm_leaderboard_budget_compare.md

python main.py task-scores \
	--log-dir logs/nonprofit_llm_benchmark_budget_compare \
	--output artifacts/nonprofit_task_scores_budget_compare.md
```

7. You review results at two levels.

- Overall rank report: [artifacts/nonprofit_llm_leaderboard_budget_compare.md](artifacts/nonprofit_llm_leaderboard_budget_compare.md)
- Slice-level report (task/language/risk/difficulty): [artifacts/nonprofit_task_scores_budget_compare.md](artifacts/nonprofit_task_scores_budget_compare.md)

This gives an auditable, repeatable loop from prompts to rubric grades to final ranking.

## Generate Reports

Overall leaderboard:

```bash
python main.py leaderboard \
	--log-dir logs/nonprofit_llm_benchmark_budget_compare \
	--output artifacts/nonprofit_llm_leaderboard_budget_compare.md
```

Task-level breakdown:

```bash
python main.py task-scores \
	--log-dir logs/nonprofit_llm_benchmark_budget_compare \
	--output artifacts/nonprofit_task_scores_budget_compare.md
```

## Repository Layout

- [main.py](main.py): CLI (`run`, `leaderboard`, `task-scores`)
- [inspect_benchmark/nonprofit/run_eval_set.py](inspect_benchmark/nonprofit/run_eval_set.py): inspect eval_set execution
- [inspect_benchmark/nonprofit/task_bank.py](inspect_benchmark/nonprofit/task_bank.py): benchmark samples
- [inspect_benchmark/nonprofit/tasks.py](inspect_benchmark/nonprofit/tasks.py): task wiring and judge prompt
- [inspect_benchmark/nonprofit/objective_metrics.py](inspect_benchmark/nonprofit/objective_metrics.py): deterministic objective rubric scorer
- [inspect_benchmark/nonprofit/leaderboard.py](inspect_benchmark/nonprofit/leaderboard.py): leaderboard + task-score aggregation

## Notes

- This is a capability comparison benchmark, not direct proof of field impact.
- For stronger rank stability, increase sample counts and multilingual high-risk cases.
