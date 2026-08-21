# Nonprofit LLM Benchmark (inspect_ai)

This repository runs a nonprofit-focused LLM-vs-LLM benchmark using [inspect_ai](https://inspect.aisi.org.uk/), the open-source eval framework from the UK AI Security Institute.

## Why This Benchmark Exists

Most LLM benchmarks test general knowledge or coding. None measure what nonprofit practitioners actually need: strategy quality under resource constraints, compliance awareness, board communication, grant writing, and impact framing. This benchmark fills that gap.

The design follows three principles, adapted from [HealthBench](https://openai.com/index/healthbench/) (the closest prior work, using physician-written rubrics):

- **Meaningful**: Tasks are realistic nonprofit scenarios, not trivia. Scores should reflect real practitioner judgment.
- **Trustworthy**: Rubric criteria are grounded in sector-specific standards, not LLM intuition alone. Deterministic objective checks anchor scoring.
- **Unsaturated**: Current models still have substantial room to improve on hard cases. The benchmark should remain a useful signal as models get stronger.

## Why Inspect AI

The valuable part of a benchmark is the tasks and grading criteria, not the runner. [Inspect AI](https://inspect.aisi.org.uk/) gives us the runner for free: fan-out across 20+ model providers, retries, reproducible logs of every prompt/response/grade, and a viewer for inspecting individual transcripts. Building on a framework from the UK AI Security Institute means we inherit credibility and don't have to defend homegrown tooling.

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

## Benchmark Rigor Improvements

The project now includes a first round of rigor upgrades meant to reflect real nonprofit workflows rather than generic reasoning. The goal is to make the current benchmark more realistic without waiting for a perfect production dataset.

Implemented improvements:

- Task-area-aware objective checks: the determinisitic scorer now looks for concepts specific to each nonprofit domain, not just generic length and keyword counts.
- Execution realism checks: answers must reflect operational constraints such as team capacity, timelines, budget constraints, and monitoring/oversight.
- High-risk safeguards: privacy, compliance, and fairness scenarios require explicit controls and reject unsafe shortcuts.
- Benchmark validation: the project now includes a built-in validation check for task coverage and schema integrity before runs are trusted.
- Flexible scoring composition: the scoring architecture in [inspect_benchmark/nonprofit/scoring.py](inspect_benchmark/nonprofit/scoring.py) is intentionally extensible and supports future benchmark-of-benchmarks weighting without rewriting leaderboard logic.

Future rigor ideas for the benchmark:

- Add task variants that include organization context, staffing limits, and budget constraints.
- Combine objective rubric signals with task- and domain-specific evidence checks (for example: grant narrative completeness, communications channel appropriateness, governance controls).
- Add a small expert-review calibration set to tune judge prompts and grade thresholds.
- Introduce a benchmark-of-benchmarks layer where external evaluator signals are added as weighted components on top of the domain-specific nonprofit score.
- Add a small suite of adversarial cases designed to catch generic, template-like outputs that pass basic keyword checks but fail on realistic nonprofit execution.

## Install

```bash
pip install -r requirements.txt
```

Set credentials as needed:

```bash
export OPENAI_API_KEY=...
```

## Validate Benchmark Health

Before running the benchmark, validate that the task bank is well-formed and covers all required nonprofit domains.

```bash
python main.py validate
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

- [main.py](main.py): CLI (`run`, `validate`, `leaderboard`, `task-scores`)
- [inspect_benchmark/nonprofit/run_eval_set.py](inspect_benchmark/nonprofit/run_eval_set.py): inspect eval_set execution
- [inspect_benchmark/nonprofit/task_bank.py](inspect_benchmark/nonprofit/task_bank.py): benchmark samples + validation helper
- [inspect_benchmark/nonprofit/tasks.py](inspect_benchmark/nonprofit/tasks.py): task wiring and judge prompt
- [inspect_benchmark/nonprofit/objective_metrics.py](inspect_benchmark/nonprofit/objective_metrics.py): deterministic objective rubric scorer
- [inspect_benchmark/nonprofit/scoring.py](inspect_benchmark/nonprofit/scoring.py): flexible weighted-scoring configuration designed for future benchmark-of-benchmarks composition
- [inspect_benchmark/nonprofit/leaderboard.py](inspect_benchmark/nonprofit/leaderboard.py): leaderboard + task-score aggregation

## Benchmark Quality Standards

This benchmark is designed against the [BetterBench](https://betterbench.stanford.edu/index.html) checklist (NeurIPS 2024), which evaluates AI benchmarks across 46 quality criteria covering task design, scoring validity, and usability. Key practices followed:

- Tasks are drawn from realistic practitioner scenarios, not synthetic edge cases.
- Scoring is split into deterministic (objective) and model-graded (judge) components to reduce grader variance.
- Logs are fully reproducible: every prompt, response, and grade is captured by Inspect AI.
- Leaderboard reports include task-slice breakdowns so results can be interrogated by domain, difficulty, language, and risk level.

## Roadmap: Community Task Contributions

This benchmark follows a crowdsourcing model inspired by [LegalBench](https://github.com/HazyResearch/legalbench), which builds tasks from a professional community. The goal is to expand the task bank with contributions from nonprofit practitioners — EDs, program directors, grant writers, and compliance leads — who can submit tasks and rubric criteria from their real work. This makes the benchmark harder to game and more representative of the sector over time.

## Notes

- This is a capability comparison benchmark, not direct proof of field impact.
- For stronger rank stability, increase sample counts and multilingual high-risk cases.
- The project is intentionally structured so that part 2 of the roadmap — a benchmark-of-benchmarks architecture — can be added by composing additional score components without reworking the scoring model or CLI.
