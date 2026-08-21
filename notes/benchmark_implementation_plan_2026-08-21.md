# Nonprofit Benchmark Roadmap

## Source plan
This is the working summary of the original plan in [notes/benchmark_plan_2026-08-07.md](notes/benchmark_plan_2026-08-07.md), updated with the current implementation status.

## Current status
The project is now in a good place to move to the second phase: a benchmark-of-benchmarks layer.

Why this is ready:
- the benchmark is structurally valid
- the task bank has full coverage across the nonprofit domains
- the objective scoring now checks for realism, not just generic length/keywords
- the scoring system is already modular and designed for weighted composition

Verified state:
- `python main.py validate` returned: Valid = True
- sample count: 24
- task areas covered: 7

## Decision
Yes — we are ready to move on to the benchmark-of-benchmarks stage.

The caveat is that this should be treated as a second-phase extension, not as a final claim that the benchmark is fully mature. The first phase made the benchmark much more realistic, but expert calibration and adversarial realism checks are still useful follow-ups before broad external claims.

## Phase 2: benchmark of benchmarks
Goal:
- keep the nonprofit benchmark as the primary signal
- layer in external benchmark signals as additional weighted components
- start with fixed weights, then allow custom weighting later

Recommended design:
- Core score: nonprofit task performance
- Additional components: external evaluation signals or reference benchmarks
- Final score: weighted combination of component scores
- Keep the framework configurable so new benchmarks can be added without rebuilding the system

## Simple replication workflow
1. Activate the environment
   ```bash
   cd /Users/Samuel/GMBench
   source .venv/bin/activate
   ```
2. Check that the benchmark is healthy
   ```bash
   python main.py validate
   ```
3. Run the nonprofit benchmark
   ```bash
   python main.py run \
     --models openai/gpt-4.1-nano,openai/gpt-4.1-mini \
     --tasks fundraising,grants,operations,communications,impact,governance,compliance \
     --judge-model openai/gpt-4.1-mini \
     --log-dir logs/nonprofit_llm_benchmark_budget_compare
   ```
4. Generate reports
   ```bash
   python main.py leaderboard \
     --log-dir logs/nonprofit_llm_benchmark_budget_compare \
     --output artifacts/nonprofit_llm_leaderboard_budget_compare.md

   python main.py task-scores \
     --log-dir logs/nonprofit_llm_benchmark_budget_compare \
     --output artifacts/nonprofit_task_scores_budget_compare.md
   ```

## Next milestones
- add external benchmark score components
- keep weights explicit and fixed at first
- add expert calibration data and adversarial realism cases
- later, build adaptive org-specific benchmark setup on top of this structure

## Working principle
This project is no longer just a toy benchmark. It has a legitimate core benchmark, a modular scoring layer, and a clear path to a broader benchmark-of-benchmarks architecture without rewriting the project.
