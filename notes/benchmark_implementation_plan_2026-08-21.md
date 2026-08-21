# Nonprofit Benchmark Expansion and Rigor Plan

## Source plan
This file consolidates the original benchmark expansion plan from [notes/benchmark_plan_2026-08-07.md](notes/benchmark_plan_2026-08-07.md) with the implementation notes captured during the first rigor pass.

## Original plan goals

### 1. Make the current benchmark more rigorous
- Improve both objective scoring and LLM-judge scoring so they better reflect real nonprofit workflows.
- Use more realistic evaluation signals, especially for nonprofit tasks where practical decision-making matters more than generic reasoning.
- Treat the current dataset as a starting point, while acknowledging that real-world nonprofit data is still limited.

### 2. Turn the project into a "benchmark of benchmarks"
- Incorporate existing external benchmarks and evaluation frameworks where appropriate.
- Use them as scoring components or reference signals.
- Combine them into a final evaluation through weighted scoring.
- Start with fixed weights and a simple architecture.

### 3. Create a more adaptive evaluation experience
- A nonprofit worker or manager should be able to input their organization context.
- A specialized agent should help interpret that input, ask follow-up questions, and refine the benchmark setup.
- The system should use that information to:
  - generate or structure the benchmark prompt/tasks for the models, and
  - assign preference-based weights or priorities relevant to the specific nonprofit context.

## Proposed product direction
### User experience
- A nonprofit user provides organization context and needs.
- A specialized agent asks a short set of clarifying questions.
- The agent translates that information into benchmark inputs and evaluation priorities.

### Benchmark design direction
- Make prompts and scoring more realistic for nonprofit decision-making.
- Expand beyond static task prompts toward more context-aware evaluation.
- Use both deterministic and human/LLM judge signals, but make them more robust and realistic.

## Strategic purpose
- Improve benchmark credibility for researchers and AI companies.
- Make the evaluation more compelling to external stakeholders.
- Push the project beyond a toy benchmark toward a more serious nonprofit AI evaluation platform.

## Implementation status as of 2026-08-21

### Completed baseline work
- Added task-bank validation and benchmark health checks.
- Added a CLI validation command: `python main.py validate`.
- Added task-specific realism checks to objective scoring.
- Added explicit high-risk safeguards for privacy, compliance, and fairness cases.
- Added a flexible weighted-scoring configuration to support future benchmark-of-benchmarks composition.
- Updated documentation in README.md so workflow and architecture are explained as the project evolves.

### Why this matters
The current benchmark is still a starting point, but the project now has a better base structure for rigorous evaluation and future extension.

## Prioritized next ideas for benchmark rigor

### 1. Domain-specific realism checks
- Score against actual nonprofit execution needs, not just generic reasoning.
- Require budget, staffing, timeline, governance, and operational constraints when relevant.
- Penalize answers that sound polished but are not executable in real nonprofit settings.

### 2. Adversarial quality testing
- Build benchmark variants designed to catch generic, template-like answers that pass fake keyword checks but fail in realistic nonprofit judgment.
- Explicitly test whether responses are operationally useful rather than rhetorically fluent.

### 3. Expert calibration layer
- Assemble a small set of expert-reviewed task responses.
- Use that set to tune judge prompts and grade thresholds before broader leaderboard runs.
- Calibrate for realistic nonprofit tradeoffs rather than abstract “best answer” patterns.

### 4. Multi-signal evaluation
- Combine objective score + judge score + external benchmark components using weighted composition.
- Keep the architecture modular so new signals can be added without rewriting the evaluator.

### 5. Context-heavy task variants
- Add organization context, staffing constraints, budget limits, and risk profile into tasks.
- Create evaluation tasks that require the model to adapt to real nonprofit conditions instead of using one-size-fits-all advice.

## Recommended next milestones
1. Create context-aware task variants.
   - Add org profile metadata and scenario constraints.
   - Ensure task scoring responds to the context rather than a static generic prompt.

2. Add expert calibration data.
   - Build a small labeled set for review and tuning.
   - Use this to refine judge instructions and scoring thresholds.

3. Add adversarial benchmark cases.
   - Create edge cases that distinguish generic fluency from operational realism.

4. Extend weighted score components.
   - Add a small external benchmark or reference signal layer behind the scoring config.
   - Keep the weights configurable and explicit.

5. Revisit adaptive evaluation experience.
   - Once the benchmark is stable, build context collection and benchmark tailoring around the scoring framework.

## Risk management and rollback guidance
- This project is still in an exploratory stage and benchmark rigor work may require iterative revisions.
- If a future implementation fails or significantly destabilizes the current framework, revert to the last known-good commit and reapply the next smallest change.
- Keep this plan and the original plan in notes so the team can resume from a clear design baseline.

## Current working principle
The benchmark should evolve from a useful nonprofit capability comparison into a more realistic, context-aware, and credible evaluation system, while keeping the architecture modular enough to support a future benchmark-of-benchmarks layer.
