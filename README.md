# Good Model Labs Benchmark Framework

This project compares different AI implementations for the same nonprofit task, using the same test data and the same scoring rules.

It is designed for product reviewers, project managers, and technical teams who need a clear answer to:

- Which system performs best for this task?
- What trade-offs do we see in quality, speed, and cost?

For a fuller walkthrough in plain language, see [GOOD_MODEL_LABS_BENCHMARK.md](GOOD_MODEL_LABS_BENCHMARK.md).

## What This Project Is And Is Not

This project is:

- A system comparison tool
- A repeatable evaluation process
- A way to justify baseline system choices with metrics

This project is not:

- A direct measure of nonprofit real-world outcomes
- A deployment analytics tool

## Simple Workflow Overview

When you run the benchmark:

1. It loads one dataset for a specific task.
2. It runs multiple AI systems on the same examples.
3. It scores outputs with a shared metric set.
4. It ranks systems with configurable weights.
5. It outputs a leaderboard and diagnostic artifacts.

## What A Workflow Means Here

In this project, a workflow is one complete way to solve the task from input to final output.

For example, a workflow can include:

- Prompting the model
- Applying rules or cleanup steps
- Running safety checks
- Formatting the final JSON output

Why this matters:

- Each workflow is a candidate solution strategy.
- The benchmark compares workflows to decide which strategy should become the baseline.
- A higher-ranked workflow is not just "a better prompt"; it is a better end-to-end process.

### Workflow Example (Plain Language)

Input example:

- "Client Maria Lopez called. Family of 4. Needs groceries and diapers. Spanish preferred."

One workflow might do this:

1. Read the note text.
2. Extract key fields such as name, household size, urgency, language, and requested services.
3. Apply cleanup rules (for example, fix missing fields or normalize wording).
4. Return a structured JSON record.

Output example:

```json
{
	"client_name": "Maria Lopez",
	"household_size": 4,
	"urgency": "high",
	"requested_services": ["groceries", "diapers"],
	"preferred_language": "Spanish",
	"notes_summary": "High-need household of 4 requesting groceries, diapers."
}
```

In this repository, Workflow A, B, and C follow this same input/output goal but use different strategies, which is why they score differently.

## Current Built-In Example

Task:

- Food Pantry Intake Structuring

Dataset:

- ID: food_pantry_intake_v1
- Size: 5 examples

Systems:

- Food Pantry Intake Workflow A
- Food Pantry Intake Workflow B
- Food Pantry Intake Workflow C
- Food Pantry Intake Workflow D

Model defaults:

- Workflow model: gpt-4o-mini
- Judge model: gpt-4o-mini

## Scoring Categories

The framework includes metrics across several themes:

- Quality: accuracy, precision, recall, hallucination_rate
- Reliability: json_validity, failure_rate
- Safety and robustness: prompt_injection, pii_leakage, robustness
- Efficiency: latency_ms, cost_usd, prompt_tokens, completion_tokens
- Cross-slice consistency: cross_org, cross_language, cross_model
- Human-like evaluation: llm_judge_quality

## Speed And Concurrency

To reduce benchmark runtime, the framework runs several operations in parallel with safe limits:

- Per-example execution concurrency in Runner: default 4
- Per-system ranking concurrency in Leaderboard: default 3
- LLM judge concurrency: default 4

## Setup

The framework auto-loads a local .env file from repository root.

Minimum real-provider setup:

```bash
OPENAI_API_KEY=your_key_here
```

Supported provider keys:

- OPENAI_API_KEY
- ANTHROPIC_API_KEY
- GOOGLE_API_KEY

If no real provider key is found, the framework uses MockProvider for local testing.

## Run

Run benchmark:

```bash
python main.py
```

Run tests:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## Outputs For Reviewers

Primary output:

- Leaderboard table with rank, score, quality, speed, and cost columns

Diagnostics:

- artifacts/traces: step-level execution traces
- artifacts/failure_logs: detailed failed-example records

These make it easier to explain why a system won or lost.

## Project Structure

```text
.
├── benchmark.py
├── runner.py
├── workflow.py
├── providers.py
├── dataset.py
├── metrics.py
├── evaluators.py
├── leaderboard.py
├── reports.py
├── registry.py
├── workflows/
├── datasets/
├── examples/
├── tests/
└── main.py
```

## Extending To New Tasks

To add a new benchmarked task:

1. Add a dataset module in datasets.
2. Add one or more system modules in workflows.
3. Register them.
4. Choose metric set and ranking weights.
5. Run benchmark and compare results.

## Adding A New Workflow Variant

To add another distinct workflow for an existing task:

1. Add a new workflow strategy class in workflows/food_pantry_intake.py.
2. Give that strategy its own behavior, such as:
	- prompt style
	- conservative vs balanced behavior
	- stronger source-text corrections
	- optional verification for risky cases
3. Add the strategy to the WORKFLOW_STRATEGIES registry.
4. Add a small registered system class that points to that strategy.
5. Re-run the benchmark and compare results.

This keeps new workflow variants easy to add without rewriting the whole pipeline or being limited to a fixed set of boolean switches.

Example added in this repository:

- Workflow D uses a recall-heavy strategy class to show how a new workflow can be created by adding a new strategy plus a small registered system class.

### Primary WorkflowStrategy Methods

When creating a new workflow strategy, these are the main methods to understand:

1. system_prompt()
	Defines the active system instructions sent to the model.

2. llm_prompt(safe_input)
	Builds the user-side task prompt from the input text.

3. sanitize_input(text)
	Lets the workflow clean or transform raw input before model use.

4. normalize_output(system, parsed, safe_input)
	Main post-processing hook. Turns model output into the final structured result.

5. fallback_output(system, safe_input)
	Defines what to return if the model call fails.

6. should_verify(system, safe_input, merged)
	Decides whether to run an extra review or repair pass.

7. verification_prompt(safe_input, merged)
	Builds the prompt for that review or repair pass.

8. verification_system_prompt()
	Defines the system instructions for the verification step.

In practice, the most important methods for making a workflow truly distinct are usually:

1. system_prompt()
2. llm_prompt(...)
3. normalize_output(...)
4. should_verify(...)
